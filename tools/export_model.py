"""fetch a coco detector and export it to onnx for the rust runtime.

offline tooling. the runtime never depends on python or torch; this runs once to
produce `models/detector.onnx` and is not part of the hot path.

model choice is a deliberate tradeoff, see DESIGN.md. the ultralytics models are
agpl-3.0, which is viral if metermate is published, so the permissive default is
preferred and the ultralytics path is opt-in.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
# the upstream checkpoints an export starts from, kept together beside the two
# models they produce rather than left wherever the export happened to run.
UPSTREAM = MODELS / "upstream"

# input size for the detector. 640 is what coco models are trained at; smaller
# is faster but loses the far side of the street, which is where the vehicle we
# care about usually appears first.
IMGSZ = 640


def export_ultralytics(name: str, out: Path, nms: bool, iou: float) -> Path:
    from ultralytics import YOLO

    # ultralytics resolves a bare name against the working directory and
    # downloads there when it is missing, which drops a 20-80 MB checkpoint into
    # the repo root and makes the next export fetch it again. prefer the kept
    # copy when there is one.
    kept = UPSTREAM / f"{name}.pt"
    model = YOLO(str(kept) if kept.exists() else f"{name}.pt")
    produced = model.export(
        format="onnx",
        imgsz=IMGSZ,
        simplify=True,
        dynamic=False,
        nms=nms,
        # class-agnostic on purpose. per-class nms cannot suppress a `car` box
        # and a `suitcase` box sitting on the same van, and metermate does not
        # use the coco label for anything: stage 2 decides what a vehicle is.
        # doing this in the graph keeps the rust side free of a dedup pass.
        agnostic_nms=nms,
        iou=iou,
    )
    # shutil.move, not Path.replace: the destination is frequently on a
    # different filesystem than the working directory, and rename(2) cannot
    # cross devices.
    shutil.move(str(produced), str(out))
    # a checkpoint just downloaded into the working directory joins the others,
    # so only the first export on a machine pays for it.
    stray = Path(f"{name}.pt")
    if stray.exists():
        UPSTREAM.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stray), str(kept))
    return out


def describe_output(path: Path) -> None:
    """report whether the export kept its nms-free head.

    yolo26 emits final detections directly, so the rust side needs no nms at all.
    that property is known to be lost in some exports, and a silently reinstated
    nms requirement shows up as duplicate overlapping boxes rather than an error,
    so check it here rather than discover it later.
    """
    import onnx

    model = onnx.load(str(path))
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  output {out.name}: {dims}")
        # [1, N, 6] is one row per detection: already final.
        # [1, 84, 8400] is the classic dense head and needs nms downstream.
        if len(dims) == 3 and dims[-1] == 6:
            print("  -> nms-free: rust needs no postprocessing beyond thresholding")
        elif len(dims) == 3 and dims[1] in (84, 116):
            print("  -> dense head: rust must implement nms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # yolo26s, not the nano. measured on a frame from this camera, nano scored
    # 0.70/0.51/0.40/0.35 on four vehicles and missed the silver sedan in the
    # foreground entirely -- the largest and most obvious car in the shot -- and
    # drew the van's box around a stretch of road. small found six at
    # 0.88/0.87/0.86/0.83/0.80, every one above 0.7, with tight boxes, for 2x
    # the inference cost. the detector is motion-gated and rate-capped, so it
    # idles at zero either way and the accuracy is worth the gated cost.
    ap.add_argument("--model", default="yolo26s", help="ultralytics model name")
    ap.add_argument("--out", type=Path, default=MODELS / "detector.onnx")
    # yolo26 is advertised as nms-free, but the exported graph came out as a
    # dense [1, 84, 8400] head anyway. baking nms into the graph gets the
    # intended result: final boxes out, no postprocessing in rust.
    ap.add_argument("--no-nms", dest="nms", action="store_false", default=True)
    # the released yolo26 weights report end2end=False, so the advertised
    # nms-free head is not available to us; forcing it on would run an untrained
    # branch. graph nms is the supported path, and this is its iou threshold.
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        print(f"{args.out} already exists, nothing to do")
        return 0

    print(
        f"exporting {args.model} at {IMGSZ}x{IMGSZ} "
        f"(nms={args.nms}, agnostic, iou={args.iou}) -> {args.out}"
    )
    path = export_ultralytics(args.model, args.out, args.nms, args.iou)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    describe_output(path)
    print("note: ultralytics models are agpl-3.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
