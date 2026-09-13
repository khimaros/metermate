"""find the moments a vehicle actually transits the scene, for hand labelling.

the recorded clips are mostly parked cars. sampling them at fixed intervals
measures the same stationary row over and over, which says nothing about the
thing the system exists to do -- notice a vehicle arriving. this pulls out the
frames where something is genuinely moving, so a person labels a few dozen real
passages instead of hundreds of redundant frames.

the motion scan is deliberately **independent of metermate** (see `motion.py`):
an event list drawn from the detector could not contain a vehicle the detector
never saw, which is exactly the quantity being measured.

an entry is one **moving object**, not one stretch of time. an earlier version
emitted one entry per event and boxed the largest blob in its peak frame, which
silently dropped every other vehicle moving at the same time -- and a clip of a
city intersection has them constantly. on one ten minute clip that hid 171 real
transits, all of which metermate had cropped correctly and all of which scored
as false positives for having done so.

    uv run tools/label.py data/eval/x-sub.mp4 data/eval/x-main.mp4 out/

writes `out/manifest.json` plus a clean and a boxed image per object. fill in
the `truth` field of each entry by looking at the clean image, then run
score.py. pass `--from` an existing manifest to carry its decisions over.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from motion import background, blobs, boxes_overlap, motion_signal, scan_frames
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "detector.onnx"

SIZE = 640
CONF = 0.25
# coco classes the pipeline acts on. anything else it says is discarded, so a
# `bench` drawn over a car counts as a miss, not as a different sort of hit.
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# this fraction of the frame must change before it is called motion. tuned so a
# car crossing registers and a tree moving in wind does not.
MOTION_FRAC = 0.004
# events closer together than this are one stretch of motion. they are no longer
# one *vehicle*: the blobs inside are tracked separately, so merging generously
# costs nothing and avoids splitting one passage into three.
MERGE_GAP_S = 2.0
# and anything shorter than this is a flicker.
MIN_EVENT_S = 0.4

# how often to look inside an event. an object crossing the frame in two seconds
# is seen ten times at this rate, which is plenty to link it up and cheap
# because the scan is already decoded.
SAMPLE_STRIDE_S = 0.2
# an object must be seen in at least this many samples. one sighting is a glint
# off a windscreen or a pedestrian's head appearing between parked cars.
MIN_LOOKS = 2


def objects(scan: np.ndarray, bg: np.ndarray, a: int, b: int, fps: float,
            main_w: int, main_h: int) -> list[dict]:
    """one entry per thing that moved during `[a, b)`, not one per event.

    blobs are linked across samples by overlap, which is enough here: the scan
    is 160x120, vehicles are tens of pixels across, and consecutive samples are
    200ms apart, so a car's silhouette always overlaps its own previous one. two
    cars passing in opposite directions briefly merge into one blob and then
    separate; that shows up as one track ending and two beginning, which
    over-counts rather than under-counts, and over-counting is the safe error
    for a list a person is about to review.
    """
    stride = max(1, int(SAMPLE_STRIDE_S * fps))
    tracks: list[dict] = []
    for i in range(a, b, stride):
        if i >= len(scan):
            break
        for box in blobs(scan, bg, i, main_w, main_h):
            area = (box[2] - box[0]) * (box[3] - box[1])
            # the newest track it continues, so a car does not get stolen by a
            # stale track it happens to drive over.
            hit = next((t for t in reversed(tracks)
                        if t["last"] >= i - stride and boxes_overlap(t["box"], box)), None)
            if hit is None:
                tracks.append({"first": i, "last": i, "looks": 1, "box": box,
                               "peak": i, "peak_area": area, "peak_box": box})
                continue
            hit["last"], hit["looks"], hit["box"] = i, hit["looks"] + 1, box
            # the frame where it is biggest is where it is clearest, which is
            # the one a person should be shown.
            if area > hit["peak_area"]:
                hit["peak"], hit["peak_area"], hit["peak_box"] = i, area, box
    return [
        {"first": t["first"], "last": t["last"], "looks": t["looks"],
         "peak": t["peak"], "box": t["peak_box"], "peak_area": t["peak_area"]}
        for t in tracks if t["looks"] >= MIN_LOOKS
    ]


def events(signal: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """contiguous runs of motion, merged, as (start, end) frame indices.

    a run says only "the frame was changing here". which things were changing,
    and how many, is `objects`.
    """
    hot = signal >= MOTION_FRAC
    runs, start = [], None
    for i, on in enumerate(hot):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(hot)))

    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) / fps <= MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if (b - a) / fps >= MIN_EVENT_S]


def carry_over(old: list[dict], t: float, box) -> str | None:
    """the `truth` a person already set for this object, if they have.

    re-running the labeller must not throw away the decisions, or nobody will
    re-run it. matched on time and position rather than on the event number,
    which changes the moment the event list does.
    """
    for e in old:
        if e.get("truth") is None or abs(e["t"] - t) > MERGE_GAP_S:
            continue
        if e.get("moved_box") and boxes_overlap(e["moved_box"], box):
            return e["truth"]
    return None


def centre_inside(box, region) -> bool:
    """is this detection *about* the thing that moved?

    plain overlap is too generous. the motion box spans where a car was and
    where it got to, so it clips whatever is parked alongside, and a passing car
    scored as three detections -- itself, the van beside it and the jeep behind.
    a detection counts only if its own centre sits in the moving region.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def letterbox(img: Image.Image):
    w, h = img.size
    scale = min(SIZE / w, SIZE / h)
    dw, dh = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(img.resize((dw, dh), Image.BILINEAR),
                 ((SIZE - dw) // 2, (SIZE - dh) // 2))
    return canvas, scale, (SIZE - dw) // 2, (SIZE - dh) // 2


def detect(sess, img: Image.Image) -> list[dict]:
    """the pipeline's whole-frame pass, on one frame."""
    canvas, scale, ox, oy = letterbox(img)
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    out = sess.run(None, {"images": np.transpose(a, (2, 0, 1))[None]})[0]
    hits = []
    for r in out.reshape(-1, 6):
        if r[4] < CONF:
            continue
        hits.append({
            "cls": int(r[5]),
            "label": VEHICLES.get(int(r[5])),
            "conf": round(float(r[4]), 3),
            "box": [round((r[0] - ox) / scale), round((r[1] - oy) / scale),
                    round((r[2] - ox) / scale), round((r[3] - oy) / scale)],
        })
    return hits


def grab(clip: Path, t: float, dest: Path) -> Image.Image:
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(clip),
                    "-frames:v", "1", "-y", str(dest)], check=True)
    return Image.open(dest).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sub", type=Path, help="substream clip, for the motion scan")
    ap.add_argument("main", type=Path, help="main stream clip, for the images")
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL)
    # no default cap. an earlier version kept the 40 strongest events and said
    # so in one line of output that nobody read again; two manifests then sat at
    # exactly 40 entries, and every transit past the cap counted as a false
    # positive when metermate cropped it.
    ap.add_argument("--max-objects", type=int, default=0,
                    help="keep only the N with most motion (0 = all)")
    ap.add_argument("--from", dest="previous", type=Path,
                    help="an existing manifest whose `truth` decisions to carry over")
    args = ap.parse_args()

    fps = float(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(args.sub)],
        capture_output=True, text=True, check=True).stdout.strip().split("/")[0]) / 1.0
    denom = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(args.sub)],
        capture_output=True, text=True, check=True).stdout.strip().split("/")[1]
    fps = fps / float(denom)

    print(f"scanning {args.sub.name} for motion at {fps:.2f} fps ...", flush=True)
    scan = scan_frames(args.sub)
    signal = motion_signal(scan)
    bg = background(scan)
    found = events(signal, fps)

    # the main stream's size decides the coordinate space every box is in.
    args.out.mkdir(parents=True, exist_ok=True)
    probe = args.out / "probe.png"
    main_w, main_h = grab(args.main, 0.0, probe).size
    probe.unlink()

    seen = [o for a, b in found for o in objects(scan, bg, a, b, fps, main_w, main_h)]
    seen.sort(key=lambda o: o["peak"])
    print(f"{len(found)} stretches of motion over {len(signal) / fps:.0f}s, "
          f"holding {len(seen)} moving objects")
    if args.max_objects and len(seen) > args.max_objects:
        dropped = len(seen) - args.max_objects
        seen = sorted(seen, key=lambda o: -o["peak_area"])[: args.max_objects]
        seen.sort(key=lambda o: o["peak"])
        print(f"  WARNING: dropping {dropped} objects to honour --max-objects. "
              f"anything metermate crops from them will score as a false positive.")

    previous = json.loads(args.previous.read_text()) if args.previous else []
    if previous:
        print(f"carrying decisions over from {args.previous.name} "
              f"({sum(e.get('truth') is not None for e in previous)} labelled)")

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    manifest, kept = [], 0
    for i, o in enumerate(seen, 1):
        t = o["peak"] / fps
        mbox = o["box"]
        raw = args.out / f"e{i:03d}.png"
        img = grab(args.main, t, raw)
        hits = detect(sess, img)
        veh = [h for h in hits if h["label"]]

        # the clean image carries the motion box and nothing else, so the
        # labeller is told where to look without being told what was found.
        clean = img.copy()
        ImageDraw.Draw(clean).rectangle(mbox, outline=(80, 170, 255), width=6)
        boxed = clean.copy()
        d = ImageDraw.Draw(boxed)
        for h in hits:
            colour = (60, 220, 90) if h["label"] else (220, 70, 70)
            d.rectangle(h["box"], outline=colour, width=5)
            d.text((h["box"][0] + 5, max(0, h["box"][1] - 18)),
                   f"{h['label'] or 'non-veh'} {h['conf']:.2f}", fill=colour)
        half = (img.size[0] // 2, img.size[1] // 2)
        clean.resize(half).save(args.out / f"e{i:03d}_clean.png")
        boxed.resize(half).save(args.out / f"e{i:03d}_boxed.png")

        # a close crop of just the moving region, with context. the labelling
        # question is only "is that a vehicle", and answering it from a full
        # frame means hunting for a small blue box in a busy street scene.
        pad = max((mbox[2] - mbox[0]), (mbox[3] - mbox[1])) // 3
        img.crop((max(0, mbox[0] - pad), max(0, mbox[1] - pad),
                  min(img.size[0], mbox[2] + pad),
                  min(img.size[1], mbox[3] + pad))).save(args.out / f"e{i:03d}_moved.png")
        raw.unlink()

        # did any vehicle box land on the thing that moved? this is the whole
        # question, and it is decided here rather than by eye so that the
        # labelling reduces to "was that actually a vehicle".
        covering = [h for h in veh if centre_inside(h["box"], mbox)]
        truth = carry_over(previous, t, mbox)
        kept += truth is not None
        manifest.append({
            "event": i,
            "t": round(t, 2),
            "span_s": round((o["last"] - o["first"]) / fps, 1),
            "motion": round(float(signal[min(o["peak"], len(signal) - 1)]), 4),
            "looks": o["looks"],
            "moved_box": mbox,
            "detected_on_moved": [{"label": h["label"], "conf": h["conf"],
                                   "w": h["box"][2] - h["box"][0]} for h in covering],
            "detected_total": len(veh),
            "non_vehicle_boxes": [{"cls": h["cls"], "conf": h["conf"]}
                                  for h in hits if not h["label"]],
            # set by a person from the clean image: is the blue box a vehicle?
            # one of "vehicle", "not-a-vehicle", "unclear".
            "truth": truth,
        })
        print(f"  e{i:03d} t={t:6.1f}s  looks={o['looks']:>2}  "
              f"{len(covering)} detections on the moving thing"
              f"{'  [truth carried over]' if truth else ''}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out / 'manifest.json'}: {len(manifest)} objects, "
          f"{kept} already labelled, {len(manifest) - kept} to review")
    print("label each entry's `truth` from the *_clean.png images, then score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
