"""export a clip vision encoder to onnx for the stage-2 classifier.

offline tooling, like `export_model.py`. the runtime never needs python.

this is the backbone that stays frozen while the head on top of it changes: at
first a nearest-neighbour match against a handful of reference crops, later a
trained head over the same cached embeddings. see DESIGN.md.

measured before choosing this route: on crops cut by the detector, a stock clip
vit-b/32 puts 13 of 15 held-out go-4s nearer another go-4 than any car, and that
survives degrading the references to this camera's fidelity.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

# clip's native input. the vision tower is trained at this size and does poorly
# away from it, so it is not a tuning knob.
IMGSZ = 224


def export(model_id: str, out: Path) -> Path:
    import torch
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(model_id).eval()

    class VisionOnly(torch.nn.Module):
        """just the image tower, l2 normalised.

        the text tower is dead weight here: metermate compares images against
        images, never against prompts. normalising inside the graph means the
        rust side can use a plain dot product as cosine similarity.
        """

        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, pixel_values):
            f = self.clip.get_image_features(pixel_values=pixel_values)
            if not torch.is_tensor(f):
                f = f.pooler_output
            return torch.nn.functional.normalize(f, dim=-1)

    wrapper = VisionOnly(model)
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
    tmp = out.parent / f".{out.name}.tmp"
    torch.onnx.export(
        wrapper,
        dummy,
        str(tmp),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,
    )
    # shutil.move, not rename: the destination is often on another filesystem.
    shutil.move(str(tmp), str(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", type=Path, default=MODELS / "embedder.onnx")
    # int8 is opt-in and NOT recommended here, see below.
    ap.add_argument("--int8", action="store_true", default=False)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        print(f"{args.out} already exists, nothing to do")
        return 0

    print(f"exporting {args.model} vision tower at {IMGSZ}x{IMGSZ} -> {args.out}")
    path = export(args.model, args.out)
    print(f"  fp32: {path.stat().st_size / 1e6:.0f} MB")

    if args.int8:
        # tempting and wrong for this task. it does shrink the model 352 -> 88 MB
        # and halve inference from 52ms to 25ms, and the embeddings agree with
        # fp32 at mean cosine 0.995. but the go-4/car margin is only about 0.02,
        # so a 0.005 perturbation eats a quarter of it: measured on the real
        # task, held-out accuracy falls from 14/15 to 12/15 and the margin
        # collapses from 0.022 to 0.006. aggregate agreement lied.
        from onnxruntime.quantization import QuantType, quantize_dynamic

        tmp = path.parent / f".{path.name}.int8"
        quantize_dynamic(str(path), str(tmp), weight_type=QuantType.QInt8)
        shutil.move(str(tmp), str(path))
        print(f"  int8: {path.stat().st_size / 1e6:.0f} MB")

    import onnx

    model = onnx.load(str(path))
    for o in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  output {o.name}: {dims}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
