"""sweep tuning parameters against the end-to-end eval, in parallel.

every threshold in the harvest was picked from a handful of logged numbers on a
few minutes of one street. this replays a labelled clip once per setting and
ranks them on the only measure that matters: **what fraction of vehicles that
actually crossed ended up in the harvest**.

one parameter at a time around the defaults, not a full grid. a grid over six
parameters is thousands of replays at twenty-odd minutes each, and the point is
to find which knobs move the number at all before spending that.

    uv run tools/sweep.py tests/e2e/fixtures/labels/<stamp>.json \\
        data/eval/<stamp>-sub.mp4 --jobs 5

each replay runs the detector on two threads, so keep jobs well under the core
count or they fight each other and every run measures contention instead.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endtoend import CROP_FPS, harvested, precision, replay, score

REPO = Path(__file__).resolve().parents[1]

# the knobs worth asking about, with the default first in each list so the
# baseline is always part of the sweep rather than remembered from elsewhere.
GRID: dict[str, list] = {
    # the over/under-capture dial.
    "harvest.must_have_moved": [0.08, 0.02, 0.04, 0.12, 0.20],
    # how often the same place may be re-cropped.
    "harvest.min_interval_secs": [5, 1, 2, 10, 20],
    # how many vehicles one frame may yield.
    "harvest.max_per_frame": [4, 1, 2, 8],
    # skew between the gate frame and the main frame the crop is cut from.
    # costs pipe bandwidth only: ffmpeg decodes every frame regardless.
    "stream.crop_fps": [4, 2, 8, 15],
    # the gate's own sensitivity. measured noise floor on a static scene is
    # 0.108%, so the default sits under 2x above it.
    "gate.min_changed_frac": [0.002, 0.001, 0.004, 0.008],
    # what counts as the same place for the rate limit.
    "harvest.same_object_tolerance": [0.5, 0.25, 0.75],
    # where a vehicle's movement has to sit inside its own box. added to stop a
    # car passing a parked one crediting the parked one with the movement, and
    # measured live afterwards turning away 43 vehicles that had moved while the
    # harvest accepted 18 -- so the threshold needs a curve rather than one
    # number. 0.0 is the filter off, which is how it behaved before it existed.
    "harvest.motion_must_be_central": [0.20, 0.0, 0.05, 0.10, 0.30],
    # how long a place stays known once nothing is seen there. forgetting now
    # needs both wall time and missed looks, which made scenery stickier and has
    # never been measured against recall.
    "scenery.forget_after_looks": [200, 50, 100, 400],
    # share of looks a place must be occupied in before it is scenery. a spot
    # traffic crosses often could accumulate this and start vetoing the traffic.
    "scenery.min_occupancy": [0.5, 0.3, 0.7, 0.9],
}


def one(job: tuple) -> dict:
    key, value, labels, clip, crop_clip, fps, binary, outdir = job
    section, _, name = key.partition(".")
    overrides = {section: {name: value}}
    tag = f"{key}={value}"
    log = Path(outdir) / f"{key.replace('.', '_')}_{value}.log"
    output, crops = replay(Path(clip), Path(binary), overrides, log,
                           Path(crop_clip) if crop_clip else None)
    result = score(labels, output, fps,
                   crop_fps=overrides.get("stream", {}).get("crop_fps", CROP_FPS))
    # ranking on recall alone picks whichever setting filters least, every time.
    # this is what that costs, judged by the motion scan rather than by the
    # labels, so a setting cannot win by cropping vehicles nobody labelled.
    clean = precision(harvested(output), Path(clip), fps)
    rows = result["rows"]
    n = max(len(rows), 1)
    total = clean["on_motion"] + clean["off_motion"]
    return {
        "param": key,
        "value": value,
        "tag": tag,
        "transits": len(rows),
        "detected": sum(r["detected"] for r in rows),
        "harvested": sum(bool(r["harvested"]) for r in rows),
        "recall": sum(bool(r["harvested"]) for r in rows) / n,
        "off_motion": clean["off_motion"],
        "precision": clean["on_motion"] / max(total, 1),
        "crops": crops,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path)
    ap.add_argument("clip", type=Path, help="substream clip, for the gate")
    ap.add_argument("--crop-source", type=Path, help="main-stream clip for crops")
    ap.add_argument("--binary", type=Path, default=REPO / "target" / "release" / "metermate")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", action="append", help="sweep only this parameter (repeatable)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/metermate-sweep"))
    args = ap.parse_args()

    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(args.clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    num, den = raw.split("/")
    fps = float(num) / float(den)

    labels = json.loads(args.labels.read_text())
    args.workdir.mkdir(parents=True, exist_ok=True)

    grid = {k: v for k, v in GRID.items() if not args.only or k in args.only}
    # crops must come from main-stream pixels, or every replay measures a
    # quarter-size pipeline rather than the one that runs.
    guess = Path(str(args.clip).replace("-sub.", "-main."))
    crop_clip = args.crop_source or (guess if guess != args.clip and guess.exists() else None)
    if crop_clip is None:
        print("WARNING: no main-stream clip; crops cut from the substream", file=sys.stderr)

    jobs = [
        (key, value, labels, str(args.clip), str(crop_clip) if crop_clip else "",
         fps, str(args.binary), str(args.workdir))
        for key, values in grid.items()
        for value in values
    ]
    print(f"{len(jobs)} replays of {args.clip.name} at {fps:.1f}fps, {args.jobs} at a time")
    print(f"logs under {args.workdir}\n", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for r in pool.map(one, jobs):
            results.append(r)
            print(f"  {r['tag']:<38} recall {r['harvested']:>3}/{r['transits']:<3} "
                  f"= {r['recall'] * 100:>3.0f}%   precision {r['precision'] * 100:>3.0f}%"
                  f"   {r['crops']} crops", flush=True)

    print(f"\n{'parameter':<32} {'value':>8} {'recall':>12} "
          f"{'precision':>10} {'crops':>7}")
    # ranked by recall, then by precision. a tie on recall is broken by the
    # setting that took fewer pictures of a street where nothing was happening.
    for key in grid:
        ranked = sorted((x for x in results if x["param"] == key),
                        key=lambda x: (-x["recall"], -x["precision"]))
        for r in ranked:
            print(f"{key:<32} {r['value']!s:>8} "
                  f"{r['harvested']:>4}/{r['transits']:<3} {r['recall'] * 100:>3.0f}% "
                  f"{r['precision'] * 100:>9.0f}% {r['crops']:>7}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
