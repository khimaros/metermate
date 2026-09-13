"""score the whole pipeline against hand-labelled transits, not just the detector.

`score.py` answers "when a vehicle crossed, did yolo see it". it said 105/105
while a bug was live that halved multi-vehicle capture, because everything after
the detector -- `must_have_moved`, `scenery`, the per-place gap, and the skew
between the gate frame and the main frame the crop is cut from -- happens
downstream of the question it asks.

this asks the question that matters instead: **when a vehicle crossed, did a crop
of it reach the harvest.** same clips, same labels, no new recording.

    uv run tools/endtoend.py tests/e2e/fixtures/labels/<stamp>.json \\
        data/eval/<stamp>-sub.mp4

replays the clip with a throwaway harvest directory, reads the `harvested frame`
lines back, and matches them against each labelled transit by frame number.

**recall is only half of it.** every filter loosened to catch one more vehicle
also lets more of the street through, so a run is scored two ways:

- *recall*, against the hand-labelled transits: did a crop of each reach the
  harvest.
- *precision*, against the independent motion scan in `motion.py`: was each crop
  taken of something that was actually moving, or of a parked car.

precision deliberately does **not** use the labels. an earlier version counted a
crop as a false positive if it fell outside any labelled transit window, which
made it a measurement of the labeller: `label.py` emitted one entry per event
and boxed only the largest blob, so 171 correctly cropped vehicles on one clip
scored as false positives for being the second car in shot. asking the motion
scan directly needs no labels and cannot be truncated by them.
"""

import argparse
import json
import re
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "target" / "release" / "metermate"

# `harvested frame 412 car 0.88 at 210,325 157x127 moved=0.142`
HARVEST = re.compile(
    r"harvested frame (\d+) (\w[\w-]*) ([\d.]+) at (\d+),(\d+) (\d+)x(\d+) moved=([\d.]+)"
)
# the detector's own report: which gate frame, which main frame it was read
# from, and what was in it. the main frame number is not decoration -- the two
# feeds are separate decoders, and a replay where they drift apart scores a
# pipeline nobody runs. `report` checks it before believing anything else.
DETECTION = re.compile(r"frame (\d+) main (\d+) region Rect \{[^}]*\} -> (.+)")
# parked vehicles are counted rather than listed (`| 5 parked`), so what this
# matches is the ones the look actually reported.
VEHICLE = re.compile(r"(\w[\w-]*) ([\d.]+) moved=([\d.]+) box=(\d+),(\d+) (\d+)x(\d+)")
# `| 5 parked`, or `-> 5 parked, nothing moving` when that is all there was.
PARKED = re.compile(r"(\d+) parked")
# `frame 412 declined car 0.61 changed=0.0412 parked=false box=... below must_have_moved`
#
# the other side of the threshold. requires `RUST_LOG=metermate=debug`, since a
# busy frame declines most of what it detects and this would otherwise be the
# noisiest line in the log.
DECLINED = re.compile(
    r"frame \d+ declined \w[\w-]* [\d.]+ changed=([\d.]+) parked=(\w+) .* below must_have_moved"
)

# the whole-run report the binary prints at exit. recall says whether a vehicle
# was caught; this says what the pipeline cost to catch it, and a change that
# holds recall while halving the frame rate is still a regression.
STATS = re.compile(r"stats (\{.*\})")


def stats(output: str) -> dict:
    """the run's own account of itself, or an empty dict for an older binary."""
    m = None
    for m in STATS.finditer(output):
        pass
    if m is None:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def stats_lines(s: dict) -> list[str]:
    """the handful worth printing beside recall and precision."""
    if not s:
        return []
    i, p, o = s.get("input", {}), s.get("processing", {}), s.get("output", {})
    return [
        f"  stream seen   {i.get('stream_seen_pct', 0):.0f}%"
        f"   ({i.get('gate_frames', 0)} frames, {i.get('gate_dropped', 0)} dropped)",
        f"  restarts      gate {i.get('gate_restarts', 0)}, crop {i.get('crop_restarts', 0)}",
        f"  inference     {p.get('inference_ms', 0):.0f}ms"
        f"   x{p.get('inferences', 0)}   ({p.get('inferences_per_s', 0):.1f}/s)",
        f"  blind looks   {p.get('blind_looks', 0)}"
        f"   (motion with no main frame: nothing was detected)",
        f"  carrying      {s.get('carrying', {}).get('places', 0)} places,"
        f" {s.get('carrying', {}).get('tracks', 0)} tracks",
        f"  harvest       {o.get('harvested', 0)} crops,"
        f" {o.get('harvest_files', 0)} on disk ({o.get('harvest_mb', 0):.0f} MB)",
    ]


# a transit is a span of frames. a harvest counts for it if it lands inside that
# span, with a little slack: the crop is cut from whichever main frame is newest,
# which at the default crop_fps can be a quarter second behind the gate.
SLACK_FRAMES = 8

# how much of a crop's own box must be moving, in the independent scan, for the
# crop to be *about* something in motion. not a majority: a good crop of a car
# carries road and kerb around it on purpose, and the scan is coarse enough that
# a car's silhouette under-covers its bounding box.
ON_MOTION_FRAC = 0.15

# must track `config::CROP_FPS`. the skew check needs it to turn a main frame
# number back into a moment in the clip.
CROP_FPS = 15


def replay(clip: Path, binary: Path, overrides: dict[str, dict], log: Path,
           crop_clip: Path | None = None,
           keep_crops: Path | None = None,
           verbose: bool = False) -> tuple[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        crops = Path(tmp) / "crops"
        sections = {
            "camera": {"host": "127.0.0.1", "username": "u", "password": "p"},
            "stream": {"gate_subtype": 1, "gate_width": 640, "gate_height": 480},
            "detector": {"model": str(REPO / "models" / "detector.onnx")},
            "harvest": {"dir": str(crops)},
            "preview": {"source": "server-sub"},
        }
        for section, values in overrides.items():
            sections.setdefault(section, {}).update(values)

        lines = []
        for section, values in sections.items():
            lines.append(f"[{section}]")
            for k, v in values.items():
                lines.append(f"{k} = {json.dumps(v)}")
        cfg = Path(tmp) / "metermate.toml"
        cfg.write_text("\n".join(lines) + "\n")

        cmd = [str(binary), "--config", str(cfg), "--source", str(clip),
               "--dry-run", "--preview", "off", "--offline"]
        # crops must come from main-stream pixels or the replay measures a
        # pipeline nobody runs: a vehicle reaches the detector a quarter size.
        if crop_clip is not None:
            cmd += ["--crop-source", str(crop_clip)]
        with log.open("w") as sink:
            subprocess.run(
                cmd,
                stdout=sink, stderr=subprocess.STDOUT, check=False,
                # always debug: what a look found is a line per gated frame, so
                # it lives there rather than in a deployment's journal -- and
                # it is the line this whole tool is built on. `verbose` decides
                # whether the declined detections are *printed*, not whether
                # they are asked for.
                env={"RUST_LOG": "metermate=debug", "PATH": "/usr/bin:/bin"},
            )
        written = sorted(crops.glob("*.jpg"))
        # the crops are the other half of the answer. recall alone rewards every
        # loosened filter, including the ones that fill the harvest with the
        # parked van, so they have to survive the replay to be looked at.
        if keep_crops is not None:
            keep_crops.mkdir(parents=True, exist_ok=True)
            for old in keep_crops.glob("*.jpg"):
                old.unlink()
            for f in written:
                shutil.copy2(f, keep_crops / f.name)
        return log.read_text(), len(written)


def boxes_overlap(a, b) -> bool:
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])


def harvested(output: str) -> list[dict]:
    """every crop the replay kept, in gate coordinates."""
    return [
        {"frame": int(m.group(1)), "label": m.group(2), "conf": float(m.group(3)),
         "box": (int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))),
         "moved": float(m.group(8))}
        for m in HARVEST.finditer(output)
    ]


def score(labels: list[dict], output: str, fps: float,
          main_size: tuple[int, int] = (2560, 1440),
          gate_size: tuple[int, int] = (640, 480),
          crop_fps: float = CROP_FPS) -> dict:
    harvests = harvested(output)
    # what this replay's own detector reported, per frame, in gate coordinates.
    # previously the "detected" column came from the label file, which was a
    # different run's detector -- so the eval compared one run's detections with
    # another run's harvest.
    seen: dict[int, list] = {}
    # how far the main frame each inspection read was from the gate frame that
    # asked for it, in seconds of video. one crop interval is the floor; more
    # than that and the replay is scoring two different moments of the street.
    skew = []
    # **a listed vehicle is one that moved.** the log stopped spelling out the
    # parked ones -- on this street they are most of every frame and they said
    # the same thing every time -- and counts them instead, so "was everything
    # here vetoed as scenery" is now "nothing was listed and the count was not
    # zero" rather than a flag on each box.
    vetoed: dict[int, int] = {}
    for m in DETECTION.finditer(output):
        f = int(m.group(1))
        skew.append((f - 1) / fps - (int(m.group(2)) - 1) / crop_fps)
        said = m.group(3)
        count = PARKED.search(said)
        vetoed[f] = int(count.group(1)) if count else 0
        for v in VEHICLE.finditer(said):
            seen.setdefault(f, []).append({
                "box": (int(v.group(4)), int(v.group(5)), int(v.group(6)), int(v.group(7))),
                "moved": float(v.group(3)),
            })
    sx, sy = main_size[0] / gate_size[0], main_size[1] / gate_size[1]

    rows = []
    for e in labels:
        if e["truth"] != "vehicle":
            continue
        # the event's span, in frames, from the manifest's seconds.
        mid = e["t"] * fps
        half = (e["span_s"] * fps) / 2
        lo, hi = mid - half - SLACK_FRAMES, mid + half + SLACK_FRAMES
        got = [h for h in harvests if lo <= h["frame"] <= hi]

        # did *this* replay detect the moving vehicle, on its own box?
        mb = e.get("moved_box")
        here = []
        if mb:
            g = (int(mb[0] / sx), int(mb[1] / sy),
                 int((mb[2] - mb[0]) / sx), int((mb[3] - mb[1]) / sy))
            here = [d for f, ds in seen.items() if lo <= f <= hi
                    for d in ds if boxes_overlap(d["box"], g)]
        rows.append({
            "event": e["event"],
            "detected": bool(here),
            # the two ways a detected vehicle is then thrown away, which need
            # opposite fixes.
            "all_parked": not here and any(
                vetoed.get(f, 0) for f in range(int(lo), int(hi) + 1)
            ),
            "best_moved": max((d["moved"] for d in here), default=0.0),
            "harvested": len(got),
            "moved": max((h["moved"] for h in got), default=None),
        })
    return {
        "rows": rows,
        "harvests": len(harvests),
        "worst_skew_s": max(skew, key=abs) if skew else 0.0,
        # one crop interval, plus a gate frame for the rounding on either side.
        "allowed_skew_s": 1.0 / crop_fps + 1.0 / fps,
        "declined": declined(output),
    }


def declined(output: str) -> dict:
    """the movement scores that fell below `must_have_moved`.

    without this the distribution is truncated at the threshold and there is no
    way to tell whether lowering it would catch real vehicles or only noise.
    scenery's vetoes are counted separately, because they score zero for a
    reason that has nothing to do with the threshold.
    """
    hits = [(float(m.group(1)), m.group(2) == "true") for m in DECLINED.finditer(output)]
    free = sorted(c for c, parked in hits if not parked)
    pct = {}
    for p in (50, 75, 90, 95, 99):
        if free:
            pct[p] = free[min(len(free) - 1, int(len(free) * p / 100))]
    return {"n": len(hits), "parked": sum(p for _, p in hits), "percentiles": pct}


def precision(harvests: list[dict], clip: Path, fps: float,
              gate_size: tuple[int, int] = (640, 480)) -> dict:
    """was each crop taken of something that was actually moving.

    the crop's box is in gate coordinates and the scan is 160x120 of the same
    substream, so the two are the same picture at different sizes -- no
    main-stream mapping is involved and nothing can be out by a stream.

    the judgement is made frame by frame against a median background, by code
    that has never seen metermate's gate. a crop of the parked van scores zero
    here however confidently the pipeline decided otherwise.
    """
    import motion

    scan = motion.scan_frames(clip)
    bg = motion.background(scan)
    sx, sy = motion.SCAN_W / gate_size[0], motion.SCAN_H / gate_size[1]

    by_frame: dict[int, list] = {}
    for h in harvests:
        by_frame.setdefault(h["frame"], []).append(h)

    on, off = [], []
    for f, hs in sorted(by_frame.items()):
        # `pump` numbers frames from one; the scan is indexed from zero.
        i = min(max(f - 1, 0), len(scan) - 1)
        mask = motion.changed_mask(scan, bg, i)
        for h in hs:
            x, y, w, hh = h["box"]
            x0, y0 = int(x * sx), int(y * sy)
            x1, y1 = max(x0 + 1, int((x + w) * sx)), max(y0 + 1, int((y + hh) * sy))
            tile = mask[y0:y1, x0:x1]
            share = float(tile.mean()) if tile.size else 0.0
            (on if share >= ON_MOTION_FRAC else off).append({**h, "moving": share})
    return {
        "on_motion": len(on),
        "off_motion": len(off),
        # where the bad ones came from, coarsely. one place repeated over and
        # over is a standing vehicle scenery is not catching; scattered ones are
        # glare and pedestrians, which cost disk and nothing else.
        "off_motion_places": len({(h["box"][0] // 32, h["box"][1] // 32) for h in off}),
        "off_motion_worst": sorted(off, key=lambda h: -h["moved"])[:5],
        **stale_places(off, fps),
    }


# a place scenery has had at least this long to settle. mirrors
# `scenery::PARKED_AFTER`.
PARKED_AFTER_S = 90


def stale_places(off: list[dict], fps: float, bucket: int = 32) -> dict:
    """off-motion crops of a place that was *already* cropped long ago.

    the difference between "scenery has not settled this yet" and "scenery had
    every chance and lost it". a crop of a still vehicle in the first ninety
    seconds is the filter warming up, and expected. the same place cropped again
    minutes later means its track was forgotten and relearned -- which is what a
    quiet street does to a wall-clock timeout, and is invisible on a busy clip
    because nothing is ever quiet long enough to be forgotten.
    """
    first: dict[tuple, float] = {}
    stale = []
    for h in sorted(off, key=lambda h: h["frame"]):
        key = (h["box"][0] // bucket, h["box"][1] // bucket)
        at = h["frame"] / fps
        if key not in first:
            first[key] = at
        elif at - first[key] > PARKED_AFTER_S:
            stale.append({**h, "known_for_s": round(at - first[key], 1)})
    return {
        "relearned": len(stale),
        "relearned_places": len({(h["box"][0] // bucket, h["box"][1] // bucket) for h in stale}),
        "relearned_worst": sorted(stale, key=lambda h: -h["known_for_s"])[:5],
    }


def report(tag: str, result: dict, crops: int, clean: dict | None = None) -> None:
    rows = result["rows"]
    n = len(rows)
    detected = sum(r["detected"] for r in rows)
    kept = sum(bool(r["harvested"]) for r in rows)
    # detected but never harvested: the filters rejected it.
    filtered = [r["event"] for r in rows if r["detected"] and not r["harvested"]]

    print(f"\n=== {tag} ===")
    # first, because if it fails nothing below is a measurement of anything.
    # a replay once ran the gate against a crop feed frozen on the last frame
    # of the clip and reported 3/26 with a straight face.
    skew, allowed = result["worst_skew_s"], result["allowed_skew_s"]
    if abs(skew) > allowed:
        print(f"  !! gate and crop feed drifted {skew:+.2f}s apart "
              f"(a crop interval allows {allowed:.2f}s). the numbers below are void.")
    # recall needs labels; precision does not. a clip can be worth replaying for
    # one without having the other -- the quiet-street clip has no transits to
    # find and everything to say about what gets cropped anyway.
    if n:
        print(f"  transits (vehicles)        {n}")
        print(f"  detected by yolo           {detected}/{n} = {detected / n * 100:.0f}%")
        print(f"  **recall, end to end**     {kept}/{n} = {kept / n * 100:.0f}%")
    else:
        print("  no labelled transits: precision only")
    print(f"  crops written              {crops} ({result['harvests']} harvest events)")
    # the counterweight to recall. every filter loosened to catch one more
    # vehicle also lets more of the street through, and only this says so.
    if clean:
        total = clean["on_motion"] + clean["off_motion"]
        pct = clean["on_motion"] / max(total, 1) * 100
        places = clean["off_motion_places"]
        print(f"  **precision, on motion**   {clean['on_motion']}/{total} = {pct:.0f}%"
              f"   (judged by the scan, not the labels)")
        if clean["off_motion"]:
            print(f"    {clean['off_motion']} crops of a still scene, across {places} "
                  f"{'place' if places == 1 else 'places'}")
            for h in clean["off_motion_worst"]:
                print(f"      frame {h['frame']:>6} {h['label']:<5} at "
                      f"{h['box'][0]},{h['box'][1]} {h['box'][2]}x{h['box'][3]}"
                      f"  moved={h['moved']:.3f} but {h['moving'] * 100:.0f}% of it changed")
        # the scenery failure specifically: a place cropped again long after it
        # was first cropped had every chance to settle and did not.
        if clean["relearned"]:
            print(f"  **scenery relearned**       {clean['relearned']} crops across "
                  f"{clean['relearned_places']} places already known for >{PARKED_AFTER_S}s")
            for h in clean["relearned_worst"]:
                print(f"      frame {h['frame']:>6} at {h['box'][0]},{h['box'][1]}"
                      f"  first cropped {h['known_for_s']:.0f}s earlier")
    if rows:
        per = [r["harvested"] for r in rows if r["harvested"]]
        if per:
            print(f"  crops per harvested transit  median {st.median(per):.0f}  max {max(per)}")
    if filtered:
        print(f"  detected then filtered out: {len(filtered)} -> {filtered[:12]}")
    # why they were thrown away, which decides what to change.
    vetoed = [r["event"] for r in rows if r["detected"] and not r["harvested"] and r["all_parked"]]
    below = [r["event"] for r in rows
             if r["detected"] and not r["harvested"] and not r["all_parked"]]
    if vetoed:
        print(f"    of those, vetoed as scenery: {len(vetoed)}")
    if below:
        print(f"    of those, moved below threshold: {len(below)}")
    moved = [r["moved"] for r in rows if r["moved"] is not None]
    if moved:
        print(f"  moved of harvested: median {st.median(moved):.3f}  min {min(moved):.3f}")
    # what the threshold turned away, which is otherwise invisible: the smallest
    # `moved` a harvested crop can report is the threshold itself, so the
    # distribution arrives pre-truncated and tuning from it is circular.
    d = result.get("declined")
    if d and d["n"]:
        print(f"  declined below the threshold: {d['n']} "
              f"({d['parked']} of them vetoed as scenery)")
        print("    changed of the rest: "
              + "  ".join(f"p{p}={v:.3f}" for p, v in d["percentiles"].items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path)
    ap.add_argument("clip", type=Path, help="substream clip, for the gate")
    ap.add_argument("--crop-source", type=Path,
                    help="main-stream clip for crops; defaults to <clip> with -sub swapped for -main")
    ap.add_argument("--binary", type=Path, default=BINARY)
    ap.add_argument("--fps", type=float, default=None, help="clip fps; probed if omitted")
    ap.add_argument("--set", action="append", default=[],
                    help="config override, section.key=value (repeatable)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--keep-crops", type=Path,
                    help="copy the harvested crops here, to look at")
    # the scan needs numpy and scipy, and decodes the clip a second time. the
    # sweep runs dozens of replays and only compares recall between them.
    ap.add_argument("--no-precision", action="store_true",
                    help="skip the motion scan that judges each crop")
    ap.add_argument("--declined", action="store_true",
                    help="log what must_have_moved turned away, to see below the threshold")
    args = ap.parse_args()

    fps = args.fps
    if fps is None:
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate", "-of", "csv=p=0", str(args.clip)],
            capture_output=True, text=True, check=True).stdout.strip()
        num, den = raw.split("/")
        fps = float(num) / float(den)

    overrides: dict[str, dict] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides.setdefault(section, {})[name] = parsed

    # a clip with no labels is still worth replaying for precision.
    labels = json.loads(args.labels.read_text()) if args.labels.exists() else []
    log = (args.out or Path(args.clip.name)).with_suffix(".e2e.log")
    tag = f"{args.clip.name} @ {fps:.1f}fps"
    if overrides:
        tag += "  " + " ".join(f"{s}.{k}={v}" for s, kv in overrides.items() for k, v in kv.items())
    print(f"replaying {tag}\n  log: {log}", flush=True)

    crop_clip = args.crop_source
    if crop_clip is None:
        guess = Path(str(args.clip).replace("-sub.", "-main."))
        crop_clip = guess if guess != args.clip and guess.exists() else None
    if crop_clip is None:
        print("  WARNING: no main-stream clip; crops will be cut from the substream",
              file=sys.stderr)
    else:
        print(f"  crops from {crop_clip.name}", flush=True)
    output, crops = replay(args.clip, args.binary, overrides, log, crop_clip,
                           args.keep_crops, args.declined)
    result = score(labels, output, fps,
                   crop_fps=overrides.get("stream", {}).get("crop_fps", CROP_FPS))
    clean = None
    if not args.no_precision:
        print("  scanning the clip for motion, to judge the crops ...", flush=True)
        clean = precision(harvested(output), args.clip, fps)
    run = stats(output)
    report(tag, result, crops, clean)
    if run:
        # what it cost, beside what it caught. a change that holds recall and
        # halves the frame rate is a regression, and until the binary reported
        # this there was no way for a replay to notice.
        print("\nwhat the run cost")
        for line in stats_lines(run):
            print(line)

    if args.out:
        args.out.write_text(json.dumps(
            {"clip": args.clip.name, "overrides": overrides, "crops": crops,
             **result, **(clean or {}), "stats": run},
            indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
