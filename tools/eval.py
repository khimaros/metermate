"""replay a recorded clip through metermate and report what it found.

every tuning decision up to now was argued from a live camera pointed at a street
that changes by the minute: the light moves, the camera has been panned twice,
and a car either drove past during the measurement or it did not. two builds have
never been compared on the same input.

this replays a fixed clip instead. it is a measurement, not a pass/fail test --
there is no ground truth for these clips, and hand-labelling every vehicle is not
worth it -- so the output is designed to be diffed between two builds:

    uv run tools/eval.py data/eval/<stamp>-sub.mp4 --out before.json
    ... change something ...
    uv run tools/eval.py data/eval/<stamp>-sub.mp4 --out after.json --compare before.json
"""

import argparse
import json
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "target" / "release" / "metermate"

# `frame 412 main 109 region Rect { .. } -> car 0.86 moved=0.12 box=120,80 90x60 | 5 parked`
#
# a debug line: it is one per gated frame, so it is not in a deployment's log.
# this tool runs the binary itself and asks for it.
REPORT = re.compile(
    r"frame (\d+) main \d+ region Rect \{ x: (\d+), y: (\d+), w: (\d+), h: (\d+) \} -> (.+)"
)
# parked vehicles are counted rather than listed, so these are the ones the
# look reported: what moved.
VEHICLE = re.compile(r"(\w[\w-]*) ([\d.]+) moved=([\d.]+) box=(\d+),(\d+) (\d+)x(\d+)")
SKIP = re.compile(r"spent this second's (\d+) inferences \((\d+)ms each\)")
FRAMES = re.compile(r"(\d+) frames examined, (\d+) skipped")

# two sightings this close, in units of the larger box, are the same vehicle.
# matches `crop::same_object`'s tolerance so tracks here mean what they mean in
# the pipeline.
SAME_PLACE = 0.5
# and a gap longer than this starts a new track even in the same place, so a
# parked car does not become one track lasting the whole clip.
#
# measured against the clip's own frame numbers, not the log timestamps. an
# offline replay takes longer than the clip it replays, so wall-clock gaps are
# always tiny and nothing ever splits: 57,000 sightings collapsed into 13 tracks
# of 8,000 looks each before this was keyed on frames.
TRACK_GAP_S = 3.0
# frames per second of the clips being replayed. the substream has run at both,
# so it is read from the log when stated and assumed otherwise.
DEFAULT_FPS = 15.0


def same_place(a, b) -> bool:
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    reach_x, reach_y = max(aw, bw) * SAME_PLACE, max(ah, bh) * SAME_PLACE
    return (abs((ax + aw / 2) - (bx + bw / 2)) <= reach_x
            and abs((ay + ah / 2) - (by + bh / 2)) <= reach_y)


def run(clip: Path, binary: Path, offline: bool, log: Path) -> tuple[str, int]:
    """replay the clip with a throwaway harvest directory.

    the log is written as it goes rather than captured and printed at the end. an
    offline replay of a ten minute clip takes twenty-odd minutes, and a run with
    no visible progress is indistinguishable from a wedged one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "metermate.toml"
        cfg.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[stream]
gate_subtype = 1
gate_width = 640
gate_height = 480

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
dir = "{Path(tmp) / "crops"}"
""")
        cmd = [str(binary), "--config", str(cfg), "--source", str(clip),
               "--dry-run", "--preview", "off"]
        if offline:
            cmd.append("--offline")
        with log.open("w") as sink:
            subprocess.run(
                cmd, stdout=sink, stderr=subprocess.STDOUT, check=False,
                env={"RUST_LOG": "metermate=debug", "PATH": "/usr/bin:/bin"},
            )
        harvested = len(list((Path(tmp) / "crops").glob("*.jpg")))
        return log.read_text(), harvested


def parse(output: str, fps: float = DEFAULT_FPS) -> dict:
    """turn the log into sightings, then group sightings into tracks.

    a track is one vehicle's passage. it is what "was this car found, and how
    quickly" is asked of, and counting raw sightings instead would just reward a
    build for running the detector more often.
    """
    sightings, rates, costs = [], [], []
    examined = skipped = 0
    for line in output.splitlines():
        if m := SKIP.search(line):
            rates.append(int(m.group(1)))
            costs.append(int(m.group(2)))
        if m := FRAMES.search(line):
            examined, skipped = int(m.group(1)), int(m.group(2))
        if not (r := REPORT.search(line)):
            continue
        # video time, from the clip's own frame number. `at` from the log
        # timestamp is wall-clock and means nothing in an offline replay.
        at = int(r.group(1)) / fps
        for v in VEHICLE.finditer(r.group(6)):
            sightings.append({
                "at": at, "label": v.group(1), "conf": float(v.group(2)),
                # listed at all means it moved: parked vehicles are counted in
                # the line rather than spelled out.
                "moved": float(v.group(3)), "parked": False,
                "box": (int(v.group(4)), int(v.group(5)), int(v.group(6)), int(v.group(7))),
            })

    sightings.sort(key=lambda s: s["at"])
    tracks: list[dict] = []
    for s in sightings:
        hit = next((t for t in reversed(tracks)
                    if same_place(t["last_box"], s["box"])
                    and s["at"] - t["last"] <= TRACK_GAP_S), None)
        if hit:
            hit["last"], hit["last_box"] = s["at"], s["box"]
            hit["n"] += 1
            hit["best"] = max(hit["best"], s["conf"])
            hit["moving"] = hit["moving"] or s["moved"] >= 0.08
        else:
            tracks.append({"first": s["at"], "last": s["at"], "last_box": s["box"],
                           "n": 1, "best": s["conf"], "moving": s["moved"] >= 0.08})

    return {
        "sightings": len(sightings),
        "tracks": len(tracks),
        "moving_tracks": sum(1 for t in tracks if t["moving"]),
        "confidences": [round(t["best"], 2) for t in tracks],
        "looks_per_track": [t["n"] for t in tracks],
        "track_seconds": [round(t["last"] - t["first"], 1) for t in tracks],
        "detector_rate": round(st.mean(rates), 1) if rates else None,
        "inference_ms": round(st.mean(costs)) if costs else None,
        "frames_examined": examined,
        "frames_skipped": skipped,
    }


def summarise(tag: str, r: dict) -> None:
    confs = r["confidences"]
    print(f"\n=== {tag} ===")
    print(f"  tracks            {r['tracks']}  ({r['moving_tracks']} moving)")
    print(f"  sightings         {r['sightings']}")
    print(f"  crops harvested   {r['harvested']}")
    if confs:
        print(f"  confidence        median {st.median(confs):.2f}  "
              f"best {max(confs):.2f}  >=0.7 {sum(1 for c in confs if c >= 0.7)}/{len(confs)}")
    if r["looks_per_track"]:
        print(f"  looks per track   median {st.median(r['looks_per_track'])}  "
              f"max {max(r['looks_per_track'])}")
    if r["detector_rate"]:
        print(f"  detector          {r['detector_rate']}/s at {r['inference_ms']}ms")
    total = r.get("frames_examined", 0) + r.get("frames_skipped", 0)
    if total:
        print(f"  frames            {r['frames_examined']} examined, "
              f"{r['frames_skipped']} skipped "
              f"({r['frames_examined'] * 100 // total}% of the stream)")


def compare(old: dict, new: dict) -> None:
    print("\n=== change ===")
    # comparing a realtime run against an offline one reads as a large
    # regression and is not one. nearly every count here scales with how often
    # the detector looked: sparse looks re-crop one car as it moves, because its
    # positions no longer overlap, and split its passage into several tracks for
    # the same reason. only same-mode comparisons say anything about quality.
    if old.get("mode") and new.get("mode") and old["mode"] != new["mode"]:
        print(f"  !! {old['mode']} vs {new['mode']}: these are not comparable.")
        print("     counts below scale with sampling density, not quality.")
        print("     re-run both in the same mode to judge a change.\n")
    for key, label, better in [
        ("tracks", "tracks found", "up"),
        ("moving_tracks", "moving tracks", "up"),
        ("harvested", "crops harvested", "up"),
        ("inference_ms", "inference ms", "down"),
    ]:
        a, b = old.get(key), new.get(key)
        if a is None or b is None:
            continue
        delta = b - a
        arrow = "same" if delta == 0 else ("up" if delta > 0 else "down")
        verdict = "" if delta == 0 else ("  better" if arrow == better else "  WORSE")
        print(f"  {label:<18} {a} -> {b}  ({delta:+}){verdict}")
    oc, nc = old.get("confidences") or [0], new.get("confidences") or [0]
    print(f"  {'median confidence':<18} {st.median(oc):.2f} -> {st.median(nc):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--binary", type=Path, default=BINARY)
    ap.add_argument("--out", type=Path, help="write the result as json")
    ap.add_argument("--compare", type=Path, help="an earlier --out to diff against")
    # replay is paced at wall-clock, so a ten minute clip costs ten minutes. a
    # mistake in the parsing above should not cost another ten, so the raw log is
    # always kept and can be re-read instead of re-run.
    ap.add_argument("--from-log", type=Path, help="re-parse a kept log, no replay")
    # the two bookends. realtime is what this machine actually manages and what a
    # deployment sees; offline examines every frame and is what a machine fast
    # enough to keep up would see. a real host sits between.
    #
    # compare like with like. counts here scale with how often the detector
    # looked, not only with how well it did: a car looked at rarely is cropped
    # several times as it moves, because its successive positions no longer
    # overlap, and the same reason splits its passage into several tracks.
    ap.add_argument("--offline", action="store_true",
                    help="examine every frame instead of pacing at wall-clock")
    args = ap.parse_args()

    if args.from_log:
        output = args.from_log.read_text()
        harvested = 0
    else:
        if not args.clip.exists():
            print(f"{args.clip} does not exist", file=sys.stderr)
            return 1
        if not args.binary.exists():
            print(f"{args.binary} missing; run `make build`", file=sys.stderr)
            return 1
        pace = "examining every frame" if args.offline else "at wall-clock rate"
        log = (args.out or Path(args.clip.name)).with_suffix(".log")
        print(f"replaying {args.clip.name} {pace}", flush=True)
        print(f"  watch progress with: tail -f {log}", flush=True)
        output, harvested = run(args.clip, args.binary, args.offline, log)

    result = parse(output) | {
        "harvested": harvested,
        "clip": args.clip.name,
        "mode": "offline" if args.offline else "realtime",
    }
    summarise(args.clip.name, result)

    if args.compare and args.compare.exists():
        compare(json.loads(args.compare.read_text()), result)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
