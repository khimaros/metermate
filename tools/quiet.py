"""build a quiet-street clip pair from recorded footage.

every clip in `data/eval` is busy daytime traffic: measured across three of
them, the longest stretch with no motion at all was 18.9 seconds, against a
`FORGET_AFTER` of 240. so nothing in the corpus can exercise what the pipeline
does when the street goes quiet -- which is most of the night, and is where
`scenery` was found to fail: it forgets a parked vehicle on wall time, then
rediscovers and re-crops it on the next burst of traffic.

this splices real footage into `traffic | still | traffic`, holding the frame at
the splice point so the two traffic halves are continuous. the parked cars are
real, so the detector finds them and `scenery` tracks them; the still stretch
produces no motion, so the gate never looks, which is exactly the condition.

    uv run tools/quiet.py data/eval/20260912-103138 --quiet-secs 300

writes `<stamp>-quiet-sub.mp4` and `-quiet-main.mp4` beside the originals.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# long enough to clear `scenery::FORGET_AFTER` with room to spare, short enough
# that the still stretch stays cheap to encode and decode.
QUIET_SECS = 300
# traffic either side. the second half is what the assertion is about: by then a
# vehicle parked throughout should be settled scenery, not a fresh discovery.
TRAFFIC_SECS = 20


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def build(src: Path, out: Path, at: float, traffic: float, quiet: float) -> Path:
    """`src` from 0..at+traffic, with `quiet` seconds of stillness spliced in."""
    work = out.parent / f".{out.stem}"
    still = work.with_suffix(".png")
    parts = [work.with_suffix(f".{n}.mp4") for n in ("a", "still", "b")]
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-i", str(src),
         "-frames:v", "1", str(still)])
    # re-encoded rather than stream-copied: a copy cuts at keyframes, and the
    # splice has to land on the exact frame the still was taken from or the
    # background model sees a jump where the test assumes continuity.
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15", "-an"]
    run(["ffmpeg", "-y", "-v", "error", "-t", str(at), "-i", str(src), *enc, str(parts[0])])
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", str(quiet), "-i", str(still),
         "-r", fps_of(src), *enc, str(parts[1])])
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-t", str(traffic), "-i", str(src),
         *enc, str(parts[2])])

    listing = work.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(out)])
    for p in [*parts, still, listing]:
        p.unlink(missing_ok=True)
    return out


def fps_of(clip: Path) -> str:
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    num, den = raw.split("/")
    return f"{float(num) / float(den):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stamp", type=Path, help="path prefix, without -sub/-main.mp4")
    ap.add_argument("--at", type=float, default=TRAFFIC_SECS,
                    help="seconds of traffic before the quiet stretch")
    ap.add_argument("--traffic-secs", type=float, default=TRAFFIC_SECS)
    ap.add_argument("--quiet-secs", type=float, default=QUIET_SECS)
    args = ap.parse_args()

    for kind in ("sub", "main"):
        src = Path(f"{args.stamp}-{kind}.mp4")
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            return 1
        out = Path(f"{args.stamp}-quiet-{kind}.mp4")
        print(f"building {out.name} ...", flush=True)
        build(src, out, args.at, args.traffic_secs, args.quiet_secs)
        print(f"  {out} ({out.stat().st_size / 1e6:.1f} MB, "
              f"{args.at + args.quiet_secs + args.traffic_secs:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
