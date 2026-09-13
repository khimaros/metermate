"""record both camera streams at once, for the eval corpus.

the evals need a **pair**: the substream is what the gate sees and the main
stream is where crops are cut from, so a clip of one without the other measures
a pipeline nobody runs. recording them separately by hand is how they end up
misnamed or minutes apart.

    uv run tools/record.py --secs 600

reads the camera from `metermate.local.toml` (falling back to `metermate.toml`)
and writes `data/eval/<stamp>-sub.mp4` and `-main.mp4`, which is the naming the
eval tooling expects. `-c copy` throughout: no decode, no re-encode, so this
costs almost nothing and the pixels are exactly what metermate would have seen.
"""

import argparse
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUB_SUBTYPE, MAIN_SUBTYPE = 1, 0


def camera() -> dict:
    for name in ("metermate.local.toml", "metermate.toml"):
        p = REPO / name
        if p.exists():
            return tomllib.loads(p.read_text())["camera"]
    raise SystemExit("no metermate.toml to read the camera from")


def url(cam: dict, subtype: int) -> str:
    return (f"rtsp://{cam['username']}:{cam['password']}@{cam['host']}:554"
            f"/cam/realmonitor?channel={cam.get('channel', 1)}&subtype={subtype}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=int, default=600)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "eval")
    ap.add_argument("--at", help="wait until this local HH:MM before starting")
    args = ap.parse_args()

    if args.at:
        hh, mm = (int(x) for x in args.at.split(":"))
        now = datetime.now()
        when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        wait = (when - now).total_seconds()
        if wait < 0:
            wait += 24 * 3600
        print(f"waiting {wait / 60:.0f} min, until {args.at} ...", flush=True)
        time.sleep(wait)

    cam = camera()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # both at once, as two processes rather than two runs: the pair has to cover
    # the same minutes or the eval is comparing different traffic.
    procs = []
    for kind, subtype in (("sub", SUB_SUBTYPE), ("main", MAIN_SUBTYPE)):
        dest = args.out / f"{stamp}-{kind}.mp4"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
               "-t", str(args.secs), "-i", url(cam, subtype), "-c", "copy", "-y", str(dest)]
        log = dest.with_suffix(".ffmpeg.txt")
        procs.append((kind, dest, subprocess.Popen(cmd, stderr=log.open("w"), stdout=subprocess.DEVNULL)))
        print(f"recording {dest.name} for {args.secs}s", flush=True)

    failed = False
    for kind, dest, p in procs:
        if p.wait() != 0 or not dest.exists() or dest.stat().st_size == 0:
            print(f"  {kind}: FAILED, see {dest.with_suffix('.ffmpeg.txt')}", file=sys.stderr)
            failed = True
        else:
            print(f"  {kind}: {dest.stat().st_size / 1e6:.0f} MB")
    if not failed:
        print(f"\nnext: make transits STAMP={stamp}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
