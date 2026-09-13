"""what the pipeline has open right now, read off the process table.

the stream window (r11.1 for the streams themselves) is defined by what is
*running*: outside the window the rtsp channels are closed, which means no
ffmpeg is pulling anything. a log line saying so is a claim; a process table
with no ffmpeg in it is the fact, and it is the one an idle machine (r3.1)
actually depends on.
"""

import time
from pathlib import Path

PROC = Path("/proc")


def ppid(pid: int) -> int | None:
    """the parent of a process, or `None` if it has already gone away."""
    try:
        stat = (PROC / str(pid) / "stat").read_text()
    except (OSError, ValueError):
        return None
    # the comm field is parenthesised and can hold spaces, so count from
    # the right: what follows the last `)` is state, ppid, ...
    tail = stat[stat.rindex(")") + 2 :].split()
    return int(tail[1]) if len(tail) > 1 else None


def ffmpegs(root: int) -> list[int]:
    """the ffmpeg processes `root` has running, direct children only.

    both feeds are children of metermate rather than grandchildren, and the
    probe that runs at startup is `ffprobe` -- so anything named ffmpeg here
    is a stream being pulled.
    """
    if not PROC.exists():
        return []
    return [
        int(pid)
        for pid in (p.name for p in PROC.iterdir() if p.name.isdigit())
        if ppid(int(pid)) == root and _comm(int(pid)) == "ffmpeg"
    ]


def _comm(pid: int) -> str:
    try:
        return (PROC / str(pid) / "comm").read_text().strip()
    except OSError:
        return ""


def wait_for_no_ffmpeg(root: int, timeout: float) -> bool:
    """whether every ffmpeg went away within `timeout`.

    closing the channels is not instantaneous -- the reader notices between
    frames -- so this waits rather than asserting on a single sample. it
    returns the first quiet sample rather than insisting on several, because
    a channel that reopens inside the same wait is the bug being tested for
    and would be caught by the caller's own assertions on frames.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not ffmpegs(root):
            return True
        time.sleep(0.2)
    return not ffmpegs(root)
