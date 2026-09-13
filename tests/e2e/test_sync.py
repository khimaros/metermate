"""the gate frame and the main frame cropped from it must be the same moment.

two decoders run side by side: the gate reads the substream, the crop feed reads
the main stream, and every detection is a main-stream frame examined because a
*substream* frame moved. nothing in the pipeline checks that those two frames
are contemporaneous.

live, the skew is bounded by the crop rate -- a couple hundred milliseconds.
offline it is unbounded and was catastrophic. `LatestFrame` keeps whichever frame
arrived last and an offline replay removes wall-clock pacing, so the crop decoder
raced through a ten minute clip in under half a minute, hit eof, and left its
slot holding the final frame forever. measured on a real clip: the inspected
image stopped changing at gate frame 200 of 6534, and every detection after that
was of the same frozen picture. the end-to-end eval scored a still photograph.

so this asserts the property directly: the main frame consulted must track the
gate frame in video time.
"""

import os
import re
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

from conftest import CLIP_FPS, CLIP_H, CLIP_W, REPO, _encode, _plate

# `frame 412 cropping from main frame 109`.
#
# the detection line carries the same pair, but only when a vehicle is found,
# and synthetic footage has none -- nor does a feed that has stopped advancing,
# which is the case this exists to catch.
INSPECTION = re.compile(r"frame (\d+) cropping from main frame (\d+)")

# the crop feed is throttled to this, so one of its frames covers this much of
# the gate's timeline.
CROP_FPS = 4

# long enough that a racing crop decoder reaches the end of the clip while the
# gate is still near the start, which is the whole failure. offline replay looks
# at every frame, so this is also the cost of the test.
CLIP_SECS = 10

# how far apart the two frames may be, in seconds of video. a crop is allowed to
# lag by the interval between crop frames plus a frame of slack; anything beyond
# that is a desync rather than a sampling rate.
MAX_SKEW_S = 2.0 / CROP_FPS


@pytest.fixture(scope="session")
def synced_pair(ffmpeg: str, tmp_path_factory) -> tuple[Path, Path]:
    """a clip in two resolutions, the same scene at the same instants.

    this is the recorded pair the evals replay: `-sub` for the gate, `-main` for
    the crops. the block sweeps the full width so its position identifies the
    moment unambiguously.
    """
    d = tmp_path_factory.mktemp("synced")
    sub = _encode(
        ffmpeg,
        d / "pair-sub.mp4",
        _plate(CLIP_SECS)
        + ["-f", "lavfi", "-i", f"color=c=white:s=90x60:r={CLIP_FPS}:d={CLIP_SECS}"],
        "[0][1]overlay=x='mod(t*120\\,600)':y=200",
    )
    main = _encode(
        ffmpeg,
        d / "pair-main.mp4",
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={CLIP_W * 2}x{CLIP_H * 2}:r={CLIP_FPS}:d={CLIP_SECS}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s=180x120:r={CLIP_FPS}:d={CLIP_SECS}",
        ],
        "[0][1]overlay=x='mod(t*240\\,1200)':y=400",
    )
    return sub, main


def replay(binary: Path, config: Path, sub: Path, main: Path, timeout: int = 300) -> str:
    proc = subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--source",
            str(sub),
            "--crop-source",
            str(main),
            "--dry-run",
            "--preview",
            "off",
            "--offline",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={"RUST_LOG": "metermate=debug", "PATH": os.environ.get("PATH", "")},
    )
    return proc.stdout + proc.stderr


@pytest.fixture
def sync_config(tmp_path: Path) -> Path:
    """written whole rather than appended to the shared fixture.

    `crop_fps` belongs to `[stream]`, which that fixture has already closed, and
    toml forbids reopening a table -- so the rate this test is entirely about
    cannot be added to it from outside.
    """
    path = tmp_path / "metermate.toml"
    path.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[stream]
gate_subtype = 1
gate_width = {CLIP_W}
gate_height = {CLIP_H}
crop_fps = {CROP_FPS}

[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 500

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
enabled = false
""")
    return path


def inspections(output: str) -> list[tuple[int, int]]:
    return [(int(m.group(1)), int(m.group(2))) for m in INSPECTION.finditer(output)]


def test_the_main_frame_keeps_pace_with_the_gate(binary, sync_config, synced_pair):
    """the failure that scored a still photograph.

    the crop decoder is not clocked by anything, so offline it empties the clip
    at whatever rate the machine decodes and then stops. every later inspection
    reads the same stale frame, which is not a slow crop feed -- it is a
    different moment of the street entirely.
    """
    sub, main = synced_pair
    seen = inspections(replay(binary, sync_config, sub, main))
    assert len(seen) > 30, f"too few inspections to judge: {len(seen)}"

    # both clips run at the same rate, so the two frame numbers are directly
    # comparable once the crop feed's throttle is undone. in absolute terms: an
    # unclocked feed races *ahead* of the gate before it freezes, and a crop
    # from the street's future is as wrong as one from its past.
    skews = [(gate - 1) / CLIP_FPS - (crop - 1) / CROP_FPS for gate, crop in seen]
    worst = max(skews, key=abs)
    assert abs(worst) <= MAX_SKEW_S, (
        f"main stream ran {worst:+.1f}s from the gate in video time "
        f"(allowed +/-{MAX_SKEW_S:.2f}s); last inspection was gate {seen[-1][0]} "
        f"against main {seen[-1][1]}"
    )


def test_the_main_frame_is_never_reused_forever(binary, sync_config, synced_pair):
    """the symptom, stated as the eval saw it.

    a frozen slot shows up as the same main frame answering hundreds of
    consecutive inspections. this catches it even if the frame numbering changes,
    which the skew assertion above would not.
    """
    sub, main = synced_pair
    seen = inspections(replay(binary, sync_config, sub, main))
    assert seen, "no inspections at all"

    runs, run = [], 1
    for (_, a), (_, b) in pairwise(seen):
        if a == b:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    # the gate looks at every frame offline; the crop feed delivers one per
    # CLIP_FPS/CROP_FPS of them, so a handful of repeats is the sampling rate.
    allowed = (CLIP_FPS // CROP_FPS) * 3
    assert max(runs) <= allowed, (
        f"one main frame answered {max(runs)} consecutive inspections "
        f"(sampling alone explains at most {allowed})"
    )
