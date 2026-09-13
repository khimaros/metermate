"""the gate has to keep up with the camera while the detector is working.

every other test here asks what the pipeline decided. this one asks whether it
was still watching while it decided, which is a different question and the one
the offline evals structurally cannot ask: they pass `--offline`, which removes
pacing so the replay examines every frame however long that takes.

the failure this exists for: the detector runs inline in the frame loop, so an
inference is also a stall. at two onnx threads that is ~130ms against a 33ms
frame period here, so sustained motion costs the gate most of its frames -- it
sees a fraction of the street, and the preview falls behind because it is
published from the same loop.
"""

import re

import pytest

from conftest import CLIP_FPS, CLIP_H, CLIP_W, _encode

# the loop reports its accounting every this many frames, so a clip has to be at
# least this long for there to be anything to assert on.
REPORT_EVERY = 300

# share of the stream the gate must still examine while the detector is running.
# generous on purpose: this is a throughput test, so it is the one measurement
# here that legitimately varies with the machine. it is set well below what a
# decoupled detector achieves and well above what an inline one does, so it
# separates the two without being a benchmark.
MIN_SEEN = 0.80

# matches both the periodic line and the one emitted when the stream ends.
SEEN = re.compile(r"(\d+) frames examined, (\d+) skipped \((\d+)% of the stream seen\)")


@pytest.fixture(scope="session")
def busy_clip(ffmpeg, tmp_path_factory):
    """something moving in nearly every frame, for long enough to be reported on.

    the block wraps rather than crossing once: the point is sustained motion, so
    that the gate is firing continuously and the detector is asked to look
    continuously. a single crossing leaves most of the clip quiet, which is
    exactly the case that does not stress the loop.
    """
    seconds = (REPORT_EVERY * 2) // CLIP_FPS
    out = tmp_path_factory.mktemp("clips") / "busy.mp4"
    inputs = ["-f", "lavfi", "-i", f"color=c=gray:s={CLIP_W}x{CLIP_H}:r={CLIP_FPS}:d={seconds}"] + [
        "-f",
        "lavfi",
        "-i",
        f"color=c=white:s=90x60:r={CLIP_FPS}:d={seconds}",
    ]
    return _encode(ffmpeg, out, inputs, f"[0][1]overlay=x='mod(t*220,{CLIP_W})':y=200")


def test_the_gate_keeps_up_while_the_detector_is_working(binary, config, busy_clip):
    import subprocess

    proc = subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--source",
            str(busy_clip),
            "--dry-run",
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        # the accounting line is debug: a busy frame would otherwise bury it.
        env={"RUST_LOG": "metermate=debug", "PATH": "/usr/bin:/bin"},
    )
    output = proc.stdout + proc.stderr
    # the end-of-run line always fires. the periodic one is keyed on frames
    # examined, so a loop far enough behind never reaches the interval -- which
    # is exactly the failure under test, and it used to hide itself.
    reports = SEEN.findall(output)
    assert reports, f"the loop never reported its frame accounting:\n{output[-2000:]}"

    examined, skipped, pct = (int(x) for x in reports[-1])
    seen = examined / max(examined + skipped, 1)
    assert seen >= MIN_SEEN, (
        f"the gate saw {pct}% of the stream ({skipped} of {examined + skipped} frames "
        f"dropped) while the detector ran. an inference is a stall, so sustained "
        f"motion costs the gate its frame rate and the preview its liveness."
    )
