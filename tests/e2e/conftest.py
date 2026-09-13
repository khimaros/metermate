"""shared fixtures for the end to end suite.

the tests drive the real release binary. video comes from generated clips rather
than the camera, so the suite is deterministic and runs anywhere (r5.5).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from browser import style

REPO = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent / "page.mjs"
# overridable so a fix can be measured against a build of the behaviour it
# replaces, which is the only way to know a regression test would have caught
# anything.
BINARY = Path(os.environ.get("METERMATE_BIN") or REPO / "target" / "release" / "metermate")

# clips are generated at the substream's native size, because that is what the
# gate consumes and metermate deliberately never rescales on that path.
CLIP_W, CLIP_H = 640, 480
CLIP_FPS = 30


@pytest.fixture(scope="session")
def binary() -> Path:
    if not BINARY.exists():
        pytest.fail(f"{BINARY} missing; run `make build` first")
    return BINARY


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        pytest.skip("ffmpeg not installed")
    return exe


@pytest.fixture(scope="session")
def node() -> str:
    exe = shutil.which("node")
    if not exe:
        pytest.skip("node not installed; `mise install` provides it")
    return exe


@pytest.fixture(scope="session")
def browser_css() -> str:
    """the preview page's stylesheet, for the tests that measure it in a browser."""
    return style()


def in_page(node: str, port: int, expr: str, path: str = "/"):
    """evaluate an expression inside a served page and return its value.

    both pages the binary serves are driven this way -- the labelling page and
    the preview -- since a rule that is right in the source and wrong in the
    rendered page is one nobody gets the benefit of.
    """
    done = subprocess.run(
        [node, str(HARNESS), f"http://127.0.0.1:{port}{path}", expr],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # the assertion below reports the page's own error
    )
    assert done.returncode == 0, f"{expr}\n{done.stdout}\n{done.stderr}"
    return json.loads(done.stdout)


def _encode(ffmpeg: str, out: Path, inputs: list[str], filter_complex: str | None) -> Path:
    cmd = [ffmpeg, "-y", "-loglevel", "error", *inputs]
    if filter_complex:
        cmd += ["-filter_complex", filter_complex]
    # a short gop keeps the clip seekable and cheap to decode, matching how the
    # camera is configured rather than libx264's defaults.
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", str(out)]
    subprocess.run(cmd, check=True)
    return out


def _plate(seconds: int) -> list[str]:
    return ["-f", "lavfi", "-i", f"color=c=gray:s={CLIP_W}x{CLIP_H}:r={CLIP_FPS}:d={seconds}"]


@pytest.fixture(scope="session")
def static_clip(ffmpeg: str, tmp_path_factory) -> Path:
    """an unchanging scene. nothing here may ever produce an alert."""
    out = tmp_path_factory.mktemp("clips") / "static.mp4"
    return _encode(ffmpeg, out, _plate(6), None)


@pytest.fixture(scope="session")
def quiet_clip(ffmpeg: str, tmp_path_factory) -> Path:
    """an unchanging scene, long enough for a test to act while it is playing.

    a page test needs two things `moving_clip` does not give: nothing harvested,
    so the only crop that arrives is the one the test writes, and a source that
    outlasts the test -- a file source that runs out ends the pipeline, and the
    preview it was serving goes with it.
    """
    out = tmp_path_factory.mktemp("clips") / "quiet.mp4"
    return _encode(ffmpeg, out, _plate(120), None)


@pytest.fixture(scope="session")
def long_clip(ffmpeg: str, tmp_path_factory) -> Path:
    """a quiet scene long enough to outlast a stream window edge.

    four minutes, because an edge can be two minutes off the moment a test
    starts: a fixture that runs out ends the pipeline, which is the one thing
    those tests must not mistake for the clock having said no.
    """
    out = tmp_path_factory.mktemp("clips") / "long.mp4"
    return _encode(ffmpeg, out, _plate(240), None)


@pytest.fixture(scope="session")
def moving_clip(ffmpeg: str, tmp_path_factory) -> Path:
    """a bright block crossing a static scene, the size of a vehicle mid-frame.

    built by overlaying a second source rather than with drawbox: drawbox
    silently rendered nothing here, and a clip that is secretly blank makes the
    gate look broken when it is fine.
    """
    out = tmp_path_factory.mktemp("clips") / "moving.mp4"
    inputs = _plate(6) + ["-f", "lavfi", "-i", f"color=c=white:s=90x60:r={CLIP_FPS}:d=6"]
    return _encode(ffmpeg, out, inputs, "[0][1]overlay=x='20+t*160':y=200")


def run_metermate(binary: Path, config: Path, source: Path, timeout: int = 60) -> str:
    """run to completion against a file source and return combined output."""
    proc = subprocess.run(
        # the preview would fight a running instance for the port; the
        # suite is about the pipeline, not the browser view.
        [
            str(binary),
            "--config",
            str(config),
            "--source",
            str(source),
            "--dry-run",
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # assertions below inspect the output, including on failure
    )
    return proc.stdout + proc.stderr


def motion_events(output: str) -> list[str]:
    return [ln for ln in output.splitlines() if "motion on" in ln or "motion off" in ln]


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """a config pointed at nothing in particular; file sources bypass the camera."""
    path = tmp_path / "metermate.toml"
    path.write_text(
        f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[stream]
gate_subtype = 1
gate_width = {CLIP_W}
gate_height = {CLIP_H}

[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 500

[detector]
# absolute: the suite runs from tests/e2e, so a repo-relative path would not
# resolve. the binary needs a model even for the gated path.
model = "{REPO / "models" / "detector.onnx"}"
"""
    )
    return path
