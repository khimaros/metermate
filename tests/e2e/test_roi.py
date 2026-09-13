"""the roi decides what the gate is allowed to see.

a polygon in config is the only thing standing between the gate and the
overhead wires this camera looks through, so it is worth proving through the
real binary rather than in a unit test: a mask that is right in `roi.rs` and
wired up wrong reaches production as a camera that watches nothing.
"""

from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO, motion_events, run_metermate

# the moving clip sweeps a block across y=200..260. these bracket it.
AROUND_THE_BLOCK = [[0, 150], [CLIP_W, 150], [CLIP_W, 300], [0, 300]]
BELOW_THE_BLOCK = [[0, 400], [CLIP_W, 400], [CLIP_W, CLIP_H], [0, CLIP_H]]


def write_config(path: Path, roi: list) -> Path:
    path.write_text(f"""
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
roi = {roi}

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
enabled = false
""")
    return path


@pytest.mark.parametrize(
    "roi,expect_motion",
    [(AROUND_THE_BLOCK, True), (BELOW_THE_BLOCK, False)],
    ids=["roi-covers-the-block", "roi-excludes-the-block"],
)
def test_the_gate_only_sees_inside_the_roi(binary, tmp_path, moving_clip, roi, expect_motion):
    cfg = write_config(tmp_path / "metermate.toml", roi)
    out = run_metermate(binary, cfg, moving_clip)
    fired = [ln for ln in motion_events(out) if "motion on" in ln]
    assert bool(fired) is expect_motion, (
        f"roi {roi} should {'fire' if expect_motion else 'stay quiet'}; got {len(fired)} events"
    )


def test_an_roi_that_covers_nothing_says_so(binary, tmp_path, moving_clip):
    """the silent failure: coordinates from the wrong frame size look valid and
    watch nothing, which is indistinguishable from a quiet street."""
    cfg = write_config(tmp_path / "metermate.toml", [[900, 900], [950, 900], [950, 950]])
    out = run_metermate(binary, cfg, moving_clip)
    assert "roi covers" in out and "gate pixels" in out, out[-2000:]


def test_no_roi_watches_the_whole_frame(binary, tmp_path, moving_clip):
    cfg = write_config(tmp_path / "metermate.toml", [])
    out = run_metermate(binary, cfg, moving_clip)
    assert "no roi" in out
    assert motion_events(out), "a block crossing an unmasked frame must fire"
