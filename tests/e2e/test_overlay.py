"""which boxes the live view draws (r8.2).

the detector runs over the whole frame, so every gated look reports every vehicle
in view -- including the ones that have sat on that kerb for years. one of them is
what the pipeline acts on, and the page has always drawn the rest, dashed, on the
argument that seeing what was *not* acted on is the point of r8.2.

that argument holds when the street is worth looking at. where most of what the
detector reports is parked, the dashes are a thicket over the one box a person
needed to check. `[preview] show_parked = false` asks for the rest only, and what
it has to keep doing is drawing nothing else: scenery is still scenery, still
counted, still vetoing a harvest -- it is just not drawn.
"""

import re
from pathlib import Path

import pytest

from conftest import REPO, in_page
from preview import Preview
from test_resolution import street_clip

MAIN = (1280, 720)
SUB = (640, 480)
# long enough to hold still in one place and be looked at twice, and to
# outlast a test that reads it for a dozen seconds of that.
SECONDS = 25
# and long enough to read the whole of it: the jeep only becomes a moving vehicle
# once `[gate] warmup_frames` is past and it starts crossing, and a test that read
# the first second alone would see a street that never moved.
READ_S = 8.0


def write_config(path: Path, url: str, show_parked: str | None) -> Path:
    preview = "" if show_parked is None else f"\n[preview]\nshow_parked = {show_parked}\n"
    path.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"
url = "{url}"

[stream]
gate_subtype = 1

[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 1500

[detector]
model = "{REPO / "models" / "detector.onnx"}"

# two seconds and two sightings, where the deployment waits 90 and three. long
# enough that a vehicle crossing is never scenery -- it is a new place every fifth
# of a second -- and short enough that the ones on the kerb are.
[scenery]
parked_after_secs = 2
min_sightings = 2
min_occupancy = 0.0
{preview}""")
    return path


@pytest.fixture
def street(ffmpeg: str, tmp_path: Path) -> str:
    dir = tmp_path / "streams"
    street_clip(ffmpeg, dir / "0.mp4", MAIN[0], MAIN[1], SECONDS)
    street_clip(ffmpeg, dir / "1.mp4", SUB[0], SUB[1], SECONDS)
    return f"{dir}/{{subtype}}.mp4"


def boxes(frames: list[dict]) -> list[dict]:
    return [d for frame in frames for d in frame["detections"]]


def test_parked_vehicles_are_drawn_until_told_otherwise(binary, street, tmp_path):
    """**the default keeps drawing them.** a parked vehicle is what the pipeline
    decided to leave alone, and a view that cannot show a decision is a webcam."""
    cfg = write_config(tmp_path / "metermate.toml", street, None)
    with Preview(binary, cfg).ready(on="/stats") as run:
        found = boxes(run.overlay(within=READ_S))
        assert any(b["parked"] for b in found), "no vehicle was ever scenery"
        assert any(not b["parked"] for b in found), "and nothing ever moved"


def test_parked_vehicles_are_left_undrawn_when_asked_that_way(binary, street, tmp_path):
    """**the option takes boxes away and nothing else.**"""
    cfg = write_config(tmp_path / "metermate.toml", street, "false")
    with Preview(binary, cfg, debug=True).ready(on="/stats") as run:
        found = boxes(run.overlay(within=READ_S))
        said = run.output()

    assert found, "nothing was drawn at all"
    assert not [b for b in found if b["parked"]], f"{len(found)} boxes, one of them scenery"
    # and the pipeline went on counting them behind the curtain: a look says what
    # moved and how many were standing there as usual, which is the same decision
    # the drawing lost. without this the test would also pass on a street where
    # nothing was ever scenery.
    assert re.search(r"\| \d+ parked", said), f"scenery was never decided:\n{said[-900:]}"


ONE = "{c: 'car', p: 0.9, x1: 10, y1: 10, x2: 60, y2: 60, m: 0.4}"


def drawn_boxes(node, port, ticked) -> int:
    """one frame, with the boxes asked for or refused."""
    return in_page(
        node,
        port,
        "(() => { const cv = document.getElementById('overlay');"
        " cv.drawn.length = 0;"
        f" boxesEl.checked = {'true' if ticked else 'false'};"
        f" draw({{w: 640, h: 480, detections: [{ONE}]}});"
        " return cv.drawn.filter(n => n === 'strokeRect').length; })()",
    )


def test_the_boxes_can_be_turned_off(binary, street, tmp_path, node):
    """**a look at the street without the boxes.** the overlay is drawn over the
    video to explain what the pipeline saw, and sometimes the question is what the
    street looks like -- so this is a checkbox and not a mode, and it leaves the
    motion dashes alone because those answer a different question.
    """
    cfg = write_config(tmp_path / "metermate.toml", street, None)
    with Preview(binary, cfg).ready(on="/stats") as run:
        assert drawn_boxes(node, run.port, True) == 1
        assert drawn_boxes(node, run.port, False) == 0


def test_refusing_the_boxes_leaves_the_last_frame_no_longer_drawn(binary, street, tmp_path, node):
    """the overlay is only repainted when something arrives, and a street with
    nothing on it sends nothing: unchecking would otherwise leave the boxes of the
    last vehicle sitting on an empty road."""
    cfg = write_config(tmp_path / "metermate.toml", street, None)
    with Preview(binary, cfg).ready(on="/stats") as run:
        cleared = in_page(
            node,
            run.port,
            "(() => { const cv = document.getElementById('overlay');"
            " draw({w: 640, h: 480,"  # a frame drawn while the boxes are on
            f"   detections: [{ONE}]}});"
            " cv.drawn.length = 0;"
            " boxesEl.checked = false; boxesEl.onchange();"
            " return cv.drawn.includes('clearRect'); })()",
        )
        assert cleared, "nothing was cleared, so the boxes outlived the tickbox"
