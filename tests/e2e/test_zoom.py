"""looking closer at one crop or clip.

a crop is a photograph of a number plate and the grid shows it 150px wide, so
clicking one has to end with the plate being readable rather than with the same
picture a fifth bigger. the window is what opening gives you and the wheel is how
the rest of it is reached, toward wherever the cursor is.

the fitting itself is css and wants a browser; what is driven here is the zoom the
page keeps on top of it, because "the point under the cursor stays under the
cursor" is arithmetic that can be wrong in exactly one place.
"""

from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO, in_page
from preview import Preview


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """a preview to drive. the page is what is under test here, so only the model
    and a stream have to be real enough for it to answer.
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

[detector]
model = "{REPO / "models" / "detector.onnx"}"
""")
    return path


def zoomed(node, port, steps, open_with="zoomCrop('/crops/x.jpg', 0)", natural=0) -> dict:
    """open one thing, then drive the wheel at the points given."""
    wheel = "".join(
        f" zoomEl.onwheel({{deltaY: {d}, clientX: {x}, clientY: {y}, preventDefault() {{}}}});"
        for d, x, y in steps
    )
    return in_page(
        node,
        port,
        f"(() => {{ zoomImg.naturalWidth = {natural};"
        f" zoomVid.videoWidth = 0; {open_with};"
        f" const at = {{k: view.k, x: view.x, y: view.y}};"
        f" {wheel}"
        f" return {{at, k: view.k, x: view.x, y: view.y,"
        f"   img: zoomImg.style.transform || '', vid: zoomVid.style.transform || ''}};"
        f" }})()",
    )


def test_opening_a_crop_gives_it_the_window(binary, config, node, moving_clip):
    """**the crop is the screen and the screen is the crop.** the grid shows two
    hundred of them, which is why each is 150px there; the one that was clicked is
    not a tile any more."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = zoomed(node, run.port, [])
    assert got["at"] == {"k": 1, "x": 0, "y": 0}, got
    # the window is the rest state, and at rest nothing is transformed at all.
    assert got["img"] == "" and got["vid"] == "", got


def test_the_wheel_zooms_toward_the_cursor(binary, config, node, moving_clip):
    """the point being looked at is the point that stays put, which is what makes
    every part of a zoomed crop reachable without a second gesture to drag it."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = zoomed(node, run.port, [(-100, 1000, 400)])
    assert abs(got["k"] - 1.2) < 1e-9, got
    # 1000,400 was at 1000,400; at 1.2 the box has to have slid back by a fifth of
    # the way to it, or the plate drifts away as it grows.
    assert abs(got["x"] + 200) < 1e-9, got
    assert abs(got["y"] + 80) < 1e-9, got
    assert got["img"] == f"translate({got['x']}px, {got['y']}px) scale({got['k']})", got
    assert not got["vid"], "the clip was moved too, so both are showing"


def test_zooming_out_stops_at_the_crop_s_own_pixels(binary, config, node, moving_clip):
    """**one screen pixel per pixel of the crop**, and no further: below that the
    screen is inventing pixels rather than showing the ones stage two saw, and the
    blur that made a verdict wrong is the reason the crop was kept."""
    with Preview(binary, config, moving_clip).ready() as run:
        # 144 of them across, in a 1440 of screen: a tenth, and notches short of it
        # left over, since the clamp is what is being measured and not the count.
        out = zoomed(node, run.port, [(100, 700, 300)] * 20, natural=144)
    assert abs(out["k"] - 0.1) < 1e-9, out


def test_a_crop_that_cannot_say_its_size_stops_at_the_window(binary, config, node, moving_clip):
    """before the bytes arrive there is no truth to zoom out to, so the window is
    the floor: a page that lets the wheel shrink a crop to nothing on the strength
    of a missing number has invented a limit rather than found one."""
    with Preview(binary, config, moving_clip).ready() as run:
        out = zoomed(node, run.port, [(100, 700, 300)] * 4, natural=0)
    assert out["k"] == 1, out


def test_the_top_is_where_the_crop_runs_out(binary, config, node, moving_clip):
    """zooming past a crop's own pixels buys nothing but a bigger blur, and a wheel
    with no top means one stray flick loses the picture entirely."""
    with Preview(binary, config, moving_clip).ready() as run:
        up = zoomed(node, run.port, [(-100, 700, 300)] * 40, natural=144)
    assert abs(up["k"] - 12) < 1e-9, up


def test_a_playing_clip_is_left_alone(binary, config, node, moving_clip):
    """**the wheel is for the crop.** a clip is the whole street at the resolution it
    was encoded at, so filling the window is as close as it gets and there is no plate
    in it that the pixels cannot already show; and its player has its own use for a
    pointer over it, which a zoom stealing the gesture would take away."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = zoomed(
            node, run.port, [(-100, 200, 600)] * 3, open_with="zoomClip('/clips/x.mp4', 'x.mp4')"
        )
    assert got["at"] == {"k": 1, "x": 0, "y": 0}, got
    assert got["k"] == 1, got
    assert not got["vid"], "the clip was transformed"
    assert not got["img"], "the crop was transformed behind the clip"


def test_the_next_crop_opens_at_the_window(binary, config, node, moving_clip):
    """whatever the last one was zoomed to is not a question about this one."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = zoomed(node, run.port, [(-100, 700, 300)] * 5)
        again = in_page(
            node,
            run.port,
            "(() => { zoomCrop('/crops/x.jpg', 0); zoomCrop("
            " '/crops/y.jpg', 0); return {k: view.k, x: view.x, y: view.y}; })()",
        )
    assert got["k"] > 2, got
    assert again == {"k": 1, "x": 0, "y": 0}, again
