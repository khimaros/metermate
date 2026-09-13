"""every view, and every crop, verdict and clip, reachable by a link.

the page is a wall of tiles and the one tile that matters is in it somewhere, so
the shareable thing has to be the crop rather than the tab: a notification, a
message pasted into a chat, and a bookmark all want a url that lands on the thing
itself.

the routing is driven through the served page's own javascript for the reason the
other page tests give -- a route that is right in the source and wrong in the
rendered page is a route nobody can follow -- and the entry a link resolves
against is asked of the real server.
"""

import json
from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO, in_page
from labelling import blank_crops, plain_name, verdict_name
from preview import Preview, get
from test_dense import FPS, street_clip

AT = 1789000000000


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    crops = tmp_path / "crops"
    names = [verdict_name(AT, "go4"), plain_name(AT + 60_000), plain_name(AT + 120_000)]
    blank_crops(crops, ffmpeg, names)
    return crops


@pytest.fixture
def events(tmp_path: Path, ffmpeg: str) -> Path:
    dir = tmp_path / "events"
    dir.mkdir()
    street_clip(ffmpeg, dir, FPS)
    return dir


@pytest.fixture
def config(tmp_path: Path, harvest: Path, events: Path) -> Path:
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
crop_fps = {FPS}

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
dir = "{harvest}"

[[subject]]
name = "go4"

[record]
dir = "{events}"
""")
    return path


def in_the_page(node, port, body: str) -> dict:
    """wait on a page action and read back what the page ended up showing."""
    js = (
        "(async () => { await (" + body + ")();"
        " return {tab: tab, open: zoomEl.classList.contains('on'),"
        " src: zoomImg.src, at: zoomAt, jump: zoomJump.hidden}; })()"
    )
    return in_page(node, port, js)


def test_every_view_has_its_own_hash(binary, config, harvest, node, moving_clip):
    """**the hash is the navigation.** one spelling per view, and a hash nothing
    knows is the live view rather than a blank page: a mistyped link should still
    land somewhere a person can read.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        views, unknown, named = in_page(
            node,
            run.port,
            "(() => [[routeOf('#/live'), routeOf('#/crops'), routeOf('#/verdicts'),"
            " routeOf('#/events'), routeOf('#/roi')],"
            " [routeOf('#'), routeOf('#/nonsense'), routeOf('garbage')],"
            " [routeOf('#/crop/x.jpg'), routeOf('#/verdict/x.jpg'),"
            "  routeOf('#/event/a.mp4'), routeOf('#/verdict')]])()",
        )
    assert [r["tab"] for r in views] == ["live", "crops", "verdicts", "events", "roi"], views
    assert all(r == {"tab": "live", "kind": "", "name": ""} for r in unknown), unknown
    # a link naming one thing carries it and says which tab it lives on: the
    # crops tab holds every crop and the verdicts tab only the named ones, and a
    # link should not have to know which of the two a crop belongs to.
    assert named[0] == {"tab": "crops", "kind": "crop", "name": "x.jpg"}, named
    assert named[1] == {"tab": "verdicts", "kind": "verdict", "name": "x.jpg"}, named
    assert named[2] == {"tab": "events", "kind": "event", "name": "a.mp4"}, named
    # `#/verdict` with nothing after it is the tab, not a broken link.
    assert named[3] == {"tab": "verdicts", "kind": "", "name": ""}, named


def test_a_verdict_link_opens_that_crop(binary, config, harvest, node, moving_clip):
    """the notification's whole point: the tap lands on the picture.

    the crop's own instant comes back with it, which is what lets the overlay go
    on to offer the recording it was taken during -- and the recording is a clip
    the page has never listed, so the offer has to be looked up.
    """
    name = verdict_name(AT, "go4")
    with Preview(binary, config, moving_clip).ready() as run:
        got = in_the_page(node, run.port, f"() => showHash('#/verdict/{name}')")
    assert got["open"], "the overlay stayed shut"
    assert got["src"].endswith("/crops/" + name), got
    assert got["tab"] == "verdicts", got
    assert got["at"] == AT, got


def test_a_crop_link_opens_the_harvest_tab(binary, config, harvest, node, moving_clip):
    """a crop with no verdict in its name belongs to the crops tab, and that is
    the tab its link opens -- both grids are one harvest asked differently."""
    name = plain_name(AT + 60_000)
    with Preview(binary, config, moving_clip).ready() as run:
        got = in_the_page(node, run.port, f"() => showHash('#/crop/{name}')")
    assert (got["tab"], got["open"]) == ("crops", True), got


def test_a_link_to_a_crop_the_budget_evicted_lands_on_its_tab(
    binary, config, harvest, node, moving_clip
):
    """**a stale link stays a stale link.** the harvest evicts oldest first, so a
    link a week old can name a crop that is gone. the tab it belonged to is still
    the honest answer; a broken image filling the screen is not.
    """
    gone = verdict_name(AT + 9_999_999, "go4")
    with Preview(binary, config, moving_clip).ready() as run:
        got = in_the_page(node, run.port, f"() => showHash('#/verdict/{gone}')")
    assert got["tab"] == "verdicts", got
    assert not got["open"], "the overlay opened on a crop that is not there"


def test_the_buttons_drive_the_url_and_the_url_drives_the_page(
    binary, config, harvest, node, moving_clip
):
    """**the wiring, which nothing else here can reach.**

    every other test in this file calls `showHash`, which would keep passing if
    nothing at all were listening to the browser. this one goes the way a person
    does: click a tab, follow a link, close the crop.
    """
    name = verdict_name(AT, "go4")
    with Preview(binary, config, moving_clip).ready() as run:
        clicked = in_page(
            node, run.port, "(() => { tabVerdicts.onclick(); return location.hash; })()"
        )
        assert clicked == "#/verdicts", clicked

        followed = in_page(
            node,
            run.port,
            "(async () => { location.hash = '#/verdict/" + name + "';"
            " await new Promise(r => setTimeout(r, 400));"
            " const opened = zoomEl.classList.contains('on');"
            " const src = zoomImg.src;"
            " closeZoom();"
            " return {opened, src, hash: location.hash, tab}; })()",
        )
    assert followed["opened"], "a hash nobody clicked did nothing"
    assert followed["src"].endswith("/crops/" + name), followed
    # and back to the tab on closing: assigning a hash its own value fires no
    # event, so a link left in the url could not be followed a second time.
    assert followed["hash"] == "#/verdicts", followed


def test_opening_a_tile_leaves_its_link_behind(binary, config, harvest, node, moving_clip):
    """**a url nobody can find is one nobody uses.** the crop a person is looking
    at should be copyable out of the location bar, and the tile is where they are
    looking at it -- so opening one names it, without routing back through the hash
    to fetch the entry and draw the same picture twice.
    """
    name = verdict_name(AT, "go4")
    with Preview(binary, config, moving_clip).ready() as run:
        got = in_page(
            node,
            run.port,
            "(async () => { showTab('verdicts');"
            " await new Promise(r => setTimeout(r, 1500));"
            " const tile = document.getElementById('verdicts-grid').children[0];"
            " tile.children[0].onclick();"
            " return {hash: location.hash, open: zoomEl.classList.contains('on'),"
            "   src: zoomImg.src}; })()",
        )
    assert got["open"], "the tile did not open a crop"
    assert got["hash"] == "#/verdict/" + name, got


def test_one_crop_answers_for_itself(binary, config, harvest, node, moving_clip):
    """`/crop?n=` is the listing entry for one name, which is what a link the
    page has never paged to resolves against.

    read off the name rather than by walking the directory: the name is the
    record, and the file has to be there for the page to show it either way.
    """
    name = verdict_name(AT, "go4")
    with Preview(binary, config, moving_clip).ready() as run:
        status, body = get(run.port, f"/crop?n={name}")
        entry = json.loads(body)
        missing = get(run.port, f"/crop?n={verdict_name(AT + 1000, 'go4')}")[0]
        escaped = get(run.port, "/crop?n=../../../etc/passwd")[0]
        junk = get(run.port, "/crop?n=not-a-crop")[0]

    assert status == 200, body
    assert entry["n"] == name, entry
    assert entry["t"] == AT, entry
    assert entry["s"] == "go4", entry
    assert missing == 404, "a crop that is not on disk was answered"
    assert escaped == 404, "a name that leaves the harvest was answered"
    assert junk == 404


def test_the_page_fits_a_phone(binary, config, harvest, node, moving_clip):
    """no viewport meta means a phone renders the page at desktop width and hands
    the person a pinch gesture, so the first thing a responsive page needs is the
    one tag that says otherwise -- and a layout for the touch pointers it names.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        html = get(run.port, "/")[1].decode()
    assert 'name="viewport"' in html, "the page still asks for a desktop viewport"
    assert "@media (pointer: coarse)" in html, "no layout for a touch pointer"
