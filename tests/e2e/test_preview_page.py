"""the live page's own javascript, run against the real server.

the same argument the labelling page's tests make: a rule that is right in the
source and wrong in the rendered page is a rule nobody gets the benefit of.
this page carries more of it now that labelling and selection happen here.
"""

import json
import shutil
import threading
import time
from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO, in_page
from labelling import blank_crops, plain_name, verdict_name
from preview import Preview, get, post
from test_dense import FPS, PREROLL_MS, STAMP, street_clip

AT = 1789000000000
# later than the clip `street_clip` writes, which is how the next clip closed by
# the recorder is named.
STAMP_LATER = STAMP + 60_000


@pytest.fixture
def events(ffmpeg: str, tmp_path: Path) -> Path:
    dir = tmp_path / "events"
    dir.mkdir()
    street_clip(ffmpeg, dir, FPS)
    return dir


# the one crop stage two named, and how far clear of its negatives it was.
MARGIN = 0.02


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    crops = tmp_path / "crops"
    blank_crops(
        crops,
        ffmpeg,
        [plain_name(AT + i * 60_000) for i in range(3)]
        + [verdict_name(AT + 3 * 60_000, "go4", margin=MARGIN)],
    )
    return crops


@pytest.fixture
def crop(harvest: Path) -> str:
    """one crop on the page, by name."""
    return plain_name(AT)


@pytest.fixture
def verdict(harvest: Path) -> str:
    """the crop stage two named, by name."""
    return verdict_name(AT + 3 * 60_000, "go4", margin=MARGIN)


def caption(node, port: int, crop: dict, recognised: bool) -> str:
    """what a card says about one crop, as the page renders it."""
    return in_page(node, port, f"captionOf({json.dumps(crop)}, {str(recognised).lower()})")


def test_a_verdict_is_listed_with_the_margin_that_named_it(
    binary, config, node, moving_clip, verdict
):
    """**the verdicts tab is tuned by a number, so it has to show it.**

    the margin is what an operator moves, and it is what a notification quotes --
    `margin +0.020`. until the crop itself carries it the two cannot be compared,
    and the only way to decide "raise the bar to +0.03" is to read a directory of
    filenames. the detector's own class and confidence are what the crops tab is
    for, and beside a margin they invite reading one as the other.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        listed = json.loads(get(run.port, "/crops?recognised=1")[1])["crops"]
        assert [c["n"] for c in listed] == [verdict], listed
        assert listed[0]["m"] == pytest.approx(MARGIN, abs=0.001), listed
        # the subject, and the number that fired it. not the detector's class,
        # which on this crop is `truck` and says nothing about stage two.
        said = caption(node, run.port, listed[0], True)
    assert said == '<b class="verdict">GO4</b> +0.02', said


def test_a_verdict_with_no_recorded_margin_says_only_that_it_fired(
    binary, config, node, moving_clip, verdict
):
    """a crop written before the margin was recorded keeps its claim and borrows
    nobody's number: printing the detector's 0.75 where the margin belongs is the
    `looks set, is not` shape again."""
    with Preview(binary, config, moving_clip).ready() as run:
        said = caption(node, run.port, {"n": verdict, "c": "truck", "p": 0.75, "s": "go4"}, True)
    assert said == '<b class="verdict">GO4</b>', said


def test_the_crops_tab_still_says_what_the_detector_saw(binary, config, node, moving_clip, verdict):
    """the other tab is a different question and keeps its own answer.

    `the detector called this a truck and stage two called it a go-4` is the case
    worth being able to find, and the harvest's size and make-up is the question
    the crops tab exists for. only the verdicts tab trades the class for the
    margin.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        named = caption(node, run.port, {"n": verdict, "c": "truck", "p": 0.75, "s": "go4"}, False)
        plain = caption(node, run.port, {"n": plain_name(AT), "c": "car", "p": 0.9}, False)
    assert named == '<b class="verdict">go4</b> &middot; truck 0.75', named
    assert plain == "<b>car</b> 0.90", plain


def standing(port: int) -> dict:
    """what the verdicts tab is listing, and the counts that explain it."""
    return json.loads(get(port, "/crops?recognised=1")[1])


def test_a_crop_labelled_the_subject_by_hand_is_on_the_verdicts_tab(
    binary, config, node, moving_clip, crop, verdict
):
    """**the tab answers "has it seen one", and a person's yes is the best answer
    there is.**

    stage two misses, and a go-4 it missed is labelled from the crops tab by
    whoever saw it. that crop then sat among two hundred cars, in neither tab
    that is about the subject. it is listed beside the verdicts now, said to be
    by hand, with no margin borrowed from anywhere.
    """
    sweeper = plain_name(AT + 60_000)
    with Preview(binary, config, moving_clip).ready() as run:
        assert post(run.port, "/label", {"n": crop, "truth": "go4"})[0] == 200
        # a name the config does not watch for is a label, not a sighting.
        assert post(run.port, "/label", {"n": sweeper, "truth": "street-sweeper"})[0] == 200
        listed = standing(run.port)["crops"]
        assert [c["n"] for c in listed] == [verdict, crop], listed
        said = caption(node, run.port, listed[1], True)

        # taken back, it leaves again.
        assert post(run.port, "/label", {"n": crop, "truth": "other"})[0] == 200
        assert [c["n"] for c in standing(run.port)["crops"]] == [verdict]
    assert said == '<b class="verdict">GO4</b> by hand', said


def test_a_verdict_marked_unclear_leaves_the_verdicts_tab(
    binary, config, node, moving_clip, verdict
):
    """**`unclear` is an answer, and the tab is a list of questions.**

    a verdict nobody could call used to stay on the tab for good, since it was
    never rejected -- so the crops that could not be settled were the ones
    looked at every time. it leaves like any other judged verdict, is counted
    apart from the rejections because it is not one, and stays on the crops tab
    and in the labelling page's second look, which is where it comes back from.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        assert [c["n"] for c in standing(run.port)["crops"]] == [verdict]
        assert post(run.port, "/label", {"n": verdict, "truth": "unclear"})[0] == 200

        after = standing(run.port)
        assert after["crops"] == [], after
        assert (after["rejected"], after["unclear"]) == (0, 1), after
        assert verdict in [c["n"] for c in run.crops()["crops"]], run.crops()

        said = in_page(
            node,
            run.port,
            "(async () => { await verdicts.refresh();"
            " return document.getElementById('verdicts-stats').innerHTML; })()",
        )
    # this config runs no classifier, which the bar says before anything else.
    assert said == "classification is off; 1 marked unclear", said


@pytest.fixture
def config(tmp_path: Path, events: Path, harvest: Path) -> Path:
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
preroll_secs = {PREROLL_MS // 1000}
""")
    return path


def picker(node, port, crop: str, keys: str = "") -> dict:
    """open a crop's label picker, press `keys`, and report what it shows.

    driven through `labelStrip` and `openPicker` rather than through the grid:
    the grid keeps its figures in a closure, and what is worth testing is the
    picker's own behaviour against the real `/labels` and `/label`.
    """
    return in_page(
        node,
        port,
        "(async () => {"
        f" const c = {{n: {crop!r}, truth: ''}};"
        " const row = labelStrip(c);"
        " row.children.find(k => k.textContent === 'edit').onclick({stopPropagation() {}});"
        " await new Promise(r => setTimeout(r, 400));"
        " const box = row.children.find(k => k.className === 'picker');"
        " const field = box.children[0], list = box.children[1];"
        f" {keys}"
        " await new Promise(r => setTimeout(r, 400));"
        " return {options: list.children.map(o => o.textContent),"
        # by token: `option` contains the letters of `on`, which a substring
        # test reads as every row being the highlighted one.
        "   on: list.children.filter(o => o.className.split(' ').includes('on'))"
        "     .map(o => o.textContent),"
        "   open: !!row.children.find(k => k.className === 'picker'),"
        "   truth: c.truth}; })()",
    )


def test_the_picker_opens_on_the_labels_in_use(binary, config, node, moving_clip, crop):
    """**offered before anything is typed.** the answer is almost always one
    of the last few labels used, and making somebody type it in full every
    time is the friction that stops crops being labelled at all."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = picker(node, run.port, crop)
    # the configured subject and the two reserved answers, with no labelling
    # done yet: a fresh deployment does not open on an empty list.
    assert any(o.startswith("go4") for o in got["options"]), got
    assert any(o.startswith("other") for o in got["options"]), got
    assert got["on"], "nothing was highlighted, so enter would do nothing"


def test_typing_filters_and_the_arrows_move(binary, config, node, moving_clip, crop):
    """type to search, arrows to choose: the list is short enough to click and
    the keyboard is faster than the mouse for the one on screen."""
    with Preview(binary, config, moving_clip).ready() as run:
        typed = picker(node, run.port, crop, "field.value = 'u'; field.oninput();")
        assert all("u" in o for o in typed["options"]), typed

        moved = picker(
            node,
            run.port,
            crop,
            "field.onkeydown({key: 'ArrowDown', stopPropagation() {}});",
        )
        assert moved["on"] == [moved["options"][1]], moved


def test_enter_labels_the_crop_with_the_highlighted_label(
    binary, config, node, moving_clip, crop, tmp_path
):
    """the whole point of the control: one key, and the decision is in
    `labels.txt` where everything else reads it."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = picker(
            node,
            run.port,
            crop,
            "field.value = 'unclear'; field.oninput();"
            " field.onkeydown({key: 'Enter', stopPropagation() {}});",
        )
        assert got["truth"] == "unclear", got
        assert not got["open"], "the picker stayed open after committing"
        after = {c["n"]: c.get("truth") for c in run.crops()["crops"]}

    assert after[crop] == "unclear", after


def test_a_name_nobody_has_used_is_offered_as_a_new_label(binary, config, node, moving_clip, crop):
    """**the case the old row of buttons could not serve.** what actually
    happens is seeing something the classifier has no name for, and a control
    that only offers last week's answers cannot take this week's."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = picker(
            node,
            run.port,
            crop,
            "field.value = 'street-sweeper'; field.oninput();"
            " field.onkeydown({key: 'Enter', stopPropagation() {}});",
        )
        assert got["truth"] == "street-sweeper", got
        after = {c["n"]: c.get("truth") for c in run.crops()["crops"]}

    assert after[crop] == "street-sweeper", after


def test_a_name_the_config_would_refuse_is_not_sent(binary, config, node, moving_clip, crop):
    """it would become an mqtt topic segment and a home assistant id if it
    grew into a subject, so it is refused in the field rather than after a
    round trip -- and the picker stays open, holding what was typed."""
    with Preview(binary, config, moving_clip).ready() as run:
        got = picker(
            node,
            run.port,
            crop,
            "field.value = 'Street Sweeper'; field.oninput();"
            " field.onkeydown({key: 'Enter', stopPropagation() {}});",
        )
        assert got["truth"] == "", got
        assert got["open"], "a refused name closed the picker anyway"


def test_marking_a_passage_does_not_close_the_clip_being_marked(
    binary, config, events, node, moving_clip
):
    """**every click in the overlay that is not the video closes it.**

    that is what makes the backdrop dismissable, and it made the first click
    of marking a passage -- `in` -- shut the clip being marked. found by using
    it rather than by reading it: the handler is three lines away from the one
    that stops the video's own controls doing the same thing.
    """
    with Preview(binary, config, moving_clip).ready(on="/clips") as run:
        open_after = in_page(
            node,
            run.port,
            # the chain a browser would walk: the button, then the bar it is
            # in, then the backdrop -- stopping wherever propagation stops.
            "(() => { zoomClip('/clips/x.mp4', 'x.mp4');"
            " const e = {stopped: false, stopPropagation() { this.stopped = true; }};"
            " markIn.onclick(e);"
            " if (!e.stopped) markEl.onclick(e);"
            " if (!e.stopped) zoomEl.onclick(e);"
            " return zoomEl.classList.contains('on'); })()",
        )
        assert open_after, "marking a passage closed the clip"


def test_marking_reads_the_players_own_clock(binary, config, events, node, moving_clip):
    """the window is `m:ss-m:ss` because that is what the player shows, and
    what `--dense` takes -- so nothing has to be converted by whoever is
    watching."""
    with Preview(binary, config, moving_clip).ready(on="/clips") as run:
        window = in_page(
            node,
            run.port,
            "(() => { zoomClip('/clips/x.mp4', 'x.mp4');"
            " zoomVid.currentTime = 63;"
            " markIn.onclick({stopPropagation() {}});"
            " zoomVid.currentTime = 128;"
            " markOut.onclick({stopPropagation() {}});"
            " return markedWindow(); })()",
        )
        assert window == "1:03-2:08", window


def test_an_out_before_an_in_is_not_a_passage(binary, config, events, node, moving_clip):
    """the server would refuse it, and a round trip is a worse way to learn
    that a mis-click was a mis-click."""
    with Preview(binary, config, moving_clip).ready(on="/clips") as run:
        window = in_page(
            node,
            run.port,
            "(() => { zoomClip('/clips/x.mp4', 'x.mp4');"
            " zoomVid.currentTime = 30;"
            " markIn.onclick({stopPropagation() {}});"
            " zoomVid.currentTime = 10;"
            " markOut.onclick({stopPropagation() {}});"
            " return markedWindow(); })()",
        )
        # the start stands and the end is still open: marking only a start is
        # a passage that runs to the end of the clip.
        assert window == "0:30-", window


def test_the_picture_is_handed_the_shape_it_is_given(binary, config, node, moving_clip):
    """**the frame keeps the encoded shape, and the css is what sizes it.**

    the live view used to cap the stage at `95vh` of width, which leaves a band of
    empty screen under a desktop picture that could have used it, and sizes a
    phone's picture by a rule that knows nothing about the header above it. the
    shape now reaches the page as a pair on the root element, unchanged from what
    the encoder sent, and the css makes it the largest frame of that shape the
    window has room for.

    the fitting itself is css and wants a browser; what is checked here is that the
    shape it needs arrives, and from what -- and that it is held in one place, since
    the shape is what keeps the overlay's boxes on the vehicles they belong to.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        encode = json.loads(get(run.port, "/config")[1])["encode"]
        got = in_page(
            node,
            run.port,
            "(() => { draw({w: 640, h: 480});"
            " return {shape: [htmlEl.style['--frame-w'] || '',"
            "   htmlEl.style['--frame-h'] || ''],"
            "   stage: document.getElementById('stage').style.aspectRatio || ''}; })()",
        )
    assert encode, "the server did not say what the stream was encoded to"
    assert got["shape"] == [str(encode[0]), str(encode[1])], got
    # the shape lives in one place: a stage that still carries it is a second
    # answer to a question the css has already been given.
    assert not got["stage"], got


def test_the_picture_is_left_the_room_the_page_spends(binary, config, node, moving_clip):
    """**what the picture gets is the window minus what the page spends.**

    the header wraps into a different height on every window, the legend is prose,
    the roi editor grows a line of toml, and a phone turns sideways -- so the height
    that is spent around the picture is measured rather than assumed. what is said
    is the sum of the three, and it is said again when any of them changes.

    a hidden live view says nothing at all, which matters most: an element nobody is
    looking at reports no heights, and a page that subtracts zero from the window
    hands the picture the whole screen and pushes the legend off it.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        got = in_page(
            node,
            run.port,
            "(() => { const stage = document.getElementById('stage'),"
            " legend = document.getElementById('live-legend'),"
            " tools = document.getElementById('roi-tools');"
            " stage.offsetTop = 40; legend.offsetHeight = 90; tools.offsetHeight = 0;"
            " fit();"
            " const plain = htmlEl.style['--room'] || '';"
            " legend.offsetHeight = 120; tools.offsetHeight = 60; fit();"
            " const grown = htmlEl.style['--room'] || '';"
            " stage.offsetParent = null; legend.offsetHeight = 10; fit();"
            " return {plain, grown, hidden: htmlEl.style['--room'] || ''}; })()",
        )
    assert got["plain"] == "130px", got
    assert got["grown"] == "220px", got
    # the hidden view leaves the last honest answer standing rather than writing 10px
    # over it, which is what the picture is sized from next time it is shown.
    assert got["hidden"] == "220px", got


# ---- arrivals and the scroll position -----------------------------------

# how far the page has to be from the top before anything counts as away from it. a
# wheel that overshot by two pixels is somebody who has arrived.
AWAY = 400

# when the card turns up, and how long the page is left polling for it. a card the
# pipeline wrote reaches the page within one poll, but one the test wrote behind the
# server's back has to wait for the server to walk the directory again -- how long
# that takes is `harvest::LISTING_TTL`'s business, so the poll is bounded rather than
# guessed at. a card that never turns up fails the test rather than passing quietly.
WRITE_MS = 1200
AWAY_POLL_MS = 1500
AWAY_WAIT_MS = 45_000

# how each grid is polled, what it draws into, and what says something is waiting.
GRIDS = {
    "crops": ("crops.refresh()", "grid", "crops-waiting", ""),
    # the stub of a `<select>` has no value of its own, and the events grid ranks
    # every clip against whatever the filter says.
    "events": ("loadClips()", "clips", "events-waiting", " worthFilter.value = 'motion';"),
}


def arrived(node, port, write, name: str) -> dict:
    """one open page, polled away from the top until a card turns up.

    the polls have to share a page: what is measured is what a grid with cards already
    on screen does when another arrives, and the harness hands every expression a
    fresh page. so the card is written from a thread while the page is open, and the
    polls are the ones the tab's timer would make -- called directly because the grid
    under test is not the one on screen.
    """
    poll, into, waits, setup = GRIDS[name]
    thread = threading.Thread(target=lambda: (time.sleep(WRITE_MS / 1000), write()), daemon=True)
    thread.start()
    try:
        return in_page(
            node,
            port,
            f"""(async () => {{
  const wait = (ms) => new Promise(r => setTimeout(r, ms));
  const grid = document.getElementById('{into}');
  const waits = document.getElementById('{waits}');
  const seen = () => ({{cards: grid.children.length, said: waits.hidden ? '' : waits.textContent}});
  {setup}
  window.scrollY = 0;
  await {poll};
  const atTop = seen();
  window.scrollY = {AWAY};
  let most = atTop.cards, waited = 0;
  while (waits.hidden && waited < {AWAY_WAIT_MS}) {{
    await wait({AWAY_POLL_MS});
    waited += {AWAY_POLL_MS};
    await {poll};
    most = Math.max(most, grid.children.length);
  }}
  const away = seen();
  window.scrollY = 0;
  await {poll};
  return {{atTop, away, back: seen(), most, waited}};
}})()""",
        )
    finally:
        thread.join(timeout=90)


def test_a_new_crop_waits_while_the_page_is_scrolled_away(
    binary, config, ffmpeg, node, quiet_clip, harvest
):
    """**a grid that grows while you read it moves what you are reading.**

    every card is prepended, so a crop arriving four seconds after the last one takes
    the row you were looking at and puts it somewhere below. the poll keeps running,
    because the bar above the grid says what the harvest costs against its budget and
    that has to stay current -- but what arrives while the page is away from the top
    waits behind a button that says how many, rather than being put in.

    this one costs half a minute, and the server's own cache is why: a crop written
    by anything but the pipeline is invisible to `/crops` until the listing is walked
    again, so the test is watching for the crop as long as `LISTING_TTL` says.
    """
    with Preview(binary, config, quiet_clip).ready() as run:
        got = arrived(
            node,
            run.port,
            lambda: blank_crops(harvest, ffmpeg, [plain_name(AT + 9_000_000)]),
            "crops",
        )
    assert got["atTop"]["cards"] > 1, got
    assert got["atTop"]["said"] == "", got
    assert got["away"]["said"] == "1 waiting", got
    # nothing was put in while the page was away, across every poll it took.
    assert got["most"] == got["atTop"]["cards"], got
    # and back at the top it goes where it was always going.
    assert got["back"] == {"cards": got["atTop"]["cards"] + 1, "said": ""}, got


def test_a_page_opened_away_from_the_top_still_shows_something(binary, config, node, quiet_clip):
    """**a button with nothing next to it is a broken page, not a held grid.**

    the freeze only makes sense once there are cards to hold still, and a tab opened
    halfway down -- from a notification, or from a reload of where the browser put the
    page back -- has nothing on screen yet. what arrives for that page is not waiting
    for anybody, so it goes straight in.
    """
    with Preview(binary, config, quiet_clip).ready() as run:
        got = in_page(
            node,
            run.port,
            f"""(async () => {{
  window.scrollY = {AWAY};
  worthFilter.value = 'motion';
  await crops.refresh();
  await loadClips();
  const at = (grid, waits) => ({{cards: document.getElementById(grid).children.length,
    said: document.getElementById(waits).hidden ? '' : 'waiting'}});
  return {{crops: at('grid', 'crops-waiting'), events: at('clips', 'events-waiting')}};
}})()""",
        )
    assert got["crops"]["cards"] > 1, got
    assert got["crops"]["said"] == "", got
    assert got["events"]["cards"] > 1, got
    assert got["events"]["said"] == "", got


def test_a_new_clip_waits_the_same_way(binary, config, ffmpeg, node, quiet_clip, events):
    """the same rule on the other grid, for the same reason.

    a clip is the thing a person scrolls back through in order to watch it again, and
    the events grid prepends too -- so an afternoon of recording turned the clip
    halfway down the screen into something that moved every six seconds.
    """
    first = min(events.iterdir())

    def write():
        # the one already there, named later: which is how the recorder names the
        # next clip it closes.
        shutil.copy(first, events / f"{STAMP_LATER}-subject-main.mp4")

    with Preview(binary, config, quiet_clip).ready(on="/clips") as run:
        got = arrived(node, run.port, write, "events")
    assert got["atTop"]["cards"] > 1, got
    assert got["atTop"]["said"] == "", got
    # the clips listing is walked every poll, so this one is held rather than late:
    # the whole wait is one poll.
    assert got["waited"] <= 2 * AWAY_POLL_MS, got
    assert got["away"]["said"] == "1 waiting", got
    assert got["most"] == got["atTop"]["cards"], got
    assert got["back"] == {"cards": got["atTop"]["cards"] + 1, "said": ""}, got
