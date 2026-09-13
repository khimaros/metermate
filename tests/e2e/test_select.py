"""choosing a passage to cut crops from, and the two commands that follow.

the loop this closes: watch an event in the preview, mark the seconds the
vehicle is in frame, and later run one command that cuts every selection into
a set and one that retrains on everything labelled. what was done by hand was
reading a clip name off a card, timing the passage with the player's clock,
and typing `--dense ... --window ... --into ...` per event.

**the selection is a note, not an extraction.** cutting crops loads the
detector and reads every frame of the clip, which is not something a page
should start while the pipeline is using the machine to watch the street
(r2.2, r3.1). so the page writes down what to do and `--prepare` does it.
"""

import subprocess
from pathlib import Path

import pytest

from conftest import REPO, in_page
from preview import Preview, post
from test_dense import FPS, PREROLL_MS, STAMP, street_clip

SUBJECT = "go4"


@pytest.fixture
def events(ffmpeg: str, tmp_path: Path) -> Path:
    """an events directory holding one recorded clip, as the recorder leaves it."""
    dir = tmp_path / "events"
    dir.mkdir()
    street_clip(ffmpeg, dir, FPS)
    return dir


@pytest.fixture
def config(tmp_path: Path, events: Path) -> Path:
    path = tmp_path / "metermate.toml"
    path.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[stream]
crop_fps = {FPS}

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
dir = "{tmp_path / "crops"}"

[record]
dir = "{events}"
preroll_secs = {PREROLL_MS // 1000}

[train]
sets = "{tmp_path / "sets"}"
""")
    return path


def run(binary: Path, config: Path, *args: str, timeout: int = 300):
    return subprocess.run(
        [str(binary), "--config", str(config), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_a_clip_and_a_window_can_be_marked_from_the_events_tab(
    binary, config, events, tmp_path, moving_clip
):
    """the card knows the clip and the player knows the seconds, so nothing
    has to be read off a screen and typed into a shell."""
    with Preview(binary, config, moving_clip).ready() as run_:
        clip = run_.clips()["clips"][0]["n"]
        status, body = post(run_.port, "/select", {"n": clip, "window": "0:03-0:08"})
        assert status == 200, (status, body, run_.output())
        assert run_.selections() == [{"n": clip, "window": "0:03-0:08"}], run_.selections()

    written = (tmp_path / "selections.txt").read_text()
    assert clip in written and "0:03-0:08" in written, written


def test_a_window_that_is_not_one_is_refused(binary, config, events, tmp_path, moving_clip):
    """the window is the argument `--dense` takes, so it is parsed here rather
    than failing hours later when the command is run."""
    with Preview(binary, config, moving_clip).ready() as run_:
        clip = run_.clips()["clips"][0]["n"]
        for bad in ("", "0:08-0:03", "later", "-", "0:03/0:08"):
            status, _ = post(run_.port, "/select", {"n": clip, "window": bad})
            assert status == 400, (bad, status)
        # `3-` is not one of them: an open end means "to the end of the clip",
        # which is what marking the start of a passage and letting it run is.
        # and a clip that is not in the events directory
        assert post(run_.port, "/select", {"n": "../x.mp4", "window": "0-1"})[0] == 404
        assert run_.selections() == [], run_.selections()


def test_a_mark_can_be_taken_back(binary, config, events, tmp_path, moving_clip):
    """**marking is a judgement made in a second, from a moving picture.**

    the window is read off a player while a vehicle crosses, so getting it
    wrong -- the wrong clip, the passage before the one meant, three seconds
    of an empty street -- is ordinary. without this the only way back is
    editing a text file on the deployment, or letting `--prepare` spend
    minutes cutting a passage nobody wants and deleting the set afterwards.
    """
    with Preview(binary, config, moving_clip).ready() as run_:
        clip = run_.clips()["clips"][0]["n"]
        assert post(run_.port, "/select", {"n": clip, "window": "1-2"})[0] == 200
        assert post(run_.port, "/select", {"n": clip, "window": "5-9"})[0] == 200

        assert post(run_.port, "/unselect", {"n": clip, "window": "1-2"})[0] == 200
        assert run_.selections() == [{"n": clip, "window": "5-9"}], run_.selections()

        # taking back one that is not there is not an error: two clicks on the
        # same `x` is a person making sure, not a mistake to report.
        assert post(run_.port, "/unselect", {"n": clip, "window": "1-2"})[0] == 200
        assert run_.selections() == [{"n": clip, "window": "5-9"}], run_.selections()

    written = (tmp_path / "selections.txt").read_text()
    assert "5-9" in written and "1-2" not in written, written


def test_what_is_marked_is_shown_on_the_card_that_marked_it(
    binary, config, events, node, moving_clip
):
    """a mark with nothing to show for it is a mark made twice. the windows
    waiting to be cut belong on the card they were taken from, which is also
    where the `x` that removes one belongs."""
    with Preview(binary, config, moving_clip).ready() as run_:
        clip = run_.clips()["clips"][0]["n"]
        assert post(run_.port, "/select", {"n": clip, "window": "1-2"})[0] == 200

        shown = in_page(
            node,
            run_.port,
            # the worth filter is a `<select>`, and its value comes from the
            # option the markup marks as selected -- which a stub document has
            # not got. everything kept, which is what the page opens on.
            "(async () => { worthFilter.value = 'motion'; showTab('events');"
            " await new Promise(r => setTimeout(r, 600));"
            " const find = e => e.className === 'marks' ? e :"
            "   (e.children || []).map(find).find(Boolean);"
            " const marks = find(document.getElementById('clips'));"
            " return marks ? marks.children.map(k => k.textContent) : null; })()",
        )
    assert shown and any("1-2" in text for text in shown), shown


def test_preparing_cuts_every_selection_into_its_own_set(
    binary, config, events, tmp_path, moving_clip
):
    """**one command for all of it.** the hand version was a `--dense` line per
    event, each with a clip path and a window copied off a card."""
    with Preview(binary, config, moving_clip).ready() as run_:
        clip = run_.clips()["clips"][0]["n"]
        assert post(run_.port, "/select", {"n": clip, "window": "1-2"})[0] == 200

    done = run(binary, config, "--prepare")
    output = done.stdout + done.stderr
    assert done.returncode == 0, output

    # sets/<subject>/<clip stamp>-<window>/crops
    sets = sorted((tmp_path / "sets").glob("*/*/crops"))
    assert len(sets) == 1, list((tmp_path / "sets").rglob("*"))
    assert list(sets[0].glob("*.jpg")), output
    # the clip travels with the crops, so the set outlives the recording.
    assert list(sets[0].parent.glob("clips/*.mp4")), output
    assert str(STAMP) in str(sets[0].parent), sets[0]


def test_a_passage_marked_again_after_cutting_is_not_cut_twice(
    binary, config, events, tmp_path, moving_clip
):
    """cutting drains the queue, so the ordinary second run has nothing to do.
    marking the same passage again is the case the guard is for: the detector
    pass is minutes, and a second copy of a passage would be counted twice by
    everything downstream."""
    with Preview(binary, config, moving_clip).ready() as run_:
        clip = run_.clips()["clips"][0]["n"]
        assert post(run_.port, "/select", {"n": clip, "window": "1-2"})[0] == 200
        assert run(binary, config, "--prepare").returncode == 0
        before = sorted(p.name for p in (tmp_path / "sets").rglob("*.jpg"))

        # cut, so nothing is waiting any more.
        assert run_.selections() == [], run_.selections()

        assert post(run_.port, "/select", {"n": clip, "window": "1-2"})[0] == 200
        again = run(binary, config, "--prepare")
        output = again.stdout + again.stderr

    assert again.returncode == 0, output
    assert sorted(p.name for p in (tmp_path / "sets").rglob("*.jpg")) == before, output
    assert "already" in output.lower(), output


def test_preparing_nothing_says_so_rather_than_failing(binary, config, events, tmp_path):
    """an empty selection list is the normal state, not an error: the command
    is run on a schedule and on a whim."""
    done = run(binary, config, "--prepare")
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert "nothing" in output.lower(), output


def test_retraining_trains_every_subject_from_the_harvest_and_the_sets(
    binary, config, tmp_path, ffmpeg
):
    """**one command, whatever has been labelled since.** the hand version is
    `--train <subject> --harvest data/crops --harvest sets/` per subject, and
    the subject list is in the config already.
    """
    from test_train import CROPS_PER_PASSAGE, NEGATIVES, PASSAGES, crop_name, solid_jpeg

    crops = tmp_path / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    at = 1789000000000
    for p in range(PASSAGES):
        for c in range(CROPS_PER_PASSAGE):
            name = crop_name(at + p * 60_000 + c * 400)
            solid_jpeg(ffmpeg, crops / name, "red")
            rows.append(f"{name} {SUBJECT} preview")
    for i in range(NEGATIVES):
        name = crop_name(at + 600_000 + i * 20_000, w=200, h=180)
        solid_jpeg(ffmpeg, crops / name, "gray")
        rows.append(f"{name} other random")
    (tmp_path / "labels.txt").write_text("\n".join(rows) + "\n")

    cfg = tmp_path / "retrain.toml"
    cfg.write_text(
        config.read_text()
        + f'\n[[subject]]\nname = "{SUBJECT}"\n'
        + f'\n[classifier]\nmodel = "{REPO / "models" / "embedder.onnx"}"\n'
        + f'references = "{tmp_path / "trained"}"\n'
    )
    done = run(binary, cfg, "--retrain", timeout=600)
    output = done.stdout + done.stderr

    assert done.returncode == 0, output
    assert (tmp_path / "trained" / SUBJECT / "references.txt").exists(), output
    assert (tmp_path / "trained" / SUBJECT / "trained.toml").exists(), output
    # it says what it did, per subject, because the answer to "did that work"
    # is the measurement rather than the exit code.
    assert SUBJECT in output and "margin" in output, output
