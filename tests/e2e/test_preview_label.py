"""labelling a crop from the live page.

the labelling page is where a sitting of two hundred crops happens, and it is
the wrong tool for the thing that happens far more often: watching the street,
seeing the thing go past, and wanting to say so about the crop that is already
on screen. that used to mean opening a second server on another port, finding
the crop again in a pool, and labelling it there.

**the labels go in the same file.** `labels.txt` beside the harvest is the only
truth there is (r6.1), so a decision made here is a decision the labelling
page, `--measure` and `--train` all see, with no import step and no second
format.
"""

import json
from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO
from labelling import blank_crops, plain_name
from preview import Preview, get, post

AT = 1789000000000
# days apart, not minutes: one of the tests below is about a label's age,
# and crops a minute apart make every label the same age.
SPACING_MS = 2 * 24 * 60 * 60 * 1000
SUBJECT = "go4"


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    """a few crops on disk, as the live page would be showing."""
    crops = tmp_path / "crops"
    blank_crops(crops, ffmpeg, [plain_name(AT + i * SPACING_MS) for i in range(4)])
    return crops


@pytest.fixture
def config(tmp_path: Path, harvest: Path) -> Path:
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

[harvest]
dir = "{harvest}"
""")
    return path


def labels_of(tmp_path: Path) -> dict[str, tuple[str, str]]:
    """`labels.txt` as `{crop: (truth, via)}`."""
    path = tmp_path / "labels.txt"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        # comments and blanks, as `Labels::load` reads them. last line wins.
        body = line.split("#")[0].strip()
        if body:
            name, truth, via = body.split()
            rows[name] = (truth, via)
    return rows


def test_a_crop_can_be_labelled_from_the_live_page(binary, config, harvest, tmp_path, moving_clip):
    """the whole point: a decision made while watching, in the file everything
    else already reads."""
    with Preview(binary, config, moving_clip).ready() as run:
        name = run.crops()["crops"][0]["n"]
        status, body = post(run.port, "/label", {"n": name, "truth": SUBJECT})
        assert status == 200, (status, body, run.output())

    assert labels_of(tmp_path)[name][0] == SUBJECT, labels_of(tmp_path)


def test_a_label_made_here_is_never_part_of_the_random_pool(
    binary, config, harvest, tmp_path, moving_clip
):
    """**the crops grid is newest-first, not a hash sample.**

    the random pool is the only one a false positive rate can honestly be
    measured on, and what makes it honest is that nothing chooses which crops
    are in it. a label made off the live page is chosen by whoever happened to
    be watching, so it carries its own provenance and the rate keeps meaning
    what it says.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        name = run.crops()["crops"][0]["n"]
        assert post(run.port, "/label", {"n": name, "truth": "other"})[0] == 200

    assert labels_of(tmp_path)[name] == ("other", "preview"), labels_of(tmp_path)


def test_a_new_subject_can_be_named_while_labelling(binary, config, harvest, tmp_path, moving_clip):
    """**a subject starts as a name and some crops.** the first waymo was
    labelled long before anything could recognise one, and requiring a config
    edit and a restart before the first label is what stops that happening at
    the moment somebody is actually looking at one."""
    with Preview(binary, config, moving_clip).ready() as run:
        name = run.crops()["crops"][0]["n"]
        assert post(run.port, "/label", {"n": name, "truth": "street-sweeper"})[0] == 200

    assert labels_of(tmp_path)[name][0] == "street-sweeper", labels_of(tmp_path)


def test_a_decision_can_be_taken_back(binary, config, harvest, tmp_path, moving_clip):
    """a misclick on a live page is likelier than on one built for labelling,
    so the last word wins and an empty truth removes the label."""
    with Preview(binary, config, moving_clip).ready() as run:
        name = run.crops()["crops"][0]["n"]
        assert post(run.port, "/label", {"n": name, "truth": SUBJECT})[0] == 200
        assert post(run.port, "/label", {"n": name, "truth": "other"})[0] == 200
        assert run.crops()["crops"][0]["truth"] == "other", run.crops()

    assert labels_of(tmp_path)[name][0] == "other", labels_of(tmp_path)


def test_the_page_is_told_what_is_already_labelled(binary, config, harvest, tmp_path, moving_clip):
    """a grid that does not show its own decisions is one that gets the same
    crop labelled twice, and the second answer is made without seeing the
    first."""
    with Preview(binary, config, moving_clip).ready() as run:
        listed = run.crops()["crops"]
        assert all("truth" not in c or c["truth"] is None for c in listed), listed
        name = listed[0]["n"]
        assert post(run.port, "/label", {"n": name, "truth": SUBJECT})[0] == 200

        after = {c["n"]: c.get("truth") for c in run.crops()["crops"]}
        assert after[name] == SUBJECT, after
        assert sum(1 for t in after.values() if t) == 1, after


def test_a_labelling_session_does_not_clobber_a_label_made_here(
    binary, config, harvest, tmp_path, moving_clip
):
    """**two processes write this file and that is the normal case.** the
    pipeline serves the preview while `metermate --label` runs a session
    against the same harvest, which is how it is actually used.

    a session holds its labels in memory and writes the whole file back, so a
    label appended by the live page while it had the file open would vanish at
    the session's next save -- and there is no second copy of `labels.txt`
    anywhere. so a save re-reads first and carries through what it did not
    know about.
    """
    session = tmp_path / "labels.txt"
    session.write_text("# opened before anything was labelled\n")

    with Preview(binary, config, moving_clip).ready() as run:
        listed = run.crops()["crops"]
        live, other = listed[0]["n"], listed[1]["n"]
        assert post(run.port, "/label", {"n": live, "truth": SUBJECT})[0] == 200

        # what a session's save looks like: everything it holds, written over
        # the file. it never saw the label above.
        session.write_text(f"# a session writing back\n{other} other seed\n")
        assert post(run.port, "/label", {"n": live, "truth": SUBJECT})[0] == 200

    rows = labels_of(tmp_path)
    assert rows[live] == (SUBJECT, "preview"), rows
    assert rows[other] == ("other", "seed"), rows


def test_the_labels_on_offer_are_the_ones_in_use(binary, config, harvest, tmp_path, moving_clip):
    """**frecency, not an alphabet.**

    on any day two or three labels are in play and the rest are history. the
    one used a minute ago is almost certainly the next one, and a list sorted
    by name buries it under whatever begins with an `a`.

    the configured subjects and the two reserved answers are always offered,
    with no uses, so a fresh deployment does not open on an empty list.
    """
    with Preview(binary, config, moving_clip).ready() as run:
        listed = [c["n"] for c in run.crops()["crops"]]
        # `ancient` is on the oldest crop and used twice; `recent` once, on the
        # newest. frecency puts the recent one first anyway.
        for name in listed[-2:]:
            assert post(run.port, "/label", {"n": name, "truth": "ancient"})[0] == 200
        assert post(run.port, "/label", {"n": listed[0], "truth": "recent"})[0] == 200

        offered = json.loads(get(run.port, "/labels")[1])["labels"]

    order = [row["truth"] for row in offered]
    assert order[0] == "recent", offered
    assert "ancient" in order, offered
    assert {"other", "unclear"} <= set(order), offered
    assert {r["truth"]: r["uses"] for r in offered}["ancient"] == 2, offered


def test_a_crop_outside_the_harvest_cannot_be_labelled(
    binary, config, harvest, tmp_path, moving_clip
):
    """the name comes from the network and picks a file. the same rule the crop
    server already applies: it names a crop in the harvest or it is refused."""
    with Preview(binary, config, moving_clip).ready() as run:
        for bad in ("../labels.txt", "subdir/x.jpg", "nothing-here.jpg", ""):
            status, _ = post(run.port, "/label", {"n": bad, "truth": SUBJECT})
            assert status == 404, (bad, status)

    assert labels_of(tmp_path) == {}, labels_of(tmp_path)


def test_a_truth_that_is_not_a_label_is_refused(binary, config, harvest, tmp_path, moving_clip):
    """a subject name becomes an mqtt topic segment and a home assistant id, so
    the same rule the config enforces applies here -- a label invented through
    a text field must not be one the config would reject."""
    with Preview(binary, config, moving_clip).ready() as run:
        name = run.crops()["crops"][0]["n"]
        for bad in ("Go 4", "-x", "GO4", "go4!"):
            status, _ = post(run.port, "/label", {"n": name, "truth": bad})
            assert status == 400, (bad, status)

    assert labels_of(tmp_path) == {}, labels_of(tmp_path)
