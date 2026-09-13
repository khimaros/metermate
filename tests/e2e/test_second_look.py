"""the second look tab: labelled crops the classifier's own vote disputes.

the tab exists to catch mislabels. a mislabel is a crop whose label the
classifier would overturn if the label were taken back -- labelled the subject
but voting with the negatives, or labelled `other` but voting with the subject
-- so each direction is ranked the way the classifier votes: the mean of a
crop's three nearest in the opposite class, less the same in its own. the top
fifty of each are offered, always, whether or not any of them crosses zero.

`unclear` is the third direction and the only one that is not about a mistake:
it is a question a person could not answer, parked in `labels.txt` and offered
by nothing else, so the ones the classifier now votes for come back here.

driven through the real binary with a seeded vector cache, so the vectors are
chosen rather than embedded and the geometry each case rests on is written down
beside it.
"""

import math
import re
from pathlib import Path

import pytest

from labelling import SUBJECT, Labelling, blank_crops, build_config, plain_name, write_cache

AT = 1789000000000
# well past `label::PASSAGE_GAP_MS`, so every group below is its own passage.
PASSAGE_MS = 60_000
# well inside it, so the crops within a group are one passage.
FRAME_MS = 400
# `server::SECOND_LOOK_OFFERED`, per direction.
OFFERED = 50
# `server::RANDOM_OFFERED`: how many undecided crops the random pool takes
# before anything is left for the ranked one.
RANDOM_POOL = 200


def unit(*xs: float) -> list[float]:
    norm = math.sqrt(sum(x * x for x in xs))
    return [x / norm for x in xs]


def passage(size: int, *direction: float) -> list[list[float]]:
    """`size` near-identical crops of one thing, as the frames of a passage are."""
    return [unit(*direction, 0.001 * f) for f in range(size)]


def build(
    tmp_path: Path, ffmpeg: str, groups: list, spare: int = 0
) -> tuple[Path, list[list[str]]]:
    """one passage per `(truth, vectors)` group, and the crop names of each.

    `spare` adds embedded crops with no label line, which is what the queue's
    ranked pool is drawn from.
    """
    names, vectors, rows, grouped = [], [], [], []
    for g, (truth, members) in enumerate(groups):
        run = [plain_name(AT + g * PASSAGE_MS + f * FRAME_MS) for f in range(len(members))]
        names += run
        vectors += members
        rows += [f"{name} {truth} seed" for name in run]
        grouped.append(run)
    for s in range(spare):
        names.append(plain_name(AT + (len(groups) + s) * PASSAGE_MS))
        # street-ish and distinct, so the ranked pool has something to order.
        vectors.append(unit(0, 1, 0, 0.01 * s))
    crops = tmp_path / "crops"
    blank_crops(crops, ffmpeg, names)
    write_cache(tmp_path / "embeddings.bin", names, vectors)
    (tmp_path / "labels.txt").write_text("".join(row + "\n" for row in rows))
    return crops, grouped


# four passages labelled the subject, six negatives, and the three single crops
# each direction's rules turn on. indices are what the tests below unpack.
LAYOUT = [
    (SUBJECT, passage(4, 1, 0, 0)),
    (SUBJECT, passage(4, 1, 0, 0)),
    # honestly the subject, just nearer everything: small, distant.
    (SUBJECT, passage(4, 1, 1, 0)),
    # labelled the subject, and not one.
    (SUBJECT, passage(4, 0, 0.6, 0.8)),
    *[("other", passage(1, 0, 1, 0)) for _ in range(6)],
    # labelled other, and is one.
    ("other", passage(1, 1, 0, 0)),
    ("waymo", passage(1, 1, 0, 0)),
    ("unclear", passage(1, 1, 0, 0)),
]
MISLABEL, PLANTED, WAYMO, UNCLEAR = 3, 10, 11, 12


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return build_config(tmp_path)


def review(run: Labelling) -> list[dict]:
    return [i for i in run.queue()["items"] if i["via"] == "review"]


def test_a_passage_labelled_wrong_leads_its_direction(binary, config, tmp_path, ffmpeg):
    """**labelled the subject, voting with the negatives: first.**

    two rankings that look reasonable both put it last, which is why the case is
    built this way:

    - by raw similarity to the negatives, the generic passage leads. it is the
      subject, just nearer everything, and it resembles the negatives more
      (0.71) than the mislabel does (0.60).
    - leaving out only the crop itself lets the rest of its passage vote for
      it. frames of one passage are near-copies, so every crop's own score
      becomes 1.0 and the ranking collapses back into raw similarity.

    leaving its whole passage out is what separates them: the mislabel is then
    measured against the subject elsewhere (0.42) and loses to the negatives by
    0.18, while the generic passage breaks even.
    """
    crops, groups = build(tmp_path, ffmpeg, LAYOUT)
    with Labelling(binary, config, crops).ready() as run:
        mine = [(i["n"], i["score"]) for i in review(run) if i["truth"] == SUBJECT]
        assert {n for n, _ in mine[:4]} == set(groups[MISLABEL]), mine
        # every crop labelled the subject is ranked, not only the ones past a line.
        assert len(mine) == 16, mine
        scores = [s for _, s in mine]
        assert scores == sorted(scores, reverse=True), mine


def test_an_other_crop_voting_with_the_subject_leads_the_other_direction(
    binary, config, tmp_path, ffmpeg
):
    """**labelled `other`, voting with the subject: first.**

    a sweep marks a screenful `other` in one key, so a subject crop swept up
    with the traffic is a mislabel this project is especially likely to make.

    only crops labelled `other` are offered this way. one labelled another
    subject still counts as a negative, but a click here would overwrite
    `waymo` with this subject or with `other` -- something false about the crop
    rather than about this subject. `unclear` is offered, in its own direction.
    """
    crops, groups = build(tmp_path, ffmpeg, LAYOUT)
    with Labelling(binary, config, crops).ready() as run:
        offered = review(run)
        theirs = [(i["n"], i["score"]) for i in offered if i["truth"] == "other"]
        assert theirs and theirs[0][0] == groups[PLANTED][0], theirs
        assert len(theirs) == 7, theirs
        assert groups[WAYMO][0] not in {i["n"] for i in offered}, offered


def test_unclear_crops_come_back_most_like_the_subject_first(binary, config, tmp_path, ffmpeg):
    """**`unclear` is a decision to come back to, and nothing came back.**

    a person marks a crop unclear when they looked and could not tell: clipped
    by the frame edge, too small to read. it is then excluded from both halves
    of every measurement -- rightly, since counting it either way invents an
    answer nobody could give -- and so it sits there forever, with no pool that
    ever offers it again.

    the classifier's vote is the thing that changed in the meantime. an unclear
    crop it now votes *for* is worth looking at twice, because the question it
    was parked on is exactly the one the vote answers.
    """
    crops, groups = build(tmp_path, ffmpeg, LAYOUT)
    with Labelling(binary, config, crops).ready() as run:
        offered = review(run)
        unclear = [(i["n"], i["score"]) for i in offered if i["truth"] == "unclear"]
        assert [n for n, _ in unclear] == groups[UNCLEAR], offered
        # ranked the way the other two are, and in the direction that makes an
        # unclear crop interesting: above zero votes with the subject.
        assert unclear[0][1] > 0, unclear


def test_opening_a_session_says_what_it_is_comparing(binary, config, tmp_path, ffmpeg):
    """**opening a session is the other long silence.**

    each direction of the second look compares every labelled crop against
    every negative, and the queue then scores every still-undecided crop
    against every known example. both grow with how much labelling has already
    been done -- the work is smallest on the day the tool is least useful -- and
    both used to announce themselves only once finished, so a harvest with
    thousands of labels looked like a server that had hung before the browser
    could reach it.
    """
    # several of each: a direction holding one crop reports only its total,
    # since a decile line at 100% would say nothing the next line does not.
    crops, _ = build(
        tmp_path,
        ffmpeg,
        [
            *[(SUBJECT, passage(2, 1, 0, 0)) for _ in range(3)],
            *[("other", passage(2, 0, 1, 0)) for _ in range(3)],
            *[("unclear", passage(2, 1, 0, 0)) for _ in range(3)],
        ],
        # past `server::RANDOM_OFFERED`, or every undecided crop lands in the
        # random pool and the ranked pass has nothing left to score.
        spare=RANDOM_POOL + 10,
    )
    with Labelling(binary, config, crops).ready() as run:
        out = run.output()

    for unit in (f"crops labelled {SUBJECT}", "crops labelled other", "crops marked unclear"):
        assert re.search(rf"comparing {unit}: \d+/\d+ \(\d+%\)", out), (unit, out)
        assert re.search(rf"compared \d+ {unit} in ", out), (unit, out)

    assert re.search(r"ranking undecided crops: \d+/\d+ \(\d+%\)", out), out
    assert re.search(r"ranked \d+ undecided crops in ", out), out


def test_a_subject_seen_in_one_passage_is_still_ranked(binary, config, tmp_path, ffmpeg):
    """a new subject starts as one passage, and that set is the likeliest to hold a mistake.

    with its own passage left out there is nothing left to compare a crop
    against, so a class seen only once leaves out just the crop itself rather
    than offering nothing.
    """
    crops, groups = build(
        tmp_path,
        ffmpeg,
        [(SUBJECT, passage(4, 1, 0, 0)), *[("other", passage(1, 0, 1, 0)) for _ in range(3)]],
    )
    with Labelling(binary, config, crops).ready() as run:
        mine = [i["n"] for i in review(run) if i["truth"] == SUBJECT]
        assert sorted(mine) == sorted(groups[0]), review(run)


def test_each_direction_offers_at_most_fifty(binary, config, tmp_path, ffmpeg):
    """a budget, not a measurement: fifty a direction is one honest sitting."""
    crops, _ = build(
        tmp_path,
        ffmpeg,
        [(SUBJECT, passage(1, 1, 0, 0)) for _ in range(OFFERED + 10)]
        + [("other", passage(1, 0, 1, 0)) for _ in range(OFFERED + 10)]
        + [("unclear", passage(1, 1, 0, 0)) for _ in range(OFFERED + 10)],
    )
    with Labelling(binary, config, crops).ready() as run:
        offered = review(run)
        assert len([i for i in offered if i["truth"] == SUBJECT]) == OFFERED
        assert len([i for i in offered if i["truth"] == "other"]) == OFFERED
        assert len([i for i in offered if i["truth"] == "unclear"]) == OFFERED
