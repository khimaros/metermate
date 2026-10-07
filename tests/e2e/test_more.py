"""another screenful, without training.

the labelling queue is a screenful of a harvest that keeps growing while somebody
labels it, and the way back to the rest of it used to be the train button -- which
also runs the embedder over the whole harvest and measures every label there is.
these ask for the crops and nothing else.

driven through the real binary against a harvest on disk, with crops added and
answers given while the server is up, because "it saw what landed since" is the
whole claim, and a server that read the directory once cannot make it.
"""

import time
from pathlib import Path

import pytest

from labelling import (
    AT,
    SPACING_MS,
    SUBJECT,
    Labelling,
    blank_crops,
    build_config,
    plain_name,
    verdict_name,
)


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    """a harvest with **no seeded vector cache**, so the server embeds what it finds.

    that is the case a refresh exists for, and the only one it can be tested under:
    the seeded caches elsewhere in the suite are four numbers wide and made by hand,
    and the embedder's 512 refuse to sit beside them.
    """
    names = [verdict_name(AT + i * SPACING_MS, SUBJECT) for i in range(3)]
    names += [plain_name(AT + (3 + i) * SPACING_MS) for i in range(4)]
    dir = tmp_path / "crops"
    blank_crops(dir, ffmpeg, names)
    return dir


@pytest.fixture
def serve(binary, tmp_path: Path, harvest: Path):
    """start the labelling server over the harvest, and stop it with the test."""
    started = []

    def start(ingest: int | None = None) -> Labelling:
        started.append(Labelling(binary, build_config(tmp_path), harvest, ingest=ingest))
        return started[-1].ready()

    yield start
    for run in started:
        run.stop()


def test_more_shows_the_crops_that_landed(ffmpeg, harvest, serve):
    """**the queue was never the harvest, and the embedder is not paid for early.**
    the disk holds whatever the pipeline has kept since the page was opened, but
    while crops already embedded are still undecided another screenful comes from
    those: embedding the newcomers first is minutes of waiting for crops nobody has
    reached. once everything embedded has an answer, asking for more looks at the
    directory again.
    """
    run = serve()
    before = run.queue()
    assert before["items"], "the queue opened empty"
    embeds = run.output().count("embedding ")

    landed = [plain_name(AT + (20 + i) * SPACING_MS) for i in range(2)]
    blank_crops(harvest, ffmpeg, landed)

    got = run.post("/refresh")
    assert got["total"] == before["total"], f"it embedded with crops still undecided: {got}"
    assert run.output().count("embedding ") == embeds, run.output()

    assert run.post("/sweep", "random")["swept"] == before["total"]
    got = run.post("/refresh")
    assert got["total"] == before["total"] + 2, got
    assert got["remaining"] == 2, got
    assert {i["n"] for i in got["items"]} >= set(landed), got


def test_more_measures_nothing(ffmpeg, harvest, serve):
    """the point of the button is the measurement it does not do. `trained/` is
    written from a row of a curve, so the surest sign that no curve was drawn is that
    a margin is still refused -- and the answer itself carries the queue, which is
    what the page redraws, and no report."""
    run = serve()
    got = run.post("/refresh")
    assert got["items"] and "total" in got, got
    assert "passages" not in got, f"a refresh answered with a measurement: {got}"
    refused = run.post("/save", "0.5")
    assert "press train first" in refused.get("error", ""), refused


def test_the_service_takes_in_crops_by_itself(ffmpeg, harvest, serve):
    """**the slow half can happen on a clock.** with `--label-ingest` the label server
    is left running for an evening while the pipeline harvests, and embedding is what
    takes time -- so the crops that landed are read and embedded without anybody
    asking, and the pool on screen is still the pool that was drawn.

    leaving the pool alone is the point of the split. which pool a crop came from is
    the provenance of its label and what decides what may be measured on it, so a
    pool rebuilt underneath an open page would misfile the next click.
    """
    run = serve(ingest=1)
    before = run.queue()

    blank_crops(harvest, ffmpeg, [plain_name(AT + (30 + i) * SPACING_MS) for i in range(2)])
    deadline = time.time() + 60
    got = before
    while time.time() < deadline:
        got = run.queue()
        if got["total"] > before["total"]:
            break
        time.sleep(0.5)

    assert got["total"] == before["total"] + 2, got
    assert got["items"] == before["items"], "the pool was redrawn under the page"
    assert "new crops taken in" in run.output(), run.output()


def test_a_quiet_harvest_says_nothing(ffmpeg, harvest, serve):
    """a pass over a harvest nobody has added to costs a directory walk and one line
    of log a minute, and a log nobody reads is worse than no log: the line only means
    something when there is something in it."""
    run = serve(ingest=1)
    time.sleep(3)
    assert "taken in" not in run.output(), run.output()[-600:]


def test_the_asking_answer_is_still_quiet(ffmpeg, harvest, serve):
    """**nothing is taken in unless somebody asks**, which is how a session starts:
    the page's button is the only thing that notices, and the clock is something
    `--label-ingest <seconds>` turns on."""
    run = serve()
    assert "taking in new crops when asked" in run.output(), run.output()
    before = run.queue()
    blank_crops(harvest, ffmpeg, [plain_name(AT + 40 * SPACING_MS)])
    time.sleep(3)
    assert run.queue()["total"] == before["total"], "it looked anyway"
    run.post("/sweep", "random")
    assert run.post("/refresh")["total"] == before["total"] + 1


def test_more_sees_answers_given_while_it_was_open(ffmpeg, harvest, serve):
    """**the live page labels the same file over the other port.** an answer given
    there while this queue is open should take that crop out of the next screenful
    rather than come back to be given twice."""
    run = serve()
    name = run.queue()["items"][0]["n"]
    (harvest.parent / "labels.txt").write_text(f"{name} {SUBJECT} seed\n")

    got = run.post("/refresh")
    assert got["labelled"] == 1, got
    assert all(it["n"] != name or it["truth"] == SUBJECT for it in got["items"]), got
