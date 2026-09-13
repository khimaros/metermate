"""marking a classifier verdict wrong, from the labelling page.

a crop's filename carries what stage two called it. that is **evidence of what
fired, never a label**: on the first evening the verdict was recorded, every
crop it named was a false positive. what a crop *is* lives in `labels.txt`, and
only a person writes it.

so rejecting a verdict is the most valuable labelling action there is.
`split_negatives` draws `Via::Alert` negatives ahead of every other tier --
measured on the deployment, five crops the classifier called a go-4 were
labelled `other`, none were drawn, and all five still scored as the subject to
four decimals. the tier only fills if the page can actually reject one.

driven through the real binary over http rather than against the queue
directly, because everything here is wiring: a verdict has to travel from a
filename, through a listing, into a pool the page renders, into a key, into a
line of `labels.txt` carrying the right `via`. a queue that is right in
`server.rs` and rendered in a view nobody can reach is a feature that does not
exist.
"""

import urllib.error
from pathlib import Path

import pytest

from labelling import (
    AT,
    PLAIN,
    SPACING_MS,
    SUBJECT,
    VERDICTS,
    Labelling,
    build_config,
    build_harvest,
    plain_name,
    verdict_name,
)


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    return build_harvest(tmp_path, ffmpeg)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return build_config(tmp_path)


def fired(queue: dict) -> list[dict]:
    """the pool of what the classifier fired on, as the page groups it."""
    return [i for i in queue["items"] if i["via"] == "alert"]


def test_the_verdicts_are_on_the_page_before_anything_is_measured(binary, config, harvest):
    """**the verdict is a listing, not a measurement.**

    the alert pool used to come out of `eval::measure`, so it was empty until
    `train` had been pressed -- and `train` needs labelled positives, which is
    exactly what a harvest full of false positives does not have. the crops
    stage two named are on disk with the verdict in their names, so they are
    offered the moment the session opens.
    """
    with Labelling(binary, config, harvest).ready() as run:
        shown = fired(run.queue())
        assert len(shown) == VERDICTS, [i["n"] for i in shown]

        named = {i["n"] for i in shown}
        assert named == {verdict_name(AT + i * SPACING_MS, SUBJECT) for i in range(VERDICTS)}
        # the control: a crop stage two said nothing about is not a verdict, and
        # rejecting it says nothing about the classifier.
        assert not any(plain_name(AT + (VERDICTS + i) * SPACING_MS) in named for i in range(PLAIN))

        # what it was called travels with it. the whole action is "this verdict
        # is wrong", which cannot be asked of a tile that does not say what the
        # verdict was.
        assert all(i["verdict"] == SUBJECT for i in shown), shown


def test_one_key_marks_a_verdict_wrong_and_the_page_shows_it_marked(binary, config, harvest):
    """the keystroke writes `other` *and* `alert`, and both halves matter.

    `other` is the judgement. `alert` is what puts the crop at the front of
    `split_negatives`, ahead of the ordinary traffic that never confused the
    classifier in the first place. a rejection recorded without it is a fact
    nobody acts on.
    """
    with Labelling(binary, config, harvest).ready() as run:
        crop = fired(run.queue())[0]["n"]
        # what the `n` key sends: the crop, the verdict, and the pool it came
        # from.
        assert run.post("/label", f"{crop} other alert")["ok"]

        assert run.entries()[crop] == ("other", "alert"), run.entries()

        # and the page can see it is done, so a screenful says which ones have
        # been dealt with rather than only how many.
        marked = [i for i in fired(run.queue()) if i["n"] == crop]
        assert marked and marked[0]["truth"] == "other", marked


def test_the_page_serves_the_favicon_it_asks_for(binary, config, harvest):
    """**a link in the head is only half of it.**

    both servers run a page and a person runs them side by side, so identical
    blank tabs is how you reload the wrong one. the markup asking for
    `/favicon.png` and the route serving it live in different files, and
    nothing else would notice if one of them moved.
    """
    with Labelling(binary, config, harvest).ready() as run:
        status, body = run.get("/favicon.png")
        assert status == 200, run.output()
        # a png, by its magic rather than by its extension
        assert body[:8] == b"\x89PNG\r\n\x1a\n", body[:16]

        page = run.get("/")[1].decode()
        assert 'href="/favicon.png"' in page, "the page does not ask for it"


def test_the_second_look_pool_cannot_be_swept(binary, config, harvest):
    """**nothing may bulk-mark the crops put up for a second look.**

    the pool holds labelled crops the classifier's own vote disputes, ranked with
    no threshold, so on a clean set most of them are correct labels. every one
    is a judgement a person has to make on the image. the page never offers the
    button, but the endpoint accepted the pool name, and a bulk `other` over it
    would undo fifty labels of the subject in one request.
    """
    with Labelling(binary, config, harvest).ready() as run:
        try:
            swept = run.post("/sweep", "review")
        except urllib.error.HTTPError as e:
            assert e.code >= 400, e
            return
        assert swept.get("swept", 0) == 0, swept


def test_a_screenful_of_verdicts_is_rejected_in_one_click(binary, config, harvest):
    """the usual case is that the whole screen is wrong, so it is one action.

    the same argument the random pool's sweep rests on: looking at a screen and
    saying "none of these" is one judgement, and making it once is the same
    decision with the clicks removed. it is the usual case here rather than an
    occasional one -- every crop named on the first evening was a false
    positive.
    """
    with Labelling(binary, config, harvest).ready() as run:
        # one picked out by hand first. a sweep that overwrote it would be
        # destroying the answers it is meant to be finishing.
        keep = fired(run.queue())[0]["n"]
        run.post("/label", f"{keep} {SUBJECT} alert")

        swept = run.post("/sweep", "alert")
        assert swept["swept"] == VERDICTS - 1, swept
        assert keep not in swept["names"]

        rows = run.entries()
        assert rows[keep] == (SUBJECT, "alert"), "a decision already made was overwritten"
        for i in range(VERDICTS):
            name = verdict_name(AT + i * SPACING_MS, SUBJECT)
            if name != keep:
                assert rows[name] == ("other", "alert"), rows

        # nothing outside the pool moved: the sweep takes one pool, not the page.
        assert len(rows) == VERDICTS, sorted(rows)

        # and the button exists to press. the server can sweep the pool either
        # way, so without this the whole action is unreachable from the browser.
        page = run.get("/")[1].decode()
        sweepable = page.split("const SWEEPABLE = ", 1)[1].split(";", 1)[0]
        assert "alert" in sweepable, sweepable
