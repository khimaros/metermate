"""the labelling page: one tab per pool, and a switch for what is decided.

the five pools answer five different questions, and **which pool a crop came
from decides what may be measured on it** -- a false positive rate is only
honest on the random one. stacked down a single scroll, that distinction is a
heading you went past rather than where you are. one tab each makes it the
thing you navigated to.

driven by running the page's own javascript against the real binary
(`page.mjs`), because a rule that is right in the source and wrong in the
rendered page is a rule nobody gets the benefit of. the parts under test are
written as pure functions of the queue, so no browser is needed to ask them
what belongs on screen.
"""

from pathlib import Path

import pytest

from conftest import in_page
from labelling import (
    AT,
    SPACING_MS,
    SUBJECT,
    VERDICTS,
    Labelling,
    build_config,
    build_harvest,
    plain_name,
    verdict_name,
)

# every pool, in the order the tab bar offers them, by the slug `Via::slug`
# writes and the page filters on.
POOLS = ["random", "seed", "ranked", "review", "alert"]


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    return build_harvest(tmp_path, ffmpeg)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return build_config(tmp_path)


def evaluate(node: str, run: Labelling, expr: str):
    """evaluate an expression inside the loaded page and return its value."""
    return in_page(node, run.port, expr)


def test_the_page_offers_one_tab_for_each_pool(binary, config, harvest, node):
    """**the pool is the navigation, not a heading inside it.**

    a rate measured on the ranked pool is measuring the ranker, and the way
    that mistake gets made is by scrolling past the boundary between two pools
    without noticing there was one. a tab cannot be scrolled past.
    """
    with Labelling(binary, config, harvest).ready() as run:
        tabs = evaluate(node, run, "TABS.map(t => t[0])")
        assert tabs == POOLS, tabs

        # the words the tabs are offered under. they are what a person navigates
        # by, so they say what the pool is for rather than what it is called
        # internally.
        named = dict(evaluate(node, run, "TABS"))
        assert named["seed"] == "nearby"
        assert named["ranked"] == "most likely"
        assert named["alert"] == "verdicts"


def test_a_tab_draws_its_own_pool_and_nothing_else(binary, config, harvest, node):
    """the whole point of the split: what is on screen came from one pool."""
    with Labelling(binary, config, harvest).ready() as run:
        for via in POOLS:
            drawn = evaluate(node, run, f"shown(items, {via!r}, true).map(i => i.via)")
            assert set(drawn) <= {via}, (via, drawn)

        # and the verdicts tab really does hold the crops stage two named, so
        # the split did not quietly empty the pool that matters most.
        named = evaluate(node, run, "shown(items, 'alert', true).map(i => i.n)")
        assert set(named) == {verdict_name(AT + i * SPACING_MS, SUBJECT) for i in range(VERDICTS)}


def test_hiding_what_is_decided_leaves_only_the_work(binary, config, harvest, node):
    """after a sweep marks two hundred crops, the pool is two hundred tiles of
    finished work. the switch collapses it to what is left.

    it turns on what a crop was drawn as, not what it is now: a decision made
    since the pool was drawn stays on screen, so a misclick can be clicked back
    rather than vanishing under the pointer.
    """
    with Labelling(binary, config, harvest).ready() as run:
        judged = verdict_name(AT, SUBJECT)
        assert run.post("/label", f"{judged} other alert")["ok"]

        # reloaded, so the decision is one the pool was drawn with.
        left = evaluate(node, run, "load().then(() => shown(items, 'alert', false).map(i => i.n))")
        assert judged not in left, left
        assert len(left) == VERDICTS - 1, left

        with_decided = evaluate(
            node, run, "load().then(() => shown(items, 'alert', true).map(i => i.n))"
        )
        assert judged in with_decided, with_decided
        assert len(with_decided) == VERDICTS

        # a decision made *after* the pool was drawn is not swept off the screen
        # under the hand that made it.
        still_there = evaluate(
            node,
            run,
            "load().then(() => { const it = shown(items, 'alert', false)[0];"
            " it.truth = 'other'; return shown(items, 'alert', false).some(i => i.n === it.n); })",
        )
        assert still_there


def test_the_switch_says_how_much_it_is_hiding(binary, config, harvest, node):
    """**a switch that does nothing must not look broken.**

    three of the five pools are built from the unlabelled crops only -- random
    and most likely come off `unlabelled`, and neighbours skip anything already
    decided -- so on a fresh page there is nothing on them for the switch to
    reveal, and flipping it changes not one tile. a count beside it is the
    difference between "nothing here is decided" and "this control is dead".
    """
    with Labelling(binary, config, harvest).ready() as run:
        assert evaluate(node, run, "hiding(items, 'random')") == 0

        judged = verdict_name(AT, SUBJECT)
        assert run.post("/label", f"{judged} other alert")["ok"]
        hidden = evaluate(node, run, "load().then(() => hiding(items, 'alert'))")
        assert hidden == 1, hidden


def test_an_empty_pool_offers_nothing_to_sweep(binary, config, harvest, node):
    """**a pool with no crops in it must not offer to mark them.**

    splitting the pools into tabs meant always drawing the one you are on, where
    before a pool with nothing in it was never built at all. the sweep button
    came with it, and on an empty pool its label -- which is written from the
    count -- was never filled in, leaving a live unlabelled stub that offered to
    sweep nothing.
    """
    with Labelling(binary, config, harvest).ready() as run:
        # nothing has ever been labelled, so the neighbour pool has no seed to
        # sit next to and is empty while the verdicts pool is not.
        assert evaluate(node, run, "shown(items, 'seed', true).length") == 0
        assert evaluate(node, run, "offersSweep(items, 'seed')") is False
        assert evaluate(node, run, "offersSweep(items, 'alert')") is True

        # and the pool that is never swept whatever is in it stays that way.
        assert evaluate(node, run, "offersSweep(items, 'review')") is False


def test_a_pool_whose_crops_are_all_decided_cannot_be_swept(binary, config, harvest, node):
    """the other way to offer to mark nothing: a pool that is full but finished.

    the page deliberately does not rebuild the queue after a label, to keep the
    scroll position, so a pool can reach "every crop decided" without the server
    being asked anything. the button has to go dead from what is on screen.
    """
    with Labelling(binary, config, harvest).ready() as run:
        state = evaluate(
            node,
            run,
            "load().then(async () => {"
            " showTab('alert');"
            " for (const it of shown(items, 'alert', true).slice()) {"
            "   at = items.indexOf(it); await label('other'); }"
            " const b = document.getElementById('sweep-alert');"
            " return {left: items.filter(i => i.via === 'alert' && !i.truth).length,"
            "         text: b.textContent, disabled: b.disabled}; })",
        )
        assert state["left"] == 0, state
        assert state["disabled"] is True, state
        assert state["text"] == f"all {VERDICTS} decided", state

        # and the server agrees, which is the half the page cannot enforce.
        assert run.post("/sweep", "alert")["swept"] == 0


def test_the_switch_is_held_on_and_greyed_for_the_second_look(binary, config, harvest, node):
    """**every crop in the second look pool is already decided by definition.**

    that is what the pool is: crops with an answer, put back because the answer
    may be wrong. so there is nothing there for the switch to hide, and letting
    it apply would empty the tab -- the control would read as having broken the
    page. it is held on and greyed out instead.
    """
    with Labelling(binary, config, harvest).ready() as run:
        state = evaluate(
            node,
            run,
            "load().then(() => { showTab('review'); showDecided(false);"
            " return {wanted: wantsDecided('review', false),"
            "         elsewhere: wantsDecided('random', false),"
            "         off: document.getElementById('decided').disabled,"
            "         beside: document.getElementById('decided-n').textContent}; })",
        )
        # the switch is ignored here and obeyed everywhere else.
        assert state["wanted"] is True
        assert state["elsewhere"] is False
        assert state["off"] is True, "the switch is still live on the second look tab"
        # and it does not claim to be hiding anything.
        assert state["beside"] == "", state


def test_the_second_look_puts_each_direction_under_its_own_heading(binary, config, harvest, node):
    """**which way a crop is disputed is the question the tab asks**, so the
    three directions are not mixed into one grid.

    grouped by what a crop was labelled when the pool was drawn, so one flipped
    by a click stays under the heading it was offered under rather than jumping
    to another group at the next draw.

    a direction with nothing in it is still returned in order, and dropped when
    the grid is drawn: the grouping says what the tab is, the render says what
    is on screen.
    """
    plain = [plain_name(AT + (VERDICTS + i) * SPACING_MS) for i in range(4)]
    (harvest.parent / "labels.txt").write_text(
        f"{plain[0]} {SUBJECT} seed\n{plain[1]} {SUBJECT} seed\n"
        f"{plain[2]} other seed\n{plain[3]} other seed\n"
    )
    with Labelling(binary, config, harvest).ready() as run:
        groups = evaluate(
            node,
            run,
            "load().then(() => { lookGroups(shown(items, 'review', true))[1][1][0].truth = SUBJECT;"
            " return lookGroups(shown(items, 'review', true))"
            "   .map(([heading, list]) => ({heading, names: list.map(i => i.n)})); })",
        )
        assert len(groups) == 3, groups
        assert SUBJECT in groups[0]["heading"], groups
        assert sorted(groups[0]["names"]) == sorted(plain[:2]), groups
        assert "other" in groups[1]["heading"], groups
        assert sorted(groups[1]["names"]) == sorted(plain[2:]), groups
        # nothing was marked unclear here, so the third heading is empty rather
        # than absent, and the grid drops it.
        assert "unclear" in groups[2]["heading"], groups
        assert groups[2]["names"] == [], groups


def test_the_highlighted_row_is_the_one_train_writes(binary, config, harvest, node):
    """**the page highlights the margin the server chose**, rather than picking
    its own over a different curve.

    the operating point comes from the near curve when there is one, and a page
    that recomputed "best" over every crop could light up a row `--train` would
    never write. every row of the curve the choice is made on can be saved.
    """
    row = (
        "{{margin: {m}, passages: {p}, passage_recall: {r}, crop_recall: {r},"
        " fpr: {f}, per_hour: 1, alerts_per_hour: 0.1}}"
    )
    report = (
        "{curve: 'near', operating_margin: -0.01, passages: [{}, {}], decision: ["
        + row.format(m=-0.01, p=2, r=0.9, f=0.005)
        + ","
        + row.format(m=0, p=1, r=0.5, f=0.001)
        + "], sweep: ["
        + row.format(m=-0.01, p=2, r=1, f=0.02)
        + ","
        + row.format(m=0, p=2, r=0.9, f=0.01)
        + "]}"
    )
    with Labelling(binary, config, harvest).ready() as run:
        html = evaluate(node, run, f"sweepTable({report})")
        best = [tr for tr in html.split("<tr") if "class='best'" in tr]
        assert len(best) == 1, html
        assert "-0.010" in best[0], best
        assert html.count("saveMargin(") == 2, html


def test_a_tile_offers_the_crops_filename_to_copy(binary, config, harvest, node):
    """**the filename is what every other tool takes**: `--gather`, a line of
    `labels.txt`, a grep of the harvest. selecting it by hand off a tile means
    catching the caption without catching the tile, and catching the tile labels
    the crop.

    `navigator.clipboard` is deliberately not used. it is only defined in a
    secure context and this page is served over plain http on the deployment's
    own address, so there the button would silently do nothing.
    """
    with Labelling(binary, config, harvest).ready() as run:
        state = evaluate(
            node,
            run,
            "load().then(() => { showTab('alert');"
            " const find = e => e.className === 'copy' ? e :"
            "   (e.children || []).map(find).find(Boolean);"
            " const button = find(document.getElementById('out'));"
            " button.onclick({stopPropagation() {}});"
            " return {copied: globalThis.__copied, said: button.textContent,"
            "   truth: items.find(i => i.n === globalThis.__copied).truth}; })",
        )
        assert state["copied"] in {
            verdict_name(AT + i * SPACING_MS, SUBJECT) for i in range(VERDICTS)
        }, state
        # a copy that failed and one that worked look identical otherwise: the
        # clipboard cannot be read back.
        assert state["said"] == "copied", state
        # and copying is not judging.
        assert state["truth"] is None, state


def test_the_training_report_holds_the_keys_that_label(binary, config, harvest, node):
    """**the report covers the grid, so the keys must not reach through it.**

    `n` is "this is not one" to the grid and an ordinary keystroke to somebody
    reading a page of numbers about negatives. with the report over the top, a
    key that fell through would mark whatever crop the cursor was left on, off
    screen and unnoticed, into the file the whole loop rests on.
    """
    with Labelling(binary, config, harvest).ready() as run:
        crop = evaluate(
            node,
            run,
            "load().then(() => { showTab('alert');"
            " showTrain(true);"
            " document.onkeydown({key: 'n', preventDefault() {}});"
            " return {open: trainOpen(), truth: items[at].truth}; })",
        )
        assert crop["open"] is True
        assert crop["truth"] is None, "a labelling key reached through the report"

        # escape is the one key that does get through, and it closes it.
        shut = evaluate(
            node,
            run,
            "load().then(() => { showTrain(true);"
            " document.onkeydown({key: 'Escape', preventDefault() {}});"
            " return trainOpen(); })",
        )
        assert shut is False
