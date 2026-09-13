"""choosing the operating point from the training report, and shipping it.

the report shows the whole curve so a person can see what each margin buys, but
only the row `--train` picks by itself could ever be written, and a margin typed
into the config is refused. the page now writes `trained/<subject>/` at the row
a person chose, with that row's own numbers beside it, through the same code
`--train` uses.
"""

import math
import tomllib
from pathlib import Path

import pytest

from labelling import SUBJECT, Labelling, blank_crops, build_config, plain_name, write_cache

AT = 1789000000000
# `classify::VOTE_K` is 3 and the reference set needs at least that many crops;
# four passages of three clear it, as in `test_train.py`.
PASSAGES, CROPS_PER_PASSAGE = 4, 3
NEGATIVES = 40


def unit(*xs: float) -> list[float]:
    norm = math.sqrt(sum(x * x for x in xs))
    return [x / norm for x in xs]


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return build_config(tmp_path)


@pytest.fixture
def harvest(tmp_path: Path, ffmpeg: str) -> Path:
    """labelled passages of the subject, and unlabelled traffic spread from far
    to fairly near it, so the curve has something different to say at each
    margin."""
    names, vectors, rows = [], [], []
    for p in range(PASSAGES):
        for c in range(CROPS_PER_PASSAGE):
            name = plain_name(AT + p * 60_000 + c * 400)
            names.append(name)
            vectors.append(unit(1, 0.05 * p, 0, 0.01 * c))
            rows.append(f"{name} {SUBJECT} seed")
    for i in range(NEGATIVES):
        names.append(plain_name(AT + 600_000 + i * 20_000))
        vectors.append(unit(i / NEGATIVES, 1, 0.1 * (i % 3), 0))
    crops = tmp_path / "crops"
    blank_crops(crops, ffmpeg, names)
    write_cache(tmp_path / "embeddings.bin", names, vectors)
    (tmp_path / "labels.txt").write_text("".join(row + "\n" for row in rows))
    return crops


def test_a_chosen_margin_is_written_with_its_own_numbers(binary, config, harvest):
    """**the row a person picked, not the one the rule would have.** the
    recall and false positive rate saved beside the margin are that row's, so
    the artifact still says what it was measured at."""
    with Labelling(binary, config, harvest).ready() as run:
        report = run.post("/train")
        assert not report.get("fatal"), report
        curve = report["decision"]
        chosen = next(r for r in curve if r["margin"] != report["operating_margin"])
        saved = run.post("/save", str(chosen["margin"]))
        assert saved.get("ok"), saved

    written = config.parent / "trained" / SUBJECT
    meta = tomllib.loads((written / "trained.toml").read_text())
    assert meta["margin"] == pytest.approx(chosen["margin"], abs=1e-6), meta
    assert meta["fpr"] == pytest.approx(chosen["fpr"], abs=1e-4), meta
    assert meta["passages"] == chosen["passages"], meta
    assert (written / "references.txt").exists()
    assert (written / "negatives.txt").exists()


def test_the_report_counts_under_the_configured_confirmation(binary, config, harvest):
    """**the curve describes the rule the deployment runs.** passages and false
    alerts are counted the way `[track] confirm_m` of `confirm_n` looks would
    confirm them, so changing those in the config changes what the report says."""
    config.write_text(config.read_text() + "\n[track]\nconfirm_m = 2\nconfirm_n = 4\n")
    with Labelling(binary, config, harvest).ready() as run:
        report = run.post("/train")
        assert (report["confirm_m"], report["confirm_n"]) == (2, 4), report


def test_nothing_is_written_before_a_measurement(binary, config, harvest):
    """a margin means something only against the curve it was read off."""
    with Labelling(binary, config, harvest).ready() as run:
        saved = run.post("/save", "0.0")
        assert "error" in saved, saved
    assert not (config.parent / "trained").exists()


def test_a_margin_that_is_not_on_the_curve_is_refused(binary, config, harvest):
    """there is no row to take its numbers from, so nothing honest to write."""
    with Labelling(binary, config, harvest).ready() as run:
        run.post("/train")
        saved = run.post("/save", "0.0123")
        assert "error" in saved, saved
    assert not (config.parent / "trained").exists()
