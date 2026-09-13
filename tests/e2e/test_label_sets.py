"""labelling across a tree of sets: `--label <subject> --harvest sets/`.

`--train` and `--measure` already walk `sets/<subject>/<id>/crops`, and
`--prepare` ends by saying "label them at `--label <subject> --harvest sets`" --
which answered `no crops in sets`, because the labelling page read one
directory. so every set cut from video had to be labelled in a session of its
own, by a path nobody remembers.

what is checked here:

- **the tree opens as one harvest**, every crop under it offered and served.
- **a label lands beside the crop it names.** each set keeps its own
  `labels.txt`, which is what lets a set travel with its judgements.
- **a crop two sets share is one crop.** offered once, and labelled in both, or
  the next run finds two sets disagreeing and refuses to open.
- **doing it again changes nothing.** opening, refreshing and repeating an
  answer leave every file under the tree byte for byte as it was.
"""

import shutil
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
    write_cache,
)

# two sets filed under two subjects, which is the shape `--prepare` leaves.
SETS = ((SUBJECT, "001"), ("waymo", "001"))
PER_SET = 4


def names_of(index: int) -> list[str]:
    return [plain_name(AT + (index * 100 + i) * SPACING_MS) for i in range(PER_SET)]


def build_tree(tmp_path: Path, ffmpeg: str, seeded: bool = True) -> Path:
    root = tmp_path / "sets"
    for index, (subject, id) in enumerate(SETS):
        crops = root / subject / id / "crops"
        crops.parent.mkdir(parents=True)
        blank_crops(crops, ffmpeg, names_of(index))
        if seeded:
            write_cache(crops.parent / "embeddings.bin", names_of(index))
    return root


@pytest.fixture
def tree(tmp_path: Path, ffmpeg: str) -> Path:
    return build_tree(tmp_path, ffmpeg)


def set_dir(root: Path, index: int) -> Path:
    subject, id = SETS[index]
    return root / subject / id


def rows(set: Path) -> dict[str, tuple[str, str]]:
    """one set's `labels.txt` as `{crop: (truth, via)}`."""
    path = set / "labels.txt"
    if not path.exists():
        return {}
    found = {}
    for line in path.read_text().splitlines():
        parts = line.split("#")[0].split()
        if len(parts) >= 3:
            found[parts[0]] = (parts[1], parts[2])
    return found


def snapshot(root: Path) -> dict[str, bytes]:
    """every file the labelling page may write under the tree."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.name in ("labels.txt", "embeddings.bin")
    }


def share(root: Path) -> str:
    """put the first set's first crop in the second set as well."""
    name = names_of(0)[0]
    shutil.copy(set_dir(root, 0) / "crops" / name, set_dir(root, 1) / "crops" / name)
    write_cache(set_dir(root, 1) / "embeddings.bin", names_of(1) + [name])
    return name


def test_a_tree_of_sets_opens_as_one_harvest(binary, tmp_path, tree):
    with Labelling(binary, build_config(tmp_path), tree).ready() as run:
        queue = run.queue()
        assert queue["total"] == len(SETS) * PER_SET, queue
        offered = {i["n"] for i in queue["items"]}
        for index in range(len(SETS)):
            assert set(names_of(index)) <= offered, (index, sorted(offered))
        # a crop is served from whichever set holds it.
        for name in offered:
            assert run.get(f"/crop/{name}")[0] == 200, name


def test_a_label_lands_beside_the_crop_it_names(binary, tmp_path, tree):
    with Labelling(binary, build_config(tmp_path), tree).ready() as run:
        name = names_of(1)[0]
        assert run.post("/label", f"{name} {SUBJECT} random")["ok"]
        assert rows(set_dir(tree, 1)) == {name: (SUBJECT, "random")}
        # the other set was not asked anything, so nothing was written into it.
        assert not (set_dir(tree, 0) / "labels.txt").exists()

        # a sweep is many labels at once and each still goes home.
        swept = run.post("/sweep", "random")
        assert swept["swept"] == len(SETS) * PER_SET - 1, swept
        for index in range(len(SETS)):
            assert set(rows(set_dir(tree, index))) == set(names_of(index)), index
        assert rows(set_dir(tree, 1))[name][0] == SUBJECT, "the sweep overwrote a decision"

        # and taking the sweep back takes it out of every set it went into.
        assert run.post("/undo")["undone"] == swept["swept"]
        assert rows(set_dir(tree, 0)) == {}
        assert rows(set_dir(tree, 1)) == {name: (SUBJECT, "random")}


def test_a_crop_two_sets_share_is_one_crop(binary, tmp_path, tree):
    shared = share(tree)
    config = build_config(tmp_path)
    with Labelling(binary, config, tree).ready() as run:
        queue = run.queue()
        assert queue["total"] == len(SETS) * PER_SET, queue
        assert [i["n"] for i in queue["items"]].count(shared) == 1, queue
        assert run.post("/label", f"{shared} {SUBJECT} random")["ok"]
        for index in range(len(SETS)):
            assert rows(set_dir(tree, index)) == {shared: (SUBJECT, "random")}, index

    # the next session reads two files that agree, and counts one decision.
    with Labelling(binary, config, tree).ready() as run:
        assert run.queue()["labelled"] == 1, run.queue()


def test_labelling_a_tree_again_changes_nothing(binary, tmp_path, tree):
    shared = share(tree)
    config = build_config(tmp_path)
    answers = [f"{shared} {SUBJECT} random", f"{names_of(1)[1]} other random"]
    with Labelling(binary, config, tree).ready() as run:
        for answer in answers:
            assert run.post("/label", answer)["ok"]
    first = snapshot(tree)
    assert any(k.endswith("labels.txt") for k in first), sorted(first)

    with Labelling(binary, config, tree).ready() as run:
        run.post("/refresh")
        for answer in answers:
            assert run.post("/label", answer)["ok"]
        run.post("/refresh")
    assert snapshot(tree) == first


def test_a_tree_is_embedded_once(binary, tmp_path, ffmpeg):
    """**the embedder is the slow half, and a crop's vector never changes.**

    with no cache anywhere the first session embeds every crop into the cache
    beside its own set. the second finds them there: it embeds nothing, writes
    nothing, and says nothing about embedding.
    """
    tree = build_tree(tmp_path, ffmpeg, seeded=False)
    config = build_config(tmp_path)
    with Labelling(binary, config, tree).ready() as run:
        assert run.queue()["total"] == len(SETS) * PER_SET
        assert "embedding" in run.output(), run.output()
    first = snapshot(tree)
    # one cache per set, each holding its own crops and nobody else's.
    for index, (subject, id) in enumerate(SETS):
        cache = first[f"{subject}/{id}/embeddings.bin"]
        for other in range(len(SETS)):
            held = [name.encode() in cache for name in names_of(other)]
            assert all(held) if other == index else not any(held), (index, other, held)
    assert len(first) == len(SETS), sorted(first)

    with Labelling(binary, config, tree).ready() as run:
        run.post("/refresh")
        assert "embedding" not in run.output(), run.output()
    assert snapshot(tree) == first
