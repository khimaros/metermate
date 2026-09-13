"""gathering labelled crops into the set, so a measurement stays reproducible.

the harvest runs a disk budget that deletes oldest first, so a label outlives the
crop it describes. nothing fails when that happens -- the set simply trains on
whatever survived, and says the same confident things about it.

this is not hypothetical. of 6213 labels in the waymo set, 1598 already name
crops that no longer exist on the machine that made them: 26% of a person's
afternoon, unrecoverable, discovered only because somebody counted.

so `--gather` copies what a set's labels name into the set, and **reports what is
already gone** rather than quietly proceeding with the remainder.
"""

import subprocess

# `{millis}_{label}_{conf}_{w}x{h}.jpg`, which is what `parse_name` reads. the
# bytes never matter here: gathering copies files, it does not decode them.
CROP = b"not a real jpeg"


def crop_name(millis: int, w: int = 240, h: int = 200) -> str:
    return f"{millis}_car_090_{w}x{h}.jpg"


def gather(binary, config, set_dir, harvest, timeout: int = 120):
    return subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--gather",
            str(set_dir),
            "--harvest",
            str(harvest),
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def make_set(tmp_path, present: list[str], gone: list[str]):
    """a set whose labels name more crops than the harvest still holds."""
    harvest = tmp_path / "crops"
    harvest.mkdir()
    for n in present:
        (harvest / n).write_bytes(CROP)

    set_dir = tmp_path / "sets" / "go4" / "learn"
    set_dir.mkdir(parents=True)
    rows = [f"{n} go4 seed" for n in present] + [f"{n} other random" for n in gone]
    (set_dir / "labels.txt").write_text("\n".join(rows) + "\n")
    return set_dir, harvest


def test_gather_copies_what_the_labels_name_into_the_set(binary, config, tmp_path):
    """the set ends up owning its crops, so rotation cannot empty it later."""
    present = [crop_name(1789000000000 + i * 1000) for i in range(3)]
    gone = [crop_name(1789000090000 + i * 1000) for i in range(2)]
    set_dir, harvest = make_set(tmp_path, present, gone)

    run = gather(binary, config, set_dir, harvest)
    output = run.stdout + run.stderr
    # without this the test passes on the binary rejecting an unknown flag,
    # which is what it does while the feature does not exist yet.
    assert "unexpected argument" not in output, output
    assert run.returncode == 0, output

    for n in present:
        assert (set_dir / "crops" / n).exists(), f"{n} was labelled and not gathered"
    assert (set_dir / "crops" / present[0]).read_bytes() == CROP


def test_gather_reports_the_labels_whose_crops_are_already_gone(binary, config, tmp_path):
    """**the count is the point.** a set quietly missing a quarter of its crops
    trains without complaint and reports numbers about a smaller thing than the
    one that was labelled."""
    present = [crop_name(1789000000000 + i * 1000) for i in range(3)]
    gone = [crop_name(1789000090000 + i * 1000) for i in range(2)]
    set_dir, harvest = make_set(tmp_path, present, gone)

    run = gather(binary, config, set_dir, harvest)
    output = run.stdout + run.stderr
    assert "unexpected argument" not in output, output
    assert run.returncode == 0, output
    assert "2" in output, f"the 2 unrecoverable labels were not reported: {output}"
    # and it must not invent them
    for n in gone:
        assert not (set_dir / "crops" / n).exists()


def test_gathering_twice_does_not_copy_twice(binary, config, tmp_path):
    """gathering is how a set is kept current, so it runs again and again as
    labelling continues. it has to be cheap and idempotent."""
    present = [crop_name(1789000000000 + i * 1000) for i in range(3)]
    set_dir, harvest = make_set(tmp_path, present, [])

    assert gather(binary, config, set_dir, harvest).returncode == 0
    first = sorted(p.name for p in (set_dir / "crops").iterdir())

    run = gather(binary, config, set_dir, harvest)
    assert run.returncode == 0, run.stdout + run.stderr
    assert sorted(p.name for p in (set_dir / "crops").iterdir()) == first
    # nothing new to do, and it should say so rather than reporting 3 again.
    assert "0" in run.stdout + run.stderr
