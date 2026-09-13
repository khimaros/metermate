"""building a reference file from labelled crops, without a browser.

`--label` and `--measure` turn a harvest into labels and into numbers, but the
artifact the classifier actually loads -- `trained/<subject>/` -- could only be
produced by the python reference builder. so the thing that ships was built by a
different toolchain from the thing that was measured.

what is checked here:

- **both halves are written from the labels**, into the subject's own directory.
  a subject holding positives and no negatives is skipped rather than judged,
  because "is this like a go-4" has no answer without "compared to what", so
  writing one half without the other is a classifier that can never fire.
- **the binary can then load what it just wrote.** a writer and a reader that
  disagree about the format would be invisible until a deployment went quiet.
- **a measurement the eval voided is not written.** the preconditions exist to
  stop confident numbers about a rule nobody ran; emitting a reference file from
  them would ship that mistake rather than print it.
"""

import re
import subprocess

import pytest

from conftest import REPO

# crops of one subject further apart than this are separate passages
# (`label::PASSAGE_GAP_MS`), so the timestamps below are spaced well past it to
# make four passages rather than one long one.
PASSAGE_GAP_MS = 10_000
SUBJECT = "sweeper"

# `classify::VOTE_K` is 3, and `preconditions` voids a reference set smaller
# than it. four passages of three crops clears that with room to spare.
PASSAGES, CROPS_PER_PASSAGE = 4, 3
# enough unlabelled crops that the negative half is not the whole harvest.
NEGATIVES = 40


def crop_name(millis: int, w: int = 240, h: int = 200) -> str:
    """`{millis}_{label}_{conf}_{w}x{h}.jpg`, which is what `parse_name` reads."""
    return f"{millis}_car_090_{w}x{h}.jpg"


def solid_jpeg(ffmpeg: str, path, colour: str, size: str = "240x200") -> None:
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s={size}",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
    )


@pytest.fixture
def harvest(tmp_path, ffmpeg: str):
    """a harvest of two visibly different kinds of crop, and a labels file.

    the embedder is real, so the two kinds have to actually differ in the
    picture: a subject that is a solid colour and negatives that are not.
    """
    dir = tmp_path / "crops"
    dir.mkdir()
    labels = []
    at = 1789000000000
    for p in range(PASSAGES):
        for c in range(CROPS_PER_PASSAGE):
            # inside a passage: a few hundred ms apart. between passages: well
            # past the gap, so they group as four separate examples.
            name = crop_name(at + p * 60_000 + c * 400)
            solid_jpeg(ffmpeg, dir / name, "red")
            labels.append(f"{name} {SUBJECT} seed")
    for i in range(NEGATIVES):
        name = crop_name(at + 600_000 + i * 20_000, w=200, h=180)
        solid_jpeg(ffmpeg, dir / name, "gray")
        labels.append(f"{name} other random")
    (tmp_path / "labels.txt").write_text("\n".join(labels) + "\n")
    return dir


@pytest.fixture
def muddled(tmp_path, ffmpeg: str):
    """a harvest the embedder cannot separate: the street looks like the subject.

    every other fixture here is solid red against solid gray, which separates
    perfectly -- so no operating point is out of reach in it, and a constraint
    that cannot be met has nowhere to show. here a margin that keeps the
    passages also fires on the street, which is the ordinary case on a real
    camera and the only one where the curve says no.
    """
    dir = tmp_path / "muddled"
    dir.mkdir()
    labels = []
    at = 1789000000000
    for p in range(PASSAGES):
        for c in range(CROPS_PER_PASSAGE):
            name = crop_name(at + p * 60_000 + c * 400)
            solid_jpeg(ffmpeg, dir / name, "red")
            labels.append(f"{name} {SUBJECT} seed")
    for i in range(NEGATIVES):
        name = crop_name(at + 600_000 + i * 20_000, w=200, h=180)
        solid_jpeg(ffmpeg, dir / name, "red", size="200x180")
        labels.append(f"{name} other random")
    (tmp_path / "labels.txt").write_text("\n".join(labels) + "\n")
    return dir


@pytest.fixture
def set_tree(tmp_path, ffmpeg: str):
    """two sets under one root: the subject's own crops, and the street's.

    **neither trains on its own.** the first has no negatives and the second has
    no subject, so a run that succeeds can only have walked both -- which is
    what makes the control in the test below worth having.
    """
    root = tmp_path / "sets"
    at = 1789000000000

    subject = root / SUBJECT / "001" / "crops"
    subject.mkdir(parents=True)
    rows = []
    for p in range(PASSAGES):
        for c in range(CROPS_PER_PASSAGE):
            name = crop_name(at + p * 60_000 + c * 400)
            solid_jpeg(ffmpeg, subject / name, "red")
            rows.append(f"{name} {SUBJECT} seed")
    (subject.parent / "labels.txt").write_text("\n".join(rows) + "\n")

    street = root / "other" / "001" / "crops"
    street.mkdir(parents=True)
    rows = []
    for i in range(NEGATIVES):
        name = crop_name(at + 600_000 + i * 20_000, w=200, h=180)
        solid_jpeg(ffmpeg, street / name, "gray")
        rows.append(f"{name} other random")
    (street.parent / "labels.txt").write_text("\n".join(rows) + "\n")
    return root


def chosen(output: str) -> dict:
    """the row `--train` says it settled on, as the run reports it."""
    said = re.search(
        r"chose margin ([+-][\d.]+): (\d+)% of passages, ([\d.]+)% of the street", output
    )
    assert said, output
    return {
        "margin": float(said.group(1)),
        "recall": int(said.group(2)),
        "fpr": float(said.group(3)),
    }


def written_margin(out, subject: str = SUBJECT) -> float:
    """the margin beside the vectors, which is the one that ships."""
    text = (out / subject / "trained.toml").read_text()
    return float(re.search(r"margin = (\S+)", text).group(1))


def train(binary, config, harvest, out, subject: str = SUBJECT, timeout: int = 300, *extra: str):
    # the embedder path has to be spelled absolutely, for the reason conftest
    # gives for the detector: the suite runs from tests/e2e, so a repo-relative
    # default resolves to nothing.
    cfg = out.parent / "train.toml"
    cfg.write_text(
        config.read_text() + f'\n[classifier]\nmodel = "{REPO / "models" / "embedder.onnx"}"\n'
    )
    roots = harvest if isinstance(harvest, list) else [harvest]
    return subprocess.run(
        [
            str(binary),
            "--config",
            str(cfg),
            *[arg for root in roots for arg in ("--harvest", str(root))],
            "--train",
            subject,
            "--train-out",
            str(out),
            "--preview",
            "off",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_the_operating_point_is_capped_by_false_positive_rate(binary, config, harvest, tmp_path):
    """**a margin is an output of a measurement, and which output is a choice.**

    the default keeps 80% of the passages and takes the cleanest margin that
    does, which is a recall bracket rather than a rate anybody committed to. an
    operator who knows what nuisance rate they will tolerate -- under one alert
    in a hundred vehicles -- wants to say that instead and buy whatever recall
    it leaves.
    """
    trained = tmp_path / "trained"
    run = train(binary, config, harvest, trained, SUBJECT, 300, "--max-fpr", "1%")
    output = run.stdout + run.stderr
    assert run.returncode == 0, output

    row = chosen(output)
    assert row["fpr"] <= 1.0, row
    # and what shipped is the row it named, not a second selection.
    assert written_margin(trained) == pytest.approx(row["margin"], abs=1e-4), output


def test_the_recall_bracket_chooses_the_cleanest_row_in_it(binary, config, harvest, tmp_path):
    """the other way round: name the hit rate, take the quietest margin that
    reaches it. the default is this rule at 80%, so the flag only moves the
    bracket rather than introducing a second kind of answer."""
    trained = tmp_path / "trained"
    run = train(binary, config, harvest, trained, SUBJECT, 300, "--min-recall", "60%")
    output = run.stdout + run.stderr
    assert run.returncode == 0, output

    row = chosen(output)
    assert row["recall"] >= 60, row
    assert written_margin(trained) == pytest.approx(row["margin"], abs=1e-4), output


def test_an_unreachable_operating_point_is_refused(binary, config, muddled, tmp_path):
    """**nothing is written when the constraint cannot be met.** falling back to
    the default would ship a rule at a rate the operator explicitly ruled out,
    and the artifact beside the vectors carries no record of what was asked
    for -- so the deployment would look exactly like one that got its way."""
    trained = tmp_path / "trained"
    run = train(
        binary, config, muddled, trained, SUBJECT, 300, "--min-recall", "100%", "--max-fpr", "0%"
    )
    output = run.stdout + run.stderr
    assert run.returncode != 0, output
    # and says what each axis actually offered, or the only move left is to
    # guess another pair of numbers.
    assert "no row" in output, output
    assert "% of the street" in output and "% of the passages" in output, output
    assert not (trained / SUBJECT / "trained.toml").exists(), output


def test_a_training_run_says_how_far_through_it_is(binary, config, harvest, tmp_path):
    """**a run that prints nothing for four minutes looks like a hung run.**

    embedding a real harvest is thousands of crops at tens of milliseconds
    each, and the cross-validation behind the curve is a pass over every
    negative for every labelled passage -- which is now every rejected verdict
    rather than a sample of thirty. announcing the total at the start says a
    wait is coming without saying how much of it is left, so both loops count
    as they go and estimate what remains.
    """
    trained = tmp_path / "trained"
    run = train(binary, config, harvest, trained)
    output = run.stdout + run.stderr
    assert run.returncode == 0, output

    embedding = re.search(r"embedding crops: (\d+)/(\d+) \((\d+)%\)", output)
    assert embedding, output
    done, total = int(embedding.group(1)), int(embedding.group(2))
    assert 0 < done <= total == PASSAGES * CROPS_PER_PASSAGE + NEGATIVES, embedding.group(0)
    assert re.search(r"embedded \d+ crops in ", output), output

    # the fold loop is the slower half on a real harvest, and it runs whether or
    # not anything needed embedding.
    assert re.search(r"scoring passages: \d+/\d+ \(\d+%\)", output), output
    assert re.search(r"scored \d+ passages in ", output), output

    # **and the probe is the longest silence of the three.** it is a gradient
    # fit per passage over every negative in the harvest, and it runs after the
    # last thing that printed anything -- which is what a stuck run looks like.
    assert re.search(r"fitting probe folds: \d+/\d+ \(\d+%\)", output), output
    assert re.search(r"fitted \d+ probe folds in ", output), output


def test_training_writes_a_reference_file_the_binary_can_load(binary, config, harvest, tmp_path):
    """the whole point: labels in, an artifact the classifier loads out."""
    trained = tmp_path / "trained"
    run = train(binary, config, harvest, trained)
    assert run.returncode == 0, run.stdout + run.stderr

    positives = trained / SUBJECT / "references.txt"
    negatives = trained / SUBJECT / "negatives.txt"
    assert positives.exists(), run.stdout + run.stderr
    assert negatives.exists(), "a subject with no negatives is skipped, not judged"

    for path, expect in ((positives, SUBJECT), (negatives, "other")):
        rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
        assert rows, f"{path.name} is empty"
        # the label must match the half holding it. a negatives.txt of go-4s
        # trains the exact inversion of the rule and fails silently.
        assert {r[0] for r in rows} == {expect}, f"{path.name}: {sorted({r[0] for r in rows})}"
        # every row is a label and an embedding of the same width, or the reader
        # will take a short row as a vector and compare across dimensions.
        widths = {len(r) - 1 for r in rows}
        assert len(widths) == 1, f"{path.name}: rows of differing width: {sorted(widths)}"
        assert widths.pop() > 1


def test_the_trained_file_is_accepted_by_the_classifier(
    binary, config, harvest, tmp_path, moving_clip
):
    """a writer and a reader that disagree would be invisible until deployment."""
    trained = tmp_path / "trained"
    assert train(binary, config, harvest, trained).returncode == 0

    enabled = tmp_path / "with-classifier.toml"
    # the subject has to be named: `cfg.subjects()` stands `[classifier]` in
    # when no `[[subject]]` block exists, so without this the startup line
    # reports the default subject and says nothing about the one just trained.
    enabled.write_text(
        config.read_text()
        + f'\n[classifier]\nenabled = true\nreferences = "{trained}"\n'
        + f'model = "{REPO / "models" / "embedder.onnx"}"\n'
        + f'\n[[subject]]\nname = "{SUBJECT}"\n'
    )
    run = subprocess.run(
        [
            str(binary),
            "--config",
            str(enabled),
            "--source",
            str(moving_clip),
            "--dry-run",
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = run.stdout + run.stderr
    # the startup line reports what the file actually holds for each configured
    # subject. a count of zero is the failure that matters: the binary starts,
    # the pipeline runs, and the subject can never fire.
    loaded = re.search(rf"subject {SUBJECT}: (\d+) references", output)
    assert loaded, output
    assert int(loaded.group(1)) > 0, output
    assert run.returncode == 0, output


def test_training_walks_a_tree_of_sets(binary, config, set_tree, tmp_path):
    """`--harvest sets/` reaches every set's crops and every set's labels.

    each set keeps its own `labels.txt` and `embeddings.bin` beside its crops,
    so the labels a person made travel with the crops they describe rather than
    with the subject they were collected for.

    without the walk, the fold of one subject's crops into another's negatives
    could never fire: a single invocation only ever saw one set, and the other
    subject's crops are by construction in a different one.
    """
    trained = tmp_path / "trained"
    run = train(binary, config, set_tree, trained)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (trained / SUBJECT / "references.txt").exists(), run.stdout + run.stderr
    assert (trained / SUBJECT / "negatives.txt").exists(), run.stdout + run.stderr

    # the control. pointed at the subject's set alone there are no negatives at
    # all, so it must refuse -- which is what proves the run above crossed sets
    # rather than finding everything it needed in the first one.
    alone = tmp_path / "alone"
    one = train(binary, config, set_tree / SUBJECT / "001" / "crops", alone)
    assert one.returncode != 0, "a set holding no negatives trained anyway"
    assert not (alone / SUBJECT).exists()


def test_training_reads_several_harvest_roots(binary, config, set_tree, tmp_path):
    """**a deployment's own harvest and a set cut from video train together.**
    the rejected verdicts live beside the live harvest and a dense set lives under
    `sets/`, so one run has to read both. neither root below trains on its own --
    one has no negatives, the other no subject -- so success means both were read.
    """
    trained = tmp_path / "trained"
    run = train(binary, config, [set_tree / SUBJECT, set_tree / "other"], trained)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (trained / SUBJECT / "references.txt").exists(), run.stdout
    assert (trained / SUBJECT / "negatives.txt").exists(), run.stdout


def test_every_rejected_verdict_is_written_as_a_negative(binary, config, harvest, tmp_path):
    """**the cap dropped exactly the negatives that mattered.** on the deployment
    658 verdicts were rejected by hand, 69 of them cargo bikes, and the thirty
    negatives drawn held no bike -- so bikes kept firing. every rejected verdict
    is written now, and only ordinary traffic is a sample."""
    labels = tmp_path / "labels.txt"
    rows = labels.read_text().splitlines()
    street = [i for i, row in enumerate(rows) if row.endswith(" other random")]
    for i in street[:35]:
        rows[i] = rows[i].replace(" other random", " other alert")
    labels.write_text("\n".join(rows) + "\n")

    trained = tmp_path / "trained"
    run = train(binary, config, harvest, trained)
    assert run.returncode == 0, run.stdout + run.stderr
    written = (trained / SUBJECT / "negatives.txt").read_text().splitlines()
    # all 35 rejected verdicts, and the 5 ordinary crops left for the sample.
    assert len(written) == len(street), (len(written), run.stdout)


def test_a_voided_measurement_is_not_written(binary, config, tmp_path, ffmpeg):
    """below `VOTE_K` references the eval calls its own numbers void.

    writing a file from them would ship the mistake instead of printing it, and
    the deployment would be classifying against two crops of one passage.
    """
    dir = tmp_path / "crops"
    dir.mkdir()
    name = crop_name(1789000000000)
    solid_jpeg(ffmpeg, dir / name, "red")
    (tmp_path / "labels.txt").write_text(f"{name} {SUBJECT} seed\n")

    trained = tmp_path / "trained"
    run = train(binary, config, dir, trained)
    output = run.stdout + run.stderr
    # it has to refuse for the measurement's sake. without this the test passes
    # on any failure at all -- including the binary rejecting the flag before
    # the feature exists, which is what it did while this was being written.
    assert "unexpected argument" not in output, output
    assert run.returncode != 0, output
    assert SUBJECT in output, output
    assert not (trained / SUBJECT).exists(), "a voided measurement still produced references"
