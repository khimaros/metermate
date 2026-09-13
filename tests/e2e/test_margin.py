"""the margin travels with the references it was measured at.

`--train` used to print the operating point it measured and then trust a person
to copy that number into a config on another machine. that is the one step of a
deployment that can silently not happen: nothing fails, the classifier simply
runs a rule nobody measured, and every number it reports is about a different
rule from the one that was scored.

so the margin is written beside the vectors it belongs to, and there is no
`[classifier] margin` at all -- an override is just a slower way to describe an
unmeasured rule, and `[classifier] enabled = false` is the honest lever when a
deployment misbehaves.

`classify::DEFAULT_MARGIN` is deliberately unchanged: it is the constant the
judge fixture (`tests/e2e/fixtures/judge`) is written in, and the fifth query
there sits at an excess of exactly 0.000000, so a negative default would flip it.
"""

import re
import subprocess
import tomllib

import pytest

from conftest import REPO
from test_train import (
    CROPS_PER_PASSAGE,
    NEGATIVES,
    PASSAGES,
    SUBJECT,
    crop_name,
    solid_jpeg,
    train,
)

TRAINED = "trained.toml"

# `classify::DEFAULT_MARGIN`: what a set trained before the margin travelled
# with it falls back to, and what a set carrying one must NOT use.
DEFAULT_MARGIN = 0.005

STARTUP = rf"subject {SUBJECT}: \d+ references, \d+ negatives, margin (-?\d+\.\d+)"


def run_binary(binary, cfg, clip, timeout: int = 300):
    return subprocess.run(
        [str(binary), "--config", str(cfg), "--source", str(clip), "--dry-run", "--preview", "off"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def startup_margin(binary, cfg, clip) -> tuple[float, str]:
    """the margin the binary reports for the subject at startup."""
    run = run_binary(binary, cfg, clip)
    output = run.stdout + run.stderr
    found = re.search(STARTUP, output)
    assert found, output
    return float(found.group(1)), output


def classifier_config(config, trained, margin: float | None = None) -> str:
    body = (
        config.read_text()
        + f'\n[classifier]\nenabled = true\nreferences = "{trained}"\n'
        + f'model = "{REPO / "models" / "embedder.onnx"}"\n'
    )
    if margin is not None:
        body += f"margin = {margin}\n"
    return body + f'\n[[subject]]\nname = "{SUBJECT}"\n'


@pytest.fixture
def harvest(tmp_path, ffmpeg: str):
    """the same shape as `test_train`'s, defined here rather than imported.

    a fixture does not travel through a python import -- pytest only finds one
    in the requesting module or a conftest -- and promoting it to conftest would
    make a novel cross-module coupling load-bearing for the whole suite.
    """
    dir = tmp_path / "crops"
    dir.mkdir()
    rows = []
    at = 1789000000000
    for p in range(PASSAGES):
        for c in range(CROPS_PER_PASSAGE):
            name = crop_name(at + p * 60_000 + c * 400)
            solid_jpeg(ffmpeg, dir / name, "red")
            rows.append(f"{name} {SUBJECT} seed")
    for i in range(NEGATIVES):
        name = crop_name(at + 600_000 + i * 20_000, w=200, h=180)
        solid_jpeg(ffmpeg, dir / name, "gray")
        rows.append(f"{name} other random")
    (tmp_path / "labels.txt").write_text("\n".join(rows) + "\n")
    return dir


@pytest.fixture
def trained_set(binary, config, harvest, tmp_path):
    trained = tmp_path / "trained"
    run = train(binary, config, harvest, trained)
    assert run.returncode == 0, run.stdout + run.stderr
    return trained, run.stdout + run.stderr


def measured_margin(trained) -> float:
    text = (trained / SUBJECT / TRAINED).read_text()
    found = re.search(r"^margin\s*=\s*(-?\d+\.\d+)", text, re.MULTILINE)
    assert found, f"no margin recorded:\n{text}"
    return float(found.group(1))


def test_training_records_the_margin_it_measured(trained_set):
    """**taken off the report that chose the references**, not decided again.

    two selections agree until the day they do not, and nothing would then say
    which of them the deployment was running.
    """
    trained, output = trained_set
    meta = trained / SUBJECT / TRAINED
    assert meta.exists(), f"no {TRAINED} beside the vectors:\n{output}"

    recorded = measured_margin(trained)
    printed = re.search(r"chose margin ([+-]\d+\.\d+)", output)
    assert printed, output
    assert recorded == pytest.approx(float(printed.group(1))), (recorded, output)

    # provenance, so "measured against what" is answerable from the artifact.
    text = meta.read_text()
    for key in ("passages", "references", "negatives", "trained_at_millis"):
        assert re.search(rf"^{key}\s*=", text, re.MULTILINE), f"{key} missing:\n{text}"


def test_training_records_the_ladder_the_margin_was_chosen_from(trained_set):
    """**one margin says where the bar is; the ladder says what moving it costs.**

    a verdict a notch under the bar raises the question "what would lowering it
    let in", and the answer was a curve printed once to a terminal on the
    machine that trained. it is written beside the margin now, every rung of
    the curve the operating point was chosen from, so the bar and its
    alternatives travel together and were measured together.
    """
    trained, output = trained_set
    meta = tomllib.loads((trained / SUBJECT / TRAINED).read_text())
    ladder = meta.get("ladder")
    assert ladder and len(ladder) > 1, f"no ladder beside the margin:\n{meta}\n{output}"

    margins = [rung["margin"] for rung in ladder]
    assert margins == sorted(set(margins)), margins
    for rung in ladder:
        assert 0.0 <= rung["fpr"] <= 1.0, rung
        assert 0.0 <= rung["passage_recall"] <= 1.0, rung
        assert rung["passages"] >= 0, rung

    # the bar is a rung of it, carrying the numbers recorded beside the margin.
    chosen = [r for r in ladder if r["margin"] == pytest.approx(meta["margin"], abs=1e-4)]
    assert len(chosen) == 1, (meta["margin"], margins)
    assert chosen[0]["fpr"] == pytest.approx(meta["fpr"]), (chosen, meta)
    assert chosen[0]["passage_recall"] == pytest.approx(meta["passage_recall"]), (chosen, meta)


def test_a_config_with_no_margin_uses_the_trained_one(
    binary, config, trained_set, tmp_path, moving_clip
):
    """the whole point: nothing to hand-copy, and no way to forget."""
    trained, _ = trained_set
    cfg = tmp_path / "no-margin.toml"
    cfg.write_text(classifier_config(config, trained))
    used, output = startup_margin(binary, cfg, moving_clip)
    assert used == pytest.approx(measured_margin(trained), abs=1e-6), output
    assert "(measured)" in output, output


def test_a_config_naming_a_margin_is_refused(binary, config, trained_set, tmp_path, moving_clip):
    """a removed key must not sit in a deployed config pretending to work.

    serde drops unknown fields silently, so without `deny_unknown_fields` the
    stale line would read as configured and do nothing -- which is exactly the
    divergence this change exists to end.
    """
    trained, _ = trained_set
    cfg = tmp_path / "stale.toml"
    cfg.write_text(classifier_config(config, trained, margin=0.25))
    run = run_binary(binary, cfg, moving_clip)
    output = run.stdout + run.stderr
    assert run.returncode != 0, output
    assert "margin" in output, output


def test_references_without_metadata_fall_back_to_the_default(
    binary, config, trained_set, tmp_path, moving_clip
):
    """a directory written before this existed still loads and still fires.

    falling back matters more than it looks: reading a missing margin as 0.0
    would lower the bar on every subject at once.
    """
    trained, _ = trained_set
    (trained / SUBJECT / TRAINED).unlink()

    cfg = tmp_path / "bare.toml"
    cfg.write_text(classifier_config(config, trained))
    used, output = startup_margin(binary, cfg, moving_clip)
    assert used == pytest.approx(DEFAULT_MARGIN, abs=1e-6), output
    assert "default" in output, output
