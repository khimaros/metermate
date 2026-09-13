"""dense crops from an event clip, for a set to be labelled by hand.

the harvest keeps a few crops a passage by design -- it rate-limits by place and
drops what did not move -- so a recorded go-4 passage leaves a dozen crops where
there were hundreds of looks. `--dense` crops every detection in a window of an
event clip into a set, named by when each frame was taken, so the passage can be
labelled whole and training gets all of it.
"""

import re
import subprocess
from pathlib import Path

import pytest

from conftest import REPO

FIXTURE = REPO / "tests" / "e2e" / "fixtures" / "street.jpg"
STAMP = 1789000010000
# `[record] preroll_secs`: an event clip starts this long before its stamp.
PREROLL_MS = 3000
FPS = 5
FRAME_MS = 1000 // FPS
CROP = re.compile(r"^(\d+)_([a-z-]+)_(\d{3})_(\d+)x(\d+)\.jpg$")


def street_clip(ffmpeg: str, dir: Path, rate: int) -> Path:
    """an event clip named as the recorder names one: a real street, held still."""
    dir.mkdir(parents=True, exist_ok=True)
    clip = dir / f"{STAMP}-subject-main.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(FIXTURE),
            "-t",
            "3",
            "-r",
            str(rate),
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    return clip


@pytest.fixture
def event(ffmpeg: str, tmp_path: Path) -> Path:
    return street_clip(ffmpeg, tmp_path / "events", FPS)


def dense(binary: Path, config: Path, clip: Path, window: str, into: Path):
    return subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--dense",
            str(clip),
            "--window",
            window,
            "--into",
            str(into),
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def offsets(into: Path) -> list[int]:
    """each crop's time, in milliseconds from the start of the clip."""
    start = STAMP - PREROLL_MS
    found = []
    for path in sorted((into / "crops").glob("*.jpg")):
        match = CROP.match(path.name)
        assert match, f"{path.name} is not a crop name the harvest can read"
        assert path.stat().st_size > 0, path.name
        found.append(int(match.group(1)) - start)
    return found


def test_every_frame_in_the_window_is_cropped_by_when_it_was_taken(binary, config, event, tmp_path):
    """**named by the frame's own time, not by when the extraction ran.** passages
    are grouped by time, so a crop stamped with the clock of the run would make
    the whole event one burst a few milliseconds long.

    and **nothing is dropped for not moving**: the scene is a still photograph,
    and a go-4 stopped at the kerb writing a ticket is a positive worth having.
    """
    into = tmp_path / "sets" / "go4" / str(STAMP)
    run = dense(binary, config, event, "1-2", into)
    assert run.returncode == 0, run.stdout + run.stderr

    found = offsets(into)
    assert found, f"no crops from a street of parked cars:\n{run.stdout}{run.stderr}"
    # the window is half-open: the frames at 1.0 through 1.8 seconds.
    assert all(1000 <= t < 2000 for t in found), found
    frames = sorted({round(t / FRAME_MS) * FRAME_MS for t in found})
    assert frames == [1000, 1200, 1400, 1600, 1800], found

    # the event clip goes with the set, because recordings rotate out under
    # their own budget and a set should outlive the clip it came from.
    assert (into / "clips" / event.name).exists()


def test_a_window_can_be_given_in_minutes_and_seconds(binary, config, event, tmp_path):
    """the preview's player shows `m:ss`, so the window is typed the way it is read."""
    seconds = tmp_path / "seconds"
    clock = tmp_path / "clock"
    assert dense(binary, config, event, "1-2", seconds).returncode == 0
    run = dense(binary, config, event, "0:01-0:02", clock)
    assert run.returncode == 0, run.stdout + run.stderr
    assert offsets(clock) == offsets(seconds)


def test_frames_are_taken_at_the_harvests_rate_not_the_containers(binary, config, ffmpeg, tmp_path):
    """**a recording's container overstates its frame rate.** clips the recorder
    writes claim 100 fps for a camera that sends 15, and reading that literally
    crops each real frame several times over, at several times the cost. frames
    are taken at `[stream] crop_fps`, the rate the live harvest decodes the main
    stream at, or the clip's own rate if that is lower.
    """
    clip = street_clip(ffmpeg, tmp_path / "fast", 25)
    config.write_text(config.read_text().replace("[stream]\n", "[stream]\ncrop_fps = 5\n"))
    into = tmp_path / "set"
    run = dense(binary, config, clip, "1-2", into)
    assert run.returncode == 0, run.stdout + run.stderr
    frames = sorted({round(t / FRAME_MS) * FRAME_MS for t in offsets(into)})
    assert frames == [1000, 1200, 1400, 1600, 1800], offsets(into)


def test_a_window_with_no_end_runs_to_the_end_of_the_clip(binary, config, event, tmp_path):
    """where a vehicle leaves is often "still there when the clip stops", and
    counting that out of the recording to type it in is work for nothing."""
    into = tmp_path / "open"
    run = dense(binary, config, event, "2-", into)
    assert run.returncode == 0, run.stdout + run.stderr
    frames = sorted({round(t / FRAME_MS) * FRAME_MS for t in offsets(into)})
    # a three second clip at five frames a second: 2.0 through 2.8.
    assert frames == [2000, 2200, 2400, 2600, 2800], offsets(into)


def test_a_window_past_the_end_of_the_clip_is_refused(binary, config, event, tmp_path):
    """an empty set would look like a street with nothing in it, which is wrong."""
    run = dense(binary, config, event, "10-12", tmp_path / "late")
    assert run.returncode != 0, run.stdout + run.stderr
    assert "no frames" in run.stdout + run.stderr
