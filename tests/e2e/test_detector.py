"""stage-1 detector quality against a real frame from this camera.

the rest of the suite uses synthetic clips so it runs anywhere. this one cannot:
what regressed was the detector's confidence on *this* street -- hazy, shot
through glass, wide-angle, vehicles at an oblique angle -- and a white rectangle
on grey says nothing about that.

the failure this guards against is specific. yolo26n scored 0.70/0.51/0.40/0.35
on the four vehicles in `fixtures/street.jpg`, missed the silver sedan in the
foreground entirely, and drew the van's box around a stretch of road. every
number below is a floor that nano failed and yolo26s clears.
"""

import re
import subprocess

import pytest

from conftest import CLIP_FPS, REPO

# the fixture is the substream's geometry: 640x480 holding a 16:9 scene,
# anamorphically squeezed, exactly as the camera emits it. detecting well on a
# clean 1440p still would prove nothing about what metermate actually consumes.
FIXTURE = REPO / "tests" / "e2e" / "fixtures" / "street.jpg"

# vehicles a human counts in the fixture: silver sedan in the foreground, the
# jeep across the street, two cars up by the garages, and the van. the partial
# car at the left edge is not counted -- it is mostly out of frame.
VEHICLES_IN_FRAME = 5

# floors, not targets. nano managed 1 of 4 above 0.5 and none above 0.7.
MIN_VEHICLES = 4
MIN_MEDIAN_CONFIDENCE = 0.60
MIN_BEST_CONFIDENCE = 0.75

VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck"}
DETECTION = re.compile(r"\b(\w+) (\d\.\d+) +(\d+)x(\d+) +at \((\d+),(\d+)\)")


@pytest.fixture(scope="session")
def street_clip(ffmpeg: str, tmp_path_factory):
    """the fixture frame held for a few seconds, so the detector sees it often.

    a still rather than motion on purpose: this measures the detector, and the
    gate is bypassed in raw detector mode anyway.
    """
    out = tmp_path_factory.mktemp("clips") / "street.mp4"
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
            str(CLIP_FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="session")
def panning_street(ffmpeg: str, tmp_path_factory):
    """the same street, drifting, so the gate fires on real vehicles.

    the still above is right for measuring the detector and useless for
    measuring what gets *logged*: nothing moves, so no look ever happens.
    """
    out = tmp_path_factory.mktemp("clips") / "panning.mp4"
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
            str(CLIP_FPS),
            "-vf",
            "crop=600:440:x='min(39,t*20)':y=20,scale=640:480",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            str(out),
        ],
        check=True,
    )
    return out


def ran(binary, config, clip, debug: bool) -> str:
    env = {"PATH": "/usr/bin:/bin"}
    if debug:
        env["RUST_LOG"] = "metermate=debug"
    done = subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--source",
            str(clip),
            "--dry-run",
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=env,
    )
    return done.stdout + done.stderr


def test_what_a_look_found_is_not_in_the_default_log(binary, config, panning_street):
    """**a line per gated frame is a line fifteen times a second.**

    it is the line to read when the question is "what did the detector see",
    and that question is asked of a replay rather than of a deployment -- where
    it is thousands of lines an hour of parked cars and passing traffic,
    through which the handful that matter have to be found. so it moves to
    debug, where the tools that parse it already look.
    """
    loud = ran(binary, config, panning_street, debug=True)
    looks = [line for line in loud.splitlines() if "region Rect" in line and "-> " in line]
    assert any(" car " in line or " truck " in line for line in looks), loud[-3000:]

    quiet = ran(binary, config, panning_street, debug=False)
    assert "region Rect" not in quiet, quiet[-3000:]


def detections(binary, config, clip) -> list[tuple[str, float, int, int]]:
    """every box the raw detector reported, ungated and unfiltered (r8.4)."""
    proc = subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--source",
            str(clip),
            "--debug-detector",
            "--dry-run",
            "--preview",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={"RUST_LOG": "metermate=debug", "PATH": "/usr/bin:/bin"},
    )
    return [
        (m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in DETECTION.finditer(proc.stdout + proc.stderr)
    ]


def best_per_place(found) -> list[tuple[str, float, int, int]]:
    """collapse the repeated per-frame detections of a still scene.

    the clip is one frame held, so every vehicle is redetected ~90 times. group
    by rounded box size so the counts below mean distinct vehicles.
    """
    by_place: dict[tuple[int, int], tuple[str, float, int, int]] = {}
    for label, conf, w, h in found:
        key = (w // 20, h // 20)
        if conf > by_place.get(key, ("", 0.0, 0, 0))[1]:
            by_place[key] = (label, conf, w, h)
    return list(by_place.values())


def test_detector_finds_the_vehicles_on_this_street(binary, config, street_clip):
    found = [d for d in detections(binary, config, street_clip) if d[0] in VEHICLE_LABELS]
    assert found, "the detector reported no vehicles at all on a street full of them"

    distinct = best_per_place(found)
    confidences = sorted((c for _, c, _, _ in distinct), reverse=True)
    median = confidences[len(confidences) // 2]

    assert len(distinct) >= MIN_VEHICLES, (
        f"found {len(distinct)} of {VEHICLES_IN_FRAME} vehicles: {confidences}"
    )
    assert median >= MIN_MEDIAN_CONFIDENCE, (
        f"median confidence {median:.2f} below {MIN_MEDIAN_CONFIDENCE}: {confidences}"
    )
    assert confidences[0] >= MIN_BEST_CONFIDENCE, (
        f"best detection only {confidences[0]:.2f}: {confidences}"
    )


def test_the_foreground_sedan_is_not_missed(binary, config, street_clip):
    """the specific miss that started this: nano did not see it at all.

    it is the largest, closest, least ambiguous car in the frame. a detector
    that misses it will miss a go-4 stopping at the kerb, which is the whole
    point of the system.
    """
    found = [d for d in detections(binary, config, street_clip) if d[0] in VEHICLE_LABELS]
    # lower-left quadrant of the 640x640 letterboxed input, and big: the sedan
    # spans roughly a fifth of the frame width.
    sedan = [(label, conf) for label, conf, w, h in found if w >= 100 and h >= 60]
    assert sedan, (
        "nothing large was detected in the foreground; "
        f"largest boxes were {sorted(((w, h) for _, _, w, h in found), reverse=True)[:5]}"
    )
    assert max(c for _, c in sedan) >= MIN_BEST_CONFIDENCE, (
        f"the foreground vehicle scored only {max(c for _, c in sedan):.2f}"
    )
