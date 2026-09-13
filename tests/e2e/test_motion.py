"""the independent motion scan the evals rest on.

`tools/motion.py` is not part of the runtime. it is the evidence: it decides
which moments count as transits for recall, and whether each harvested crop was
taken of something that was actually moving for precision. a bug in here does
not break the pipeline, it silently changes what every number about the pipeline
means -- which is worse, because nothing fails.
"""

import sys

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "tools"))
import motion

# a vehicle-sized silhouette in scan pixels: a car on the far side of this
# street is about 22x10 at 160x120.
CAR_W, CAR_H = 20, 9


def scene(*blobs) -> tuple[np.ndarray, np.ndarray]:
    """an empty street and one frame with things on it.

    returns `(frames, bg)` shaped as `motion` expects, with the moving thing in
    frame 1 and nothing in frame 0.
    """
    bg = np.full((motion.SCAN_H, motion.SCAN_W), 100, dtype=np.int16)
    frame = bg.copy()
    for x, y, w, h in blobs:
        frame[y : y + h, x : x + w] = 100 + motion.PIXEL_DELTA * 4
    return np.stack([bg, frame]), bg.astype(float)


def test_a_vehicle_entering_the_frame_is_not_eaten_by_the_edge():
    """**the bug that made the precision number meaningless.**

    `binary_closing` erodes with `border_value=0` by default, so anything
    touching the array border is treated as bordered by background and eaten.
    measured: a solid 8x5 blob flush against column zero lost 40% of itself and
    column zero vanished entirely.

    the consequence was not a slightly noisy mask. crops of vehicles entering or
    leaving the frame were scored as crops of a still scene, and they clustered:
    73-89% of everything the proxy called a false positive touched a frame edge,
    against 3% of what it accepted. the pipeline was being blamed for the
    measurement.
    """
    frames, bg = scene((0, 50, CAR_W, CAR_H))
    mask = motion.changed_mask(frames, bg, 1)
    assert mask[:, 0].any(), "the column against the edge was erased"
    # and the whole silhouette survives, not just a remnant of it.
    assert mask.sum() >= CAR_W * CAR_H, f"only {mask.sum()} of {CAR_W * CAR_H}px survived"


@pytest.mark.parametrize(
    "x,y",
    [(0, 50), (motion.SCAN_W - CAR_W, 50), (60, 0), (60, motion.SCAN_H - CAR_H)],
    ids=["left", "right", "top", "bottom"],
)
def test_every_edge_of_the_frame_reports_motion(x, y):
    """all four, because a vehicle leaves by a different edge than it entered."""
    frames, bg = scene((x, y, CAR_W, CAR_H))
    box = [x * 4, y * 4, (x + CAR_W) * 4, (y + CAR_H) * 4]  # scan -> substream
    found = motion.blobs(frames, bg, 1, motion.SCAN_W * 4, motion.SCAN_H * 4)
    assert found, "nothing was found at all"
    assert motion.boxes_overlap(found[0], box), f"{found[0]} missed {box}"


def test_an_interior_blob_is_unchanged_by_the_border_rule():
    """the counterweight: relaxing the border must not inflate the middle."""
    frames, bg = scene((70, 50, CAR_W, CAR_H))
    assert motion.changed_mask(frames, bg, 1).sum() == CAR_W * CAR_H


def test_every_moving_thing_is_reported_not_just_the_largest():
    """what made 171 correctly cropped vehicles look like false positives.

    an event is a stretch of time, and a city intersection puts several vehicles
    in one. taking the largest blob labelled one of them; metermate cropped all
    of them and was marked wrong for the rest.
    """
    frames, bg = scene((10, 20, 24, 12), (90, 70, CAR_W, CAR_H), (60, 40, 14, 8))
    found = motion.blobs(frames, bg, 1, motion.SCAN_W * 4, motion.SCAN_H * 4)
    assert len(found) == 3, f"expected three moving things, got {len(found)}"
    # largest first, so a caller taking `[0]` gets the clearest one.
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in found]
    assert areas == sorted(areas, reverse=True), areas


def test_overhead_wires_are_not_transits():
    """what flooded the label set once every blob was reported.

    the camera looks over a block strung with utility wires, and they move in
    wind. on one ten minute clip 60 of the 80 unlabelled objects were wires
    against the sky -- each one a question a person had to answer.

    area alone does not separate them: a wire spanning the frame covers more
    scan pixels than a distant car. thickness does not either, because a
    diagonal wire has a large square bounding box. what does is how much of that
    box the thing actually fills: measured over 131 objects, vehicles filled
    0.35 of their box at the 5th percentile against 0.16 at the 75th for
    everything else.
    """
    # three scan pixels thick and running diagonally, which is what one looks
    # like at 160x120 once the codec has smeared it. thin *and connected*: a
    # one-pixel line falls apart into disconnected fragments that the size floor
    # already rejects, so testing with one proves nothing.
    bg = np.full((motion.SCAN_H, motion.SCAN_W), 100, dtype=np.int16)
    frame = bg.copy()
    for col in range(30, 110):
        row = 20 + (col - 30) // 2
        frame[row : row + 3, col] = 100 + motion.PIXEL_DELTA * 4
    frames = np.stack([bg, frame])

    found = motion.blobs(frames, bg.astype(float), 1, motion.SCAN_W * 4, motion.SCAN_H * 4)
    assert found == [], f"a wire was called a transit: {found}"


def test_a_vehicle_sized_silhouette_survives_the_fill_rule():
    """the counterweight, at the thinnest a real vehicle gets.

    the far side of this street reads as roughly 20x9 scan pixels, and a car is
    not a solid rectangle -- wheels and windows punch holes in the difference.
    """
    frames, bg = scene((40, 40, CAR_W, CAR_H))
    frames[1][42, 44:52] = 100  # a windscreen that matches the road behind it
    found = motion.blobs(frames, bg, 1, motion.SCAN_W * 4, motion.SCAN_H * 4)
    assert len(found) == 1, f"a vehicle was filtered out as a wire: {found}"


def test_grain_below_the_blob_floor_is_not_a_transit():
    """the counterweight to the above: every pixel of noise is not a vehicle."""
    frames, bg = scene((30, 30, 2, 2), (80, 80, 3, 2))
    assert motion.blobs(frames, bg, 1, motion.SCAN_W * 4, motion.SCAN_H * 4) == []


def test_a_static_scene_reports_no_motion_anywhere():
    """the scene is mostly parked cars; none of them may ever count."""
    frames, bg = scene()
    assert not motion.changed_mask(frames, bg, 1).any()
    assert motion.blobs(frames, bg, 1, motion.SCAN_W * 4, motion.SCAN_H * 4) == []


def test_overlap_frac_asks_how_much_of_the_crop_is_moving():
    """asymmetric on purpose: a crop of one car in a queue is a good crop."""
    crop = [0, 0, 100, 100]
    assert motion.overlap_frac(crop, [0, 0, 50, 100]) == pytest.approx(0.5)
    assert motion.overlap_frac(crop, [0, 0, 1000, 1000]) == pytest.approx(1.0)
    assert motion.overlap_frac(crop, [200, 200, 300, 300]) == 0.0
