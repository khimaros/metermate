"""an independent answer to "was anything moving here, and where".

shared by `label.py`, which turns it into a list of transits to hand label, and
`endtoend.py`, which uses it to ask whether each harvested crop was taken of
something that actually moved.

**independent on purpose.** metermate's own gate is an exponential background
model in 8.8 fixed point at the substream's native size; this is a median
background differenced at 160x120. neither can inherit the other's blind spots,
which is the only reason either is evidence. an eval built on the pipeline's own
motion signal would score the pipeline against itself.
"""

import subprocess
from pathlib import Path

import numpy as np

# heavily downscaled luma: enough to see a car move, cheap enough to run over
# ten minutes of video in a few seconds.
SCAN_W, SCAN_H = 160, 120
# a pixel must change by this much to count, which rejects sensor grain and the
# gentle exposure drift of a camera pointed at a sunlit street.
PIXEL_DELTA = 18
# blobs smaller than this are glare, leaf shadow, and the tops of wires. a car
# on the far side of this street is about 22x10 in scan pixels, so this admits
# something a quarter that size.
MIN_BLOB_PX = 24

# and this much of its own bounding box must actually be the thing.
#
# the camera looks over a block strung with utility wires, and they move in
# wind: 60 of the 80 unlabelled objects on one ten minute clip were wire against
# sky. size does not separate them, because a wire spanning the frame covers
# more scan pixels than a distant car, and neither does thickness, because a
# diagonal wire has a large square bounding box. how much of that box is filled
# does. measured over 131 objects: vehicles filled 0.35 of their box at the 5th
# percentile, everything else 0.16 at the 75th. at 0.20 not one vehicle is lost
# and the rest is nearly halved.
MIN_BLOB_FILL = 0.20


def scan_frames(clip: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip),
         "-vf", f"scale={SCAN_W}:{SCAN_H},format=gray", "-fps_mode", "passthrough",
         "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(raw) // (SCAN_W * SCAN_H)
    return raw[: n * SCAN_W * SCAN_H].reshape(n, SCAN_H, SCAN_W).astype(np.int16)


def motion_signal(frames: np.ndarray) -> np.ndarray:
    """fraction of the frame that changed, per frame."""
    return (np.abs(frames[1:] - frames[:-1]) > PIXEL_DELTA).mean(axis=(1, 2))


def background(frames: np.ndarray, samples: int = 200) -> np.ndarray:
    """the empty street: a per-pixel median over the clip.

    parked cars are in the median and so vanish from the difference, which is
    the point -- what is left is whatever was not there before.
    """
    idx = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    return np.median(frames[idx], axis=0)


def changed_mask(frames: np.ndarray, bg: np.ndarray, i: int) -> np.ndarray:
    """which scan pixels differ from the empty street in frame `i`.

    against the background rather than the neighbouring frame. a frame-to-frame
    difference marks where a car came from *and* where it got to, so the region
    straddles the road beside the car instead of covering it: one passing car
    was scored a miss when the detector had in fact boxed it at 0.9.
    """
    from scipy import ndimage

    changed = np.abs(frames[i] - bg) > PIXEL_DELTA
    # close small gaps: a windscreen and a bonnet difference as two blobs.
    #
    # `border_value=1` because closing erodes, and eroding with a background
    # border eats anything touching the edge of the frame -- measured at 40% of
    # a solid blob flush against column zero, with the column itself erased. a
    # vehicle entering or leaving the scene is exactly that shape, so crops of
    # them were being scored as crops of a still street: 73-89% of everything
    # this called a false positive touched a frame edge, against 3% of what it
    # accepted. the pipeline was being blamed for the measurement.
    return ndimage.binary_closing(
        changed, structure=np.ones((3, 3)), iterations=2, border_value=1
    )


def blobs(frames: np.ndarray, bg: np.ndarray, i: int, main_w: int, main_h: int,
          min_px: int = MIN_BLOB_PX, min_fill: float = MIN_BLOB_FILL) -> list[list[int]]:
    """**every** moving thing in frame `i`, in main-stream pixels, largest first.

    this used to return only the largest, which is what a global bounding box
    over all changed pixels would have to be replaced with -- that spans from a
    shimmering wire to a shadow on the far kerb. but largest-only has its own
    failure, and it is the one that mattered: an event is a stretch of *time*
    when the frame was changing, not one vehicle. two cars crossing together,
    or one following another closely enough to merge, arrived as a single event
    and only one of them was ever labelled. the rest became "false positives"
    the moment metermate cropped them correctly.
    """
    from scipy import ndimage

    labelled, n = ndimage.label(changed_mask(frames, bg, i))
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(labelled), labelled, range(1, n + 1))
    sx, sy = main_w / SCAN_W, main_h / SCAN_H
    out = []
    # largest first, so a caller taking the first gets the clearest thing. the
    # size floor cannot end the loop early any more: a wire is large and still
    # not a transit, so each candidate is judged on its own.
    for k in np.argsort(sizes)[::-1]:
        if sizes[k] < min_px:
            break
        ys, xs = np.nonzero(labelled == int(k) + 1)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if sizes[k] / (w * h) < min_fill:
            continue
        out.append([int(xs.min() * sx), int(ys.min() * sy),
                    int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)])
    return out


def boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def overlap_frac(inner, outer) -> float:
    """how much of `inner` sits inside `outer`.

    asymmetric on purpose. a crop is judged by whether *it* is full of moving
    pixels, not by whether it covers all of them: a crop of one car in a queue
    of three is a good crop.
    """
    w = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1)
    return w * h / area
