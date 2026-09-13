"""end to end behaviour of the motion gate, driving the real binary."""

from conftest import motion_events, run_metermate


def test_static_scene_produces_no_alerts(binary, config, static_clip):
    """the expensive half of the requirement: never cry wolf on an empty street."""
    out = run_metermate(binary, config, static_clip)
    assert motion_events(out) == [], out


def test_moving_object_fires_exactly_one_event_pair(binary, config, moving_clip):
    """one object crossing the frame is one event, not one per frame (r4.1)."""
    events = motion_events(run_metermate(binary, config, moving_clip))
    assert events, "a vehicle-sized object crossing the frame must fire"

    ons = [e for e in events if "motion on" in e]
    assert len(ons) == 1, f"expected a single latched event, got {len(ons)}:\n" + "\n".join(events)


def test_regions_are_bounded_not_whole_frame(binary, config, moving_clip):
    """regression guard.

    a single bounding box over every changed pixel made one speck of noise plus a
    real object span the entire frame, which would hand the detector the whole
    image and defeat the gate. regions must stay near the object's own size.
    """
    import re

    out = run_metermate(binary, config, moving_clip)
    widths = [int(w) for w in re.findall(r"w: (\d+)", out)]
    heights = [int(h) for h in re.findall(r"h: (\d+)", out)]
    assert widths, f"no regions reported:\n{out}"

    # the drawn block is 90x60; allow generous slack for motion blur and the
    # trailing edge, but nothing close to the 640x480 frame.
    assert max(widths) < 400, f"region spans most of the frame: {max(widths)}"
    assert max(heights) < 300, f"region spans most of the frame: {max(heights)}"


def test_file_source_runs_without_a_camera_or_broker(binary, config, static_clip):
    """r5.5: the same binary must be exercisable on a laptop with neither."""
    out = run_metermate(binary, config, static_clip)
    assert "stream ended" in out, out
    assert "panic" not in out.lower(), out
