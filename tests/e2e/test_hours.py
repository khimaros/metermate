"""time restrictions: the harvest and the recorder keep office hours (r11.1).

enforcement is a daytime phenomenon, and a night of empty road spends the same
disk budget as a day of traffic. both `active_hours` knobs are driven here
against the real binary, with windows written *relative to the moment the test
runs* -- the point is that the clock decides, so a fixed window would either
expire or prove nothing.

what is deliberately not restricted is alerting: the window sits on the two
decisions that *start* collecting, so a confirmation outside it still writes
its evidence crop and still publishes, because r4.5 does not keep office
hours. the existing alert tests (`test_evidence.py`) are the ones that would
break if that placement ever moves.
"""

from pathlib import Path

import pytest

from alerts import base, clips, run
from conftest import run_metermate
from test_events import Running, free_port, write_config
from windows import closed_window, on_another_day, on_today, open_window


@pytest.fixture(scope="module")
def street(tmp_path_factory) -> Path:
    """the street clips once; every run below points at the same pair."""
    root = tmp_path_factory.mktemp("hours")
    clips(root / "big")
    return root / "big"


def harvest_run(binary, street: Path, crops: Path, hours: str) -> str:
    cfg = street.parent / f"harvest-{crops.name}.toml"
    cfg.write_text(
        base(f"{street}/{{subtype}}.mp4")
        + f'\n[harvest]\nenabled = true\ndir = "{crops}"\n'
        + f'active_hours = "{hours}"\n'
    )
    code, out = run(binary, cfg, "--dry-run")
    assert code == 0, out
    return out


def test_crops_flow_while_the_window_is_open(binary, street, tmp_path):
    """the control: same street, same binary, same everything bar the clock."""
    hours = open_window()
    crops = tmp_path / "open"
    out = harvest_run(binary, street, crops, hours)
    assert f"only {hours} local" in out, out[-2000:]
    assert list(crops.glob("*.jpg")), f"the window was open and nothing was kept:\n{out[-2000:]}"


def test_no_crops_flow_while_the_window_is_shut(binary, street, tmp_path):
    """**the clock, not the street, said no.** the gate fired over the same
    clips the control run above harvested from, and the startup line says the
    window is shut -- because an empty harvest directory otherwise reads as a
    quiet street.
    """
    hours = closed_window()
    crops = tmp_path / "shut"
    out = harvest_run(binary, street, crops, hours)
    assert f"only {hours} local" in out, out[-2000:]
    assert "motion on" in out, f"nothing happened at all, so nothing was tested:\n{out[-2000:]}"
    assert not list(crops.glob("*.jpg")), (
        f"crops were harvested outside the window: {list(crops.glob('*.jpg'))}"
    )
    assert "harvested frame" not in out, out[-2000:]


def test_a_window_on_a_day_that_is_not_today_keeps_nothing(binary, street, tmp_path):
    """**the day, not the hour, said no.** the clock is standing inside this
    window and it is written for a weekday the run is not on, so the harvest
    keeps nothing: one word in front of the same knob, and the same decision.
    """
    hours = on_another_day(open_window())
    crops = tmp_path / "wrong-day"
    out = harvest_run(binary, street, crops, hours)
    assert f"only {hours} local" in out, out[-2000:]
    assert "motion on" in out, f"nothing happened at all, so nothing was tested:\n{out[-2000:]}"
    assert not list(crops.glob("*.jpg")), (
        f"crops were harvested on the wrong day: {list(crops.glob('*.jpg'))}"
    )


def test_a_window_that_names_today_keeps_what_the_hour_keeps(binary, street, tmp_path):
    """the control for the control: saying today out loud is not a restriction.

    a day list that quietly shut everything on the day it named would be the
    failure this feature exists to have, and it would look exactly like a quiet
    street.
    """
    hours = on_today(open_window())
    crops = tmp_path / "today"
    out = harvest_run(binary, street, crops, hours)
    assert f"only {hours} local" in out, out[-2000:]
    assert list(crops.glob("*.jpg")), f"today was refused its own window:\n{out[-2000:]}"


def record_run(binary, tmp_path: Path, moving_clip, hours: str) -> tuple[list[Path], str]:
    events = tmp_path / f"events-{hours.replace(':', '')}"
    port = free_port()
    cfg = write_config(
        tmp_path / f"metermate-{events.name}.toml", events, active_hours=f'"{hours}"'
    )
    with Running(binary, cfg, moving_clip, port) as r:
        out = r.wait()
    return sorted(events.glob("*.mp4")) if events.exists() else [], out


def test_recording_starts_inside_the_window(binary, tmp_path, moving_clip):
    kept, out = record_run(binary, tmp_path, moving_clip, open_window())
    assert kept, f"the window was open and no clip was written:\n{out[-2000:]}"
    # the line printed when a clip opens, not the startup line about
    # recording, which mentions clips whatever the clock says.
    assert "opened" in out, out[-2000:]


def test_recording_stays_shut_outside_the_window(binary, tmp_path, moving_clip):
    """the gate fires -- the same fixture triggers every other test here -- and
    no clip opens. the trigger is what the window closes, so a clip already
    running keeps its own pre-roll and hangover."""
    kept, out = record_run(binary, tmp_path, moving_clip, closed_window())
    assert "motion on" in out, f"nothing happened at all, so nothing was tested:\n{out[-2000:]}"
    assert not kept, f"clips were recorded outside the window: {kept}"
    # the line printed when a clip *opens*; the startup line about recording
    # is expected and already asserts on the window itself.
    assert "opened" not in out, out[-2000:]


@pytest.mark.parametrize("section", ["harvest", "record"])
def test_a_window_that_is_not_one_is_refused(binary, config, static_clip, section):
    """a typo in a window silently keeps nothing -- or everything -- and both
    read as whatever the street was doing. refused at load, like `keep`."""
    config.write_text(config.read_text() + f'\n[{section}]\nactive_hours = "07:00-25:00"\n')
    out = run_metermate(binary, config, static_clip)
    assert "25:00" in out, out[-2000:]
    assert "0-23" in out, out[-2000:]
