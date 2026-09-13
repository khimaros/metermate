"""the stream keeps office hours: outside the window the channels close.

r11.1 gave the harvest and the recorder a window each, but both sit *downstream*
of the streams: a shut harvest still leaves two ffmpeg processes pulling a
street nobody is looking at, which is r3.1 spent on nothing. `[stream]
active_hours` closes the channels themselves, and the harvest and the recorder
go quiet along with them because both are driven by frames.

what must not go quiet is the process. the crops and the clips on disk outlive
the window that wrote them (r8.5, r8.6), so every test here asserts against a
preview that keeps answering while nothing is being pulled at all.

the windows come from `windows.py` and are placed a minute or two ahead of the
moment the test runs: the edges are the whole point, and the clock -- not the
fixture -- is what decides them.
"""

import datetime
import json
import re
import time
from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO, in_page, run_metermate
from ingest import ffmpegs, wait_for_no_ffmpeg
from test_events import Running, free_port, get
from windows import (
    between,
    boundary,
    closed_window,
    closing_soon,
    now,
    on_another_day,
    on_today,
    open_window,
    opening_soon,
)

# longer than the binary takes to load a model, probe both streams and spawn
# its first ffmpeg: enough that "no ffmpeg" means the clock said no rather
# than the process not having got there yet.
START_GRACE_S = 20
# a window edge lands on a minute and is noticed on a roughly one second poll,
# and a closed channel is noticed between frames. this is the slack all of
# those add up to, on top of waiting out the boundary itself.
EDGE_GRACE_S = 20
# the longest a boundary can be away: windows.boundary keeps it at least a
# minute out, and it lands on a minute, so a minute and two is the ceiling.
BOUNDARY_S = 120


def stream_config(path: Path, crops: Path, events: Path, hours: str, unload: bool = False) -> Path:
    """harvest and recording both on, so their silence can be attributed."""
    # the key is only written when it is wanted: an unknown field is refused at
    # load, and a config that names one the binary does not have is this test's
    # red state, not every test in the file's.
    models = "unload_models = true\n" if unload else ""
    path.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[stream]
gate_subtype = 1
gate_width = {CLIP_W}
gate_height = {CLIP_H}
active_hours = "{hours}"
{models}
[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 500

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
enabled = true
dir = "{crops}"

[record]
enabled = true
dir = "{events}"
max_clip_secs = 2
hangover_secs = 1
# the fixtures are a white block rather than a vehicle, so nothing is ever
# detected in them and the shipped default would discard every clip.
keep = []
""")
    return path


def memory_mb(pid: int) -> int:
    """resident set, from the kernel rather than from the process's own idea."""
    with open(f"/proc/{pid}/status", encoding="ascii") as report:
        for line in report:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    raise AssertionError(f"no resident set for pid {pid}")


def started(binary: Path, tmp_path: Path, source: Path, hours: str, unload: bool = False):
    crops, events = tmp_path / "crops", tmp_path / "events"
    cfg = stream_config(tmp_path / "metermate.toml", crops, events, hours, unload)
    return Running(binary, cfg, source, free_port()), crops, events


def stats(port: int) -> dict | None:
    """`/stats`, or nothing while the process is still starting."""
    status, body, _ = get(port, "/stats")
    return json.loads(body) if status == 200 else None


def gate_frames(port: int) -> int:
    """frames so far, from a preview that is expected to be answering."""
    report = stats(port)
    assert report is not None, "the preview stopped answering while the street was not being pulled"
    return report["input"]["gate_frames"]


def wait_for_frames(port: int, timeout: float) -> int:
    """the gate frame count once it has moved off zero, or after `timeout`.

    polling rather than sleeping first, because the preview only starts
    answering a second or so after the process does and a test that asks
    before then is measuring its own impatience.
    """
    deadline = time.monotonic() + timeout
    frames = 0
    while time.monotonic() < deadline:
        report = stats(port)
        if report is not None:
            frames = report["input"]["gate_frames"]
            if frames > 0:
                return frames
        time.sleep(0.5)
    return frames


def test_the_channels_stay_shut_outside_the_window(binary, tmp_path, moving_clip):
    """**the clock, not the street, said no.**

    the same fixture the harvest tests cut crops from, the same binary and the
    same gate: what differs is a window the clock is outside of, and what
    follows is that nothing is even asked of the camera.
    """
    hours = closed_window()
    run, crops, events = started(binary, tmp_path, moving_clip, hours)
    with run:
        time.sleep(START_GRACE_S)
        out = run.output()
        assert run.proc.poll() is None, (
            f"the process gave up instead of waiting for the window:\n{out[-2000:]}"
        )
        # both halves of the sentence: the window, and the fact that it is the
        # window and not the street that is holding the streams shut.
        assert f"only {hours} local" in out, out[-2000:]
        assert "stream shut until" in out, out[-2000:]
        assert not ffmpegs(run.proc.pid), "a stream was pulled outside the window"
        assert gate_frames(run.port) == 0, "frames arrived outside the window"
        assert not list(crops.glob("*.jpg")), "the harvest ran outside the window"
        assert not list(events.glob("*.mp4")), "the recorder ran outside the window"
        # and the browser is still welcome: clips outlive the window (r8.6).
        assert get(run.port, "/")[0] == 200


def test_the_channels_open_when_the_window_opens(binary, tmp_path, long_clip):
    """reopening is not a restart: the same process starts pulling again.

    the window shuts when this run starts and opens at a minute boundary a
    minute or two in, so the test first proves the channels are shut and then
    that frames arrive with nothing started by hand.
    """
    run, _, _ = started(binary, tmp_path, long_clip, opening_soon())
    with run:
        time.sleep(5)
        assert not ffmpegs(run.proc.pid), "the window was shut and a stream was pulled anyway"
        assert gate_frames(run.port) == 0, "the window was shut and frames arrived anyway"

        seen = wait_for_frames(run.port, BOUNDARY_S + EDGE_GRACE_S)
        out = run.output()
        assert run.proc.poll() is None, out[-2000:]
        assert seen > 0, f"the window opened and nothing was pulled:\n{out[-2000:]}"
        assert "rtsp channels starting" in out, out[-2000:]


def test_the_channels_close_when_the_window_shuts(binary, tmp_path, long_clip):
    """closing is not a restart either: the process stays and the streams go.

    this is the half that saves the machine. an idle ffmpeg pulling 1.8 mbit
    and decoding it is invisible until it is twelve hours of it, and it is the
    same twelve hours r3.1 says belongs to nothing.
    """
    run, _, _ = started(binary, tmp_path, long_clip, closing_soon())
    with run:
        # prove it was open first, or the closing proves nothing.
        assert wait_for_frames(run.port, 60) > 0, f"the stream never started:\n{run.output()}"

        assert wait_for_no_ffmpeg(run.proc.pid, BOUNDARY_S + EDGE_GRACE_S), (
            f"ffmpeg outlived the window:\n{run.output()[-2000:]}"
        )
        out = run.output()
        assert run.proc.poll() is None, (
            f"the process took the closed channels as its own end:\n{out[-2000:]}"
        )
        assert "shut until" in out, out[-2000:]

        # nothing new arrives. the first wait is for whatever was already
        # queued when the channels went down to drain through the loop; after
        # that the number has to stop moving entirely, or the window closed
        # and the street kept coming.
        time.sleep(3)
        settled = gate_frames(run.port)
        time.sleep(3)
        assert gate_frames(run.port) == settled, "frames kept arriving after the window shut"
        assert get(run.port, "/")[0] == 200


def wait_for_preview(port: int, timeout: float = 30.0):
    """hold until the page can be served, which is a second or so after start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stats(port) is not None:
            return
        time.sleep(0.2)
    pytest.fail(f"the preview never answered on port {port}")


# what the live tab says about a stream that is not being pulled. the page's own
# scope is what the node harness evaluates this in, so the assertion is about
# the element the page writes rather than about a function it calls.
SAYS = """(async () => {
  await showStreamState();
  return {hidden: streamEl.hidden, said: streamMsg.textContent};
})()"""


def test_the_live_page_says_the_stream_is_closed(binary, tmp_path, node, moving_clip):
    """**a page holding the last frame of a street it stopped watching is a page
    making a claim that is no longer true.**

    the frozen picture is the failure here, not the notice: nothing about it says
    when it stopped, and a still street looks exactly like a quiet one. what the
    server knows -- that the channels are shut, and until when -- has to be on the
    picture rather than only in the log.
    """
    hours = closed_window()
    run, _, _ = started(binary, tmp_path, moving_clip, hours)
    with run:
        wait_for_preview(run.port)
        # the markup, not just the behaviour the stubbed dom of the node harness
        # would have run against anything.
        status, body, _ = get(run.port, "/")
        assert status == 200 and b'id="stream-shut"' in body, "the live page has no notice"
        got = in_page(node, run.port, SAYS)
        assert got["hidden"] is False, f"the page said nothing about a closed stream: {got}"
        assert_says_when_it_reopens(got["said"], hours.split("-")[0])


# the notice is one of two shapes, and which one is correct is a fact about the
# clock the test set the window by rather than about the page: a wait that runs
# past midnight names the day, and one that does not must not.
DAYS = r"sun|mon|tue|wed|thu|fri|sat"


def assert_says_when_it_reopens(said: str, opening: str) -> None:
    """the notice, read as a sentence: `until 07:00 local`, or `until mon 07:00`."""
    assert said.startswith("until ") and said.endswith(" local"), said
    body = said[len("until ") : -len(" local")]
    minutes = int(opening[:2]) * 60 + int(opening[3:])
    now_at = now()
    later_today = now_at.hour * 60 + now_at.minute < minutes
    if later_today:
        assert body == opening, f"a wait that ends tonight named a day: {said}"
    else:
        assert re.fullmatch(rf"({DAYS}) {re.escape(opening)}", body), (
            f"a wait that runs past midnight did not name the day: {said}"
        )


def test_the_page_says_nothing_while_the_stream_runs(binary, tmp_path, node, long_clip):
    """the same page, the same binary, the window open: no notice. the long
    fixture because this stream really is pulled, and a six second clip would
    end the run before the page was ever asked."""
    run, _, _ = started(binary, tmp_path, long_clip, open_window())
    with run:
        wait_for_preview(run.port)
        got = in_page(node, run.port, SAYS)
        assert got["hidden"] is True, f"a running stream was reported closed: {got}"


def test_a_window_on_a_day_that_is_not_today_keeps_the_stream_shut(
    binary, tmp_path, node, moving_clip
):
    """**the time says open and the day says no.**

    this window is one the clock is standing inside, written for a weekday the
    run is not on -- which is the whole reason a day list exists, and the thing
    a window of clock time cannot say. what has to follow is every symptom of a
    shut stream: no ffmpeg, no frames, and a report naming the day it is
    waiting for rather than an hour that is not coming.
    """
    hours = on_another_day(open_window())
    run, _, _ = started(binary, tmp_path, moving_clip, hours)
    with run:
        time.sleep(START_GRACE_S)
        out = run.output()
        assert run.proc.poll() is None, out[-2000:]
        assert f"only {hours} local" in out, out[-2000:]
        assert not ffmpegs(run.proc.pid), f"a stream was pulled on the wrong day:\n{out[-2000:]}"
        assert gate_frames(run.port) == 0, "frames arrived on the wrong day"
        report = stats(run.port)["input"]
        assert report["stream_shut"] is True, report
        # "until 07:00" three days from now is a wrong time, not a far one.
        assert re.fullmatch(r"(sun|mon|tue|wed|thu|fri|sat) \d\d:\d\d", report["stream_opens"]), (
            report
        )
        # and the page says the same sentence, day and all: the notice is what
        # an operator actually reads, and a wait that names the wrong shape is
        # the same wrong claim in bigger type.
        got = in_page(node, run.port, SAYS)
        assert got["hidden"] is False, got
        assert re.fullmatch(r"until (sun|mon|tue|wed|thu|fri|sat) \d\d:\d\d local", got["said"]), (
            got
        )


def test_a_window_that_names_today_ends_when_the_hour_says_so(binary, tmp_path, long_clip):
    """a day in front of a window changes nothing about the edge it ends on.

    this is the regression a day grammar could easily have: a window that says
    "wed 07:00-18:00" on a wednesday is the window it always was, and the
    channels still go down at the end of it.
    """
    run, _, _ = started(binary, tmp_path, long_clip, on_today(closing_soon()))
    with run:
        assert wait_for_frames(run.port, 60) > 0, f"the stream never started:\n{run.output()}"
        assert wait_for_no_ffmpeg(run.proc.pid, BOUNDARY_S + EDGE_GRACE_S), (
            f"ffmpeg outlived a window with a day on the front:\n{run.output()[-2000:]}"
        )
        assert get(run.port, "/")[0] == 200


def test_two_windows_open_one_day(binary, tmp_path, long_clip):
    """a morning and an evening, and the middles of the day shut.

    the shape of a school street, and the one a single window forces you to pay
    for in twelve hours of decoding. the second window is eight hours off, so
    what the shut stream reports is the *other* window's start: two windows
    means the wait is to whichever one comes next, not to the first one
    somebody wrote down.
    """
    first = closing_soon()
    edge = boundary()
    later = edge + datetime.timedelta(hours=8)
    hours = f"{first},{between(later, later + datetime.timedelta(hours=2))}"

    run, _, _ = started(binary, tmp_path, long_clip, hours)
    with run:
        assert wait_for_frames(run.port, 60) > 0, f"the stream never started:\n{run.output()}"
        assert wait_for_no_ffmpeg(run.proc.pid, BOUNDARY_S + EDGE_GRACE_S), (
            f"the first of two windows outlived itself:\n{run.output()[-2000:]}"
        )
        report = stats(run.port)["input"]
        assert report["stream_shut"] is True, report
        assert report["stream_opens"].endswith(f"{later:%H:%M}"), (
            f"waiting for the wrong window: {report}"
        )
        assert get(run.port, "/")[0] == 200


# the least that has to come back for an "unloaded" to mean anything. the
# detector the tests load is 37 MB of weights and a 5 MB input buffer, and the
# assertion is that the process let go of something on that order rather than of
# a few pages of drift.
FLOOR_MB = 10


def wait_for_release(port: int, timeout: float = 15.0) -> int:
    """the megabytes `/stats` says the last unload gave back.

    reported by the code that made the drop, because watching the resident set
    from here cannot tell a model's megabytes from an evening of harvesting --
    and the question an operator asks is how much the switch bought.
    """
    deadline = time.monotonic() + timeout
    released = 0
    while time.monotonic() < deadline:
        report = stats(port)
        if report is not None:
            released = report["input"]["models_released_mb"]
            if released > 0:
                return released
        time.sleep(0.5)
    return released


def test_a_shut_window_can_take_the_models_with_it(binary, tmp_path, long_clip):
    """**the box that does something else at night.**

    `active_hours` stops the asking and `unload_models` gives the weights back
    with it: several hundred megabytes of onnx held from midnight to seven for a
    street nobody is watching is the same r3.1 argument the channels were. what
    is not on the table is unloading while frames are arriving -- the models go
    because nothing can look at a frame, not because it would be convenient.
    """
    run, _, _ = started(binary, tmp_path, long_clip, closing_soon(), unload=True)
    with run:
        assert wait_for_frames(run.port, 60) > 0, f"the stream never started:\n{run.output()}"
        assert wait_for_no_ffmpeg(run.proc.pid, BOUNDARY_S + EDGE_GRACE_S), (
            f"ffmpeg outlived the window:\n{run.output()[-2000:]}"
        )
        released = wait_for_release(run.port)
        out = run.output()
        assert released >= FLOOR_MB, (
            f"the window shut with the switch on and gave back {released} MB:\n{out[-2000:]}"
        )
        assert "models unloaded" in out, out[-2000:]
        # and the window that took the models still serves the ones it kept.
        assert get(run.port, "/")[0] == 200, "the preview went with the models"


def test_a_shut_window_leaves_the_models_alone_by_default(binary, tmp_path, long_clip):
    """the switch is off, so a closed window costs the memory it always cost.

    unloading is a decision a deployment makes: the model stays warm, the box is
    not shared, and a model load every morning is not worth having. the window
    closing is not on its own a licence to take anything away, and a run that
    unloaded by default would be a surprise at 07:00 for the cost of a clock.
    """
    run, _, _ = started(binary, tmp_path, long_clip, closing_soon())
    with run:
        assert wait_for_frames(run.port, 60) > 0, f"the stream never started:\n{run.output()}"
        loaded = memory_mb(run.proc.pid)
        assert wait_for_no_ffmpeg(run.proc.pid, BOUNDARY_S + EDGE_GRACE_S), (
            f"ffmpeg outlived the window:\n{run.output()[-2000:]}"
        )
        time.sleep(2)
        out = run.output()
        assert "models unloaded" not in out, out[-2000:]
        report = stats(run.port)["input"]
        assert report["models_released_mb"] == 0, (
            f"the switch was off and {report['models_released_mb']} MB were released: {report}"
        )
        assert memory_mb(run.proc.pid) > loaded - FLOOR_MB, (
            f"{loaded} MB before the edge and {memory_mb(run.proc.pid)} after, "
            f"with the switch off: {out[-2000:]}"
        )


def test_the_models_come_back_with_the_window(binary, tmp_path, long_clip):
    """an unload nobody can undo is a shutdown, not a feature.

    this run starts outside its window, so the models are never loaded at all --
    which is the point of the switch on a box that boots at three in the morning
    -- and the reopening has to build them before a single crop can be cut. the
    proof is frames arriving, not a line saying it would try.
    """
    run, _, _ = started(binary, tmp_path, long_clip, opening_soon(), unload=True)
    with run:
        time.sleep(5)
        assert gate_frames(run.port) == 0, "the window was shut and frames arrived anyway"
        quiet = memory_mb(run.proc.pid)
        assert wait_for_frames(run.port, BOUNDARY_S + EDGE_GRACE_S) > 0, (
            f"the window opened and nothing was pulled:\n{run.output()[-2000:]}"
        )
        out = run.output()
        assert run.proc.poll() is None, out[-2000:]
        # the models are back, and back *before* the frame that needed them: they
        # come up with the channels rather than being built on the way past a
        # look, because the first vehicle through a window that opened at seven
        # is not the one to pay a second of model load for.
        resident, at = 0, time.monotonic() + 30
        while time.monotonic() < at:
            resident = memory_mb(run.proc.pid)
            if resident >= quiet + FLOOR_MB:
                break
            time.sleep(0.5)
        assert resident >= quiet + FLOOR_MB, (
            f"{quiet} MB before the edge, {resident} after: {run.output()[-2000:]}"
        )


def test_unloading_without_a_window_says_so(binary, config, static_clip):
    """a knob that cannot do anything is a knob somebody thinks is working.

    `unload_models` without `active_hours` is that: the stream never shuts, so
    the memory is held exactly as it was, and the line says which of the two
    settings is missing.
    """
    text = config.read_text().replace("[stream]\n", "[stream]\nunload_models = true\n")
    assert "unload_models = true" in text, "the config has no [stream] section to extend"
    config.write_text(text)
    out = run_metermate(binary, config, static_clip)
    assert "no window to unload with" in out, out[-2000:]


def test_a_stream_window_that_is_not_one_is_refused(binary, config, static_clip):
    """a typo here keeps nothing all day and looks like a quiet street, so it
    is refused at load like every other window (r11.1)."""
    text = config.read_text().replace("[stream]\n", '[stream]\nactive_hours = "07:00-25:00"\n')
    assert 'active_hours = "07:00-25:00"' in text, "the config has no [stream] section to break"
    config.write_text(text)
    out = run_metermate(binary, config, static_clip)
    assert "25:00" in out, out[-2000:]
    assert "0-23" in out, out[-2000:]
