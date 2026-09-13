"""windows of local time, written relative to the moment the test runs.

shared because several test files now ask the same thing of the clock. a fixed
window would either expire or prove nothing: what is under test is that the
clock decides, so every window here is built from `now` -- the same local time
the binary reads through `localtime_r`.
"""

import datetime

_MINUTE = datetime.timedelta(minutes=1)

# how far ahead of the moment the test runs an edge is placed. two minutes
# leaves time to assert the state on one side of an edge before the run reaches
# it, and no longer than a test is willing to wait for the other side.
LEAD_MINUTES = 2


def now() -> datetime.datetime:
    # astimezone() with no argument is the local zone.
    return datetime.datetime.now(datetime.UTC).astimezone()


def between(start: datetime.datetime, end: datetime.datetime) -> str:
    """`"HH:MM-HH:MM"`: what the config parses, wrapping midnight as it does."""
    return f"{start:%H:%M}-{end:%H:%M}"


def open_window(shift_minutes: int = -LEAD_MINUTES, length_minutes: int = 120) -> str:
    """a window the clock is inside, and stays inside for the whole run."""
    start = now() + datetime.timedelta(minutes=shift_minutes)
    return between(start, start + datetime.timedelta(minutes=length_minutes))


def closed_window(shift_minutes: int = 360, length_minutes: int = 60) -> str:
    """a window the clock is nowhere near: hours ahead, so it cannot open."""
    start = now() + datetime.timedelta(minutes=shift_minutes)
    return between(start, start + datetime.timedelta(minutes=length_minutes))


# how far ahead of the moment a test starts an edge is placed. the binary needs
# a few seconds to load a model and probe a stream before it could pull anything,
# and the test needs those seconds to prove it did not.
LEAD_SECONDS = 60


def boundary(lead_seconds: int = LEAD_SECONDS) -> datetime.datetime:
    """the next minute boundary at least `lead_seconds` away.

    a window edge lands on a minute, so the boundary is the edge. the lead is
    what leaves time to assert the state on the other side of it first, and it is
    why the wait is a minute to two rather than a second to two -- a boundary one
    second away is a test that asserts against a race.
    """
    edge = now().replace(second=0, microsecond=0) + _MINUTE
    while (edge - now()).total_seconds() < lead_seconds:
        edge += _MINUTE
    return edge


def opening_soon(length_minutes: int = 60) -> str:
    """shut now, opening at that boundary: the opening edge, on its own.

    a window that was already open would prove nothing about a stream that has
    to start.
    """
    edge = boundary()
    return between(edge, edge + datetime.timedelta(minutes=length_minutes))


def closing_soon(behind_minutes: int = 120) -> str:
    """open now, shutting at that boundary: the closing edge, on its own."""
    start = now() - datetime.timedelta(minutes=behind_minutes)
    return between(start.replace(second=0, microsecond=0) + _MINUTE, boundary())


# `strftime`'s `%a` is in the running locale, and the config's day names are
# English, so the tests say them in English too rather than trusting the box.
DAYS = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")


def _wday(day: datetime.datetime) -> int:
    """the day index the config and `tm_wday` both use: sunday first."""
    return day.isoweekday() % 7


def today() -> str:
    """this weekday, as the config spells it: `"sat"`, `"wed"`."""
    return DAYS[_wday(now())]


def not_today() -> str:
    """a weekday the clock is not on: three days ahead, so whichever way a week
    is counted it is a day that is not this one and not tomorrow."""
    return DAYS[(_wday(now()) + 3) % 7]


def on_today(window: str) -> str:
    """the same window, but saying so out loud: `"wed 07:00-19:00"`."""
    return f"{today()} {window}"


def on_another_day(window: str) -> str:
    """a window whose clock time the run falls inside, on a day it is not.

    the case a day list exists for, and the one a window of clock time alone
    cannot express: the time says open and the day says no.
    """
    return f"{not_today()} {window}"
