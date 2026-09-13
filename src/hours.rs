//! windows of local time: when a collection is allowed to run (r11.1, r11.3).
//!
//! enforcement is a daytime phenomenon and the street is a different one at
//! 3am -- so the harvest and the recorder want a window, not a round-the-clock
//! spin that spends the disk budget on empty road.
//!
//! the grammar is groups of `"[days ]window[,window]*"`, separated by `;`:
//!
//! ```text
//! "07:00-19:00"                          every day
//! "mon-fri 07:00-19:00"                  weekdays only
//! "mon-fri 07:00-09:00,16:00-19:00"      a morning and an evening
//! "mon-fri 07:00-19:00; sat 09:00-17:00" a week's worth of them
//! ```
//!
//! a window with no days in front of it is the ordinary daily one and stays
//! written the way it has always been; the days in front of it are what makes a
//! window that is shut thursday evening and open monday morning, which is what a
//! weekday window is for.
//!
//! the decision is minutes since local midnight, which keeps the arithmetic
//! pure: reading the clock is one small side-effecting call, and everything
//! else -- parsing, laying the week out, covering, waiting -- is testable
//! without one.

use serde::de::Error as _;
use std::time::{SystemTime, UNIX_EPOCH};

/// minutes in a day, in the units this whole module thinks in.
pub const MINUTES_PER_DAY: u32 = 1_440;

/// the week, in the order a clock reports it: `tm_wday` counts from sunday.
const DAYS: [(&str, &str); 7] = [
    ("sun", "sunday"),
    ("mon", "monday"),
    ("tue", "tuesday"),
    ("wed", "wednesday"),
    ("thu", "thursday"),
    ("fri", "friday"),
    ("sat", "saturday"),
];

/// every day of the week, as the bit set a group's days are held in.
const ALL_DAYS: u8 = 0x7f;

/// a window that repeats.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Hours {
    /// no restriction, which is what every deployment had before this existed.
    #[default]
    Always,
    /// open minutes laid out over a week, which includes the ordinary case of
    /// the same window every day.
    Weekly(Week),
}

/// one open span of a single day, in minutes since local midnight.
///
/// spans never wrap: a window written across midnight is laid down twice, once
/// on each of the two days it is really open on, which is what makes covering a
/// minute a lookup rather than a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    from: u32,
    to: u32,
}

impl Span {
    fn covers(&self, minute: u32) -> bool {
        self.from <= minute && minute < self.to
    }
}

/// a week's worth of open minutes, and the groups they were written as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Week {
    /// as written, so the startup line and `/stats` can print back the window
    /// somebody configured rather than a reconstruction of it.
    groups: Vec<Group>,
    /// the same windows laid out by day, in `tm_wday` order.
    open: [Vec<Span>; 7],
}

impl Default for Week {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            open: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// one group: the days it speaks for and the windows open on them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Group {
    /// a bit per day; a group with no days means every day, which is what a
    /// window written on its own has always meant.
    days: u8,
    /// as written, so `09:00-09:00` prints back as `09:00-09:00` rather than as
    /// the whole day it means.
    windows: Vec<Span>,
}

impl Hours {
    /// parse a window, refusing anything that is not one: a typo in a window is
    /// a collection that never stops, or never starts, and both read as whatever
    /// the street was doing.
    ///
    /// an empty string is `Always`, so a config can say so explicitly.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Ok(Hours::Always);
        }
        let mut week = Week::default();
        for group in raw.split(';') {
            week.push(Group::parse(group)?);
        }
        Ok(Hours::Weekly(week))
    }

    /// whether nothing restricts this at all, which is the one case that
    /// never has to explain itself to anyone reading an empty directory.
    pub fn is_always(&self) -> bool {
        matches!(self, Hours::Always)
    }

    /// the window as a startup-line suffix, empty when there is no window
    /// to report.
    pub fn note(&self) -> String {
        match self {
            Hours::Always => String::new(),
            hours => format!(", only {hours} local"),
        }
    }

    /// whether `minute` on `day` (in `tm_wday` order, sunday first) falls
    /// inside a window. pure, and the whole decision; reading the clock is
    /// `open_at`.
    pub fn covers_at(&self, day: u32, minute: u32) -> bool {
        match self {
            Hours::Always => true,
            Hours::Weekly(week) => week.covers_at(day, minute),
        }
    }

    /// whether the window is open right now, by the operating system's clock
    /// and its notion of local -- the street's timezone, dst included.
    pub fn open_at(&self, now: SystemTime) -> bool {
        // the common case never touches the clock at all.
        match self {
            Hours::Always => true,
            Hours::Weekly(_) => {
                let (day, minute) = local_clock(now);
                self.covers_at(day, minute)
            }
        }
    }

    /// minutes of real time until the window next opens: 0 while it is open,
    /// and up to a week for a window that keeps, say, wednesdays. `None` only
    /// for a window that never shuts.
    ///
    /// the wait is real time rather than a minute of the day, because a window
    /// that opens on monday is not open at 07:00 today.
    pub fn minutes_until_open(&self, day: u32, minute: u32) -> Option<u32> {
        match self {
            Hours::Always => None,
            Hours::Weekly(week) => week.minutes_until_open(day, minute),
        }
    }

    /// when a shut window opens again, written the way a person asks for it:
    /// `"07:00"` later today, `"mon 07:00"` when the wait runs past midnight.
    ///
    /// `None` while it is open, and for a window that never shuts -- there is
    /// no time to point at and no reason to look for one. The day is half the
    /// answer: "until 07:00" on a friday evening about a weekday window is a
    /// wrong time, not an early one.
    pub fn opens_at(&self, now: SystemTime) -> Option<String> {
        if self.is_always() {
            return None;
        }
        let (day, minute) = local_clock(now);
        let wait = self.minutes_until_open(day, minute)?;
        (wait > 0).then(|| label(day, minute, wait))
    }
}

impl Week {
    /// lay one group's windows onto the days it names.
    fn push(&mut self, group: Group) {
        for day in 0..7u32 {
            if group.days & (1 << day) == 0 {
                continue;
            }
            for w in &group.windows {
                // one clock time is the whole day and none of it, and it has
                // always meant the whole day: an empty window that silently
                // keeps nothing is the failure this module exists to avoid.
                if w.from == w.to {
                    self.open[day as usize].push(Span {
                        from: 0,
                        to: MINUTES_PER_DAY,
                    });
                    continue;
                }
                if w.from < w.to {
                    self.open[day as usize].push(*w);
                } else {
                    // a window over midnight is open on the morning after as
                    // well. that morning belongs to the day the window was
                    // written for: "mon 22:00-06:00" covers tuesday at three
                    // and not wednesday at three.
                    self.open[day as usize].push(Span {
                        from: w.from,
                        to: MINUTES_PER_DAY,
                    });
                    self.open[((day + 1) % 7) as usize].push(Span { from: 0, to: w.to });
                }
            }
        }
        self.groups.push(group);
    }

    fn covers_at(&self, day: u32, minute: u32) -> bool {
        self.day(day).iter().any(|span| span.covers(minute))
    }

    fn minutes_until_open(&self, day: u32, minute: u32) -> Option<u32> {
        if self.covers_at(day, minute) {
            return Some(0);
        }
        // the first opening on the days still to come, today included. today
        // counts only the openings that have not already gone past, since a
        // window already running was caught just above.
        // a week plus a day: a window that keeps only saturdays is found again
        // on the seventh day ahead, and one day short of that is no opening.
        for ahead in 0..=7u32 {
            let mut best: Option<u32> = None;
            for span in self.day((day + ahead) % 7) {
                if ahead == 0 && span.from <= minute {
                    continue;
                }
                let at = ahead * MINUTES_PER_DAY + span.from;
                if best.is_none_or(|soonest| at < soonest) {
                    best = Some(at);
                }
            }
            if let Some(at) = best {
                return Some(at - minute);
            }
        }
        // a week with no open minute at all, which the parser does not accept.
        None
    }

    fn day(&self, day: u32) -> &[Span] {
        &self.open[(day % 7) as usize]
    }
}

impl Group {
    /// one `;` group: the days, then the windows. a token with a `:` in it is a
    /// window and one without is a day, which is why the days come first --
    /// after the first window a comma is only ever between two windows.
    fn parse(raw: &str) -> Result<Self, String> {
        let mut days: u8 = 0;
        let mut windows = Vec::new();
        let mut any_days = false;
        let mut any_windows = false;
        for token in raw
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|token| !token.is_empty())
        {
            if token.contains(':') {
                any_windows = true;
                windows.push(window(token)?);
            } else {
                if any_windows {
                    return Err(format!(
                        "{raw:?}: the days go in front of the windows they open"
                    ));
                }
                days |= day_range(token)?;
                any_days = true;
            }
        }
        if !any_windows {
            return Err(format!(
                "{raw:?} is not a window; expected \"HH:MM-HH:MM\", e.g. \"07:00-19:00\""
            ));
        }
        Ok(Group {
            days: if any_days { days } else { ALL_DAYS },
            windows,
        })
    }

    /// as written, but tidied: padded times, merged days, lower case.
    fn text(&self) -> String {
        let windows: Vec<String> = self
            .windows
            .iter()
            .map(|w| format!("{}-{}", clock(w.from), clock(w.to)))
            .collect();
        let days = days_text(self.days);
        if days.is_empty() {
            windows.join(",")
        } else {
            format!("{days} {}", windows.join(","))
        }
    }
}

impl std::fmt::Display for Hours {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hours::Always => Ok(()),
            Hours::Weekly(week) => {
                let groups: Vec<String> = week.groups.iter().map(Group::text).collect();
                f.write_str(&groups.join("; "))
            }
        }
    }
}

/// config is a string: what a person wrote is what a person reads back.
impl<'de> serde::Deserialize<'de> for Hours {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Self::parse(&raw).map_err(D::Error::custom)
    }
}

impl serde::Serialize for Hours {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

/// `"07:05"` as minutes past midnight.
fn minutes(raw: &str) -> Result<u32, String> {
    let (h, m) = raw
        .trim()
        .split_once(':')
        .ok_or_else(|| format!("{raw:?} is not a time; expected \"HH:MM\""))?;
    let h: u32 = h
        .trim()
        .parse()
        .map_err(|_| format!("{raw:?} is not a time"))?;
    let m: u32 = m
        .trim()
        .parse()
        .map_err(|_| format!("{raw:?} is not a time"))?;
    if h > 23 {
        return Err(format!(
            "{raw:?} is not a time of day: the hour must be 0-23"
        ));
    }
    if m > 59 {
        return Err(format!(
            "{raw:?} is not a time of day: the minute must be 0-59"
        ));
    }
    Ok(h * 60 + m)
}

/// `"mon-fri 07:00-19:00"` as its two clock times, in the order written: a
/// window whose start is past its end wraps midnight rather than meaning
/// nothing.
fn window(raw: &str) -> Result<Span, String> {
    let (from, to) = raw.split_once('-').ok_or_else(|| {
        format!("{raw:?} is not a window; expected \"HH:MM-HH:MM\", e.g. \"07:00-19:00\"")
    })?;
    Ok(Span {
        from: minutes(from)?,
        to: minutes(to)?,
    })
}

/// a day by name, in `tm_wday` order. both `"sat"` and `"saturday"` are
/// written, and neither cares about case.
fn day_of(name: &str) -> Option<u32> {
    let name = name.trim().to_ascii_lowercase();
    DAYS.iter()
        .position(|(short, long)| name == *short || name == *long)
        .map(|day| day as u32)
}

fn day_name(day: u32) -> &'static str {
    DAYS[(day % 7) as usize].0
}

/// one day, or a run of them: `"sat"`, `"mon-fri"`. a run wraps the week, since
/// `"fri-mon"` is a thing a person means and no year-boundary is involved.
fn day_range(raw: &str) -> Result<u8, String> {
    let (from, to) = match raw.split_once('-') {
        Some((from, to)) => (one_day(from)?, one_day(to)?),
        None => {
            let day = one_day(raw)?;
            (day, day)
        }
    };
    let mut bits = 0u8;
    let mut at = from;
    loop {
        bits |= 1 << at;
        if at == to {
            break;
        }
        at = (at + 1) % 7;
    }
    Ok(bits)
}

fn one_day(raw: &str) -> Result<u32, String> {
    day_of(raw).ok_or_else(|| {
        if !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit()) {
            // "07-19" is this module's own most likely mistake.
            format!("{raw:?} is not a day; a time is written \"HH:MM-HH:MM\"")
        } else {
            format!("{raw:?} is not a day; expected sun, mon, ... or a range like mon-fri")
        }
    })
}

/// the days a group speaks for, as they would have been written: consecutive
/// days merged into a range, and no prefix at all for a group that means every
/// day.
fn days_text(days: u8) -> String {
    if days == ALL_DAYS {
        return String::new();
    }
    // anchored on the day that opens a run -- a set day whose day before is not
    // set -- so that a run over the end of the week prints as the one range it
    // was written as: "sat-sun", not "sun,sat".
    let first = (0..7u32)
        .find(|d| days & (1 << d) != 0 && days & (1 << ((d + 6) % 7)) == 0)
        .unwrap_or(0);
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for n in 0..7u32 {
        let day = (first + n) % 7;
        if days & (1 << day) == 0 {
            continue;
        }
        match runs.last_mut() {
            Some(run) if run.1 == (day + 6) % 7 => run.1 = day,
            _ => runs.push((day, day)),
        }
    }
    runs.iter()
        .map(|(from, to)| {
            if from == to {
                day_name(*from).to_string()
            } else {
                format!("{}-{}", day_name(*from), day_name(*to))
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// minutes since local midnight as `"07:05"`, the form a config is written in.
pub fn clock(minute: u32) -> String {
    format!("{:02}:{:02}", (minute % MINUTES_PER_DAY) / 60, minute % 60)
}

/// the opening time, given that `day` is today, `minute` is now and `wait` is
/// minutes of real time from here: `"07:00"`, or `"mon 07:00"` once the wait
/// runs past midnight.
fn label(day: u32, minute: u32, wait: u32) -> String {
    let at = minute + wait;
    let ahead = at / MINUTES_PER_DAY;
    let at = at % MINUTES_PER_DAY;
    if ahead == 0 {
        return clock(at);
    }
    format!("{} {}", day_name((day + ahead) % 7), clock(at))
}

/// the local day of the week and minute of the day, as one clock reading: a
/// window that read them separately could straddle midnight between the two.
#[cfg(unix)]
fn local_clock(now: SystemTime) -> (u32, u32) {
    let Ok(since) = now.duration_since(UNIX_EPOCH) else {
        return (0, 0);
    };
    let secs = since.as_secs() as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r writes the struct we hand it, touches no global
    // state, and reads only the value `secs` points at.
    if unsafe { libc::localtime_r(&secs, &mut tm) }.is_null() {
        // no timezone to consult; UTC keeps the pipeline running and is
        // already what an unset TZ resolves to.
        return utc_clock(since);
    }
    let day = (tm.tm_wday).rem_euclid(7) as u32;
    let minute = (tm.tm_hour.max(0) as u32) * 60 + (tm.tm_min.max(0) as u32);
    (day, minute)
}

#[cfg(not(unix))]
fn local_clock(now: SystemTime) -> (u32, u32) {
    let Ok(since) = now.duration_since(UNIX_EPOCH) else {
        return (0, 0);
    };
    utc_clock(since)
}

fn utc_clock(since: std::time::Duration) -> (u32, u32) {
    let days = (since.as_secs() / 86_400) as u32;
    // the epoch was a thursday, and `tm_wday` counts from sunday.
    (
        (days + 4) % 7,
        ((since.as_secs() / 60) % (MINUTES_PER_DAY as u64)) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(h: u32, m: u32) -> u32 {
        h * 60 + m
    }

    /// a day by name rather than by number, which is how a window is written
    /// and therefore how a test about one should read.
    fn day_of(name: &str) -> u32 {
        super::day_of(name).expect("a day of the week")
    }

    /// any one day will do, for a window that named none: it means all seven.
    const ANY_DAY: u32 = 0;

    #[test]
    fn empty_is_always_and_never_needs_the_clock() {
        assert_eq!(Hours::parse("").unwrap(), Hours::Always);
        assert_eq!(Hours::parse("  ").unwrap(), Hours::Always);
        assert!(Hours::Always.is_always());
        for day in 0..7 {
            assert!(Hours::Always.covers_at(day, 0));
            assert!(Hours::Always.covers_at(day, 1_439));
        }
    }

    #[test]
    fn a_window_with_no_days_is_open_every_day() {
        // the form every deployment was configured with before days existed.
        let h = Hours::parse("07:00-19:00").unwrap();
        assert!(!h.is_always());
        for day in 0..7 {
            assert!(h.covers_at(day, at(12, 0)), "day {day} was shut");
            assert!(!h.covers_at(day, at(19, 0)), "day {day} was open");
        }
    }

    #[test]
    fn a_window_holds_its_start_and_loses_its_end() {
        let h = Hours::parse("07:00-19:00").unwrap();
        assert!(h.covers_at(ANY_DAY, at(7, 0)));
        assert!(h.covers_at(ANY_DAY, at(18, 59)));
        assert!(!h.covers_at(ANY_DAY, at(6, 59)));
        assert!(!h.covers_at(ANY_DAY, at(19, 0)));
    }

    #[test]
    fn a_window_wraps_midnight() {
        let h = Hours::parse("22:00-06:00").unwrap();
        assert!(h.covers_at(day_of("mon"), at(23, 30)));
        assert!(h.covers_at(day_of("tue"), at(0, 0)));
        assert!(h.covers_at(day_of("tue"), at(5, 59)));
        assert!(!h.covers_at(day_of("tue"), at(6, 0)));
        assert!(!h.covers_at(day_of("tue"), at(12, 0)));
        // and every morning of the week, because every evening was open.
        assert!(h.covers_at(day_of("sun"), at(3, 0)));
    }

    #[test]
    fn a_point_is_a_whole_day_not_a_day_off() {
        // nobody writes "09:00-09:00" meaning "never", and a window that
        // silently keeps nothing is the failure mode of this whole feature.
        let h = Hours::parse("09:00-09:00").unwrap();
        for day in 0..7 {
            for minute in [0, 540, 1_439] {
                assert!(h.covers_at(day, minute));
            }
        }
    }

    #[test]
    fn times_need_not_be_padded_but_must_be_times() {
        let h = Hours::parse("7:05-9:00").unwrap();
        assert!(h.covers_at(ANY_DAY, at(7, 5)));
        assert!(!h.covers_at(ANY_DAY, at(9, 0)));
        for bad in [
            "07:00",
            "24:00-08:00",
            "07:60-19:00",
            "07-19",
            "half-past",
            "07:00-",
            "-07:00",
        ] {
            assert!(Hours::parse(bad).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn waiting_for_a_window_is_the_gap_to_its_start() {
        let day = day_of("wed");
        let h = Hours::parse("07:00-19:00").unwrap();
        assert_eq!(h.minutes_until_open(day, at(7, 0)), Some(0));
        assert_eq!(h.minutes_until_open(day, at(18, 59)), Some(0));
        assert_eq!(h.minutes_until_open(day, at(19, 0)), Some(720));
        assert_eq!(h.minutes_until_open(day, at(6, 59)), Some(1));
        // wrapping: the wait is the shut part of the day, not the open part.
        let night = Hours::parse("22:00-06:00").unwrap();
        assert_eq!(night.minutes_until_open(day, at(23, 30)), Some(0));
        assert_eq!(night.minutes_until_open(day, at(12, 0)), Some(600));
        // a window that never shuts is never worth a clock read for.
        assert_eq!(Hours::Always.minutes_until_open(day, at(12, 0)), None);
    }

    #[test]
    fn waiting_counts_days_as_well_as_minutes() {
        let h = Hours::parse("sat 09:00-17:00").unwrap();
        // friday noon to saturday nine is a day and nine hours: the shut half of
        // the weekend is not time anyone is enforcing anything in.
        assert_eq!(h.minutes_until_open(day_of("fri"), at(12, 0)), Some(1_260));
        assert_eq!(h.minutes_until_open(day_of("sat"), at(8, 0)), Some(60));
        // and once the afternoon has gone, the next opening is a week away less
        // the afternoon itself.
        assert_eq!(h.minutes_until_open(day_of("sat"), at(19, 0)), Some(9_480));
        assert_eq!(h.minutes_until_open(day_of("sun"), at(12, 0)), Some(8_460));
    }

    #[test]
    fn a_shut_window_names_when_it_stops_being_shut() {
        let now = UNIX_EPOCH + std::time::Duration::from_secs(123_456_789);
        assert_eq!(Hours::Always.opens_at(now), None);
        let (day, minute) = local_clock(now);
        // open for one minute of every day, and not this one: what the line
        // that explains an empty directory needs is the time it starts being
        // wrong, and `/stats` needs the same string.
        let shut = Hours::parse(&format!("{}-{}", clock(minute + 1), clock(minute + 2))).unwrap();
        let until = shut.opens_at(now).expect("a shut window opens sometime");
        assert!(
            until.ends_with(&clock(minute + 1)),
            "{until} is not the opening minute"
        );
        // a minute of a day to go and the day joins the time, which is the
        // whole reason this is not a minute of the day.
        if minute == MINUTES_PER_DAY - 1 {
            assert!(until.starts_with(day_name((day + 1) % 7)));
        } else {
            assert!(!until.contains(' '));
        }
        // and while it is open there is nothing to name.
        let open =
            Hours::parse(&format!("{}-{}", clock(minute + 1439), clock(minute + 1))).unwrap();
        assert_eq!(open.opens_at(now), None);
    }

    #[test]
    fn a_parsed_window_prints_as_one_written() {
        assert_eq!(Hours::parse("7:5-9:0").unwrap().to_string(), "07:05-09:00");
        assert_eq!(Hours::Always.to_string(), "");
    }

    #[test]
    fn a_day_list_keeps_the_days_it_names() {
        let h = Hours::parse("mon-fri 07:00-19:00").unwrap();
        assert!(h.covers_at(day_of("mon"), at(7, 0)));
        assert!(h.covers_at(day_of("fri"), at(18, 59)));
        // the weekend is the whole point of a weekday window.
        assert!(!h.covers_at(day_of("sat"), at(12, 0)));
        assert!(!h.covers_at(day_of("sun"), at(12, 0)));
        // and a range that wraps the week wraps the list, not the calendar year.
        let weekend = Hours::parse("sat-sun 09:00-17:00").unwrap();
        assert!(weekend.covers_at(day_of("sat"), at(9, 0)));
        assert!(weekend.covers_at(day_of("sun"), at(16, 59)));
        assert!(!weekend.covers_at(day_of("fri"), at(12, 0)));
    }

    #[test]
    fn two_windows_open_one_day() {
        // a morning and an evening, which is the shape of a school street: the
        // middles of the day are the part nobody is enforcing.
        let h = Hours::parse("mon-fri 07:00-09:00,16:00-19:00").unwrap();
        let tue = day_of("tue");
        assert!(h.covers_at(tue, at(8, 0)));
        assert!(!h.covers_at(tue, at(12, 0)));
        assert!(h.covers_at(tue, at(17, 0)));
        assert!(!h.covers_at(day_of("sat"), at(8, 0)));
        assert_eq!(h.minutes_until_open(tue, at(12, 0)), Some(240));
        // the evening window belongs to the same day: after it, the wait is
        // across the night to the next morning and not to some other window.
        assert_eq!(h.minutes_until_open(tue, at(19, 0)), Some(720));
    }

    #[test]
    fn a_week_of_groups_adds_up() {
        let h = Hours::parse("mon-fri 07:00-19:00; sat 09:00-17:00").unwrap();
        assert!(h.covers_at(day_of("wed"), at(12, 0)));
        assert!(h.covers_at(day_of("sat"), at(10, 0)));
        assert!(!h.covers_at(day_of("sat"), at(19, 0)));
        // nothing said about sunday, so sunday is closed whatever the clock says.
        for minute in [at(0, 0), at(12, 0), at(23, 59)] {
            assert!(!h.covers_at(day_of("sun"), minute));
        }
    }

    #[test]
    fn a_wrapping_window_belongs_to_the_day_that_wrote_it() {
        // "22:00-06:00" on monday is open on tuesday morning. it is not open on
        // wednesday morning, which is what a day list is for and what a single
        // wrapping window used to get for free.
        let h = Hours::parse("mon 22:00-06:00").unwrap();
        assert!(h.covers_at(day_of("mon"), at(23, 0)));
        assert!(h.covers_at(day_of("tue"), at(3, 0)));
        assert!(!h.covers_at(day_of("tue"), at(6, 0)));
        assert!(!h.covers_at(day_of("wed"), at(3, 0)));
        // from sunday noon the wait is the rest of the day, the night, and the
        // morning: a day and a half to monday's evening.
        assert_eq!(h.minutes_until_open(day_of("sun"), at(12, 0)), Some(2_040));
    }

    #[test]
    fn a_window_that_opens_next_week_names_its_day() {
        // "until 07:00" said on a friday evening about a weekday window is a
        // wrong time, not an early one: the day is the half that matters.
        // friday 20:00 to monday 07:00 is the rest of the evening, a day, and
        // a morning: 59 hours.
        assert_eq!(label(day_of("fri"), at(20, 0), 3_540), "mon 07:00");
        // one that opens later the same day is still just an hour.
        assert_eq!(label(day_of("fri"), at(19, 0), 60), "20:00");
        assert_eq!(label(day_of("sat"), at(23, 0), 60), "sun 00:00");
    }

    #[test]
    fn days_and_windows_that_are_not_ones_are_refused() {
        for bad in [
            "mon-fax 07:00-19:00",
            // a window is what a group is for: days alone name no time.
            "mon-fri",
            "07:00-19:00 sat",
            "mon-fri 07:00-25:00",
            // a group with nothing in it names no window, however it got empty.
            "07:00-19:00;",
        ] {
            assert!(Hours::parse(bad).is_err(), "{bad:?} was accepted");
        }
        // the day grammar is the only new one, so the mistakes are new too:
        // nothing in the old form stops working.
        assert!(Hours::parse("mon-fri 07:00-19:00").is_ok());
        assert!(Hours::parse("07:00-19:00").is_ok());
        // what is between the days and the windows is a separator, and any
        // number of them: a space, a comma, both. no reading of the window
        // turns on it, so refusing one reads as a bug.
        for loose in [
            "mon-fri,07:00-19:00",
            "sat,  sun   09:00-17:00",
            " mon-fri 07:00-09:00 , 16:00-19:00 ",
        ] {
            assert!(Hours::parse(loose).is_ok(), "{loose:?} was refused");
        }
    }

    #[test]
    fn a_week_prints_as_it_was_written() {
        // the startup line and /stats are where a window gets read back, and a
        // person checks what they wrote against it.
        for spec in [
            "07:00-19:00",
            "mon-fri 07:00-19:00",
            "mon-fri 07:00-09:00,16:00-19:00",
            "mon-fri 07:00-19:00; sat 09:00-17:00",
        ] {
            assert_eq!(Hours::parse(spec).unwrap().to_string(), spec);
        }
        // written loosely, printed as a window: padded, merged, lower case.
        assert_eq!(
            Hours::parse("Sat,Sun 7:0-9:00").unwrap().to_string(),
            "sat-sun 07:00-09:00"
        );
    }

    #[test]
    fn a_day_list_prints_as_a_run_of_days() {
        // what comes back is a canonical day list rather than what was typed:
        // consecutive days merge into a range, and a run over the end of the
        // week stays one range, because "sat-sun" is how it would be written.
        // two separate runs print from the first one, so the order is the week's.
        let printed = |days: &str| {
            Hours::parse(&format!("{days} 07:00-19:00"))
                .unwrap()
                .to_string()
        };
        for days in ["sun", "mon-fri", "sat-sun", "fri-sun", "mon,wed", "mon,sat"] {
            assert_eq!(printed(days), format!("{days} 07:00-19:00"));
        }
        // every day named is no day list at all.
        assert_eq!(printed("sun-sat"), "07:00-19:00");
        // written as a list, read back as the range it names.
        assert_eq!(printed("mon,tue,wed"), "mon-wed 07:00-19:00");
    }

    #[test]
    fn the_clock_says_whether_it_is_open() {
        let epoch = UNIX_EPOCH + std::time::Duration::from_secs(0);
        // 1970-01-01 00:00 UTC is a local time of its own, whatever this test
        // machine thinks local is; all that is fixed here is that Always and
        // a whole-day window agree with it and an empty hour does not claim
        // every minute there has ever been.
        assert!(Hours::Always.open_at(epoch));
        assert!(Hours::parse("00:00-00:00").unwrap().open_at(epoch));
        // one minute of every day open, including this one.
        let (day, minute) = local_clock(epoch);
        let narrow = Hours::parse(&format!("{}-{}", clock(minute), clock(minute + 1))).unwrap();
        assert!(narrow.open_at(epoch));
        assert!(!narrow.covers_at((day + 1) % 7, minute + 1));
    }
}
