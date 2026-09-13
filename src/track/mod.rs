//! following one vehicle across frames, so "what happened" has a subject.
//!
//! everything phase 3 publishes is about a thing over time -- it arrived, it
//! stopped, someone got out -- and none of that is answerable from a single
//! frame. the tracker supplies the identity those statements are about.
//!
//! **association is iou here, and that is not a contradiction.** DESIGN records
//! iou being rejected for matching a *motion region* to a detection, where it
//! measured below 0.25 for every real vehicle because a motion region is the
//! bounding box of changed pixels rather than of the object. detection against
//! detection, one frame apart, is the case iou is actually good at: two boxes
//! around the same car, barely moved.
//!
//! this deliberately knows nothing about subjects. it tracks whatever the
//! detector reports, and a verdict is attached per look, so adding a subject
//! needs no change here (r10.1, r10.3).

use crate::config::TrackCfg;
use crate::gate::Rect;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub rect: Rect,
    /// when this vehicle was first seen, which is what an arrival is dated by.
    pub first: Instant,
    pub last: Instant,
    /// looks this track was found in, and consecutive looks it was missed in.
    pub seen: u32,
    missed: u32,
    /// when it was last observed moving, which is what dwell counts from.
    moving_at: Instant,
    history: VecDeque<(Instant, Rect)>,
    /// which subject the classifier named in each of the last `confirm_n`
    /// looks, `None` for none of them. **the name, not a flag**: counting "was
    /// a subject" would confirm a track that alternated between two of them and
    /// then publish under whichever won the final look.
    verdicts: VecDeque<Option<String>>,
    /// the moving/stopped state last published for this track, so mqtt carries
    /// transitions rather than a message per frame. `None` = never reported.
    reported_stopped: Option<bool>,
}

fn centre(r: Rect) -> (f32, f32) {
    (r.x as f32 + r.w as f32 / 2.0, r.y as f32 + r.h as f32 / 2.0)
}

pub fn iou(a: Rect, b: Rect) -> f32 {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let inter = ((x1 - x0) * (y1 - y0)) as f32;
    let union = (a.w * a.h + b.w * b.h) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

impl Track {
    /// gate pixels per second, over the recent window.
    pub fn speed(&self) -> f32 {
        let (Some((t0, r0)), Some((t1, r1))) = (self.history.front(), self.history.back()) else {
            return 0.0;
        };
        let secs = t1.saturating_duration_since(*t0).as_secs_f32();
        if secs <= 0.0 {
            return 0.0;
        }
        let (x0, y0) = centre(*r0);
        let (x1, y1) = centre(*r1);
        ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt() / secs
    }

    /// how long it has been stationary. zero while it is still moving.
    pub fn dwell(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.moving_at)
    }

    /// has it been recognised as *this* subject in enough of the recent looks?
    ///
    /// one frame is not evidence (r1.4). a white van at the wrong angle reads as
    /// a go-4 for a frame or two; it does not do so for three of the last five.
    pub fn confirmed(&self, subject: &str, m: u32) -> bool {
        self.verdicts
            .iter()
            .filter(|v| v.as_deref() == Some(subject))
            .count() as u32
            >= m
    }

    /// the subject this track is confirmed as, if any. what the publish path
    /// needs: which name to report it under, rather than merely that something
    /// matched.
    pub fn confirmed_subject(&self, m: u32) -> Option<&str> {
        self.verdicts
            .iter()
            .flatten()
            .find(|name| self.confirmed(name, m))
            .map(|s| s.as_str())
    }
}

pub struct Tracker {
    cfg: TrackCfg,
    tracks: Vec<Track>,
    next_id: u64,
    /// tracks that expired on the last look, waiting to be reported once.
    gone: Vec<Track>,
}

impl Tracker {
    pub fn new(cfg: &TrackCfg) -> Self {
        Self {
            cfg: cfg.clone(),
            tracks: Vec::new(),
            next_id: 1,
            gone: Vec::new(),
        }
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// one look at the scene: associate `seen` with existing tracks, start
    /// tracks for what is new, and drop what has gone.
    ///
    /// returns the id assigned to each input box, in order, so a caller can
    /// attach its own per-detection result to the right track.
    pub fn update(&mut self, now: Instant, seen: &[Rect]) -> Vec<u64> {
        let mut ids = Vec::with_capacity(seen.len());
        let mut taken = vec![false; self.tracks.len()];

        for &rect in seen {
            // the best unclaimed track, so two cars side by side cannot both
            // match the same one.
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken[*i])
                .map(|(i, t)| (i, iou(t.rect, rect)))
                .filter(|(_, s)| *s >= self.cfg.min_iou)
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, _)) => {
                    taken[i] = true;
                    ids.push(self.extend(i, now, rect));
                }
                None => {
                    // `taken` is indexed by track, and starting one appends to
                    // `self.tracks`, so it has to grow with it -- and a track
                    // created this look has by definition been seen in it.
                    taken.push(true);
                    ids.push(self.start(now, rect));
                }
            }
        }

        for (i, t) in self.tracks.iter_mut().enumerate() {
            if !taken[i] {
                t.missed += 1;
            }
        }
        // expired on elapsed time *and* consecutive misses, for the reason
        // `scenery` forgets that way: the detector is only asked when something
        // moved, so a stretch with no looks is not evidence anything left.
        let max_gap = Duration::from_secs_f32(self.cfg.max_gap_secs);
        let max_missed = self.cfg.max_missed_looks;
        let (live, gone): (Vec<Track>, Vec<Track>) = self.tracks.drain(..).partition(|t| {
            now.saturating_duration_since(t.last) < max_gap || t.missed < max_missed
        });
        self.tracks = live;
        // only ones that were ever reported: a track nobody published an
        // arrival for needs no departure.
        self.gone
            .extend(gone.into_iter().filter(|t| t.reported_stopped.is_some()));
        ids
    }

    /// tracks that have ended since this was last called, reported once each.
    pub fn departed(&mut self) -> Vec<Track> {
        std::mem::take(&mut self.gone)
    }

    /// has this track's moving/stopped state changed since it was last
    /// reported? records the new state, so a caller that publishes on `true`
    /// emits transitions rather than a message every frame.
    pub fn transition(&mut self, id: u64, stopped: bool) -> bool {
        match self.tracks.iter_mut().find(|t| t.id == id) {
            Some(t) if t.reported_stopped != Some(stopped) => {
                t.reported_stopped = Some(stopped);
                true
            }
            _ => false,
        }
    }

    fn start(&mut self, now: Instant, rect: Rect) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.tracks.push(Track {
            id,
            rect,
            first: now,
            last: now,
            seen: 1,
            missed: 0,
            moving_at: now,
            history: VecDeque::from([(now, rect)]),
            verdicts: VecDeque::new(),
            reported_stopped: None,
        });
        id
    }

    fn extend(&mut self, i: usize, now: Instant, rect: Rect) -> u64 {
        let stopped_below = self.cfg.stopped_below_px_per_sec;
        let window = self.cfg.velocity_window;
        let t = &mut self.tracks[i];
        t.rect = rect;
        t.last = now;
        t.seen += 1;
        t.missed = 0;
        t.history.push_back((now, rect));
        while t.history.len() > window {
            t.history.pop_front();
        }
        if t.speed() >= stopped_below {
            t.moving_at = now;
        }
        t.id
    }

    /// record which subject the classifier named for a track this look.
    ///
    /// called with every judgement, including `None`: feeding it only the
    /// matches would make the ratio a count and confirm on the first frame that
    /// guessed right.
    pub fn saw_subject(&mut self, id: u64, subject: Option<&str>) {
        let n = self.cfg.confirm_n as usize;
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.verdicts.push_back(subject.map(str::to_string));
            while t.verdicts.len() > n {
                t.verdicts.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TrackCfg {
        TrackCfg::default()
    }

    fn at(x: u32, y: u32) -> Rect {
        Rect {
            x,
            y,
            w: 100,
            h: 80,
        }
    }

    #[test]
    fn a_car_keeps_one_id_as_it_crosses_the_frame() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        let first = tr.update(t0, &[at(10, 200)])[0];
        for i in 1..40u64 {
            let ids = tr.update(
                t0 + Duration::from_millis(i * 66),
                &[at(10 + i as u32 * 4, 200)],
            );
            assert_eq!(ids, vec![first], "lost the car at step {i}");
        }
        assert_eq!(tr.tracks().len(), 1);
    }

    #[test]
    fn two_cars_side_by_side_keep_separate_ids() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        let ids = tr.update(t0, &[at(10, 100), at(400, 100)]);
        assert_ne!(ids[0], ids[1]);
        let later = tr.update(t0 + Duration::from_millis(66), &[at(14, 100), at(404, 100)]);
        assert_eq!(later, ids, "the two cars swapped identity");
    }

    #[test]
    fn a_moving_car_has_speed_and_no_dwell() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        for i in 0..10u64 {
            tr.update(
                t0 + Duration::from_millis(i * 100),
                &[at(10 + i as u32 * 20, 200)],
            );
        }
        let t = &tr.tracks()[0];
        assert!(t.speed() > 100.0, "speed was {}", t.speed());
        assert!(t.dwell(t0 + Duration::from_secs(1)) < Duration::from_millis(200));
    }

    /// what `dwell` exists for: a vehicle that stops at the kerb, which is what
    /// enforcement does before a ticket is written.
    #[test]
    fn a_car_that_stops_accumulates_dwell() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        for i in 0..10u64 {
            tr.update(
                t0 + Duration::from_millis(i * 100),
                &[at(10 + i as u32 * 20, 200)],
            );
        }
        let stopped = at(190, 200);
        for i in 0..40u64 {
            tr.update(
                t0 + Duration::from_secs(1) + Duration::from_millis(i * 100),
                &[stopped],
            );
        }
        let now = t0 + Duration::from_secs(1) + Duration::from_millis(39 * 100);
        assert!(
            tr.tracks()[0].dwell(now) >= Duration::from_secs(3),
            "dwell was {:?}",
            tr.tracks()[0].dwell(now)
        );
    }

    #[test]
    fn a_track_is_dropped_once_it_is_gone() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        tr.update(t0, &[at(10, 200)]);
        assert_eq!(tr.tracks().len(), 1);
        // past both the gap and the miss count, since either alone keeps it.
        for i in 1..=(cfg().max_missed_looks + 1) {
            let now =
                t0 + Duration::from_secs_f32(cfg().max_gap_secs) + Duration::from_millis(i as u64);
            tr.update(now, &[]);
        }
        assert!(tr.tracks().is_empty(), "{:?}", tr.tracks());
    }

    /// a quiet street produces no looks, and no looks is not evidence the car
    /// left. the same mistake wall-clock forgetting made in `scenery`.
    #[test]
    fn a_long_quiet_stretch_alone_does_not_drop_a_track() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        tr.update(t0, &[at(10, 200)]);
        // hours later, one look, and the car is still there.
        let ids = tr.update(t0 + Duration::from_secs(3600), &[at(10, 200)]);
        assert_eq!(
            tr.tracks().len(),
            1,
            "the track was dropped over a quiet gap"
        );
        assert_eq!(ids[0], 1, "and it kept its identity");
    }

    /// **confirmation is per subject, not "was it any of them".**
    ///
    /// a track that alternates between two subjects has not been recognised as
    /// either; it has been guessed at. counting "was a subject" would confirm
    /// on five looks that never agreed, and then publish under whichever name
    /// happened to win the last one.
    #[test]
    fn alternating_subjects_confirm_neither() {
        let mut tr = Tracker::new(&cfg());
        let id = tr.update(Instant::now(), &[at(10, 200)])[0];
        // four of five looks named *a* subject -- which is what counting
        // `is_some()` saw, and it cleared three. no single name gets past two.
        for name in [
            Some("go4"),
            Some("sweeper"),
            Some("go4"),
            Some("sweeper"),
            None,
        ] {
            tr.saw_subject(id, name);
        }
        let t = &tr.tracks()[0];
        assert!(
            !t.confirmed("go4", cfg().confirm_m),
            "go4 confirmed on two looks"
        );
        assert!(!t.confirmed("sweeper", cfg().confirm_m));
        assert_eq!(t.confirmed_subject(cfg().confirm_m), None);
    }

    /// and one that is consistently the same subject confirms, by name.
    #[test]
    fn a_consistent_subject_confirms_under_its_own_name() {
        let mut tr = Tracker::new(&cfg());
        let id = tr.update(Instant::now(), &[at(10, 200)])[0];
        for _ in 0..cfg().confirm_m {
            tr.saw_subject(id, Some("sweeper"));
        }
        let t = &tr.tracks()[0];
        assert!(t.confirmed("sweeper", cfg().confirm_m));
        assert!(
            !t.confirmed("go4", cfg().confirm_m),
            "confirmed the wrong subject"
        );
        assert_eq!(t.confirmed_subject(cfg().confirm_m), Some("sweeper"));
    }

    /// one frame is not evidence (r1.4).
    #[test]
    fn a_subject_is_confirmed_only_after_enough_looks() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        let id = tr.update(t0, &[at(10, 200)])[0];
        tr.saw_subject(id, Some("go4"));
        assert!(
            !tr.tracks()[0].confirmed("go4", cfg().confirm_m),
            "one look was enough"
        );
        for _ in 0..cfg().confirm_m {
            tr.saw_subject(id, Some("go4"));
        }
        assert!(tr.tracks()[0].confirmed("go4", cfg().confirm_m));
    }

    /// and a flicker does not survive the window sliding past it.
    #[test]
    fn a_single_misread_does_not_confirm() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        let id = tr.update(t0, &[at(10, 200)])[0];
        tr.saw_subject(id, Some("go4"));
        for _ in 0..cfg().confirm_n {
            tr.saw_subject(id, None);
        }
        assert!(!tr.tracks()[0].confirmed("go4", cfg().confirm_m));
    }

    /// mqtt carries transitions, not a message a frame. at fifteen frames a
    /// second the second shape would be fifteen publishes per vehicle per
    /// second, all saying the same thing.
    #[test]
    fn only_a_change_of_state_is_worth_reporting() {
        let mut tr = Tracker::new(&cfg());
        let id = tr.update(Instant::now(), &[at(10, 200)])[0];
        assert!(tr.transition(id, false), "the first report is a change");
        assert!(!tr.transition(id, false), "same state reported twice");
        assert!(tr.transition(id, true), "moving -> stopped");
        assert!(!tr.transition(id, true));
        assert!(tr.transition(id, false), "stopped -> moving again");
    }

    /// a departure is only interesting for a vehicle something was said about.
    #[test]
    fn only_a_reported_track_departs() {
        let mut tr = Tracker::new(&cfg());
        let t0 = Instant::now();
        let ids = tr.update(t0, &[at(10, 200), at(400, 200)]);
        let (quiet, loud) = (ids[0], ids[1]);
        // only one of them was ever published about.
        tr.transition(loud, false);
        assert_ne!(quiet, loud);

        for i in 1..=(cfg().max_missed_looks + 1) {
            tr.update(
                t0 + Duration::from_secs_f32(cfg().max_gap_secs) + Duration::from_millis(i as u64),
                &[],
            );
        }
        let gone = tr.departed();
        assert_eq!(gone.len(), 1, "{gone:?}");
        assert_eq!(gone[0].id, loud);
        assert!(tr.departed().is_empty(), "a departure was reported twice");
    }

    #[test]
    fn iou_is_zero_for_boxes_that_do_not_touch() {
        assert_eq!(iou(at(0, 0), at(500, 500)), 0.0);
        assert!((iou(at(0, 0), at(0, 0)) - 1.0).abs() < 1e-6);
        assert!(
            iou(at(0, 0), at(50, 0)) > 0.3,
            "half-overlap should associate"
        );
    }
}
