//! telling parked vehicles from passing ones, by how long they stay put.
//!
//! the harvest wants vehicles in motion. the street is full of vehicles that are
//! not, and two in particular dominated it: a van parked in the foreground whose
//! bounding box encloses a stretch of road, and a jeep that cars drive past.
//!
//! three earlier approaches failed, and the reasons are worth keeping:
//!
//! - **a hard-coded rectangle** for the van. it worked until the camera was
//!   panned, at which point it silently pointed at empty road. position is not a
//!   property of the thing.
//! - **intersection over union** between the detection and the motion region.
//!   a motion region is the bounding box of *changed* pixels, which for a moving
//!   car is often just its leading and trailing edges, so the region is a subset
//!   of the vehicle rather than a match for it. every real vehicle measured
//!   below 0.25.
//! - **changed pixels inside the detection box.** this does separate a moving
//!   car (0.11-0.17) from a stationary one (0.00-0.03), and it is still used.
//!   but a box large enough to contain moving road -- the van -- passes any
//!   threshold that also admits real cars.
//!
//! what actually distinguishes them is time. a vehicle seen repeatedly in the
//! same place over minutes is scenery, whatever it is and wherever it sits. that
//! names no object, covers the jeep as well as the van, and re-learns by itself
//! when the camera moves.
//!
//! a fourth approach failed inside *this* one, and is the subtlest of the lot.
//! a track that drifts to its newest sighting -- so a vehicle shuffling slightly
//! stays one track -- works while the detector looks a few times a second, and
//! fails once it looks at every frame: consecutive sightings of a moving car are
//! always within `same_place` of each other, so the track walks across the scene
//! with the car and eventually calls it parked. it was found by replaying a clip
//! with every frame examined, where the harvest fell from 58 crops to 10, and it
//! would have shown up on a faster deployment host as vehicles quietly going
//! missing. the anchor is therefore fixed at first sighting and never moves.

use crate::gate::Rect;
use std::time::{Duration, Instant};

/// is this sighting in the same place as that anchor?
///
/// not `crop::same_object`, which scales its reach per axis by that axis's
/// extent. a vehicle against the frame edge is clipped by a different amount
/// each look -- one parked car measured 17px to 70px wide -- so a width-derived
/// reach gave the narrowest box the least tolerance, and it started a fresh
/// track every time. the reach here is isotropic, from the largest dimension of
/// either box. the anchor is still fixed, which is what stops a track following
/// a moving car.
/// has this place been both unobserved for long enough *and* looked past often
/// enough to count as empty?
///
/// free rather than a method so the prune can call it while holding the place
/// list. as a method it borrowed all of `self`, which is why the prune used to
/// allocate a `Vec<bool>` of answers first and then apply them.
fn forgotten(s: &Standing, now: Instant, looks: u64, after_secs: u64, after_looks: u64) -> bool {
    now.duration_since(s.last) >= Duration::from_secs(after_secs)
        && looks.saturating_sub(s.last_look) >= after_looks
}

fn same_place(anchor: Rect, seen: Rect, tolerance: f32) -> bool {
    let centre = |r: Rect| (r.x as f32 + r.w as f32 / 2.0, r.y as f32 + r.h as f32 / 2.0);
    let (ax, ay) = centre(anchor);
    let (sx, sy) = centre(seen);
    let scale = anchor.w.max(anchor.h).max(seen.w).max(seen.h) as f32;
    let reach = scale * tolerance;
    (ax - sx).abs() <= reach && (ay - sy).abs() <= reach
}

struct Standing {
    /// where it was **first** seen, and never updated afterwards.
    ///
    /// this used to drift to the newest sighting, so that a vehicle shuffling
    /// slightly stayed one track. that is fine when the detector looks a few
    /// times a second and wrong when it looks at every frame: each sighting of a
    /// moving car is within `same_place` of the one before it, so the track
    /// walked across the scene with the car and eventually called it parked. an
    /// anchor cannot follow anything; a vehicle either keeps coming back to
    /// where it started or it is not standing there.
    anchor: Rect,
    first: Instant,
    last: Instant,
    /// the look this place was first seen in, and how many looks since have
    /// found something here. the ratio of the two is what separates a parked
    /// vehicle from a patch of road that traffic keeps crossing.
    first_look: u64,
    last_look: u64,
    looks_seen: u64,
}

pub struct Scenery {
    cfg: crate::config::SceneryCfg,
    standing: Vec<Standing>,
    /// the look the place list was last pruned in, so it happens once per look
    /// rather than once per detection.
    pruned_at_look: u64,
    /// how many times the detector has been asked about the scene. this is the
    /// clock occupancy is measured against, rather than wall time, because what
    /// matters is "present in how many of the looks taken", and the rate of
    /// looking varies with the machine and the traffic.
    looks: u64,
}

impl Scenery {
    pub fn new(cfg: &crate::config::SceneryCfg) -> Self {
        Self {
            cfg: cfg.clone(),
            standing: Vec::new(),
            pruned_at_look: u64::MAX,
            looks: 0,
        }
    }

    fn parked_after(&self) -> Duration {
        Duration::from_secs(self.cfg.parked_after_secs)
    }

    /// begin a new look at the scene. call once per inspection, before the
    /// detections from it are observed.
    ///
    pub fn tick(&mut self) {
        self.looks += 1;
    }

    /// how much of this place's known life something has been standing in it.
    ///
    /// **the discriminator.** a parked car is found in nearly every look, so its
    /// ratio approaches one. a stretch of road is found only while traffic is
    /// crossing it -- perhaps one look a minute -- so its ratio is near zero,
    /// however long the place has been known.
    ///
    /// the previous rule asked only how long ago a place was first seen, which
    /// every busy place satisfies eventually. the whole road became scenery and
    /// the harvest kept 12% of real vehicles.
    fn occupancy(&self, s: &Standing) -> f32 {
        let span = self.looks.saturating_sub(s.first_look).max(1);
        s.looks_seen as f32 / span as f32
    }

    fn is_settled(&self, s: &Standing, now: Instant) -> bool {
        s.looks_seen >= self.cfg.min_sightings as u64
            && now.duration_since(s.first) >= self.parked_after()
            && self.occupancy(s) >= self.cfg.min_occupancy
    }

    /// record a sighting and say whether this place counts as scenery.
    pub fn observe(&mut self, rect: Rect, now: Instant) -> bool {
        let looks = self.looks;
        // **once per look, not once per detection.** whether a place has been
        // forgotten can only change when a look is taken, and `observe` runs
        // once per detection -- eight or so times per look on this street. it
        // used to prune on every one of them, rebuilding the place list and
        // allocating a `Vec<bool>` the length of it each time, to reach the same
        // answer it reached a microsecond earlier.
        if self.pruned_at_look != looks {
            self.pruned_at_look = looks;
            let (secs, missed) = (self.cfg.forget_after_secs, self.cfg.forget_after_looks);
            self.standing
                .retain(|s| !forgotten(s, now, looks, secs, missed));
        }
        if let Some(i) = self
            .standing
            .iter()
            .position(|s| same_place(s.anchor, rect, self.cfg.same_place))
        {
            {
                let s = &mut self.standing[i];
                s.last = now;
                // once per look, however many vehicles are reported in it:
                // occupancy asks in how many *looks* something was here, not how
                // many detections happened to overlap.
                if s.last_look != looks {
                    s.last_look = looks;
                    s.looks_seen += 1;
                }
            }
            return self.is_settled(&self.standing[i], now);
        }

        self.standing.push(Standing {
            anchor: rect,
            first: now,
            last: now,
            first_look: looks,
            last_look: looks,
            looks_seen: 1,
        });
        false
    }

    /// read-only view of the same decision, for diagnostics. does not record a
    /// sighting, so logging cannot change what the harvest decides.
    pub fn is_scenery(&self, rect: Rect, now: Instant) -> bool {
        self.standing
            .iter()
            .find(|s| same_place(s.anchor, rect, self.cfg.same_place))
            .is_some_and(|s| self.is_settled(s, now))
    }

    /// how many places are being remembered.
    ///
    /// worth reporting rather than inferring: every sighting is compared against
    /// all of them, so this is a per-detection cost, and it grows with traffic.
    pub fn tracked(&self) -> usize {
        self.standing.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FORGET_AFTER_LOOKS, FORGET_AFTER_SECS, PARKED_AFTER_SECS};

    const PARKED_AFTER: Duration = Duration::from_secs(PARKED_AFTER_SECS);
    const FORGET_AFTER: Duration = Duration::from_secs(FORGET_AFTER_SECS);

    fn at(x: u32, y: u32) -> Rect {
        Rect {
            x,
            y,
            w: 120,
            h: 80,
        }
    }

    /// the van and the jeep: seen again and again in the same spot.
    #[test]
    fn a_vehicle_that_stays_put_becomes_scenery() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let spot = at(400, 200);

        // one look per second, with the car present in every one of them:
        // occupancy is what marks it as standing, so the looks have to happen.
        s.tick();
        assert!(!s.observe(spot, t0), "not scenery on first sight");
        for i in 1..30 {
            s.tick();
            assert!(
                !s.observe(spot, t0 + Duration::from_secs(i)),
                "too soon at {i}s"
            );
        }
        for i in 30..=91 {
            s.tick();
            s.observe(spot, t0 + Duration::from_secs(i));
        }
        s.tick();
        assert!(
            s.observe(spot, t0 + PARKED_AFTER + Duration::from_secs(2)),
            "should be scenery by now"
        );
    }

    /// a car driving through is seen in a different place every time, so it
    /// never accumulates a history anywhere.
    #[test]
    fn a_vehicle_that_moves_never_becomes_scenery() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        for (i, x) in (0..600).step_by(60).enumerate() {
            let when = t0 + Duration::from_secs(i as u64 * 20);
            s.tick();
            assert!(
                !s.observe(at(x, 200), when),
                "a moving car was called scenery"
            );
        }
    }

    /// the same car, looked at often instead of rarely.
    ///
    /// found by replaying a clip with every frame examined: the harvest fell
    /// from 58 crops to 10. each sighting of a moving car is within `same_place`
    /// of the one before it, so a track that drifts to the newest position walks
    /// across the frame with the car, accumulates `parked_after_secs`, and calls a
    /// moving vehicle scenery -- which drops it from the harvest.
    ///
    /// this matters more the faster the machine is: dense sampling is the
    /// deployment host's normal condition, not an artificial one.
    #[test]
    fn a_car_crossing_the_frame_is_not_scenery_however_often_it_is_seen() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        // 15fps for two minutes, moving 3px a frame: 45px a second, a slow car.
        for i in 0..1800u64 {
            s.tick();
            let when = t0 + Duration::from_millis(i * 66);
            let x = 10 + (i as u32 * 3);
            assert!(
                !s.observe(at(x, 200), when),
                "a moving car was called scenery at x={x} after {}s",
                i * 66 / 1000
            );
        }
    }

    /// **the failure that made the harvest useless.**
    ///
    /// a stretch of road is not scenery just because vehicles keep appearing on
    /// it. but a track is created the first time anything is seen somewhere, and
    /// every later vehicle passing the same stretch matches it by proximity --
    /// refreshing it, so it never expires, and inheriting its verdict. after
    /// `parked_after_secs` the whole road is condemned.
    ///
    /// measured on a ten minute clip: the first three transits were judged
    /// moving and harvested, and every one of the twenty-two after the ninety
    /// second mark was called parked and discarded. the harvest kept 12%.
    ///
    /// the discriminator is not *how long* a place has been known, but how
    /// *continuously* it has been occupied. a parked car is there in nearly
    /// every look; a patch of road is there only when traffic is crossing it.
    #[test]
    fn a_busy_stretch_of_road_is_not_scenery() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let road = at(400, 200);

        // ten minutes of looks, with a vehicle crossing this spot in only one
        // look out of every ten -- busy road, nothing parked.
        let mut verdicts = Vec::new();
        for i in 0..600 {
            s.tick();
            if i % 10 == 0 {
                verdicts.push(s.observe(road, t0 + Duration::from_secs(i)));
            }
        }
        assert!(
            !verdicts.iter().any(|&v| v),
            "traffic crossing a spot made the spot -- and everything on it -- scenery"
        );
    }

    /// the counterweight: something genuinely standing there must still be
    /// recognised, or the van in the foreground floods the harvest again.
    #[test]
    fn a_car_that_actually_stays_is_still_scenery() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let spot = at(400, 200);

        let mut last = false;
        for i in 0..600 {
            s.tick();
            // present in every single look, which is what parked looks like.
            last = s.observe(spot, t0 + Duration::from_millis(i * 250));
        }
        assert!(last, "a car parked for two minutes was not called scenery");
    }

    /// real boxes from one clip: the same parked car at the right frame edge,
    /// clipped differently each look. widths 17 to 70, centres barely moving.
    /// a width-derived reach gives the 17px box 4px of tolerance, so every look
    /// started a fresh track and the car was cropped forever -- 20 of the 283
    /// crops on that clip, and all of its remaining precision loss.
    #[test]
    fn a_vehicle_clipped_by_the_frame_edge_is_still_one_place() {
        let looks = [
            Rect {
                x: 602,
                y: 0,
                w: 37,
                h: 96,
            },
            Rect {
                x: 622,
                y: 0,
                w: 17,
                h: 83,
            },
            Rect {
                x: 603,
                y: 2,
                w: 36,
                h: 63,
            },
            Rect {
                x: 569,
                y: 0,
                w: 70,
                h: 120,
            },
            Rect {
                x: 613,
                y: 0,
                w: 26,
                h: 91,
            },
        ];
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let mut last = false;
        for i in 0..600u64 {
            s.tick();
            last = s.observe(
                looks[i as usize % looks.len()],
                t0 + Duration::from_millis(i * 250),
            );
        }
        assert_eq!(
            s.tracked(),
            1,
            "one parked car became {} places",
            s.tracked()
        );
        assert!(last, "a car parked against the frame edge never settled");
    }

    /// the counterweight: tolerating a clipped box must not let a track follow
    /// a car across the scene.
    #[test]
    fn tolerating_a_clipped_box_does_not_let_a_track_follow_a_car() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        for i in 0..1800u64 {
            s.tick();
            let x = 10 + (i as u32 * 3);
            assert!(
                !s.observe(at(x, 200), t0 + Duration::from_millis(i * 66)),
                "a moving car was called scenery at x={x}"
            );
        }
    }

    /// **the overnight failure.**
    ///
    /// the gate only looks when something moves, and at night nothing does for
    /// minutes at a time. forgetting on wall time reads "nobody looked" as "it
    /// has gone": measured on one night, 16% of the gaps between crops exceeded
    /// `forget_after_secs` against 0% by day, with a longest gap of 35 minutes. so
    /// the jeep was dropped and rediscovered on every burst of traffic, and
    /// cropped again each time, all night.
    #[test]
    fn a_parked_vehicle_survives_a_quiet_night() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let spot = at(400, 200);
        let mut settled = false;
        // eight bursts six minutes apart: a car passes, the gate fires for a
        // couple of seconds, then the street is empty again.
        for burst in 0..8u64 {
            let base = t0 + Duration::from_secs(burst * 360);
            for i in 0..20u64 {
                s.tick();
                settled = s.observe(spot, base + Duration::from_millis(i * 100));
            }
        }
        assert!(settled, "a car parked all night was never called parked");
        assert_eq!(s.tracked(), 1, "the same car became {} places", s.tracked());
    }

    /// a single stale sighting must not condemn a place.
    #[test]
    fn one_old_sighting_is_not_enough() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        s.tick();
        s.observe(at(400, 200), t0);
        // long enough, but only the second sighting ever.
        s.tick();
        assert!(!s.observe(at(400, 200), t0 + PARKED_AFTER + Duration::from_secs(1)));
    }

    /// when the parked car leaves, its place stops being scenery -- or a go-4
    /// pulling into the space it vacated would be vetoed on arrival.
    ///
    /// the looks in the middle are the point. "not seen for four minutes" is
    /// only evidence the car has gone if something was looking; without them
    /// this passed while describing a street nobody watched.
    #[test]
    fn scenery_is_forgotten_once_it_goes_away() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let spot = at(400, 200);
        for i in 0..=95 {
            s.tick();
            s.observe(spot, t0 + Duration::from_secs(i));
        }
        s.tick();
        assert!(s.observe(spot, t0 + PARKED_AFTER + Duration::from_secs(6)));

        // it drives off, and traffic keeps crossing elsewhere.
        let gone = t0 + PARKED_AFTER + Duration::from_secs(10);
        for i in 0..FORGET_AFTER_LOOKS + 10 {
            s.tick();
            s.observe(at(50, 400), gone + Duration::from_millis(i * 250));
        }
        let later = gone + FORGET_AFTER + Duration::from_secs(10);
        s.tick();
        assert!(
            !s.observe(spot, later),
            "a new arrival inherited the old history"
        );
    }

    /// a vehicle stopping at the kerb -- which is what enforcement does -- must
    /// stay interesting long enough to be harvested and classified.
    #[test]
    fn a_vehicle_that_just_stopped_is_still_interesting() {
        let mut s = Scenery::new(&crate::config::SceneryCfg::default());
        let t0 = Instant::now();
        let kerb = at(300, 250);
        for i in 0..6 {
            s.tick();
            let when = t0 + Duration::from_secs(i * 10);
            assert!(!s.observe(kerb, when), "condemned after only {}s", i * 10);
        }
    }
}
