//! event clips: record the moments, not the hours.
//!
//! the recorded eval clips are fixed windows started by hand, and the thing they
//! need to contain arrives about twice an hour. measured against the labels, **0
//! of 24 labelled waymo crops fall inside any recorded clip** -- the nearest
//! miss is a clip that starts 85 seconds after a passage. so no change to crop
//! framing or to the detector can be tested against a known positive, only
//! against counts of things nobody has identified.
//!
//! this records when something happens instead. it costs no inference and no
//! second camera session: the main stream is already remuxed to fragmented mp4
//! as a second output of the ffmpeg feeding the crops, `-c copy`, and
//! `preview::fmp4::Split` already cuts that into an `init` header and fragments.
//! a ring of recent fragments written out behind that header is a playable mp4.
//!
//! two things here are load-bearing and neither is an optimisation.
//!
//! **the pre-roll is the feature.** motion fires *after* the subject is already
//! in frame, so a file opened at the trigger starts on a vehicle halfway across.
//! recall at first appearance cannot be measured from a clip like that, and that
//! is one of the questions the corpus exists to answer.
//!
//! **what a clip is worth is decided after the pipeline has judged it**, not
//! when the trigger fires. at the moment motion starts, the detector and the
//! classifier have not run; by the time they have, the ring still holds the
//! beginning. that is the real argument for buffering rather than streaming
//! straight to a file, and it is what makes the retention policy possible: a
//! busy ten minutes holds 27 motion events, so keeping all of them is about
//! 1.6 GB an hour, while keeping the ones a subject was recognised in is about
//! 15 MB an hour.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub mod select;

/// how much of the stream to hold before a trigger, and how long a clip may run.
///
/// three seconds. five was the original guess at how long a vehicle takes to
/// cross from the frame edge and read as padding; one was tried and was too
/// short -- clips opened with the vehicle already fully in shot, confirmed by
/// decoding the deployment's clips frame by frame, and a clip that opens on a
/// vehicle halfway across cannot say when it first appeared, which is the
/// question the pre-roll exists to answer.
///
/// it is also the margin that absorbs skew between the gate's clock and the
/// main-stream fragments, so it is not only about how fast the gate fires.
///
/// the ceiling stops a stretch of continuous traffic becoming one enormous
/// file, which would be neither an event nor reviewable.
pub const PREROLL: Duration = Duration::from_secs(3);
pub const MAX_CLIP: Duration = Duration::from_secs(60);

/// the stream name the remuxed main feed is recorded under. one stream today:
/// the substream the gate reads is decoded to rgb rather than remuxed, so there
/// is no byte copy of it to buffer, and re-encoding it would cost more than the
/// rest of the loop put together. a replay that wants the pair needs a `-c copy`
/// tee on the gate ingest as well, which is the natural next step.
pub const MAIN_STREAM: &str = "main";
/// the gate's substream, recorded alongside it.
///
/// **an end-to-end replay needs both.** the substream drives the gate and the
/// motion scan that judges precision; the main stream supplies the crops. and
/// the substream cannot be derived by downscaling main -- the gate sees the
/// camera's own encode, anamorphic and at its own frame rate, and metermate
/// deliberately never rescales on that path. a clip of main alone can be
/// watched but not replayed, which is most of what `[record]` is for.
pub const SUB_STREAM: &str = "sub";

/// how long after the last trigger a clip stays open, so one vehicle is one
/// clip rather than a burst of them.
///
/// **this is the second latch, not the first.** the trigger is the gate's
/// *latched* motion, which already holds for `[gate] latch_ms` past the last
/// frame that qualified -- so the quiet tail on a clip is that plus this, and
/// four seconds here meant six before a departing vehicle stopped extending it.
/// measured on the deployment: tails of six to ten seconds, between 46% and 74%
/// of each clip's length. one second here leaves the gate doing the job it is
/// already doing and keeps the tail near the latch.
pub const HANGOVER: Duration = Duration::from_secs(1);

/// what was found in a clip, worst first. retention evicts in this order, so the
/// disk fills with the clips worth replaying rather than with the newest ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Worth {
    /// something moved in the roi and nothing more was ever established.
    Motion,
    /// stage one found something it was configured to care about.
    Detection,
    /// stage two recognised a subject. the reason the feature exists.
    Subject,
}

impl Worth {
    pub fn slug(self) -> &'static str {
        match self {
            Worth::Motion => "motion",
            Worth::Detection => "detection",
            Worth::Subject => "subject",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "motion" => Some(Worth::Motion),
            "detection" => Some(Worth::Detection),
            "subject" => Some(Worth::Subject),
            _ => None,
        }
    }
}

struct Fragment {
    bytes: Vec<u8>,
    /// the fragment's own decode time, not when it was received.
    ///
    /// **the ring is bounded by the stream's clock rather than the wall's.**
    /// they are the same thing on a live camera and nothing like it in a
    /// replay, where the crop feed is pulled to the gate's position: five
    /// seconds of wall-clock there covered eighty-seven seconds of video, so a
    /// wall-bounded ring produced clips whose length depended on how fast the
    /// machine happened to be. the repo already learned this once, in `--offline`.
    at: u64,
}

/// ticks per second in the remuxed stream, read from the init segment's `mdhd`.
/// 90 kHz is the h.264 convention and what this camera's remux uses, but it is
/// read rather than assumed because a wrong one silently rescales the ring.
const DEFAULT_TIMESCALE: u64 = 90_000;

fn timescale_of(init: &[u8]) -> u64 {
    fn walk(buf: &[u8]) -> Option<u64> {
        let mut at = 0;
        while at + 8 <= buf.len() {
            let size =
                u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
            let kind = [buf[at + 4], buf[at + 5], buf[at + 6], buf[at + 7]];
            if size < 8 || at + size > buf.len() {
                return None;
            }
            match &kind {
                b"moov" | b"trak" | b"mdia" => {
                    if let Some(v) = walk(&buf[at + 8..at + size]) {
                        return Some(v);
                    }
                }
                b"mdhd" => {
                    // version 1 widens the two times before it to 64 bits.
                    let off = if buf[at + 8] == 1 { at + 28 } else { at + 20 };
                    if off + 4 <= at + size {
                        return Some(u32::from_be_bytes(buf[off..off + 4].try_into().ok()?) as u64);
                    }
                }
                _ => {}
            }
            at += size;
        }
        None
    }
    walk(init).filter(|t| *t > 0).unwrap_or(DEFAULT_TIMESCALE)
}

/// shift a fragment's decode time so a clip can begin at zero.
///
/// **without this every clip is unplayable in the way that matters.** a
/// fragment carries an *absolute* `baseMediaDecodeTime` in its `tfdt` box,
/// counted from the start of the stream ffmpeg has been remuxing since startup.
/// write a handful of those behind an init header and the file claims a
/// timeline running from the origin: measured on a real replay, clips a few
/// seconds long reported durations of 49s, 87s and 180s, rising with uptime.
/// a player shows minutes of nothing, and anything computing a frame number
/// from a timestamp -- which every eval here does -- is wrong by however long
/// metermate had been running.
///
/// so the first fragment's time is subtracted from all of them. this walks
/// `moof > traf > tfdt` and rewrites the field in place, touching nothing else:
/// sizes do not change, because the value is rewritten in its existing width.
fn rebase(fragment: &mut [u8], origin: u64) {
    fn walk(buf: &mut [u8], origin: u64) {
        let mut at = 0;
        while at + 8 <= buf.len() {
            let size =
                u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
            let kind = [buf[at + 4], buf[at + 5], buf[at + 6], buf[at + 7]];
            if size < 8 || at + size > buf.len() {
                return;
            }
            match &kind {
                // containers: descend.
                b"moof" | b"traf" => walk(&mut buf[at + 8..at + size], origin),
                b"tfdt" => {
                    // a full box: one version byte then three of flags.
                    let body = at + 12;
                    if buf[at + 8] == 1 && body + 8 <= at + size {
                        let v = u64::from_be_bytes(buf[body..body + 8].try_into().unwrap());
                        buf[body..body + 8]
                            .copy_from_slice(&v.saturating_sub(origin).to_be_bytes());
                    } else if body + 4 <= at + size {
                        let v = u32::from_be_bytes(buf[body..body + 4].try_into().unwrap());
                        let shifted = (v as u64).saturating_sub(origin) as u32;
                        buf[body..body + 4].copy_from_slice(&shifted.to_be_bytes());
                    }
                }
                _ => {}
            }
            at += size;
        }
    }
    walk(fragment, origin);
}

/// is this fragment's first sample one a decoder can start from?
///
/// **a clip may only begin here.** every other frame is coded against pictures
/// before it, so a file that opens mid-gop opens with frames nothing can show:
/// measured on the deployment, an 8.73s clip whose first decodable frame was at
/// 1.33s, and which played as a held still over the first third of its pre-roll.
///
/// the flag is `sample_is_non_sync_sample`, bit 16 of the sample flags, which
/// arrive either as `trun`'s `first_sample_flags` or as `tfhd`'s
/// `default_sample_flags`. a fragment declaring neither is taken as a keyframe:
/// that is what an all-intra stream looks like -- the substream's mjpeg, where
/// every frame is a sync sample -- and being wrong in that direction costs one
/// decodable frame at the front rather than an undecodable clip.
fn is_keyframe(fragment: &[u8]) -> bool {
    const NON_SYNC: u32 = 0x0001_0000;

    /// `(trun first_sample_flags, tfhd default_sample_flags)`.
    ///
    /// **both, rather than whichever is met first.** this camera writes `tfhd`
    /// ahead of `trun` inside the `traf`, and its `tfhd` default says non-sync
    /// on every fragment of the stream -- keyframes included. only `trun`
    /// distinguishes them, and it carries `first_sample_flags` *only* on the
    /// keyframes. reading the first box that answered therefore called every
    /// frame a non-keyframe, which emptied the ring on every push and removed
    /// the pre-roll entirely.
    fn flags_in(buf: &[u8]) -> (Option<u32>, Option<u32>) {
        let (mut from_trun, mut from_tfhd) = (None, None);
        let mut at = 0;
        while at + 8 <= buf.len() {
            let size =
                u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
            let kind = [buf[at + 4], buf[at + 5], buf[at + 6], buf[at + 7]];
            if size < 8 || at + size > buf.len() {
                break;
            }
            match &kind {
                b"moof" | b"traf" => {
                    let (t, h) = flags_in(&buf[at + 8..at + size]);
                    from_trun = from_trun.or(t);
                    from_tfhd = from_tfhd.or(h);
                }
                b"trun" if at + 16 <= at + size => {
                    let tr = u32::from_be_bytes([0, buf[at + 9], buf[at + 10], buf[at + 11]]);
                    let mut off = at + 16; // past version+flags and sample_count
                    if tr & 0x01 != 0 {
                        off += 4; // data_offset
                    }
                    if tr & 0x04 != 0
                        && off + 4 <= at + size
                        && let Ok(b) = buf[off..off + 4].try_into()
                    {
                        from_trun = from_trun.or(Some(u32::from_be_bytes(b)));
                    }
                }
                b"tfhd" if at + 16 <= at + size => {
                    let tf = u32::from_be_bytes([0, buf[at + 9], buf[at + 10], buf[at + 11]]);
                    let mut off = at + 16; // past version+flags and track_ID
                    for (bit, width) in [(0x01, 8), (0x02, 4), (0x08, 4), (0x10, 4)] {
                        if tf & bit != 0 {
                            off += width;
                        }
                    }
                    if tf & 0x20 != 0
                        && off + 4 <= at + size
                        && let Ok(b) = buf[off..off + 4].try_into()
                    {
                        from_tfhd = from_tfhd.or(Some(u32::from_be_bytes(b)));
                    }
                }
                _ => {}
            }
            at += size;
        }
        (from_trun, from_tfhd)
    }

    let (from_trun, from_tfhd) = flags_in(fragment);
    from_trun.or(from_tfhd).is_none_or(|f| f & NON_SYNC == 0)
}

/// the decode time a fragment starts at, for choosing a clip's origin.
fn decode_time(fragment: &[u8]) -> Option<u64> {
    fn walk(buf: &[u8]) -> Option<u64> {
        let mut at = 0;
        while at + 8 <= buf.len() {
            let size =
                u32::from_be_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
            let kind = [buf[at + 4], buf[at + 5], buf[at + 6], buf[at + 7]];
            if size < 8 || at + size > buf.len() {
                return None;
            }
            match &kind {
                b"moof" | b"traf" => {
                    if let Some(v) = walk(&buf[at + 8..at + size]) {
                        return Some(v);
                    }
                }
                b"tfdt" => {
                    let body = at + 12;
                    return if buf[at + 8] == 1 && body + 8 <= at + size {
                        Some(u64::from_be_bytes(buf[body..body + 8].try_into().ok()?))
                    } else if body + 4 <= at + size {
                        Some(u32::from_be_bytes(buf[body..body + 4].try_into().ok()?) as u64)
                    } else {
                        None
                    };
                }
                _ => {}
            }
            at += size;
        }
        None
    }
    walk(fragment)
}

/// one stream's ring and, while a clip is open, the file it is being written to.
struct Stream {
    name: String,
    init: Vec<u8>,
    ring: VecDeque<Fragment>,
    open: Option<std::fs::File>,
    /// the decode time the open clip began at, subtracted from every fragment
    /// written so the file's timeline starts at zero.
    origin: u64,
    timescale: u64,
    /// the newest decode time seen, so a stream that restarts can be noticed.
    last_at: Option<u64>,
}

/// a clip being written across every stream at once.
struct Open {
    started: Instant,
    last_trigger: Instant,
    worth: Worth,
    stamp: u128,
    bytes: u64,
}

pub struct Recorder {
    dir: PathBuf,
    /// where derived files for these clips live, so eviction takes the poster
    /// with the clip rather than leaving a still of something unwatchable.
    cache_dir: PathBuf,
    streams: Vec<Stream>,
    open: Option<Open>,
    preroll: Duration,
    max_clip: Duration,
    hangover: Duration,
    budget_bytes: u64,
    /// outcomes worth a file, or empty for all of them. **this cannot be
    /// decided before recording**, only before keeping: at the trigger the
    /// detector has not run, so the choice is between buffering every event and
    /// missing the start of the ones that matter. that is the same argument the
    /// ring buffer rests on.
    keep: Vec<Worth>,
}

impl Recorder {
    pub fn new(cfg: &crate::config::RecordCfg, names: &[&str]) -> Result<Self> {
        let dir = cfg.dir.as_path();
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        adopt_partials(dir);
        Ok(Self {
            dir: dir.to_path_buf(),
            cache_dir: cfg.cache_dir.clone(),
            streams: names
                .iter()
                .map(|n| Stream {
                    name: n.to_string(),
                    init: Vec::new(),
                    ring: VecDeque::new(),
                    open: None,
                    origin: 0,
                    timescale: DEFAULT_TIMESCALE,
                    last_at: None,
                })
                .collect(),
            open: None,
            preroll: Duration::from_secs(cfg.preroll_secs),
            max_clip: Duration::from_secs(cfg.max_clip_secs),
            hangover: Duration::from_secs(cfg.hangover_secs),
            budget_bytes: cfg.max_bytes,
            keep: cfg.keep.iter().filter_map(|s| Worth::parse(s)).collect(),
        })
    }

    /// is a clip that turned out to hold `worth` worth keeping?
    fn wanted(&self, worth: Worth) -> bool {
        self.keep.is_empty() || self.keep.contains(&worth)
    }

    pub fn recording(&self) -> bool {
        self.open.is_some()
    }

    /// true until the stream's header has been seen.
    pub fn init_needed(&self) -> bool {
        self.streams.iter().any(|s| s.init.is_empty())
    }

    /// a clip written before the header arrives is undecodable, and ffmpeg emits
    /// it only once at the start of the stream -- so a trigger in the first
    /// second or two has nothing to build a file from and is better skipped than
    /// written unplayable.
    pub fn ready(&self) -> bool {
        !self.init_needed()
    }

    /// the `ftyp`+`moov` header for a stream. every clip of it begins with this;
    /// fragments without it are undecodable.
    pub fn set_init(&mut self, stream: &str, init: Vec<u8>) {
        if let Some(s) = self.streams.iter_mut().find(|s| s.name == stream) {
            s.timescale = timescale_of(&init);
            s.init = init;
        }
    }

    /// has this stream's clock gone backwards? only a restarted ffmpeg does
    /// that: with a fragment per frame the decode time is otherwise monotonic.
    fn restarted(&self, stream: &str, at: Option<u64>) -> bool {
        let (Some(at), Some(s)) = (at, self.streams.iter().find(|s| s.name == stream)) else {
            return false;
        };
        s.last_at.is_some_and(|last| at < last)
    }

    /// feed one fragment. it joins the ring, and is appended to an open clip.
    pub fn push(&mut self, stream: &str, bytes: &[u8], _now: Instant) -> Result<()> {
        let incoming = decode_time(bytes);
        // **the clip no player would show more than one frame of.** the ingest
        // supervisor restarts ffmpeg when the camera stalls, and the remux then
        // begins a fresh timeline at zero. rebasing those against the old
        // origin drives every one of them negative, `saturating_sub` floors
        // them at zero, and the file claims that all of its frames happen at
        // once: a clip on the deployed box held 103 fragments that way, 1.4 MB
        // of video that decoded to nothing.
        //
        // nothing spans that boundary. the open clip is finished with what it
        // has, the ring is dropped because it describes a stream that no longer
        // exists, and the header is dropped so the new one is asked for -- a
        // restarted ffmpeg may have renegotiated the stream, and fragments
        // behind a stale `moov` are undecodable whatever their timestamps say.
        if self.restarted(stream, incoming) {
            tracing::info!("the recorded stream restarted; finishing the clip that spanned it");
            self.close()?;
            if let Some(s) = self.streams.iter_mut().find(|s| s.name == stream) {
                s.ring.clear();
                s.init.clear();
                s.last_at = None;
            }
        }
        let preroll = self.preroll;
        let Some(s) = self.streams.iter_mut().find(|s| s.name == stream) else {
            return Ok(());
        };
        s.last_at = incoming.or(s.last_at);
        if let Some(file) = s.open.as_mut() {
            // rebased like the pre-roll was, or the clip's timeline jumps back
            // to the stream's origin the moment the ring runs out.
            let mut shifted = bytes.to_vec();
            rebase(&mut shifted, s.origin);
            file.write_all(&shifted)
                .context("appending to an event clip")?;
            if let Some(open) = self.open.as_mut() {
                open.bytes += shifted.len() as u64;
            }
        }
        // **the ring keeps filling while a clip is open.** it used to be emptied
        // at the trigger, which left the next event to buffer its pre-roll from
        // nothing: on a street where traffic arrives in bursts, the clip after a
        // clip opens with the vehicle already halfway across, which is the exact
        // failure the pre-roll exists to prevent. it costs one extra copy of at
        // most `preroll` seconds of video -- a couple of megabytes.
        //
        // a fragment with no decode time cannot be placed on the stream's
        // clock, so it is held against the newest one rather than dropped.
        let at = incoming.unwrap_or_else(|| s.ring.back().map_or(0, |f| f.at));
        s.ring.push_back(Fragment {
            bytes: bytes.to_vec(),
            at,
        });
        // **the ring may only begin on a keyframe.** anything before the first
        // one is coded against a picture the ring does not hold, so a clip
        // built from it opens with frames no decoder can show. measured on the
        // deployment before this: a 2.00s keyframe interval on the main stream,
        // and an 8.73s clip whose first decodable frame was at 1.33s -- a third
        // of the pre-roll played as a held still, while the container went on
        // claiming it started at zero.
        while s.ring.front().is_some_and(|f| !is_keyframe(&f.bytes)) {
            s.ring.pop_front();
        }
        // bounded by time rather than by count: fragment size varies with how
        // much of the frame is moving, so a count would hold ten seconds of an
        // empty street and one second of a busy one, which is backwards.
        //
        // whole gops at a time, for the same reason: dropping one fragment
        // would leave the ring starting mid-gop again. so the ring runs from a
        // keyframe and holds between `preroll` and `preroll` plus a gop -- an
        // extra second or two of the stream, against a clip that decodes from
        // its first frame. on an all-intra stream every fragment is a keyframe
        // and this is exactly the eviction it always did.
        let window = preroll.as_secs() * s.timescale;
        while let Some(next) = s
            .ring
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, f)| is_keyframe(&f.bytes))
            .map(|(i, _)| i)
        {
            if at.saturating_sub(s.ring[next].at) < window {
                break;
            }
            s.ring.drain(..next);
        }
        Ok(())
    }

    /// something moved in the roi. opens a clip if one is not already running,
    /// writing the ring out first so the clip starts before the trigger did.
    pub fn trigger(&mut self, now: Instant, stamp_millis: u128) -> Result<()> {
        if let Some(open) = self.open.as_mut() {
            open.last_trigger = now;
            return Ok(());
        }
        for i in 0..self.streams.len() {
            let path = partial_path(&self.dir, stamp_millis, &self.streams[i].name);
            let mut file = std::fs::File::create(&path)
                .with_context(|| format!("creating {}", path.display()))?;
            file.write_all(&self.streams[i].init)?;
            // the clip's timeline begins at its first fragment, which is the
            // oldest thing in the ring rather than the trigger.
            let origin = self.streams[i]
                .ring
                .front()
                .and_then(|f| decode_time(&f.bytes))
                .unwrap_or(0);
            self.streams[i].origin = origin;
            // borrowed rather than taken: the ring goes on holding these for
            // the *next* event, which is what gives back-to-back clips a
            // pre-roll each. clips overlapping by a few seconds is the point.
            for f in &self.streams[i].ring {
                let mut shifted = f.bytes.clone();
                rebase(&mut shifted, origin);
                file.write_all(&shifted)?;
            }
            self.streams[i].open = Some(file);
        }
        self.open = Some(Open {
            started: now,
            last_trigger: now,
            worth: Worth::Motion,
            stamp: stamp_millis,
            bytes: 0,
        });
        Ok(())
    }

    /// raise what the open clip is worth. only ever upwards: a clip that held a
    /// subject for one frame is a clip that held a subject.
    pub fn saw(&mut self, worth: Worth) {
        if let Some(open) = self.open.as_mut() {
            open.worth = open.worth.max(worth);
        }
    }

    /// close the clip if it has run long enough or gone quiet. call every frame.
    pub fn tick(&mut self, now: Instant) -> Result<Option<PathBuf>> {
        let Some(open) = self.open.as_ref() else {
            return Ok(None);
        };
        let quiet = now.saturating_duration_since(open.last_trigger) >= self.hangover;
        let long = now.saturating_duration_since(open.started) >= self.max_clip;
        if !quiet && !long {
            return Ok(None);
        }
        self.close()
    }

    /// finish the open clip, renaming it to carry what was found in it.
    ///
    /// the name is written at the end rather than the beginning because that is
    /// when it is known, which is the whole shape of this module.
    pub fn close(&mut self) -> Result<Option<PathBuf>> {
        let Some(open) = self.open.take() else {
            return Ok(None);
        };
        let wanted = self.wanted(open.worth);
        let mut first = None;
        for s in self.streams.iter_mut() {
            s.open = None; // dropping the handle flushes and closes it
            let from = partial_path(&self.dir, open.stamp, &s.name);
            if !wanted {
                let _ = std::fs::remove_file(&from);
                continue;
            }
            let to = clip_path_in(&self.dir, open.stamp, open.worth, &s.name);
            if from.exists() {
                std::fs::rename(&from, &to)
                    .with_context(|| format!("renaming {} to {}", from.display(), to.display()))?;
            }
            first.get_or_insert(to);
        }
        tracing::info!(
            "event clip {} held {} ({:.1} MB){}",
            open.stamp,
            open.worth.slug(),
            open.bytes as f64 / 1e6,
            if wanted { "" } else { ", discarded" }
        );
        self.enforce_budget()?;
        Ok(first)
    }

    /// delete the least useful clips until the directory is inside its budget.
    ///
    /// **by worth first and only then by age**, which is the opposite of the
    /// harvest's oldest-first rule and deliberately so. a crop is one of
    /// thousands of interchangeable examples, so the oldest is the cheapest to
    /// lose. an event clip containing a recognised subject may be the only one
    /// that exists, and a newer clip of an empty street is worth nothing beside
    /// it.
    fn enforce_budget(&self) -> Result<()> {
        let mut clips = list(&self.dir);
        let mut total: u64 = clips.iter().map(|c| c.bytes).sum();
        if total <= self.budget_bytes {
            return Ok(());
        }
        clips.sort_by_key(|c| (c.worth, c.stamp));
        for c in clips {
            if total <= self.budget_bytes {
                break;
            }
            if std::fs::remove_file(&c.path).is_ok() {
                // the cached poster goes with it, or the directory fills with
                // stills of clips nobody can watch.
                let _ = std::fs::remove_file(poster_path(&self.cache_dir, &c.path));
                tracing::info!(
                    "evicted event clip {} ({})",
                    c.path.display(),
                    c.worth.slug()
                );
                total = total.saturating_sub(c.bytes);
            }
        }
        Ok(())
    }
}

/// `<epoch_ms>-<worth>-<stream>.mp4`.
///
/// the stamp leads so a directory listing is chronological, and the stream name
/// trails so the pair a replay needs -- the substream the gate saw and the main
/// stream its crops came from -- sits side by side under one stamp.
fn clip_path_in(dir: &Path, stamp: u128, worth: Worth, stream: &str) -> PathBuf {
    dir.join(format!("{stamp}-{}-{stream}.mp4", worth.slug()))
}

/// the name an open clip is written under, until it closes.
///
/// **a clip being written is not yet a clip.** its worth is still being decided,
/// so its final name is not known, and its last bytes are a fragment that is
/// still arriving. writing it under a name `parse_name` refuses keeps it out of
/// the preview -- which would otherwise show a growing file labelled with the
/// floor value rather than with whatever turns up in it -- and out of the disk
/// budget, which would otherwise be counting bytes that have not landed.
fn partial_path(dir: &Path, stamp: u128, stream: &str) -> PathBuf {
    dir.join(format!("{stamp}-{stream}.mp4.part"))
}

/// adopt whatever a process that died mid-clip left behind.
///
/// its worth was never decided, so it is taken at the floor: `motion` is the
/// one thing known to have happened, and being the floor it is also the first
/// thing the budget evicts, which is the right way to be wrong about it. the
/// alternative is a file that is never listed, never evicted, and never
/// noticed.
fn adopt_partials(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let Some(name) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some((stamp, stream)) = name
            .strip_suffix(".mp4.part")
            .and_then(|rest| rest.split_once('-'))
        else {
            continue;
        };
        let Ok(stamp) = stamp.parse::<u128>() else {
            continue;
        };
        let to = clip_path_in(dir, stamp, Worth::Motion, stream);
        if std::fs::rename(e.path(), &to).is_ok() {
            tracing::info!("adopted an unfinished event clip as {}", to.display());
        }
    }
}

/// one clip on disk. `name` is what `/clips/<name>` serves.
pub struct Clip {
    pub path: PathBuf,
    pub name: String,
    pub stamp: u128,
    pub worth: Worth,
    pub stream: String,
    pub bytes: u64,
    /// when the clip stopped being written, from the file's own mtime.
    ///
    /// **the only record of how long a clip actually ran.** the name carries
    /// when it started and nothing carries when it stopped, so anything wanting
    /// the span had to guess -- from `max_clip_secs`, which is the ceiling
    /// rather than this clip's length, or from when the next clip began, which
    /// is an upper bound with the whole quiet gap inside it. the mtime is the
    /// close, exactly, for free, and the rename that finishes a clip preserves
    /// it. `stamp` again if it cannot be read, which reads as a clip of no
    /// duration rather than as one of unbounded duration.
    pub ended_ms: u128,
}

/// the clip that was recording when something happened, and how far into it
/// that moment falls.
///
/// **this is a containment test, not a resemblance one.** matching the other
/// way round -- picking a crop to illustrate a clip -- was removed because
/// every rule for "which crop belongs to this clip" was wrong somewhere. this
/// direction asks a question with a definite answer: was a clip open at this
/// instant. a clip covers `[stamp - preroll, ended_ms]`, both ends known, and
/// either the moment is inside one or it is not.
///
/// `None` rather than a guess when the clip's span cannot be trusted. `ended_ms`
/// comes from the file's mtime, which a copy does not preserve, so a directory
/// moved with `cp` claims every clip ran until the moment it was copied -- and
/// a clip claiming hours would swallow every crop of the day.
pub fn containing(
    all: &[Clip],
    at_millis: u128,
    preroll_ms: u128,
    max_clip_ms: u128,
) -> Option<(&Clip, f64)> {
    all.iter()
        .filter(|c| c.stream == MAIN_STREAM)
        .filter(|c| c.ended_ms >= c.stamp && c.ended_ms - c.stamp <= max_clip_ms)
        .find_map(|c| {
            let opened = c.stamp.saturating_sub(preroll_ms);
            (at_millis >= opened && at_millis <= c.ended_ms)
                .then(|| (c, (at_millis - opened) as f64 / 1000.0))
        })
}

/// a still from inside a clip, for the events grid.
///
/// **the browser cannot do this itself.** the obvious card is a `<video>` with
/// `#t=` pointing at the interesting moment, and it does not work: a fragmented
/// mp4 written `empty_moov` with no `sidx` carries no map from time to byte
/// offset, so there is nothing to seek with. measured in chrome against real
/// clips, `seekable` is `[0, 0]` and `currentTime` stays at zero however the
/// fragment is written -- every card showed frame one, which is the pre-roll,
/// which is the one part of a clip guaranteed to have nothing in it.
///
/// so the frame is cut here and served as an image: a few kilobytes a card
/// instead of however many seconds of video the seek would have needed, and any
/// moment of the clip is reachable rather than only the beginning.
///
/// cached beside the clip, and deleted with it. `-ss` is deliberately *not*
/// used: seeking these files is what does not work, and it silently lands
/// somewhere else. this decodes from the start and takes the first frame at or
/// after `at_secs`, which for a few seconds of video costs a fraction of one.
/// **one at a time, because this decodes video inside a memory-capped service.**
///
/// a browser opens six connections to a host, so a cold events tab asks for six
/// posters at once, and the server answers each on its own thread. measured
/// against a real clip, peak rss went 256 MB for one concurrent extraction to
/// 1476 MB for six -- which against the unit's `MemoryHigh` put the whole
/// service into direct reclaim, and against `MemoryMax` had the kernel kill it.
/// it restarted twice in twenty minutes on the deployment.
///
/// the lock is the structural fix and the stream choice below is only the
/// constant: sixty cards is sixty extractions however cheap each one is. they
/// are cached after the first, so the cost is paid once per clip.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// cut the frame, given the file to decode and where to cache the result.
///
/// `from` is deliberately separate from `cache`: the frame is taken from the
/// substream when there is one, because 640x480 mjpeg costs 55 MB to decode
/// against 256 MB for 2560x1440 h.264, while the cache is keyed on the main
/// clip so that eviction finds it.
pub fn poster(from: &Path, cache: &Path, at_secs: f64) -> Result<Vec<u8>> {
    if let Ok(bytes) = std::fs::read(cache)
        && !bytes.is_empty()
    {
        return Ok(bytes);
    }
    // a poisoned lock means some other extraction panicked; the queue is still
    // the thing keeping this inside its memory budget, so take it either way.
    let _serialised = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // re-checked under the lock: six browsers asking at once queue here, and
    // the first through leaves the answer for the rest.
    if let Ok(bytes) = std::fs::read(cache)
        && !bytes.is_empty()
    {
        return Ok(bytes);
    }
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            // **one thread, which is both smaller and faster here.** ffmpeg
            // defaults to one decode thread per core, and each holds its own
            // decoded-picture buffers -- at 2560x1440 a frame is 5.5 MB, so
            // sixteen threads measured 256 MB against 92 for one, and took
            // longer: for a single frame, setting up the parallelism costs more
            // than it returns. free on the substream, which is mjpeg and has no
            // frame threading to disable, so this is a straight win on the main
            // fallback and a no-op on the common path.
            "-threads",
            "1",
            "-i",
            &from.to_string_lossy(),
            // **decoding forward rather than `-ss`.** seeking a fragmented mp4
            // lands silently on the wrong frame, which is the failure this
            // whole endpoint exists to end.
            //
            // measured, `-ss` is exact on the substream today and five times
            // faster -- 0.04s against 0.21s, and identical on memory, which is
            // the axis that matters. it is exact because that stream is
            // currently mjpeg, where every frame stands alone. but mjpeg is a
            // camera setting rather than a property of being the substream: it
            // is mjpeg only because a browser once asked its encoder for mjpeg
            // and it stuck, and switching it back to h.264 is under active
            // discussion. a fast path gated on the stream name would go on
            // seeking the day that changes, and start being wrong quietly.
            // gating on the codec instead costs an ffprobe, which spends the
            // 0.17s it was meant to save.
            "-vf",
            &format!("select='gte(t\\,{at_secs:.2})',scale=640:-2"),
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-f",
            "mjpeg",
            "-",
        ])
        .output()
        .context("running ffmpeg to cut a poster frame")?;
    anyhow::ensure!(
        out.status.success() && !out.stdout.is_empty(),
        "ffmpeg produced no poster from {}: {}",
        from.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    // best effort: a poster that cannot be cached is still worth serving once.
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(cache, &out.stdout);
    Ok(out.stdout)
}

/// where a clip's cached poster lives: `<cache_dir>/<clip name>.jpg`.
///
/// a directory of its own rather than a sibling of the clip, so the events
/// directory holds clips and nothing else -- everything under the cache is
/// derived, disposable, and rebuilt on demand. named after the clip so
/// eviction can find it from the clip alone.
pub fn poster_path(cache_dir: &Path, clip: &Path) -> PathBuf {
    let name = clip.file_name().unwrap_or_default().to_string_lossy();
    cache_dir.join(format!("{name}.jpg"))
}

/// the same event on another stream, or `None` if it was not recorded.
///
/// used to take a poster from the substream rather than from main: the frame is
/// the same moment of the same event, and decoding 640x480 mjpeg costs a
/// fraction of decoding 2560x1440 h.264.
pub fn sibling(clip: &Path, stream: &str) -> Option<PathBuf> {
    let name = clip.file_name()?.to_str()?;
    let (stamp, worth, _) = parse_name(name)?;
    let path = clip.with_file_name(format!("{stamp}-{}-{stream}.mp4", worth.slug()));
    path.exists().then_some(path)
}

/// `<stamp>-<worth>-<stream>.mp4` split back up, or `None` for anything else
/// that happens to be in the directory.
pub fn parse_name(name: &str) -> Option<(u128, Worth, &str)> {
    let mut parts = name.strip_suffix(".mp4")?.splitn(3, '-');
    let stamp = parts.next()?.parse().ok()?;
    let worth = Worth::parse(parts.next()?)?;
    let stream = parts.next()?;
    (!stream.is_empty()).then_some((stamp, worth, stream))
}

/// every finished clip in `dir`, newest first.
pub fn list(dir: &Path) -> Vec<Clip> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut clips: Vec<Clip> = entries
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_str()?.to_string();
            let (stamp, worth, stream) = parse_name(&name)?;
            let meta = e.metadata().ok()?;
            let ended_ms = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(stamp, |d| d.as_millis().max(stamp));
            Some(Clip {
                path: e.path(),
                stamp,
                worth,
                stream: stream.to_string(),
                bytes: meta.len(),
                ended_ms,
                name,
            })
        })
        .collect();
    clips.sort_by(|a, b| b.stamp.cmp(&a.stamp).then_with(|| a.name.cmp(&b.name)));
    clips
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **a crop knows when it was taken, and a clip knows what it spans.** so
    /// "which recording was running then" is a containment test rather than the
    /// resemblance guess that was removed: either the moment is inside a clip or
    /// no clip was open.
    #[test]
    fn a_moment_inside_a_clip_finds_it_and_says_how_far_in() {
        let clips = vec![
            clip_spanning(10_000, 20_000, "main"),
            clip_spanning(40_000, 50_000, "main"),
        ];
        // 3s pre-roll, so the first clip opens at 7_000 on the wall clock.
        let (found, at) = containing(&clips, 12_000, 3_000, 60_000).expect("inside the first");
        assert_eq!(found.stamp, 10_000);
        assert!(
            (at - 5.0).abs() < 1e-6,
            "12_000 is 5s into a clip opened at 7_000: {at}"
        );

        // the pre-roll counts: a crop taken before the trigger is still in the file.
        let (found, at) = containing(&clips, 8_000, 3_000, 60_000).expect("inside the pre-roll");
        assert_eq!(found.stamp, 10_000);
        assert!((at - 1.0).abs() < 1e-6, "{at}");
    }

    #[test]
    fn a_moment_in_the_quiet_between_clips_finds_nothing() {
        let clips = vec![
            clip_spanning(10_000, 20_000, "main"),
            clip_spanning(40_000, 50_000, "main"),
        ];
        assert!(containing(&clips, 30_000, 3_000, 60_000).is_none());
        assert!(containing(&clips, 1_000, 3_000, 60_000).is_none());
        assert!(containing(&clips, 90_000, 3_000, 60_000).is_none());
    }

    /// `ended_ms` is an mtime and an mtime does not survive a copy, so an events
    /// directory moved with `cp` reports every clip as having run until the
    /// moment it was copied. one clip claiming hours would otherwise swallow
    /// every crop of the day.
    #[test]
    fn a_clip_with_an_implausible_span_claims_nothing() {
        let clips = vec![clip_spanning(10_000, 10_000 + 86_400_000, "main")];
        assert!(containing(&clips, 20_000, 3_000, 60_000).is_none());
    }

    /// the events tab shows main, so this has to agree with it: a moment lands
    /// on the card a person can actually open.
    #[test]
    fn a_moment_lands_on_the_main_stream_half_of_the_pair() {
        let clips = vec![
            clip_spanning(10_000, 20_000, "sub"),
            clip_spanning(10_000, 20_000, "main"),
        ];
        let (found, _) = containing(&clips, 12_000, 3_000, 60_000).unwrap();
        assert_eq!(found.stream, "main");
    }

    fn clip_spanning(stamp: u128, ended_ms: u128, stream: &str) -> Clip {
        Clip {
            path: PathBuf::from(format!("{stamp}-detection-{stream}.mp4")),
            name: format!("{stamp}-detection-{stream}.mp4"),
            stamp,
            worth: Worth::Detection,
            stream: stream.to_string(),
            bytes: 1,
            ended_ms,
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("metermate-record-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// keeps everything, because these tests are about the ring and the
    /// timeline rather than about retention. what `keep` does is measured on
    /// its own below, where the default is exercised deliberately.
    fn cfg_for(dir: &Path) -> crate::config::RecordCfg {
        crate::config::RecordCfg {
            dir: dir.to_path_buf(),
            cache_dir: dir.join("cache"),
            max_bytes: 1 << 30,
            keep: Vec::new(),
            ..Default::default()
        }
    }

    fn rec(dir: &Path) -> Recorder {
        with_cfg(&cfg_for(dir))
    }

    fn with_cfg(cfg: &crate::config::RecordCfg) -> Recorder {
        let mut r = Recorder::new(cfg, &["sub", "main"]).unwrap();
        r.set_init("sub", init_of(TEST_TIMESCALE));
        r.set_init("main", init_of(TEST_TIMESCALE));
        r
    }

    const TEST_TIMESCALE: u64 = 1000;

    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    /// a real `moof > traf > tfdt` plus an `mdat` carrying a marker, so the
    /// tests exercise the same box walking the camera's stream does. feeding
    /// plain strings would have let the ring silently stop evicting.
    fn fragment(decode: u32, marker: &[u8]) -> Vec<u8> {
        let mut tfdt = vec![0u8; 4]; // version 0, no flags
        tfdt.extend_from_slice(&decode.to_be_bytes());
        let traf = boxed(b"traf", &boxed(b"tfdt", &tfdt));
        let mut out = boxed(b"moof", &traf);
        out.extend_from_slice(&boxed(b"mdat", marker));
        out
    }

    /// a fragment shaped exactly like the camera's, which is the only shape
    /// worth testing against.
    ///
    /// **`tfhd` comes first and always says non-sync.** dumping 900 fragments
    /// off the deployment: every one carries both boxes, every `tfhd` default
    /// says `sample_is_non_sync_sample`, and `trun` carries
    /// `first_sample_flags` *only* on the keyframes -- indices 25, 55, 85, one
    /// every thirty frames. so a reader that takes whichever box answers first
    /// calls every frame a non-keyframe and is wrong about the whole stream. a
    /// fixture carrying only `trun` cannot catch that, and did not.
    fn fragment_keyed(decode: u32, keyframe: bool, marker: &[u8]) -> Vec<u8> {
        let mut tfdt = vec![0u8; 4];
        tfdt.extend_from_slice(&decode.to_be_bytes());

        // version 0, flags 0x020020: default-base-is-moof, default-sample-flags.
        let mut tfhd = vec![0u8, 0x02, 0x00, 0x20];
        tfhd.extend_from_slice(&1u32.to_be_bytes()); // track_ID
        tfhd.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // non-sync, always

        // version 0, flags 0x000005: data-offset and first-sample-flags, the
        // latter written only for a keyframe, as the camera does.
        let mut trun = vec![0u8, 0x00, 0x00, if keyframe { 0x05 } else { 0x01 }];
        trun.extend_from_slice(&1u32.to_be_bytes()); // sample_count
        trun.extend_from_slice(&0i32.to_be_bytes()); // data_offset
        if keyframe {
            trun.extend_from_slice(&0x0200_0000u32.to_be_bytes());
        }

        let traf = boxed(
            b"traf",
            &[
                boxed(b"tfhd", &tfhd),
                boxed(b"tfdt", &tfdt),
                boxed(b"trun", &trun),
            ]
            .concat(),
        );
        let mut out = boxed(b"moof", &traf);
        out.extend_from_slice(&boxed(b"mdat", marker));
        out
    }

    /// **a clip that opens mid-gop begins with frames no decoder can show.**
    /// measured on the deployment: the main stream's keyframe interval is
    /// 2.00s, the ring began wherever the time window happened to fall, and the
    /// first decodable frame of an 8.73s clip was at 1.33s. the container still
    /// claimed `start_time=0`, so a player held a single frame through the
    /// first third of the pre-roll -- which is exactly what a pre-roll exists
    /// not to do. the substream never showed it, because mjpeg is all-intra and
    /// every fragment is a sync sample.
    #[test]
    fn the_ring_begins_on_a_keyframe_so_every_frame_of_the_preroll_decodes() {
        let d = tmpdir("keyframe-preroll");
        let mut r = rec(&d);
        let t0 = Instant::now();

        // 10 seconds at 4 fragments a second, a keyframe every 2s: the
        // deployment's shape at this timescale.
        for i in 0..40u32 {
            let key = i % 8 == 0;
            r.push(
                "main",
                &fragment_keyed(i * 250, key, format!("f{i}").as_bytes()),
                t0,
            )
            .unwrap();
        }

        let s = r.streams.iter().find(|s| s.name == "main").unwrap();
        let front = s.ring.front().expect("the ring is empty");
        assert!(
            is_keyframe(&front.bytes),
            "the ring starts on a frame that depends on one already evicted, so \
             the clip opens undecodable"
        );

        // and it must still be a pre-roll: trimming to a keyframe may not cost
        // so much that the clip no longer starts before the trigger.
        let span = s.ring.back().unwrap().at - front.at;
        assert!(
            span >= 3 * TEST_TIMESCALE,
            "trimming to a keyframe left only {span} of a {} pre-roll",
            3 * TEST_TIMESCALE
        );
    }

    /// `moov > trak > mdia > mdhd`, enough for the timescale to be read.
    fn init_of(timescale: u64) -> Vec<u8> {
        let mut mdhd = vec![0u8; 4]; // version 0
        mdhd.extend_from_slice(&0u32.to_be_bytes()); // creation
        mdhd.extend_from_slice(&0u32.to_be_bytes()); // modification
        mdhd.extend_from_slice(&(timescale as u32).to_be_bytes());
        mdhd.extend_from_slice(&0u32.to_be_bytes()); // duration
        let mdia = boxed(b"mdia", &boxed(b"mdhd", &mdhd));
        boxed(b"moov", &boxed(b"trak", &mdia))
    }

    /// the whole reason to buffer. motion fires after the subject is already in
    /// frame, so a clip that starts at the trigger opens on a vehicle halfway
    /// across and cannot answer when it first appeared.
    #[test]
    fn a_clip_begins_before_the_trigger_that_opened_it() {
        let d = tmpdir("preroll");
        let mut r = rec(&d);
        let t0 = Instant::now();

        r.push("main", b"before-1", t0).unwrap();
        r.push("main", b"before-2", t0 + Duration::from_secs(1))
            .unwrap();
        r.trigger(t0 + Duration::from_secs(2), 1789).unwrap();
        r.push("main", b"after", t0 + Duration::from_secs(2))
            .unwrap();
        r.close().unwrap();

        let body = std::fs::read(clip_path_in(&d, 1789, Worth::Motion, "main")).unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            body.starts_with(&init_of(TEST_TIMESCALE)),
            "clip does not begin with its init header"
        );
        assert!(text.contains("before-1"), "pre-roll was dropped: {text}");
        assert!(
            text.contains("before-2") && text.contains("after"),
            "{text}"
        );
    }

    /// fragments older than the pre-roll window are dropped, or an idle street
    /// grows the ring without bound.
    #[test]
    fn the_ring_forgets_what_is_older_than_the_preroll() {
        let d = tmpdir("ring");
        let mut r = rec(&d);
        let t0 = Instant::now();
        // decode times, not wall-clock: six seconds of stream against a three
        // second window. the wall-clock arguments are deliberately identical so
        // a regression to timing by arrival fails this rather than passing it.
        //
        // fed at a stream-like cadence rather than as two fragments six seconds
        // apart. eviction drops whole gops and stops once what remains would no
        // longer cover the pre-roll, so with only two fragments the honest
        // answer is to keep both -- dropping one would leave nothing to roll
        // back to. the cadence is what makes "older than the window" mean
        // anything.
        let six = ((PREROLL.as_secs() + 1) * TEST_TIMESCALE) as u32;
        r.push("main", &fragment(0, b"ancient"), t0).unwrap();
        for at in (250..six).step_by(250) {
            r.push("main", &fragment(at, b"middle"), t0).unwrap();
        }
        r.push("main", &fragment(six, b"recent"), t0).unwrap();
        r.trigger(t0, 1789).unwrap();
        r.close().unwrap();

        let body = std::fs::read(clip_path_in(&d, 1789, Worth::Motion, "main")).unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(!text.contains("ancient"), "the ring kept a stale fragment");
        assert!(text.contains("recent"), "{text}");
    }

    /// what a clip is worth is not knowable when it starts. the name is written
    /// at the end, when the detector and classifier have actually run.
    #[test]
    fn a_clip_is_named_for_what_was_found_in_it_not_for_what_started_it() {
        let d = tmpdir("worth");
        let mut r = rec(&d);
        let t0 = Instant::now();
        r.trigger(t0, 1789).unwrap();
        r.push("main", &fragment(0, b"x"), t0).unwrap();
        r.saw(Worth::Detection);
        r.saw(Worth::Subject);
        r.saw(Worth::Detection); // never downgrades
        r.close().unwrap();

        assert!(clip_path_in(&d, 1789, Worth::Subject, "main").exists());
        assert!(!clip_path_in(&d, 1789, Worth::Motion, "main").exists());
    }

    /// one vehicle is one clip. re-triggering while open extends it rather than
    /// starting a second file.
    #[test]
    fn a_second_trigger_extends_the_clip_rather_than_starting_another() {
        let d = tmpdir("extend");
        let mut r = rec(&d);
        let t0 = Instant::now();
        r.trigger(t0, 1789).unwrap();
        r.trigger(t0 + Duration::from_secs(2), 9999).unwrap();
        assert!(r.recording());
        // quiet is measured from the *last* trigger, so it is not yet due.
        assert!(
            r.tick(t0 + HANGOVER + Duration::from_secs(1))
                .unwrap()
                .is_none()
        );
        assert!(
            r.tick(t0 + Duration::from_secs(2) + HANGOVER)
                .unwrap()
                .is_some()
        );
        assert!(!clip_path_in(&d, 9999, Worth::Motion, "main").exists());
    }

    #[test]
    fn a_clip_that_never_goes_quiet_is_closed_by_the_ceiling() {
        let d = tmpdir("ceiling");
        let mut r = rec(&d);
        let t0 = Instant::now();
        r.trigger(t0, 1789).unwrap();
        // re-triggered constantly, so it never goes quiet.
        for s in 1..=MAX_CLIP.as_secs() {
            r.trigger(t0 + Duration::from_secs(s), 1789).unwrap();
        }
        assert!(
            r.tick(t0 + MAX_CLIP).unwrap().is_some(),
            "ran past the ceiling"
        );
    }

    /// every stream is cut at the same moment, so the pair a replay needs -- the
    /// substream the gate saw and the main stream its crops came from -- covers
    /// the same seconds.
    #[test]
    fn a_trigger_opens_every_stream_under_one_stamp() {
        let d = tmpdir("pair");
        let mut r = rec(&d);
        let t0 = Instant::now();
        r.trigger(t0, 1789).unwrap();
        r.push("sub", &fragment(0, b"s"), t0).unwrap();
        r.push("main", &fragment(0, b"m"), t0).unwrap();
        r.saw(Worth::Subject);
        r.close().unwrap();

        for stream in ["sub", "main"] {
            let p = clip_path_in(&d, 1789, Worth::Subject, stream);
            assert!(p.exists(), "{} missing", p.display());
        }
    }

    /// **by worth first, then by age** -- the opposite of the harvest's rule.
    /// a crop is one of thousands of interchangeable examples so the oldest is
    /// cheapest to lose; an event clip holding a recognised subject may be the
    /// only one there is.
    #[test]
    fn the_budget_evicts_the_least_useful_clip_not_the_oldest() {
        let d = tmpdir("budget");
        std::fs::create_dir_all(&d).unwrap();
        let big = vec![b'x'; 400];
        // oldest and most valuable, newest and least.
        std::fs::write(clip_path_in(&d, 1, Worth::Subject, "main"), &big).unwrap();
        std::fs::write(clip_path_in(&d, 2, Worth::Detection, "main"), &big).unwrap();
        std::fs::write(clip_path_in(&d, 3, Worth::Motion, "main"), &big).unwrap();

        let r = Recorder::new(
            &crate::config::RecordCfg {
                max_bytes: 900,
                ..cfg_for(&d)
            },
            &["main"],
        )
        .unwrap();
        r.enforce_budget().unwrap();

        assert!(
            clip_path_in(&d, 1, Worth::Subject, "main").exists(),
            "the only clip holding a subject was evicted"
        );
        assert!(
            !clip_path_in(&d, 3, Worth::Motion, "main").exists(),
            "the least useful clip survived"
        );
    }

    /// **the bug that made every clip lie about its length.** a fragment's
    /// decode time is absolute, counted from when ffmpeg started remuxing, so
    /// clips written straight out claimed a timeline running from the stream's
    /// origin: measured on a real replay, clips a few seconds long reported
    /// 49s, 87s and 180s, growing with uptime. it played, which is why nothing
    /// caught it until a file was probed.
    #[test]
    fn a_clip_timeline_starts_at_zero_rather_than_at_the_streams_origin() {
        let d = tmpdir("rebase");
        let mut r = rec(&d);
        let t0 = Instant::now();
        // an hour into the stream, so a missed rebase is unmistakable.
        let origin = 3_600 * TEST_TIMESCALE as u32;
        r.push("main", &fragment(origin, b"a"), t0).unwrap();
        r.trigger(t0, 1789).unwrap();
        r.push("main", &fragment(origin + 1000, b"b"), t0).unwrap();
        r.close().unwrap();

        let body = std::fs::read(clip_path_in(&d, 1789, Worth::Motion, "main")).unwrap();
        let times = all_decode_times(&body);
        assert_eq!(
            times,
            vec![0, 1000],
            "the clip kept the stream's absolute timeline"
        );
    }

    /// every `tfdt` in a file, so a rebase can be checked rather than assumed.
    fn all_decode_times(buf: &[u8]) -> Vec<u64> {
        let mut out = Vec::new();
        let mut at = 0;
        while at + 8 <= buf.len() {
            let size = u32::from_be_bytes(buf[at..at + 4].try_into().unwrap()) as usize;
            if size < 8 || at + size > buf.len() {
                break;
            }
            if let Some(v) =
                decode_time(&buf[at..at + size]).filter(|_| &buf[at + 4..at + 8] == b"moof")
            {
                out.push(v);
            }
            at += size;
        }
        out
    }

    #[test]
    fn the_timescale_is_read_from_the_init_rather_than_assumed() {
        assert_eq!(timescale_of(&init_of(1000)), 1000);
        assert_eq!(timescale_of(&init_of(90_000)), 90_000);
        // a header that has not arrived, or one this does not understand, must
        // not silently become a zero-length window.
        assert_eq!(timescale_of(b""), DEFAULT_TIMESCALE);
        assert_eq!(timescale_of(&init_of(0)), DEFAULT_TIMESCALE);
    }

    /// **the config said one thing and the recorder did another.**
    /// `preroll_secs`, `hangover_secs` and `max_clip_secs` were parsed from the
    /// file, printed in the startup line, and then never passed to the
    /// recorder, which used its own compiled-in constants -- so metermate
    /// announced the configured pre-roll while recording the default one. the
    /// values here are deliberately nothing like the defaults, or a recorder
    /// ignoring them still passes.
    #[test]
    fn the_configured_limits_are_the_ones_used() {
        let d = tmpdir("limits");
        let cfg = crate::config::RecordCfg {
            preroll_secs: 1,
            hangover_secs: 1,
            max_clip_secs: 3,
            ..cfg_for(&d)
        };
        let t0 = Instant::now();

        // pre-roll: two seconds of stream against a one second window, fed at a
        // stream-like cadence so that "older than the window" has fragments to
        // roll back to. eviction keeps whole gops and will not cut the ring
        // below the pre-roll it exists to guarantee.
        let mut r = with_cfg(&cfg);
        let two = 2 * TEST_TIMESCALE as u32;
        r.push("main", &fragment(0, b"ancient"), t0).unwrap();
        for at in (250..two).step_by(250) {
            r.push("main", &fragment(at, b"middle"), t0).unwrap();
        }
        r.push("main", &fragment(two, b"recent"), t0).unwrap();
        r.trigger(t0, 1).unwrap();
        r.close().unwrap();
        let body = std::fs::read(clip_path_in(&d, 1, Worth::Motion, "main")).unwrap();
        assert!(
            !String::from_utf8_lossy(&body).contains("ancient"),
            "the ring used the compiled-in pre-roll, not the configured one"
        );

        // hangover: quiet at one second rather than at four.
        let mut r = with_cfg(&cfg);
        r.trigger(t0, 2).unwrap();
        assert!(r.tick(t0 + Duration::from_millis(900)).unwrap().is_none());
        assert!(
            r.tick(t0 + Duration::from_secs(1)).unwrap().is_some(),
            "the clip stayed open past the configured hangover"
        );

        // ceiling: closed at three seconds rather than at sixty, even while
        // being re-triggered.
        let mut r = with_cfg(&cfg);
        r.trigger(t0, 3).unwrap();
        for ms in [1000, 2000, 3000] {
            r.trigger(t0 + Duration::from_millis(ms), 3).unwrap();
        }
        assert!(
            r.tick(t0 + Duration::from_secs(3)).unwrap().is_some(),
            "the clip ran past the configured ceiling"
        );
    }

    /// **the second vehicle of a burst got no pre-roll.** the ring was emptied
    /// when a clip opened and only refilled once it closed, so an event
    /// arriving inside the next few seconds -- which on a street with traffic
    /// in bursts is most of them -- opened on a vehicle already halfway across.
    /// that is the exact failure the pre-roll exists to prevent, reported from
    /// the deployed box as clips that start and stop mid-road.
    #[test]
    fn a_clip_that_follows_a_clip_still_gets_its_preroll() {
        let d = tmpdir("burst");
        let mut r = rec(&d);
        let t0 = Instant::now();
        let mut at = 0u32;
        let step = TEST_TIMESCALE as u32 / 10;

        // five seconds of quiet street, then an event.
        for _ in 0..50 {
            r.push("main", &fragment(at, b"first-roll"), t0).unwrap();
            at += step;
        }
        r.trigger(t0, 1).unwrap();
        for _ in 0..20 {
            r.push("main", &fragment(at, b"during-one"), t0).unwrap();
            at += step;
        }
        r.saw(Worth::Detection);
        r.close().unwrap();

        // a second vehicle immediately behind the first.
        r.trigger(t0 + Duration::from_secs(1), 2).unwrap();
        r.push("main", &fragment(at, b"during-two"), t0).unwrap();
        r.saw(Worth::Detection);
        r.close().unwrap();

        let second = std::fs::read(clip_path_in(&d, 2, Worth::Detection, "main")).unwrap();
        let text = String::from_utf8_lossy(&second);
        assert!(
            text.contains("during-one"),
            "the second clip opened with an empty ring, so it has no pre-roll"
        );
        let times = all_decode_times(&second);
        assert!(
            times.len() > 2 && times.windows(2).all(|w| w[1] > w[0]),
            "the second clip's timeline is not a timeline: {times:?}"
        );
    }

    /// **the clip no player would show more than one frame of.** ffmpeg is
    /// restarted when the camera stalls, and the remux then begins a fresh
    /// timeline at zero. rebased against the old origin every fragment goes
    /// negative and floors at zero: a clip on the deployed box held 103
    /// fragments all claiming t=0, 1.4 MB of video that decoded to nothing.
    #[test]
    fn a_stream_that_restarts_does_not_flatten_the_timeline() {
        let d = tmpdir("restart");
        let mut r = rec(&d);
        let t0 = Instant::now();
        let hour = 3_600 * TEST_TIMESCALE as u32;

        // an hour into the stream, with a clip open across the break.
        r.push("main", &fragment(hour, b"old-a"), t0).unwrap();
        r.trigger(t0, 1).unwrap();
        r.push("main", &fragment(hour + 1000, b"old-b"), t0)
            .unwrap();
        r.saw(Worth::Detection);

        // ffmpeg restarts: the clock goes back to the beginning.
        r.push("main", &fragment(0, b"new-a"), t0).unwrap();
        assert!(!r.recording(), "a clip was carried across a stream restart");
        let old = std::fs::read(clip_path_in(&d, 1, Worth::Detection, "main")).unwrap();
        assert!(
            String::from_utf8_lossy(&old).contains("old-b"),
            "the clip open at the restart was lost rather than finished"
        );

        // the header is asked for again, because a restarted ffmpeg may have
        // renegotiated the stream.
        assert!(r.init_needed(), "the stale header was kept");
        r.set_init("main", init_of(TEST_TIMESCALE));
        r.push("main", &fragment(1000, b"new-b"), t0).unwrap();
        r.trigger(t0, 2).unwrap();
        r.push("main", &fragment(2000, b"new-c"), t0).unwrap();
        r.saw(Worth::Detection);
        r.close().unwrap();

        let body = std::fs::read(clip_path_in(&d, 2, Worth::Detection, "main")).unwrap();
        let times = all_decode_times(&body);
        assert!(
            times.windows(2).all(|w| w[1] > w[0]),
            "the timeline did not advance after the restart: {times:?}"
        );
        assert!(
            !String::from_utf8_lossy(&body).contains("old-"),
            "the ring carried fragments from the stream that ended"
        );
    }

    /// on a street with parked cars permanently in frame, a motion-only clip is
    /// one where the gate fired and nothing was ever established. measured on
    /// the deployed box those are the clips with no vehicle in them at all, so
    /// the default does not keep them.
    #[test]
    fn only_the_outcomes_asked_for_are_kept() {
        let d = tmpdir("keep");
        let cfg = crate::config::RecordCfg {
            dir: d.clone(),
            max_bytes: 1 << 30,
            keep: vec!["detection".into(), "subject".into()],
            ..Default::default()
        };
        let t0 = Instant::now();

        let mut r = with_cfg(&cfg);
        r.trigger(t0, 1).unwrap();
        r.push("main", &fragment(0, b"x"), t0).unwrap();
        r.close().unwrap(); // motion only
        assert!(list(&d).is_empty(), "a motion-only clip was kept");
        assert_eq!(
            std::fs::read_dir(&d).unwrap().count(),
            0,
            "the discarded clip was left on disk under its provisional name"
        );

        let mut r = with_cfg(&cfg);
        r.trigger(t0, 2).unwrap();
        r.push("main", &fragment(0, b"x"), t0).unwrap();
        r.saw(Worth::Detection);
        r.close().unwrap();
        assert_eq!(list(&d).len(), 2, "a detection clip was discarded");
    }

    /// the escape hatch, and the shape `[detector] classes` already uses: an
    /// empty list is "no filter" rather than "nothing".
    #[test]
    fn an_empty_keep_list_keeps_everything() {
        let d = tmpdir("keepall");
        let cfg = crate::config::RecordCfg {
            dir: d.clone(),
            max_bytes: 1 << 30,
            keep: Vec::new(),
            ..Default::default()
        };
        let mut r = with_cfg(&cfg);
        r.trigger(Instant::now(), 1).unwrap();
        r.push("main", &fragment(0, b"x"), Instant::now()).unwrap();
        r.close().unwrap();
        assert_eq!(list(&d).len(), 2, "an empty keep list discarded a clip");
    }

    /// a clip is listed when it is true, not while it is being written. the
    /// worth in the name is decided at close, so a file still open is a file
    /// whose name is provisional -- and the preview reads exactly this listing.
    #[test]
    fn an_open_clip_is_neither_listed_nor_charged_to_the_budget() {
        let d = tmpdir("partial");
        let mut r = rec(&d);
        let t0 = Instant::now();
        r.trigger(t0, 1789).unwrap();
        r.push("main", &fragment(0, b"x"), t0).unwrap();
        r.saw(Worth::Subject);

        assert!(list(&d).is_empty(), "an unfinished clip was listed");
        assert!(
            partial_path(&d, 1789, "main").exists(),
            "nothing is being written"
        );

        r.close().unwrap();
        let listed = list(&d);
        assert_eq!(listed.len(), 2, "both streams should be listed");
        assert!(listed.iter().all(|c| c.worth == Worth::Subject));
        assert!(!partial_path(&d, 1789, "main").exists());
    }

    /// a process that dies mid-clip leaves a file whose worth was never decided.
    /// taken at the floor it is listed, evictable, and honest about what is
    /// known; left as it is it would be none of those.
    #[test]
    fn a_clip_left_behind_by_a_crash_is_adopted_at_the_floor() {
        let d = tmpdir("crash");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(partial_path(&d, 1789, "main"), b"fragments").unwrap();

        let _ = rec(&d);

        assert!(!partial_path(&d, 1789, "main").exists());
        let listed = list(&d);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].worth, Worth::Motion);
        assert_eq!(listed[0].stamp, 1789);
        assert_eq!(listed[0].stream, "main");
    }

    /// the listing drives the preview, so the order is the order clips appear
    /// in, and the name is what `/clips/<name>` is asked for.
    #[test]
    fn clips_are_listed_newest_first() {
        let d = tmpdir("listing");
        std::fs::create_dir_all(&d).unwrap();
        for (stamp, worth) in [
            (1, Worth::Motion),
            (3, Worth::Subject),
            (2, Worth::Detection),
        ] {
            std::fs::write(clip_path_in(&d, stamp, worth, "main"), b"x").unwrap();
        }
        std::fs::write(d.join("notes.txt"), b"x").unwrap();

        let listed = list(&d);
        assert_eq!(
            listed.iter().map(|c| c.stamp).collect::<Vec<_>>(),
            vec![3, 2, 1],
            "clips are not newest first"
        );
        assert_eq!(listed[0].name, "3-subject-main.mp4");
        assert_eq!(listed[0].bytes, 1);
    }

    #[test]
    fn only_clip_names_parse() {
        assert_eq!(
            parse_name("1789268224097-subject-main.mp4"),
            Some((1789268224097, Worth::Subject, "main"))
        );
        // a stream name may itself contain a dash, so the split stops after two.
        assert_eq!(
            parse_name("1789-motion-sub-low.mp4"),
            Some((1789, Worth::Motion, "sub-low"))
        );
        for bad in [
            "1789-main.mp4.part", // still being written
            "1789-guess-main.mp4",
            "later-subject-main.mp4",
            "1789-subject-.mp4",
            "1789-subject-main.mkv",
            "metermate.toml",
        ] {
            assert!(parse_name(bad).is_none(), "{bad} parsed as a clip");
        }
    }

    #[test]
    fn nothing_is_written_until_something_triggers() {
        let d = tmpdir("idle");
        let mut r = rec(&d);
        let t0 = Instant::now();
        for i in 0..50 {
            r.push("main", b"frag", t0 + Duration::from_millis(i * 100))
                .unwrap();
        }
        assert!(!r.recording());
        assert_eq!(list(&d).len(), 0, "an idle street wrote a file");
    }
}
