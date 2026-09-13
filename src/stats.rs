//! what the pipeline is doing, as numbers, over http.
//!
//! **this exists because diagnosing it from outside took an evening.** every
//! question that mattered -- is the camera delivering, is the detector running,
//! is the harvest writing, is anything stalled right now -- was answerable only
//! by sshing in and running something, and each round trip cost a wrong theory.
//!
//! the design follows from that. **ages, not just totals.** a counter says what
//! has happened since boot; an age says what is happening now. "the last main
//! frame arrived 47 seconds ago" identifies a dead crop feed instantly, where
//! "1.2 million frames received" identifies nothing. every stall this program
//! can have shows up as one of these ages growing.
//!
//! it is deliberately cheap: relaxed atomics on the frame loop, read and
//! formatted only when somebody asks.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

/// never counted: a stage that has not run yet is not a stage that is late.
const NEVER: u64 = u64::MAX;

#[derive(Debug)]
pub struct Stats {
    started: Instant,

    // what arrives
    gate_frames: AtomicU64,
    gate_dropped: AtomicU64,
    gate_restarts: AtomicU64,
    crop_restarts: AtomicU64,

    // when each stage last did anything, in millis since `started`
    last_gate_frame: AtomicU64,
    last_crop_frame: AtomicU64,
    last_motion: AtomicU64,
    last_inference: AtomicU64,
    last_classify: AtomicU64,
    last_recognised: AtomicU64,
    last_harvest: AtomicU64,

    // what the loop does with it
    motion_frames: AtomicU64,
    inspected_frames: AtomicU64,
    /// motion fired and there was no main-stream frame to look at, so nothing
    /// was detected. the symptom is turquoise boxes with nothing in them.
    blind_looks: AtomicU64,
    inferences: AtomicU64,
    inference_us: AtomicU64,
    detections: AtomicU64,
    declined: AtomicU64,
    harvested: AtomicU64,
    /// **both stages run inline on the frame loop**, so what they cost is not a
    /// curiosity: it is frame rate. the per-call number says how expensive one
    /// look is and the running total says what share of the wall clock the loop
    /// spent inside them, which is the only one of the two that can explain a
    /// live feed slowing down.
    inference_total_us: AtomicU64,
    classifications: AtomicU64,
    classify_us: AtomicU64,
    classify_total_us: AtomicU64,
    /// verdicts that named a subject, of those classifications.
    recognised: AtomicU64,

    // how much the loop is carrying
    places: AtomicU64,
    tracks: AtomicU64,
    overlay: AtomicU64,

    /// whether the rtsp channels are shut by `[stream] active_hours`, and when
    /// they open again. without these a stream closed on purpose is
    /// indistinguishable from one that has died, which is the one question every
    /// age here exists to answer -- and the page needs both to say so.
    ///
    /// the opening is a `"07:00"` or a `"mon 07:00"` rather than a number, which
    /// is why this one field is a mutex and not an atomic: a window that shuts
    /// for a weekend is not a minute of the day. nothing on the frame loop reads
    /// or writes either half of this.
    stream_shut: AtomicBool,
    stream_opens: Mutex<Option<String>>,
    /// megabytes the last model unload gave back (r11.4), 0 until one happens.
    /// measured by the code that made the drop, because nothing else can say
    /// which megabytes were the models'.
    models_released: AtomicU64,

    // what is on disk
    harvest_files: AtomicU64,
    harvest_bytes: AtomicU64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            gate_frames: AtomicU64::new(0),
            gate_dropped: AtomicU64::new(0),
            gate_restarts: AtomicU64::new(0),
            crop_restarts: AtomicU64::new(0),
            stream_shut: AtomicBool::new(false),
            stream_opens: Mutex::new(None),
            models_released: AtomicU64::new(0),
            last_gate_frame: AtomicU64::new(NEVER),
            last_crop_frame: AtomicU64::new(NEVER),
            last_motion: AtomicU64::new(NEVER),
            last_inference: AtomicU64::new(NEVER),
            last_classify: AtomicU64::new(NEVER),
            last_recognised: AtomicU64::new(NEVER),
            last_harvest: AtomicU64::new(NEVER),
            motion_frames: AtomicU64::new(0),
            inspected_frames: AtomicU64::new(0),
            blind_looks: AtomicU64::new(0),
            inferences: AtomicU64::new(0),
            inference_us: AtomicU64::new(0),
            detections: AtomicU64::new(0),
            declined: AtomicU64::new(0),
            harvested: AtomicU64::new(0),
            inference_total_us: AtomicU64::new(0),
            classifications: AtomicU64::new(0),
            classify_us: AtomicU64::new(0),
            classify_total_us: AtomicU64::new(0),
            recognised: AtomicU64::new(0),
            places: AtomicU64::new(0),
            tracks: AtomicU64::new(0),
            overlay: AtomicU64::new(0),
            harvest_files: AtomicU64::new(0),
            harvest_bytes: AtomicU64::new(0),
        }
    }
}

impl Stats {
    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn stamp(&self, field: &AtomicU64) {
        field.store(self.now_ms(), Ordering::Relaxed);
    }

    /// millis since `field` was last stamped, or `None` if it never was.
    fn age(&self, field: &AtomicU64) -> Option<u64> {
        match field.load(Ordering::Relaxed) {
            NEVER => None,
            at => Some(self.now_ms().saturating_sub(at)),
        }
    }

    pub fn gate_frame(&self, dropped_since_last: u64) {
        self.gate_frames.fetch_add(1, Ordering::Relaxed);
        self.gate_dropped
            .fetch_add(dropped_since_last, Ordering::Relaxed);
        self.stamp(&self.last_gate_frame);
    }

    /// how old the newest main-stream frame is, not when we last looked at one.
    ///
    /// the distinction is the whole value of this number. `LatestFrame` hands
    /// back the last frame it received rather than nothing, so a crop feed that
    /// has died keeps answering -- the pipeline goes on examining one still
    /// photograph, every detection lands in the same place, the harvest dedupes
    /// them all, and no counter anywhere moves. stamping this when we *looked*
    /// would report a healthy feed throughout; stamping it with the frame's own
    /// age reports the truth.
    pub fn crop_frame(&self, age: std::time::Duration) {
        let at = self.now_ms().saturating_sub(age.as_millis() as u64);
        self.last_crop_frame.store(at, Ordering::Relaxed);
    }

    /// the stream window's own state: whether the channels are shut, and the
    /// minute they open. held here rather than recomputed from a clock per
    /// request, because the time worth naming is the one the channels were shut
    /// against -- and because the page is the only reader, and it asks about a
    /// window that changes twice a day.
    /// what an `unload_models` window edge gave back, for `/stats` to report.
    pub fn models_released(&self, mb: u64) {
        self.models_released.store(mb, Ordering::Relaxed);
    }

    pub fn stream(&self, shut: bool, opens: Option<String>) {
        self.stream_shut.store(shut, Ordering::Relaxed);
        // written at the two edges of a window and read by whoever is asking:
        // a panic somewhere else in the pipeline is no reason to stop
        // answering, and the next edge overwrites whatever it finds.
        *self
            .stream_opens
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = opens;
    }

    pub fn motion(&self) {
        self.motion_frames.fetch_add(1, Ordering::Relaxed);
        self.stamp(&self.last_motion);
    }

    pub fn blind_look(&self) {
        self.blind_looks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inspected(&self) {
        self.inspected_frames.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inference(&self, took: std::time::Duration, found: usize) {
        self.inferences.fetch_add(1, Ordering::Relaxed);
        self.inference_us
            .store(took.as_micros() as u64, Ordering::Relaxed);
        self.inference_total_us
            .fetch_add(took.as_micros() as u64, Ordering::Relaxed);
        self.detections.fetch_add(found as u64, Ordering::Relaxed);
        self.stamp(&self.last_inference);
    }

    /// one stage-two look, and whether it named anything.
    ///
    /// counted even when it recognises nothing, which is nearly always: the cost
    /// is paid on every look and only the total of those can explain a frame
    /// rate. `recognised` is the rarer number and the one the verdicts view is
    /// about.
    pub fn classified(&self, took: std::time::Duration, recognised: bool) {
        self.classifications.fetch_add(1, Ordering::Relaxed);
        self.classify_us
            .store(took.as_micros() as u64, Ordering::Relaxed);
        self.classify_total_us
            .fetch_add(took.as_micros() as u64, Ordering::Relaxed);
        self.stamp(&self.last_classify);
        if recognised {
            self.recognised.fetch_add(1, Ordering::Relaxed);
            self.stamp(&self.last_recognised);
        }
    }

    pub fn declined(&self) {
        self.declined.fetch_add(1, Ordering::Relaxed);
    }

    pub fn harvested(&self) {
        self.harvested.fetch_add(1, Ordering::Relaxed);
        self.stamp(&self.last_harvest);
    }

    pub fn restart(&self, gate: bool) {
        let field = if gate {
            &self.gate_restarts
        } else {
            &self.crop_restarts
        };
        field.fetch_add(1, Ordering::Relaxed);
    }

    /// sizes of what the loop carries between frames, refreshed periodically
    /// rather than per frame: they change slowly and nobody watches them live.
    pub fn carrying(&self, places: usize, tracks: usize, overlay: usize) {
        self.places.store(places as u64, Ordering::Relaxed);
        self.tracks.store(tracks as u64, Ordering::Relaxed);
        self.overlay.store(overlay as u64, Ordering::Relaxed);
    }

    /// what the harvester last reported on disk, for the page that shows the
    /// harvest against its budget. read from here rather than walked again: the
    /// harvester already knows, and the crops grid asks once per page of a
    /// scroll.
    pub fn harvest_on_disk_now(&self) -> (u64, u64) {
        (
            self.harvest_files.load(Ordering::Relaxed),
            self.harvest_bytes.load(Ordering::Relaxed),
        )
    }

    pub fn harvest_on_disk(&self, files: u64, bytes: u64) {
        self.harvest_files.store(files, Ordering::Relaxed);
        self.harvest_bytes.store(bytes, Ordering::Relaxed);
    }

    /// everything, as json.
    ///
    /// rates are averages over the whole run rather than a rolling window: a
    /// window needs upkeep on the frame loop, and the ages above already answer
    /// "is it stalled right now" better than any rate can.
    pub fn json(&self, viewers: usize) -> String {
        let up = self.started.elapsed().as_secs_f64().max(0.001);
        let g = |f: &AtomicU64| f.load(Ordering::Relaxed);
        let age = |f: &AtomicU64| match self.age(f) {
            Some(ms) => ms.to_string(),
            None => "null".to_string(),
        };
        let frames = g(&self.gate_frames);
        let dropped = g(&self.gate_dropped);
        // what is not being held *right now*: the moment the models are back a
        // figure from last night next to a running stream is a figure somebody
        // reads as missing memory.
        let released = if self.stream_shut.load(Ordering::Relaxed) {
            g(&self.models_released)
        } else {
            0
        };
        // quoted when there is a time to name and bare `null` when there is not,
        // so the page never has to tell "open" from "closed, no idea when".
        let opens = match self
            .stream_opens
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_deref()
        {
            Some(at) => format!("\"{at}\""),
            None => "null".to_string(),
        };
        format!(
            concat!(
                r#"{{"uptime_s":{:.0},"#,
                r#""input":{{"gate_frames":{},"gate_fps":{:.1},"gate_dropped":{},"#,
                r#""stream_seen_pct":{:.0},"gate_restarts":{},"crop_restarts":{},"#,
                r#""last_gate_frame_ms":{},"last_crop_frame_ms":{},"stream_shut":{},"#,
                r#""stream_opens":{},"models_released_mb":{}}},"#,
                r#""processing":{{"motion_frames":{},"last_motion_ms":{},"#,
                r#""inspected_frames":{},"blind_looks":{},"inferences":{},"#,
                r#""inference_ms":{:.0},"inferences_per_s":{:.1},"last_inference_ms":{},"#,
                r#""inference_pct":{:.1},"#,
                r#""classifications":{},"classify_ms":{:.0},"classify_pct":{:.1},"#,
                r#""last_classify_ms":{},"recognised":{},"last_recognised_ms":{},"#,
                r#""detections":{},"declined":{}}},"#,
                r#""output":{{"harvested":{},"harvest_per_min":{:.1},"last_harvest_ms":{},"#,
                r#""harvest_files":{},"harvest_mb":{:.0}}},"#,
                r#""carrying":{{"places":{},"tracks":{},"overlay_boxes":{}}},"#,
                r#""preview":{{"viewers":{}}},"limits":{}}}"#
            ),
            up,
            frames,
            frames as f64 / up,
            dropped,
            100.0 * frames as f64 / (frames + dropped).max(1) as f64,
            g(&self.gate_restarts),
            g(&self.crop_restarts),
            age(&self.last_gate_frame),
            age(&self.last_crop_frame),
            self.stream_shut.load(Ordering::Relaxed),
            opens,
            released,
            g(&self.motion_frames),
            age(&self.last_motion),
            g(&self.inspected_frames),
            g(&self.blind_looks),
            g(&self.inferences),
            g(&self.inference_us) as f64 / 1000.0,
            g(&self.inferences) as f64 / up,
            age(&self.last_inference),
            // share of wall clock spent inside each stage. both run inline on
            // the frame loop, so these are the two numbers that say whether the
            // pipeline is slow because of what it is thinking about.
            100.0 * g(&self.inference_total_us) as f64 / 1e6 / up,
            g(&self.classifications),
            g(&self.classify_us) as f64 / 1000.0,
            100.0 * g(&self.classify_total_us) as f64 / 1e6 / up,
            age(&self.last_classify),
            g(&self.recognised),
            age(&self.last_recognised),
            g(&self.detections),
            g(&self.declined),
            g(&self.harvested),
            g(&self.harvested) as f64 * 60.0 / up,
            age(&self.last_harvest),
            g(&self.harvest_files),
            g(&self.harvest_bytes) as f64 / 1e6,
            g(&self.places),
            g(&self.tracks),
            g(&self.overlay),
            viewers,
            limits_json(),
        )
    }
}

/// what the kernel says about the limits this process is running under.
///
/// **both of the faults that cost the most tonight were limits, and neither
/// looked like one.** `MemoryHigh` was crossed 165,334 times: the kernel does
/// not kill anything when that happens, it throttles every allocation into
/// direct reclaim, so the symptom was ffmpeg delivering nothing for ten seconds
/// at a stretch with no error, the cpu idle and the disk idle. `CPUQuota` is the
/// same shape -- throttling shows up as a program that is slow while the machine
/// looks unloaded, which is the one thing a top-level cpu graph cannot show.
///
/// read on request rather than on the frame loop: nobody asks often, and each
/// call is a handful of small reads from sysfs.
/// what this process is holding in memory, in megabytes: `VmRSS` from
/// `/proc/self/status`, which is the kernel's count rather than an allocator's
/// estimate of its own books.
///
/// 0 where there is no `/proc` to ask, which reads as "nothing came back" rather
/// than as a wrong number.
pub fn resident_mb() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.trim().strip_suffix(" kB"))
        .and_then(|kb| kb.parse::<u64>().ok())
        .map_or(0, |kb| kb / 1024)
}

fn limits_json() -> String {
    let Some(base) = cgroup_dir() else {
        return r#"{"cgroup":null}"#.to_string();
    };
    let read = |name: &str| std::fs::read_to_string(base.join(name)).unwrap_or_default();
    let num = |name: &str| -> Option<u64> { read(name).trim().parse().ok() };
    // `max` means unlimited, which is not a number and must not read as zero.
    let limit = |name: &str| match read(name).trim() {
        "" => "null".to_string(),
        "max" => "null".to_string(),
        v => v.parse::<u64>().map_or("null".into(), |n| n.to_string()),
    };
    // split, because `memory.current` alone cannot say whether sitting at the
    // limit is a problem. page cache is reclaimable -- writing thousands of
    // crops fills it, the kernel takes it back under pressure, and the pipeline
    // never notices. anonymous memory is not: growth there at the limit is a
    // leak, and it is the one that ends in an oom kill. the two are
    // indistinguishable in the total, which is how a benign sawtooth and a leak
    // came to look alike.
    let memory = read("memory.stat");
    format!(
        concat!(
            r#"{{"memory_mb":{:.0},"memory_anon_mb":{:.0},"memory_file_mb":{:.0},"#,
            r#""memory_high_mb":{},"memory_max_mb":{},"#,
            r#""memory_throttled":{},"memory_stall_pct":{:.1},"#,
            r#""cpu_throttled":{},"cpu_throttled_s":{:.1},"cpu_stall_pct":{:.1}}}"#
        ),
        num("memory.current").unwrap_or(0) as f64 / 1e6,
        field(&memory, "anon").unwrap_or(0) as f64 / 1e6,
        field(&memory, "file").unwrap_or(0) as f64 / 1e6,
        mb(&limit("memory.high")),
        mb(&limit("memory.max")),
        field(&read("memory.events"), "high").unwrap_or(0),
        pressure(&read("memory.pressure")),
        field(&read("cpu.stat"), "nr_throttled").unwrap_or(0),
        field(&read("cpu.stat"), "throttled_usec").unwrap_or(0) as f64 / 1e6,
        pressure(&read("cpu.pressure")),
    )
}

/// this process's cgroup directory, on cgroup v2.
fn cgroup_dir() -> Option<std::path::PathBuf> {
    let mine = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = mine.lines().find_map(|l| l.strip_prefix("0::"))?.trim();
    let dir = std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    dir.is_dir().then_some(dir)
}

/// bytes to megabytes, keeping `null` as `null`.
fn mb(v: &str) -> String {
    v.parse::<u64>()
        .map_or_else(|_| v.to_string(), |n| format!("{:.0}", n as f64 / 1e6))
}

/// `"<key> <value>"` out of a flat sysfs table.
fn field(text: &str, key: &str) -> Option<u64> {
    text.lines()
        .find_map(|l| l.strip_prefix(key)?.trim().parse().ok())
}

/// `full avg10` out of a pressure file: the share of the last ten seconds in
/// which *everything* was stalled waiting for this resource.
fn pressure(text: &str) -> f64 {
    text.lines()
        .find(|l| l.starts_with("full"))
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|f| f.strip_prefix("avg10=")?.parse().ok())
        })
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the whole point is being readable by something that is not a person, so
    /// the one thing that must never break is that it parses.
    #[test]
    fn the_report_is_json_and_says_what_has_not_happened_yet() {
        let s = Stats::default();
        let fresh = s.json(0);
        assert!(fresh.starts_with('{') && fresh.ends_with('}'), "{fresh}");
        // nothing has run, so every age is null rather than zero. zero would read
        // as "it just happened", which is the opposite of the truth.
        assert!(
            fresh.contains(r#""last_crop_frame_ms":null"#),
            "a stage that never ran reported an age: {fresh}"
        );
        assert!(fresh.contains(r#""last_inference_ms":null"#), "{fresh}");
        assert!(fresh.contains(r#""last_classify_ms":null"#), "{fresh}");
        assert!(fresh.contains(r#""last_recognised_ms":null"#), "{fresh}");

        s.gate_frame(3);
        s.motion();
        s.inference(std::time::Duration::from_millis(126), 4);
        s.classified(std::time::Duration::from_millis(48), false);
        s.classified(std::time::Duration::from_millis(52), true);
        s.harvested();
        s.carrying(61, 12, 3);
        s.harvest_on_disk(412_093, 19_847_000_000);
        let busy = s.json(2);
        assert!(busy.contains(r#""gate_frames":1"#), "{busy}");
        assert!(busy.contains(r#""gate_dropped":3"#), "{busy}");
        assert!(busy.contains(r#""detections":4"#), "{busy}");
        assert!(busy.contains(r#""places":61"#), "{busy}");
        assert!(busy.contains(r#""harvest_files":412093"#), "{busy}");
        assert!(!busy.contains(r#""last_inference_ms":null"#), "{busy}");
        // **what the two inline stages cost.** a look that recognises nothing
        // still costs a look, so the count is every classification and only
        // `recognised` is the rare one.
        assert!(busy.contains(r#""classifications":2"#), "{busy}");
        assert!(busy.contains(r#""recognised":1"#), "{busy}");
        assert!(
            busy.contains(r#""classify_ms":52"#),
            "the last look: {busy}"
        );
        assert!(!busy.contains(r#""last_recognised_ms":null"#), "{busy}");
        // the shares are of wall clock, so over a test that takes no time they
        // are large; what matters is that they are reported at all.
        assert!(busy.contains(r#""inference_pct":"#), "{busy}");
        assert!(busy.contains(r#""classify_pct":"#), "{busy}");
        // 1 frame of 4 offered.
        assert!(busy.contains(r#""stream_seen_pct":25"#), "{busy}");
        // the limits block is always present, even where there is no cgroup to
        // read: a field that vanishes is a field nothing can alert on.
        assert!(busy.contains(r#""limits":{"#), "{busy}");
    }

    #[test]
    fn a_shut_stream_says_so_rather_than_looking_stalled() {
        let s = Stats::default();
        let idle = s.json(0);
        assert!(
            idle.contains(r#""stream_shut":false,"stream_opens":null"#),
            "{idle}"
        );
        s.stream(true, Some("07:00".into()));
        let shut = s.json(0);
        // the time is the whole point: a stream that is closed and a stream that
        // has died look the same until something says when it comes back.
        assert!(
            shut.contains(r#""stream_shut":true,"stream_opens":"07:00""#),
            "{shut}"
        );
        // a window that shuts for a weekend is not a minute of the day, and the
        // page prints whatever this says.
        s.stream(true, Some("mon 07:00".into()));
        assert!(
            s.json(0).contains(r#""stream_opens":"mon 07:00""#),
            "{}",
            s.json(0)
        );
        s.stream(false, None);
        assert!(
            s.json(0).contains(r#""stream_opens":null"#),
            "{}",
            s.json(0)
        );
    }

    #[test]
    fn a_released_model_says_what_it_was_worth() {
        let s = Stats::default();
        assert!(
            s.json(0).contains(r#""models_released_mb":0"#),
            "nothing has been released yet: {}",
            s.json(0)
        );
        s.stream(true, Some("07:00".into()));
        s.models_released(17);
        assert!(
            s.json(0).contains(r#""models_released_mb":17"#),
            "{}",
            s.json(0)
        );
        // and the moment the models are back it stops being a figure at all:
        // the memory is being used again, not missing.
        s.stream(false, None);
        assert!(
            s.json(0).contains(r#""models_released_mb":0"#),
            "{}",
            s.json(0)
        );
    }

    #[test]
    fn a_process_can_ask_the_kernel_how_much_it_is_holding() {
        // the number the unload reports is the difference between two of these,
        // so a read that returned nothing would report a free 0 MB every night.
        assert!(resident_mb() > 0, "no resident size to be found");
    }

    /// the parsers, against the shapes the kernel actually writes. these are the
    /// numbers that would have named both of tonight's limit faults in one
    /// request, so they are worth being sure of.
    #[test]
    fn the_kernel_tables_are_read_the_way_the_kernel_writes_them() {
        let events = "low 0\nhigh 165334\nmax 0\noom 0\noom_kill 0\n";
        assert_eq!(field(events, "high"), Some(165334));
        assert_eq!(field(events, "oom_kill"), Some(0));
        assert_eq!(field(events, "absent"), None);

        let cpu = "usage_usec 12\nnr_periods 40\nnr_throttled 7\nthrottled_usec 1500000\n";
        assert_eq!(field(cpu, "nr_throttled"), Some(7));
        assert_eq!(field(cpu, "throttled_usec"), Some(1_500_000));

        // `some` is any task stalled; `full` is all of them, which is the one
        // that means the program is not running rather than merely contended.
        let psi = "some avg10=15.42 avg60=20.04 avg300=19.56 total=24390246841\n\
                   full avg10=15.38 avg60=19.93 avg300=19.38 total=24170442749\n";
        assert!((pressure(psi) - 15.38).abs() < 1e-6, "{}", pressure(psi));
        assert_eq!(pressure(""), 0.0);

        // **`memory.stat` keys are prefixes of each other.** `anon` leads
        // `anon_thp` and `file` leads `file_mapped`, `file_dirty` and
        // `file_thp`, so a prefix match that did not also have to parse would
        // take whichever came first and be quietly wrong by a factor. this is
        // the table that says whether sitting at the limit is page cache doing
        // its job or anonymous growth that is not coming back.
        let memory = "anon 1021108224\nfile 11004477440\nkernel 98783232\n\
                      kernel_stack 819200\nshmem 9098207232\nfile_mapped 262144\n\
                      file_dirty 4096\nfile_writeback 0\nanon_thp 0\nfile_thp 0\n";
        assert_eq!(field(memory, "anon"), Some(1_021_108_224));
        assert_eq!(field(memory, "file"), Some(11_004_477_440));
        assert_eq!(field(memory, "file_mapped"), Some(262_144));

        // an unlimited cgroup writes `max`, which must not read as zero bytes.
        assert_eq!(mb("null"), "null");
        assert_eq!(mb("1073741824"), "1074");
    }
}
