//! video ingest: supervised ffmpeg subprocesses producing raw frames.
//!
//! ffmpeg is a subprocess rather than a linked library on purpose. it keeps the
//! build free of a c++ toolchain, it lets hwaccel flags become a config change,
//! and a wedged decoder can be killed and respawned without taking us with it.

use crate::config::{
    Config, FRAME_STALL_TIMEOUT_MS, RESTART_BACKOFF_MAX_MS, RESTART_BACKOFF_MIN_MS,
};
use anyhow::{Context, Result};
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// how often a feed looks at the clock it does not have, in milliseconds: the
/// channels may have been closed since the last frame, and the reader has to
/// notice in about that long rather than after a whole stall timeout.
const CHANNEL_POLL_MS: u64 = 250;

/// whether the streams are wanted right now.
///
/// `[stream] active_hours` closes the rtsp channels by taking this away: a
/// feed that is not wanted kills its ffmpeg and waits to be wanted again
/// rather than exiting, because the point of closing the channels is that
/// something is still there to reopen them -- the preview, the crops on disk,
/// and the process that owns both (r8.5, r8.6).
#[derive(Clone)]
pub struct Channels(Arc<AtomicBool>);

impl Channels {
    /// channels nobody is holding shut, which is every run without a window.
    pub fn open() -> Self {
        Self(Arc::new(AtomicBool::new(true)))
    }

    /// channels that start shut: a run that begins outside its window pulls
    /// nothing until the window opens, and says so while it waits.
    pub fn shut() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// whether a feed should be pulling anything. cheap enough to ask on
    /// every frame, which is what makes closing prompt.
    pub fn wanted(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn close(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    pub fn reopen(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// raw pixels: rgb24 for the gate and crop feeds, so `width * height * 3`.
    pub data: Arc<Vec<u8>>,
    pub seq: u64,
    pub received: Instant,
}

/// where video comes from. a file source is not a test-only affordance: it is
/// how the e2e suite runs deterministically and how the binary is exercised on a
/// development laptop with no camera (r5.5).
#[derive(Debug, Clone)]
pub enum Source {
    Rtsp { url: String, redacted: String },
    File(PathBuf),
}

impl Source {
    fn describe(&self) -> String {
        match self {
            Source::Rtsp { redacted, .. } => redacted.clone(),
            Source::File(p) => p.display().to_string(),
        }
    }

    /// whether this is a live stream rather than something finite that ends.
    ///
    /// keyed on the url rather than on the variant, because an `[camera] url`
    /// template can name a recording: what makes a source live is that it never
    /// runs out, which `rtsp://` is and `file:` is not (r5.6).
    fn live(&self) -> bool {
        matches!(self, Source::Rtsp { url, .. } if url.starts_with("rtsp://"))
    }

    /// a file source is finite, so the supervisor must not treat clean exit as a
    /// failure to retry forever.
    fn restarts_on_exit(&self) -> bool {
        self.live()
    }
}

pub struct Ingest {
    source: Source,
    width: u32,
    height: u32,
    /// colour planes per frame: 3 for the rgb24 every feed is asked for.
    planes: usize,
    filter: String,
    transport: String,
    threads: u32,
    /// how many decoded frames may queue before frames start being dropped.
    /// carried per ingest rather than read from a constant, because the gate and
    /// the crop feed are different sizes and a queue is measured in frames.
    queue_depth: usize,
    offline: bool,
    /// a second output, written as a copy of the input bitstream.
    remux_to: Option<String>,
    /// counted restarts, reported at `/stats`. a feed that keeps dying and
    /// coming back is invisible at info level and is the single most useful
    /// number when the pipeline has gone quiet.
    watch: Option<(std::sync::Arc<crate::stats::Stats>, bool)>,
    /// the window this feed is allowed to run inside.
    channels: Channels,
}

impl Ingest {
    /// replay every frame, however long that takes.
    ///
    /// live ingest is built around never falling behind: it paces a file at
    /// wall-clock and drops whatever the consumer cannot keep up with, because a
    /// stale frame is worth less than a current one. that is right for a camera
    /// and wrong for an eval, where dropping frames measures the machine rather
    /// than the change being tested. offline replay removes the pacing and the
    /// dropping, so two builds see exactly the same frames.
    /// report restarts of this feed to `stats`. `gate` distinguishes the two,
    /// because which one is dying is the whole question: the gate feeds motion
    /// and the crop feed feeds detection, and losing them looks completely
    /// different from outside.
    pub fn watched_by(mut self, stats: std::sync::Arc<crate::stats::Stats>, gate: bool) -> Self {
        self.watch = Some((stats, gate));
        self
    }

    pub fn offline(mut self) -> Self {
        self.offline = true;
        self
    }

    /// also write the input bitstream, unchanged, to `url` as fragmented mp4.
    ///
    /// a *second output on the same ffmpeg*, not a second process, because the
    /// expensive part of a stream is pulling it off the camera rather than
    /// muxing it. a separate rtsp session for the preview pulled another
    /// 4 mbit/s and starved the gate over wifi, dropping detection from fifteen
    /// frames a second to one. this costs a mux and nothing else.
    pub fn remux_to(mut self, url: String) -> Self {
        self.remux_to = Some(url);
        self
    }

    /// run this feed over channels the caller can close.
    ///
    /// both feeds take the same handle, because the rtsp session is what the
    /// window closes and the camera counts the gate and the crop feed as two
    /// of those: stopping one and leaving the other running saves the cpu and
    /// none of the bandwidth, and leaves the preview showing a street the
    /// gate has stopped watching.
    pub fn over(mut self, channels: Channels) -> Self {
        self.channels = channels;
        self
    }

    /// the gate feed: the stream's native size, never rescaled.
    ///
    /// colour, even though the gate itself only needs luma. asking for gray was
    /// cheaper by about 9 MB/s of pipe, but it made the preview grey and left it
    /// running at the crop rate of a few frames per second. luma is derived in
    /// rust for a millisecond or so per frame, and the preview gets all fifteen
    /// frames a second in colour.
    pub fn gate(cfg: &Config, source: Source) -> Self {
        Self {
            source,
            width: cfg.stream.gate_width,
            height: cfg.stream.gate_height,
            planes: 3,
            filter: "format=rgb24".to_string(),
            transport: cfg.stream.rtsp_transport.clone(),
            threads: cfg.stream.decode_threads,
            queue_depth: cfg.stream.frame_queue_depth,
            offline: false,
            remux_to: None,
            watch: None,
            channels: Channels::open(),
        }
    }

    /// the crop feed: full-resolution colour from the main stream.
    ///
    /// **its queue is deliberately shallow, whatever the config says.** a queued
    /// gate frame is 921 KB and a queued main frame is 2560x1440x3 = 11 MB, and
    /// the gate is the only one of the two that has to see every frame -- its
    /// background model is built from them. this feed is drained into a single
    /// latest-frame slot, so depth beyond a frame in hand and a frame arriving
    /// buys nothing and costs eleven megabytes each.
    ///
    /// that is not academic. the deployment runs under `MemoryHigh=768M`, and
    /// raising the shared depth from 2 to 8 put 88 MB of main frames in flight
    /// where there had been 22. the cgroup went over its limit, the kernel
    /// throttled the whole service into direct reclaim -- 165,334 times -- and
    /// `sock_throttled` throttled the rtsp receive path itself. the main stream
    /// then delivered nothing for ten seconds at a time, every forty-five
    /// seconds, with no error anywhere, while the cpu sat idle.
    ///
    /// output is throttled well below the capture rate. throttling does not
    /// reduce decode cost, it bounds pipe bandwidth: 2560x1440 rgb at 15fps is
    /// 166 MB/s of frames nothing would read. a few per second is ample, since
    /// this exists to crop vehicles that are stopping rather than to track.
    /// the crop feed's declared size. what it is *really* reading comes from
    /// [`resolve_size`], which the caller applies with [`Ingest::at_size`].
    pub fn crops(cfg: &Config, source: Source, fps: u32) -> Self {
        let (width, height) = (cfg.stream.main_width, cfg.stream.main_height);
        Self {
            source,
            width,
            height,
            planes: 3,
            filter: format!("fps={fps},format=rgb24"),
            transport: cfg.stream.rtsp_transport.clone(),
            threads: cfg.stream.decode_threads,
            queue_depth: CROP_QUEUE_DEPTH,
            offline: false,
            remux_to: None,
            watch: None,
            channels: Channels::open(),
        }
    }

    /// the detector feed: colour, un-squeezed from the anamorphic substream and
    /// letterboxed to the model's square input.
    ///
    /// ffmpeg does the geometry because it is operating on an already small
    /// image, where rescaling is cheap. doing it in rust would mean shipping
    /// three times the pixels over the pipe to then throw most of them away.
    /// boxes come back in letterboxed coordinates, which is also what the
    /// preview draws on, so nothing has to be mapped back.
    ///
    /// `scene` is the main stream's size: the two streams share a field of view,
    /// so that is the shape the substream's pixels have to be pulled back to.
    pub fn detector(cfg: &Config, source: Source, size: u32, scene: (u32, u32)) -> Self {
        let (w, h) = (size, undistorted(size, scene.0, scene.1));
        let pad_y = (size - h) / 2;
        Self {
            source,
            width: size,
            height: size,
            planes: 3,
            filter: format!("scale={w}:{h},pad={size}:{size}:0:{pad_y}:black,format=rgb24"),
            transport: cfg.stream.rtsp_transport.clone(),
            threads: cfg.stream.decode_threads,
            queue_depth: cfg.stream.frame_queue_depth,
            offline: false,
            remux_to: None,
            watch: None,
            channels: Channels::open(),
        }
    }

    fn ffmpeg_args(&self) -> Vec<String> {
        let mut a: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into()];
        a.extend(["-threads".into(), self.threads.to_string()]);
        let path = match &self.source {
            Source::Rtsp { url, .. } => url.clone(),
            Source::File(p) => p.display().to_string(),
        };
        if self.source.live() {
            a.extend(["-rtsp_transport".into(), self.transport.clone()]);
            // drop anything that arrives late rather than buffering it.
            a.extend(["-fflags".into(), "nobuffer".into()]);
            a.extend(["-flags".into(), "low_delay".into()]);
        } else {
            // pace a file at wall-clock rate, so a replay behaves as the camera
            // does -- unless this is an offline eval, which wants every frame
            // examined rather than a faithful clock.
            if !self.offline {
                a.extend(["-re".into()]);
            }
        }
        a.extend(["-i".into(), path]);
        a.extend(["-an".into(), "-sn".into()]);
        // emit exactly the frames that arrive, and no others.
        //
        // this camera's rtsp stream declares `100 tbr` while actually sending
        // 15 fps. left to its own devices ffmpeg pads the output towards that
        // nominal rate by duplicating frames: measured at 84 fps out of a 15
        // fps camera, so five out of every six frames were copies. everything
        // downstream then paid to decode, difference and encode them.
        a.extend(["-fps_mode".into(), "passthrough".into()]);
        a.extend(["-vf".into(), self.filter.clone()]);
        a.extend(["-f".into(), "rawvideo".into(), "pipe:1".into()]);

        // a second output, stream-copied. `frag_every_frame` rather than
        // fragmenting on keyframes: at a fifteen frame gop that would hold a
        // whole second back before anything reached the browser.
        if let Some(url) = &self.remux_to {
            a.extend(["-an".into(), "-sn".into()]);
            a.extend(["-c".into(), "copy".into()]);
            a.extend([
                "-movflags".into(),
                "empty_moov+default_base_moof+frag_every_frame".into(),
            ]);
            a.extend(["-f".into(), "mp4".into(), url.clone()]);
        }
        a
    }

    fn frame_bytes(&self) -> usize {
        (self.width as usize) * (self.height as usize) * self.planes
    }

    /// read the feed at a size the stream confirmed, from [`resolve_size`].
    ///
    /// this is the number everything downstream is built at: it is the frame
    /// size the pipe is cut into, so a size left behind here is sheared
    /// half-frames rather than an error.
    pub fn at_size(mut self, size: (u32, u32)) -> Self {
        self.width = size.0;
        self.height = size.1;
        self
    }

    fn spawn(&self) -> Result<Child> {
        Command::new("ffmpeg")
            .args(self.ffmpeg_args())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .context("spawning ffmpeg (is it installed and on PATH?)")
    }

    /// run the supervisor until the source ends or the consumer hangs up.
    /// frames are delivered on the returned receiver.
    pub fn run(self) -> Receiver<Frame> {
        let (tx, rx) = sync_channel(self.queue_depth);
        std::thread::spawn(move || {
            if let Err(e) = self.supervise(tx) {
                tracing::error!("ingest supervisor stopped: {e:#}");
            }
        });
        rx
    }

    fn supervise(&self, tx: SyncSender<Frame>) -> Result<()> {
        let mut backoff = RESTART_BACKOFF_MIN_MS;
        let mut seq = 0u64;
        loop {
            self.wait_until_wanted();
            tracing::info!("ingest starting: {}", self.source.describe());
            let attempt = self.attempt(&tx, &mut seq);
            // **the window closing is not the feed failing.** it arrives here
            // as an ordinary unclean exit, and reading it as one would both
            // warn and count a restart -- burying the restarts that do mean
            // something, which is the whole reason the number is reported.
            if !self.channels.wanted() {
                tracing::debug!("ingest closed: {}", self.source.describe());
                continue;
            }
            // logged here rather than propagated: for a live source this is a
            // bad moment, not the end, and it used to be the end.
            if let Err(e) = &attempt {
                tracing::warn!("ingest attempt failed: {e:#}");
            }
            if feed_is_over(&attempt, self.source.restarts_on_exit()) {
                if matches!(attempt, Ok(true)) {
                    tracing::info!("source finished after {seq} frames");
                }
                return attempt.map(|_| ());
            }
            // a full queue means the consumer is alive but busy, which is fine.
            // only a disconnect means there is nothing left to supervise for.
            if let Err(TrySendError::Disconnected(_)) = tx.try_send(probe_frame()) {
                return Ok(());
            }
            if let Some((stats, gate)) = &self.watch {
                stats.restart(*gate);
            }
            tracing::warn!("ingest restarting in {backoff}ms");
            std::thread::sleep(Duration::from_millis(backoff));
            backoff = (backoff * 2).min(RESTART_BACKOFF_MAX_MS);
        }
    }

    /// sit without an ffmpeg until the channels are wanted again.
    ///
    /// the waiting belongs to the feed rather than to its owner because the
    /// owner has no way to restart what has already finished: an ingest that
    /// exited when the window shut would have to be rebuilt from here, and
    /// everything built at it -- the preview's video, the recorder's rings --
    /// with it.
    fn wait_until_wanted(&self) {
        if self.channels.wanted() {
            return;
        }
        tracing::debug!(
            "ingest held shut by [stream] active_hours: {}",
            self.source.describe()
        );
        while !self.channels.wanted() {
            std::thread::sleep(Duration::from_millis(CHANNEL_POLL_MS));
        }
    }

    /// one run of ffmpeg, from spawn to exit. true on clean eof.
    fn attempt(&self, tx: &SyncSender<Frame>, seq: &mut u64) -> Result<bool> {
        let mut child = self.spawn()?;
        let clean = self.pump(&mut child, tx, seq);
        let _ = child.kill();
        let _ = child.wait();
        clean
    }

    /// read frames until the stream stalls or ends. returns true on clean eof.
    fn pump(&self, child: &mut Child, tx: &SyncSender<Frame>, seq: &mut u64) -> Result<bool> {
        let stdout = child.stdout.take().context("ffmpeg stdout missing")?;
        let stderr = child.stderr.take();
        let (ftx, frx) = sync_channel(self.queue_depth);
        let (w, h, bytes) = (self.width, self.height, self.frame_bytes());
        let offline = self.offline;
        std::thread::spawn(move || read_frames(stdout, w, h, bytes, ftx, offline));
        if let Some(err) = stderr {
            std::thread::spawn(move || log_stderr(err));
        }

        let stall = Duration::from_millis(FRAME_STALL_TIMEOUT_MS);
        // polled faster than the stall it must not be confused with, so that a
        // window closing reads as a window closing within a quarter of a
        // second instead of arriving as a stalled stream ten seconds later.
        let poll = stall.min(Duration::from_millis(CHANNEL_POLL_MS));
        let mut last = Instant::now();
        loop {
            if !self.channels.wanted() {
                // closed between frames: not a stall and not the end.
                return Ok(false);
            }
            match frx.recv_timeout(poll) {
                Ok(mut f) => {
                    last = Instant::now();
                    *seq += 1;
                    f.seq = *seq;
                    if self.offline {
                        // block instead: an offline replay is finite and is
                        // measuring the pipeline, so the decoder waiting is
                        // exactly right. the send only fails once the consumer
                        // is gone.
                        if tx.send(f).is_err() {
                            return Ok(true);
                        }
                    } else if let Err(TrySendError::Disconnected(_)) = tx.try_send(f) {
                        // same reasoning as the reader: drop rather than stall.
                        return Ok(true); // consumer hung up
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    // ffmpeg will happily hold a dead rtsp socket open, so a
                    // silent stream is judged by frame arrival, not liveness.
                    if last.elapsed() >= stall {
                        tracing::warn!(
                            "no frame for {}ms, treating stream as stalled",
                            stall.as_millis()
                        );
                        return Ok(false);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(true),
            }
        }
    }
}

/// a zero-sized sentinel used only to test whether the consumer is still there.
fn probe_frame() -> Frame {
    Frame {
        width: 0,
        height: 0,
        data: Arc::new(Vec::new()),
        seq: 0,
        received: Instant::now(),
    }
}

impl Frame {
    /// supervisor liveness probes are not real frames and must be ignored.
    pub fn is_probe(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// how far into the clip this frame is, for a feed emitting at `fps`.
    ///
    /// the only common ground between two decoders reading the same recording
    /// at different rates. `pump` numbers from one, so the first frame is at
    /// zero.
    pub fn video_secs(&self, fps: f64) -> f64 {
        self.seq.saturating_sub(1) as f64 / fps
    }
}

/// what the stream says it is: `ffprobe`ed width and height, either of which
/// the stream may refuse to give.
///
/// this used to be a compiled-in constant per stream, which held until the day
/// the camera was swapped. asking is the only answer that survives that: an
/// operator changes a resolution in the camera's own page, and metermate follows
/// it, rather than cutting 4096x1856 pixels into 2560x1440 chunks and reading
/// sheared half-frames forever with no error to find.
///
/// costs about 1.6s against this camera, so once per stream at startup.
/// `-rtsp_transport tcp` goes only to a url that wants it: ffmpeg rejects the
/// option outright for anything else, including the `file:` urls the tests use
/// to reach this path.
pub fn probe_size(source: &Source) -> Option<(u32, u32)> {
    let url = match source {
        Source::File(p) => p.display().to_string(),
        Source::Rtsp { url, .. } => url.clone(),
    };
    let mut args = vec!["-v".to_string(), "error".to_string()];
    if url.starts_with("rtsp://") {
        args.extend(["-rtsp_transport".into(), "tcp".into()]);
    }
    args.extend([
        "-select_streams".into(),
        "v:0".into(),
        "-show_entries".into(),
        "stream=width,height".into(),
        "-of".into(),
        "csv=p=0:s=x".into(),
    ]);
    let out = Command::new("ffprobe").args(args).arg(&url).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (w, h) = text.trim().split_once('x')?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

/// the size to read `source` at, preferring the stream over the config and
/// saying which won.
///
/// the declared size stays in play because a camera that is rebooting cannot be
/// asked, and refusing to start over that would be worse than starting at a size
/// that a restart will correct. the fallback is loud for the same reason.
pub fn resolve_size(what: &str, declared: (u32, u32), source: &Source) -> (u32, u32) {
    match probe_size(source) {
        Some(size) => {
            if size != declared {
                tracing::warn!(
                    "{what} stream is {}x{}, not the declared {}x{}: reading what it sends \
                     (update [stream] to match)",
                    size.0,
                    size.1,
                    declared.0,
                    declared.1
                );
            }
            tracing::info!("{what} feed: {}x{}", size.0, size.1);
            size
        }
        None => {
            tracing::warn!(
                "{what} stream at {} did not answer, reading it as {}x{}",
                source.describe(),
                declared.0,
                declared.1
            );
            declared
        }
    }
}

/// how tall the model's square input is once a scene of `scene_w`x`scene_h` is
/// stretched to `size` wide.
///
/// the substream is stored at whatever aspect the encoder offers and shown at
/// the scene's, so the pixels have to be pulled back out to the scene's shape
/// before the model sees them -- and the scene's shape is the main stream's,
/// since the two share a field of view.
fn undistorted(size: u32, scene_w: u32, scene_h: u32) -> u32 {
    if scene_w == 0 {
        return size;
    }
    (((size as u64 * scene_h as u64 + scene_w as u64 / 2) / scene_w as u64).min(size as u64) as u32)
        .max(2)
}

/// a clip's real frame rate, for driving an offline replay's clock.
///
/// `avg_frame_rate` rather than `r_frame_rate`: this camera declares a nominal
/// `100 tbr` it does not send, which is the same lie that made ffmpeg duplicate
/// frames until `-fps_mode passthrough` was set. the average is measured from
/// the frames actually in the file.
pub fn probe_fps(path: &std::path::Path) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (num, den) = text.trim().split_once('/')?;
    let (num, den): (f64, f64) = (num.parse().ok()?, den.parse().ok()?);
    (den > 0.0 && num > 0.0).then(|| num / den)
}

/// holds whichever frame arrived most recently, for consumers that want
/// "whatever is current" rather than every frame.
///
/// the crop stream is read this way: nothing is waiting on main-stream frames,
/// they are only consulted when the gate has already decided something moved.
#[derive(Clone)]
pub struct LatestFrame {
    slot: Arc<Mutex<Option<Frame>>>,
}

impl LatestFrame {
    /// start draining an ingest into the slot, keeping only the newest frame.
    pub fn spawn(ingest: Ingest) -> Self {
        let slot: Arc<Mutex<Option<Frame>>> = Arc::new(Mutex::new(None));
        let sink = slot.clone();
        let rx = ingest.run();
        std::thread::spawn(move || {
            while let Some(frame) = latest(&rx) {
                if frame.is_probe() {
                    continue;
                }
                if let Ok(mut s) = sink.lock() {
                    *s = Some(frame);
                }
            }
        });
        Self { slot }
    }

    pub fn get(&self) -> Option<Frame> {
        self.slot.lock().ok().and_then(|s| s.clone())
    }

    /// forget whatever is held, because nothing holding an old frame should
    /// get to call it current.
    pub fn clear(&self) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = None;
        }
    }
}

/// the main-stream feed, addressed by *when* a frame happened.
///
/// the gate and the crops are separate decoders with no clock between them.
/// live that is survivable: both are fed by real time, so the newest main frame
/// is at most one crop interval behind the gate frame that asked for it.
///
/// offline it is not. a replay deliberately has no wall clock, so the crop
/// decoder empties the clip at whatever rate the machine manages and then
/// repeats its last frame for the rest of the run. measured on a ten minute
/// clip: the inspected image stopped changing at gate frame 200 of 6534, and
/// every end-to-end number after that was scored against a still photograph --
/// including the recall the harvest filters were being tuned against.
pub enum Crops {
    /// whichever frame arrived most recently. arrival order is the only clock a
    /// live stream has, and it is the right one.
    Live(LatestFrame),
    /// pulled forward to the gate's position in the clip, which also
    /// backpressures the decoder into staying there.
    Replay(Replay),
}

impl Crops {
    pub fn live(ingest: Ingest) -> Self {
        Crops::Live(LatestFrame::spawn(ingest))
    }

    /// replay in lockstep with the gate, from a feed emitting at `fps`.
    pub fn replay(ingest: Ingest, fps: f64) -> Self {
        Crops::Replay(Replay {
            rx: ingest.run(),
            fps,
            current: None,
            peeked: None,
        })
    }

    /// forget the held main-stream frame.
    ///
    /// the feed hands back the newest frame it has *received* rather than
    /// nothing, which is right between two live decoders and wrong across a
    /// closed window: the first vehicle through a window that reopened at
    /// seven would otherwise be cropped from the image that was last there
    /// the evening before, and presented as what is there now. a crop asked
    /// for before the feed refills is a blind look, which is what the
    /// "motion with no main-stream frame" line is already there to say.
    pub fn discard_stale(&mut self) {
        match self {
            Crops::Live(latest) => latest.clear(),
            // a replay is pulled forward to the gate rather than left behind,
            // and has no wall clock for stale to mean anything against.
            Crops::Replay(_) => {}
        }
    }

    /// the main-stream frame covering `at` seconds into the clip. live, where
    /// there is no clip to be at a point in, the newest frame.
    pub fn at(&mut self, at: f64) -> Option<Frame> {
        match self {
            Crops::Live(l) => l.get(),
            Crops::Replay(r) => r.at(at),
        }
    }
}

/// a crop feed pulled to a requested point in the clip rather than pushed.
pub struct Replay {
    rx: Receiver<Frame>,
    fps: f64,
    /// the newest frame at or before the last request, and the one after it --
    /// already taken off the channel, because finding the boundary means
    /// looking past it.
    current: Option<Frame>,
    peeked: Option<Frame>,
}

impl Replay {
    fn at(&mut self, want: f64) -> Option<Frame> {
        loop {
            let next = match self.peeked.take() {
                Some(f) => f,
                // the clip ran out. holding the last frame is the honest
                // answer; the alternative is refusing to crop the end of every
                // clip, which would quietly cost the eval its final transits.
                None => match self.rx.recv() {
                    Ok(f) => f,
                    Err(_) => break,
                },
            };
            if next.is_probe() {
                continue;
            }
            if next.video_secs(self.fps) <= want {
                self.current = Some(next);
            } else {
                self.peeked = Some(next);
                break;
            }
        }
        self.current.clone()
    }
}

/// take the freshest available frame, discarding any backlog.
///
/// this is the other half of never blocking the decoder. if the consumer took
/// one frame per call it would work through a queue of stale frames, converting
/// cpu into latency and never catching up (r2.1). blocks until at least one
/// frame exists; returns `None` once the stream is finished.
pub fn latest(rx: &Receiver<Frame>) -> Option<Frame> {
    let mut frame = rx.recv().ok()?;
    while let Ok(newer) = rx.try_recv() {
        frame = newer;
    }
    Some(frame)
}

/// a frame that is already decoded and waiting, if there is one.
///
/// the loop uses this to tell "caught up" from "behind" without giving up the
/// frame it is holding. it is how the cheap stage can run on every frame while
/// the expensive one runs only on the newest: being behind is a fact about the
/// queue, not something to infer from a clock.
pub fn pending(rx: &Receiver<Frame>) -> Option<Frame> {
    rx.try_recv().ok()
}

/// what waiting a while for a frame found.
#[derive(Debug)]
pub enum Wait {
    /// a frame.
    Frame(Frame),
    /// nothing for the whole wait. the stream is still there, and the loop can
    /// use the gap to look at a clock.
    Idle,
    /// the feed has ended: there is nothing further to wait for.
    Ended,
}

/// the next frame in order, or nothing if none arrived within `poll`.
///
/// `next` blocks until the feed is over, which is the right thing until the
/// stream itself has hours to keep (r11.3): a closed window is quiet by
/// definition, and a loop parked in `recv` cannot notice that the window it is
/// waiting inside has just opened.
pub fn wait(rx: &Receiver<Frame>, poll: Duration) -> Wait {
    match rx.recv_timeout(poll) {
        Ok(f) => Wait::Frame(f),
        Err(RecvTimeoutError::Timeout) => Wait::Idle,
        Err(RecvTimeoutError::Disconnected) => Wait::Ended,
    }
}

/// take the next frame in order, skipping nothing.
///
/// the counterpart to `latest` for an offline replay, where the point is that
/// every frame is examined. paired with the blocking sends upstream, this makes
/// a replay independent of how fast the machine is, so an eval measures the
/// change rather than the hardware.
pub fn next(rx: &Receiver<Frame>) -> Option<Frame> {
    rx.recv().ok()
}

/// frames the crop feed may queue, regardless of `[stream] frame_queue_depth`.
///
/// not a knob: it follows from the feed being latest-only. see `Ingest::crops`.
const CROP_QUEUE_DEPTH: usize = 2;

/// whether a finished attempt means the feed is over, or merely interrupted.
///
/// only a source that will not restart, having ended cleanly, is over. an
/// unclean exit, a stall, or a failure to spawn at all is the camera or the
/// machine having a bad moment -- which is the case this supervisor exists for.
///
/// errors used to propagate straight out of the retry loop, which made one bad
/// moment permanent. on the gate feed that is visible, because the video stops.
/// on the crop feed it is silent: the last main-stream frame is handed out
/// forever, so the detector examines a still photograph and the harvest dedupes
/// every crop of it as the same object.
fn feed_is_over(attempt: &Result<bool>, restarts_on_exit: bool) -> bool {
    match attempt {
        Ok(clean) => *clean && !restarts_on_exit,
        // a file source that cannot be opened is over; a camera never is.
        Err(_) => !restarts_on_exit,
    }
}

fn read_frames(
    mut out: impl Read,
    width: u32,
    height: u32,
    size: usize,
    tx: SyncSender<Frame>,
    offline: bool,
) {
    let mut dropped = 0u64;
    loop {
        let mut buf = vec![0u8; size];
        // a partial read at eof is a truncated frame, never a usable one.
        if out.read_exact(&mut buf).is_err() {
            return;
        }
        // numbered by `pump`, not here: it is the one place every surviving
        // frame passes through, so its numbering is the one whose gaps tell the
        // consumer what it skipped. drops on *this* side are counted below
        // instead, because they happen before any number is assigned.
        let frame = Frame {
            width,
            height,
            data: Arc::new(buf),
            seq: 0,
            received: Instant::now(),
        };
        // never block the decoder. blocking here backpressures ffmpeg, which
        // then buffers on its rtsp input, and the whole stream drifts further
        // behind real time with no way to recover. dropping keeps it live.
        //
        // an offline replay wants the opposite: backpressure is how every frame
        // survives, and there is no real time to fall behind.
        if offline {
            if tx.send(frame).is_err() {
                return;
            }
            continue;
        }
        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                dropped += 1;
                if dropped.is_multiple_of(crate::config::DROP_REPORT_EVERY) {
                    tracing::debug!("{dropped} frames dropped; consumer is slower than the stream");
                }
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
    }
}

fn log_stderr(err: impl Read) {
    use std::io::{BufRead, BufReader};
    for line in BufReader::new(err).lines().map_while(Result::ok) {
        if !line.trim().is_empty() {
            tracing::warn!("ffmpeg: {line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **the supervisor died of its own ffmpeg failing to start.**
    ///
    /// it exists to survive a camera that reboots and a wifi link that drops,
    /// and both of those it handled. but `spawn()?` and `pump()?` propagated out
    /// of the retry loop, so one transient failure -- a fork that returns EAGAIN
    /// under load is enough -- ended the feed permanently.
    ///
    /// on the gate that would be obvious, because the video stops. on the crop
    /// feed it is silent: `LatestFrame` keeps handing out the last frame it
    /// received, so the detector goes on examining one still photograph, every
    /// detection lands in the same place, and the harvest dedupes them all as
    /// the same object. measured on the deployment: crops arriving every 0.6s
    /// stopped dead for eight minutes while the preview stayed live, because the
    /// preview is fed by the *other* ffmpeg.
    ///
    /// so an error is a reason to retry, never a reason to stop.
    #[test]
    fn a_failed_attempt_does_not_end_a_feed_that_restarts() {
        assert!(
            !feed_is_over(&Err(anyhow::anyhow!("spawning ffmpeg")), true),
            "a transient failure ended a live feed"
        );
        assert!(
            !feed_is_over(&Ok(false), true),
            "an unclean exit ends nothing: that is what a camera reboot looks like"
        );
        // a file that genuinely ran out is the one case that is over.
        assert!(feed_is_over(&Ok(true), false));
        // and the same file, asked to loop, is not.
        assert!(!feed_is_over(&Ok(true), true));
        // an error against a file source stops, rather than respawning forever
        // on a path that does not exist.
        assert!(feed_is_over(&Err(anyhow::anyhow!("no such file")), false));
    }
    use crate::config::{Camera, ClassifierCfg, DetectorCfg, Gate, HarvestCfg, PreviewCfg, Stream};

    fn cfg() -> Config {
        Config {
            camera: Camera {
                host: "h".into(),
                username: "u".into(),
                password: "p".into(),
                url: None,
                channel: 1,
                ptz_poll_secs: 0,
                ptz_moved_degrees: 1.0,
                cgi_timeout_secs: 5,
                cgi_retry_secs: 30,
            },
            stream: Stream::default(),
            preview: PreviewCfg::default(),
            gate: Gate::default(),
            detector: DetectorCfg::default(),
            harvest: HarvestCfg::default(),
            record: crate::config::RecordCfg::default(),
            classifier: ClassifierCfg::default(),
            scenery: crate::config::SceneryCfg::default(),
            track: crate::config::TrackCfg::default(),
            crop: crate::config::CropCfg::default(),
            subjects: Vec::new(),
            mqtt: None,
            ntfy: crate::config::NtfyCfg::default(),
            train: crate::config::TrainCfg::default(),
        }
    }

    /// the gate path must never rescale. rescaling was measured at 30% of a core
    /// against 6% without, so a scale filter creeping back in here is a
    /// five-times regression that nothing else would notice.
    ///
    /// colour, not luma: the preview shares this stream, and asking ffmpeg for
    /// gray made the preview grey and left it running at the crop rate.
    #[test]
    fn gate_args_take_the_native_frame_without_rescaling() {
        let ing = Ingest::gate(
            &cfg(),
            Source::Rtsp {
                url: "rtsp://x".into(),
                redacted: "rtsp://x".into(),
            },
        );
        let args = ing.ffmpeg_args().join(" ");
        assert!(args.contains("-vf format=rgb24"), "{args}");
        // this camera declares 100 tbr while sending 15 fps; without this
        // ffmpeg duplicates frames up towards the nominal rate, and everything
        // downstream pays to process copies. measured at 84 fps out of 15.
        assert!(args.contains("-fps_mode passthrough"), "{args}");
        assert!(
            !args.contains("scale="),
            "gate path must not rescale: {args}"
        );
        assert!(args.contains("-rtsp_transport tcp"), "{args}");
        assert!(args.contains("rawvideo"), "{args}");
    }

    #[test]
    fn file_args_pace_at_wallclock_and_skip_rtsp_flags() {
        let ing = Ingest::gate(&cfg(), Source::File("/tmp/x.mp4".into()));
        let args = ing.ffmpeg_args().join(" ");
        assert!(args.contains("-re"), "{args}");
        assert!(!args.contains("rtsp_transport"), "{args}");
    }

    /// a clip of a known size, reached through a url so that the rtsp path --
    /// the one that had no way to be tested -- is the one under test. none when
    /// ffmpeg is not installed, which is the same rule the e2e suite runs by.
    fn clip(width: u32, height: u32) -> Option<PathBuf> {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return None;
        }
        let path = std::env::temp_dir().join(format!(
            "metermate-size-{}x{}-{}.mp4",
            width,
            height,
            std::process::id()
        ));
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("color=c=gray:s={width}x{height}:r=15:d=1"))
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&path)
            .output()
            .ok()?;
        out.status.success().then_some(path)
    }

    fn rtsp(url: &str) -> Source {
        Source::Rtsp {
            url: url.into(),
            redacted: url.into(),
        }
    }

    /// every stream used to be read at whatever size the source code said. a
    /// 4096x1856 main stream read in 2560x1440 chunks is not an error and not a
    /// stall: it is half a frame of sheared pixels per frame, so the detector
    /// finds nothing and the harvest writes no crops while every log line
    /// reports a healthy pipeline (r5.6).
    #[test]
    fn a_stream_is_read_at_its_own_size() {
        let Some(clip) = clip(1280, 720) else {
            eprintln!("skipped: no ffmpeg");
            return;
        };
        let source = rtsp(&format!("file:{}", clip.display()));
        assert_eq!(probe_size(&source), Some((1280, 720)));
        assert_eq!(resolve_size("crop", (2560, 1440), &source), (1280, 720));

        // and the feed really reads at it: the frame size is what the reader
        // cuts the pipe into, so a size left behind here is the bug again.
        let ing = Ingest::crops(&cfg(), source, 15).at_size((1280, 720));
        assert_eq!(ing.frame_bytes(), 1280 * 720 * 3);
        std::fs::remove_file(clip).ok();
    }

    /// a camera that is off, or rebooting, cannot be probed. the declared size
    /// is then the only thing left, and saying which one is in use is the point:
    /// a fallback that is silent is a wrong size that is silent.
    #[test]
    fn an_unprobeable_stream_falls_back_to_the_declared_size() {
        let source = rtsp("rtsp://127.0.0.1:1/nothing");
        assert_eq!(probe_size(&source), None);
        assert_eq!(resolve_size("crop", (4096, 1856), &source), (4096, 1856));
    }

    /// the substream's pixels are not the scene's shape: the camera squeezes its
    /// field of view into whatever aspect the encoder offers, and the scene's
    /// real aspect is the main stream's, because the two share a field of view.
    /// 9/16 was hardcoded, which is right for a 16:9 main and puts a box above
    /// every vehicle for any other -- what a person sees as a box that will not
    /// sit down on the car it is on top of.
    #[test]
    fn the_detector_feed_undistorts_to_the_scene() {
        // the development camera: 4096x1856, so 64:29, so 640x290 of scene.
        let args = Ingest::detector(&cfg(), rtsp("rtsp://x"), 640, (4096, 1856))
            .ffmpeg_args()
            .join(" ");
        assert!(args.contains("scale=640:290"), "{args}");
        assert!(args.contains("pad=640:640:0:175"), "{args}");

        // and unchanged for a 16:9 main, which is what the hardcoded 9/16 was.
        let args = Ingest::detector(&cfg(), rtsp("rtsp://x"), 640, (2560, 1440))
            .ffmpeg_args()
            .join(" ");
        assert!(args.contains("scale=640:360,pad=640:640:0:140"), "{args}");
    }

    /// the fix for a stream that drifts behind real time: a consumer slower than
    /// the source must skip to the newest frame, not work through a backlog.
    #[test]
    fn latest_skips_the_backlog_and_returns_the_newest_frame() {
        let (tx, rx) = sync_channel(16);
        for seq in 1..=5u64 {
            tx.send(Frame {
                width: 4,
                height: 4,
                data: Arc::new(vec![0; 16]),
                seq,
                received: Instant::now(),
            })
            .unwrap();
        }
        assert_eq!(super::latest(&rx).map(|f| f.seq), Some(5));

        // and once the sender is gone, it reports the stream as finished.
        drop(tx);
        assert!(super::latest(&rx).is_none());
    }

    /// an offline replay must examine every frame, or the eval measures the
    /// machine it ran on rather than the change being tested.
    #[test]
    fn offline_replay_does_not_pace_itself_at_wallclock() {
        let live = Ingest::gate(&cfg(), Source::File("/tmp/x.mp4".into()));
        assert!(live.ffmpeg_args().contains(&"-re".to_string()));

        let offline = Ingest::gate(&cfg(), Source::File("/tmp/x.mp4".into())).offline();
        assert!(
            !offline.ffmpeg_args().contains(&"-re".to_string()),
            "offline replay is still paced at wall-clock"
        );
    }

    fn numbered(seq: u64) -> Frame {
        Frame {
            width: 4,
            height: 4,
            data: Arc::new(vec![0; 48]),
            seq,
            received: Instant::now(),
        }
    }

    /// the e2e suite asserts the two feeds stay together within a crop interval,
    /// which a one-frame slip would pass. this pins the boundary itself.
    #[test]
    fn replay_returns_the_newest_frame_at_or_before_the_asked_for_moment() {
        let (tx, rx) = sync_channel(16);
        // a feed at 4fps: frames 1..=5 cover 0.00, 0.25, 0.50, 0.75, 1.00.
        for seq in 1..=5 {
            tx.send(numbered(seq)).unwrap();
        }
        let mut r = Replay {
            rx,
            fps: 4.0,
            current: None,
            peeked: None,
        };

        assert_eq!(r.at(0.0).map(|f| f.seq), Some(1), "the clip starts at zero");
        assert_eq!(
            r.at(0.2).map(|f| f.seq),
            Some(1),
            "0.25 has not happened yet"
        );
        assert_eq!(
            r.at(0.25).map(|f| f.seq),
            Some(2),
            "a frame on the boundary"
        );
        assert_eq!(r.at(0.9).map(|f| f.seq), Some(4), "not 0.75's successor");
        // a request the feed cannot reach holds the last frame rather than
        // refusing to crop the end of every clip.
        drop(tx);
        assert_eq!(r.at(99.0).map(|f| f.seq), Some(5));
    }

    /// the gate only asks when something moved, so requests skip forward. the
    /// feed must skip with them rather than hand back a frame per call.
    #[test]
    fn replay_skips_forward_over_moments_nothing_asked_about() {
        let (tx, rx) = sync_channel(16);
        for seq in 1..=9 {
            tx.send(numbered(seq)).unwrap();
        }
        // the last frame covers exactly the moment asked for below, so without
        // an end to the feed there is no way to know nothing follows it.
        drop(tx);
        let mut r = Replay {
            rx,
            fps: 4.0,
            current: None,
            peeked: None,
        };
        assert_eq!(r.at(0.0).map(|f| f.seq), Some(1));
        assert_eq!(r.at(2.0).map(|f| f.seq), Some(9), "did not catch up");
    }

    #[test]
    fn only_rtsp_sources_restart_on_clean_exit() {
        let file = Source::File("/tmp/x.mp4".into());
        let rtsp = Source::Rtsp {
            url: "rtsp://cam/live".into(),
            redacted: "r".into(),
        };
        assert!(!file.restarts_on_exit());
        assert!(rtsp.restarts_on_exit());
    }

    /// what makes a source live is that it never runs out, and an `[camera] url`
    /// naming a recording is not that -- so it is paced at wall-clock and given
    /// no rtsp options, which ffmpeg rejects outright for any other input.
    #[test]
    fn a_url_that_is_not_rtsp_behaves_as_a_file() {
        let source = Source::Rtsp {
            url: "file:/var/clips/0.mp4".into(),
            redacted: "file:/var/clips/0.mp4".into(),
        };
        assert!(!source.restarts_on_exit(), "a recording ran out");
        let args = Ingest::gate(&cfg(), source).ffmpeg_args();
        assert!(args.contains(&"-re".to_string()), "{args:?}");
        assert!(!args.contains(&"-rtsp_transport".to_string()), "{args:?}");
    }
}
