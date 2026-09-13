//! configuration and the defaults behind it.
//!
//! every tunable lives here so retuning never needs a recompile (r5.3). defaults
//! are chosen so a config naming only a camera and a broker already works.

use crate::hours::Hours;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// the gate runs on the substream at its native size with no rescaling at all.
// measured: rescaling the main stream to gate size cost 30% of a core, while
// decoding the substream natively costs 6%. software rescaling 3.7M pixels at
// 30fps is more expensive than decoding them, so we never do it.
//
// a substream may be anamorphic (16:9 squeezed into 4:3) and that is fine here:
// frame differencing does not care about aspect ratio; only the detector does,
// and it reads main-stream pixels.
//
// these are the *declared* size, used when the stream cannot be asked its own
// (r5.6). what it answers, or the last time it could not, is what runs.
pub const GATE_WIDTH: u32 = 640;
pub const GATE_HEIGHT: u32 = 480;

// dahua-family cameras expose main as subtype 0 and sub as subtype 1.
pub const MAIN_SUBTYPE: u8 = 0;
pub const SUB_SUBTYPE: u8 = 1;
pub const DEFAULT_CHANNEL: u8 = 1;

/// how often to ask the camera where it is pointed. cheap -- one lan request --
/// and the thing it catches is silent: every position-derived assumption in the
/// pipeline goes stale the moment the camera moves.
pub const PTZ_POLL_SECS: u64 = 5;

/// degrees of reported movement treated as the camera having been repointed.
/// measured against this camera, a still camera reports the same value every
/// time, so this only has to clear whatever a firmware might round to.
pub const PTZ_MOVED_DEGREES: f32 = 1.0;

// scenery. see `src/scenery` for what each one is protecting against; the
// measurements behind them are in DESIGN.md.
pub const PARKED_AFTER_SECS: u64 = 90;
pub const FORGET_AFTER_SECS: u64 = 240;
pub const FORGET_AFTER_LOOKS: u64 = 200;
pub const MIN_OCCUPANCY: f32 = 0.5;
pub const SAME_PLACE: f32 = 0.25;

/// share of a vehicle's movement that must sit in the middle half of its box.
/// 0.20 keeps about 90% of real crops and rejects most corner intrusions.
pub const MOTION_MUST_BE_CENTRAL: f32 = 0.20;

// the gate's notion of an object. measured on this street: 70% of the regions
// reaching the detector were under 2000px with a median of 357px, mostly wire
// and glass shimmer, and each one cost an inference a real vehicle then missed.
pub const BG_SHIFT: u32 = 5;
pub const MIN_REGION_PX: u32 = 12;
pub const MIN_REGION_THICKNESS_PX: u32 = 4;
pub const MIN_COMPONENT_PX: u32 = 4;
pub const MERGE_GAP_PX: u32 = 4;
pub const MAX_REGIONS: usize = 8;

// crop geometry. a 659x1440 strip letterboxed to 640x640 scaled a 300px car to
// 133px in a mostly black frame, which is where junk classes came from.
pub const CONTEXT_MARGIN: f32 = 0.35;
pub const MIN_CROP_PX: u32 = 96;
pub const MAX_CROP_PX: u32 = 960;
pub const MAX_REGION_FRACTION: f32 = 0.40;

// how much of the surroundings a harvested crop keeps.
pub const HARVEST_CONTEXT: f32 = 0.18;
pub const CLIPPED_CONTEXT: f32 = 0.45;
pub const EDGE_SLACK_PX: u32 = 3;

// the detector's budget, and when the hybrid looks again.
pub const MAX_REGIONS_PER_FRAME: usize = 2;
pub const DETECTOR_DUTY_CYCLE: f32 = 0.6;
pub const MIN_DETECTIONS_PER_SEC: usize = 2;
pub const MAX_DETECTIONS_PER_SEC: usize = 12;
pub const INFERENCE_SMOOTHING: f32 = 0.2;
pub const REINSPECT_MS: u64 = 200;
pub const REINSPECT_TOLERANCE: f32 = 0.5;
pub const SMALL_DETECTION_PX: u32 = 96;

pub const MIN_SIGHTINGS: u32 = 3;

// the preview. cosmetic rather than behavioural -- these decide what a person
// sees, not what is detected or kept.
pub const OVERLAY_LINGER_MS: u64 = 700;
pub const OVERLAY_SAME_BOX: f32 = 0.5;
pub const PREVIEW_WIDTH: u32 = 640;
pub const PREVIEW_HEIGHT: u32 = 360;
pub const PREVIEW_JPEG_QUALITY: u8 = 70;
/// how long a viewer waits for the remuxed stream's header before being told it
/// is not there. covers ffmpeg connecting and emitting `ftyp`+`moov`, which on
/// this camera is a second or two after start; past that the stream really is
/// missing and saying so beats holding the connection.
pub const PREVIEW_START_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
pub const CROPS_LISTED: usize = 120;
/// fewer than the crops: each one is a video element rather than an image, and
/// a clip is minutes of street where a crop is one vehicle.
pub const CLIPS_LISTED: usize = 60;
pub const VIEWER_BACKLOG: usize = 240;
/// how far behind the live edge the h.264 player deliberately sits.
///
/// it exists so a fragment arriving late has something in hand to play through
/// rather than stalling. it was 2 seconds, which is a long time to be behind a
/// street you are watching for a vehicle that stops for ninety.
/// **a floor, not a target.** the player measures how late fragments actually
/// arrive and sizes the cushion from that; this is as small as it may conclude.
/// measured on this camera, fragments are nominally 61ms apart and the worst
/// lateness seen was 167ms, so 250ms covers everything observed with margin.
pub const TARGET_BUFFER_MS: u64 = 250;
/// and as large. a cushion beyond this is latency nobody agreed to: past it the
/// stream is too erratic to smooth over and reconnecting is the better answer.
pub const MAX_BUFFER_MS: u64 = 2000;
/// and the floor on how often the player may reopen the stream.
///
/// reopening costs about a second of black, so a reconnect that itself triggers
/// a reconnect stutters forever -- that is what this prevents. but it also
/// paces the retry after a `503`, which is what the server returns while ffmpeg
/// has not yet produced a header, so at 5 seconds every restart began with five
/// seconds of nothing.
pub const RECONNECT_MIN_MS: u64 = 1000;

// harvest output.
pub const HARVEST_JPEG_QUALITY: u8 = 92;

// roi validation. `suspiciously_small` is a warning threshold, not a limit.
pub const ROI_MIN_VERTICES: usize = 3;
pub const ROI_SUSPICIOUSLY_SMALL: f32 = 0.02;

// ingest. deep enough to hold the frames that arrive during one inference.
//
// it was 2, on the reasoning that a frame we are late to is worth less than the
// one behind it. that is true of the *expensive* stage and false of the gate,
// whose background model is built from every frame: at 30fps with a 130ms
// inference, four frames arrive while the loop is busy and two of them were
// being dropped by the sender before the gate ever saw them.
pub const FRAME_QUEUE_DEPTH: usize = 8;

// how often the periodic reports fire, in events rather than seconds.
pub const DEBUG_SUMMARY_EVERY: u64 = 60;
pub const HARVEST_REPORT_EVERY: u64 = 25;
pub const FRAME_REPORT_EVERY: u64 = 300;
pub const DROP_REPORT_EVERY: u64 = 150;

// camera cgi. the camera is on the lan and answers in milliseconds; these only
// bound a wedged socket and a reboot.
pub const CGI_TIMEOUT_SECS: u64 = 5;
pub const CGI_RETRY_SECS: u64 = 30;

/// most detections the model may report in one pass. a property of the exported
/// graph's nms rather than a preference.
pub const MAX_DETECTIONS: usize = 300;
pub const VELOCITY_WINDOW: usize = 8;

// the tracker. a car at 30km/h crosses ~40 gate pixels a second at this scale,
// so a box a frame apart overlaps heavily; 0.2 tolerates a jittery detection.
pub const TRACK_MIN_IOU: f32 = 0.2;
pub const TRACK_MAX_GAP_SECS: f32 = 5.0;
pub const TRACK_MAX_MISSED_LOOKS: u32 = 30;
pub const TRACK_STOPPED_BELOW: f32 = 8.0;
pub const TRACK_STOPPED_AFTER_SECS: f32 = 3.0;
pub const TRACK_CONFIRM_M: u32 = 3;
pub const TRACK_CONFIRM_N: u32 = 5;

/// the declared main size, fallen back to the same way as the gate's. crops are
/// taken at main resolution because the detector is unreliable on vehicles under
/// 48px and a far-side vehicle is only ~50px in the gate stream; the same
/// vehicle is several times wider in main pixels.
pub const MAIN_WIDTH: u32 = 2560;
pub const MAIN_HEIGHT: u32 = 1440;

/// default full-resolution frames per second pulled for cropping.
///
/// bounds **pipe bandwidth, not decode cost** -- ffmpeg decodes every frame
/// regardless -- and sets the skew between the gate frame and the crop taken
/// from it: 250ms at 4fps, 67ms at 15. measured over 600s of 2560x1440, the
/// whole range costs about a point of one core (3.7% at 4, 4.9% at 15), so it
/// is set for the skew and the bandwidth is affordable.
pub const CROP_FPS: u32 = 15;

// a frame is considered late once this elapses. ffmpeg will sit on a dead rtsp
// socket indefinitely, so liveness is judged by frame arrival, never by the
// process still being alive.
pub const FRAME_STALL_TIMEOUT_MS: u64 = 10_000;

/// how long the frame loop waits for a frame before looking at the clock.
///
/// the stream keeps hours of its own (r11.3), and a closed window is quiet by
/// definition: without a poll the loop would sit in `recv` and only notice the
/// window had changed when the street put something in front of the camera.
pub const STREAM_POLL_MS: u64 = 1_000;

// restart backoff for the ingest supervisor, doubling up to the ceiling.
pub const RESTART_BACKOFF_MIN_MS: u64 = 500;
pub const RESTART_BACKOFF_MAX_MS: u64 = 30_000;

/// config filenames, in the order they are preferred. the local file is
/// gitignored and holds real credentials, so an operator's checkout picks it up
/// without passing a flag, while the committed example stays the fallback.
pub const LOCAL_CONFIG: &str = "metermate.local.toml";
pub const DEFAULT_CONFIG: &str = "metermate.toml";

/// the detector's square input. 640 is what coco models are trained at. smaller
/// is quadratically cheaper but loses the far side of the street, which is where
/// a vehicle appears first and is already only tens of pixels wide.
pub const DETECTOR_INPUT_SIZE: u32 = 640;

pub const DEFAULT_MQTT_PORT: u16 = 1883;
pub const MQTT_KEEPALIVE_SECS: u64 = 30;
pub const MQTT_CHANNEL_CAPACITY: usize = 64;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub camera: Camera,
    #[serde(default)]
    pub stream: Stream,
    #[serde(default)]
    pub gate: Gate,
    #[serde(default)]
    pub detector: DetectorCfg,
    #[serde(default)]
    pub harvest: HarvestCfg,
    #[serde(default)]
    pub record: RecordCfg,
    #[serde(default)]
    pub classifier: ClassifierCfg,
    #[serde(default)]
    pub scenery: SceneryCfg,
    #[serde(default)]
    pub track: TrackCfg,
    #[serde(default)]
    pub crop: CropCfg,
    #[serde(default)]
    pub preview: PreviewCfg,
    /// what metermate watches for, as `[[subject]]` blocks (r10.1).
    ///
    /// empty is the ordinary case rather than a broken one: `subjects()`
    /// synthesises a single subject from `[classifier]` so a config written
    /// before this existed keeps working and there is still only one code path.
    #[serde(default, rename = "subject")]
    pub subjects: Vec<SubjectCfg>,
    /// omit the whole `[mqtt]` section to run without alerting. useful while
    /// tuning, and while no broker exists yet.
    #[serde(default)]
    pub mqtt: Option<Mqtt>,
    #[serde(default)]
    pub ntfy: NtfyCfg,
    #[serde(default)]
    pub train: TrainCfg,
}

/// the subject a config that names none is assumed to mean. the go-4 is the
/// first subject, not a special case -- this is only what keeps a config
/// written before `[[subject]]` existed working unchanged.
pub const DEFAULT_SUBJECT: &str = "go4";

/// one thing worth publishing about (r10).
///
/// there is no `references` here on purpose. every subject is named in the one
/// reference file, whose format is already `<label> <floats>` per line, and they
/// all share a single negative set -- the street supplies one pool of "none of
/// these" and splitting it per subject would multiply the labelling for nothing.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SubjectCfg {
    /// the label in the reference file, the mqtt topic segment, and the home
    /// assistant `unique_id`. see `Config::validate`.
    pub name: String,
    /// detector classes this subject can be. empty means whatever
    /// `[detector] classes` allows, which is the usual case.
    #[serde(default)]
    pub detector_classes: Vec<String>,
}

/// where the preview page gets its pixels.
///
/// the overlay is drawn in the browser from a separate json feed and rescales
/// itself to whatever it is drawn over, so the video layer can be swapped
/// without touching any of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PreviewSource {
    /// the camera's own h.264 main stream, remuxed into fragmented mp4 and
    /// played by the browser's native decoder.
    ///
    /// the default, because it is both the best picture and nearly the cheapest.
    /// the bitstream is passed through untouched -- no decode, no encode --
    /// which measured at 0.16s of cpu to remux ten minutes of 2560x1440, so the
    /// server does about as little as it does for the mjpeg it re-encodes today,
    /// and the browser gets sixteen times the pixels.
    ///
    /// it also leaves the detector alone: this is the *main* stream, whereas the
    /// camera has only one substream encoder and the gate is using it.
    ///
    /// **the objection to this has been answered.** it read: playing an endless
    /// fragmented mp4 in a plain `<video>` gives no honest measure of how far
    /// behind live it is, and doing it properly means media source extensions,
    /// where the buffer is ours to evict and seek. that was exactly right, and
    /// measured: `seekable` reported `[0, 0]` while `buffered` was
    /// `[13.67, 17.27]`, so assigning `currentTime` clamped to zero and was
    /// discarded -- latency could only be closed by playing slightly fast, old
    /// data could never be dropped, and a backgrounded window came back and
    /// replayed everything it had missed.
    ///
    /// the page now feeds the element through a `MediaSource`. seeking works,
    /// the buffer is evicted behind the playhead, and the cushion is sized from
    /// measured fragment jitter rather than guessed. measured after: 15fps,
    /// 0.3-0.6s behind, buffer steady at six seconds.
    ///
    /// still opt-in rather than default, because all of that was verified in
    /// chrome and the deployment runs firefox.
    MainH264,
    /// metermate decodes, downscales and re-encodes the gate frame to jpeg.
    ///
    /// the default: it asks least of the browser -- an `<img>` and nothing else
    /// -- and an mjpeg frame either arrives or does not, with no buffer to
    /// mismanage and no latency to lose track of. 640x360 and a jpeg encode per
    /// frame is the price of that reliability.
    #[default]
    ServerSub,
    /// the browser fetches mjpeg straight from the camera's substream.
    ///
    /// removes metermate's jpeg encoding entirely, but it is not free:
    ///
    /// - the camera reports `MaxExtraStream=1`. there is one substream encoder
    ///   and the gate is already using it, so asking it for mjpeg **changes
    ///   what the detector sees** -- this is how the gate silently dropped from
    ///   h.264 at 15fps to mjpeg at 10fps once already.
    /// - browsers refuse credentials in a subresource url, so the camera has to
    ///   allow anonymous access for this to work at all.
    /// - the browser must be able to reach the camera, which rules out a vpn.
    CameraSub,
}

/// how the h.264 player closes a gap between playback and the live edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CatchUp {
    /// seek forward, dropping whatever was between. the picture jumps and is
    /// current.
    ///
    /// the default, because this is a view of a street being watched for a
    /// vehicle that may stop for ninety seconds: a frame that is late is worth
    /// less than a frame that is now, and the frames skipped over are ones
    /// nobody was going to act on. the pipeline sees every frame regardless --
    /// this is the picture, not the detector.
    #[default]
    Skip,
    /// play 10% fast until the gap closes. every frame is shown, in order, and
    /// the picture stays late for as long as it takes to walk back.
    Speed,
}

impl CatchUp {
    pub fn as_str(self) -> &'static str {
        match self {
            CatchUp::Skip => "skip",
            CatchUp::Speed => "speed",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewCfg {
    #[serde(default)]
    pub source: PreviewSource,
    /// size metermate encodes preview frames at, when it encodes them at all.
    #[serde(default = "default_preview_width")]
    pub width: u32,
    #[serde(default = "default_preview_height")]
    pub height: u32,
    /// jpeg quality for those frames. lower is cheaper and uglier.
    #[serde(default = "default_preview_jpeg_quality")]
    pub jpeg_quality: u8,
    /// how long a detection box stays drawn after the detector last reported
    /// it, so boxes do not flicker between runs.
    #[serde(default = "default_overlay_linger_ms")]
    pub overlay_linger_ms: u64,
    /// how close two boxes must be to be redrawn as the same vehicle rather
    /// than accumulating into a thicket.
    #[serde(default = "default_overlay_same_box")]
    pub overlay_same_box: f32,
    /// draw the vehicles scenery has decided are part of the street. on, because
    /// a parked vehicle is what the pipeline chose to leave alone and seeing the
    /// decision is r8.2; off, to leave only the boxes a person might still
    /// disagree with, which is what a street full of parked cars needs.
    #[serde(default = "default_show_parked")]
    pub show_parked: bool,
    /// crops returned per page of the harvest viewer.
    #[serde(default = "default_crops_listed")]
    pub crops_listed: usize,
    /// event clips returned to the events viewer. the page says how many were
    /// kept back, so a cap can never read as an empty directory.
    #[serde(default = "default_clips_listed")]
    pub clips_listed: usize,
    /// frames a slow viewer may fall behind before it is dropped.
    #[serde(default = "default_viewer_backlog")]
    pub viewer_backlog: usize,
    /// how far behind the live edge the h.264 player sits, and how often it may
    /// reopen the stream. both are the browser's behaviour rather than the
    /// pipeline's, and both are latency the viewer feels directly.
    #[serde(default = "default_target_buffer_ms")]
    pub target_buffer_ms: u64,
    #[serde(default = "default_max_buffer_ms")]
    pub max_buffer_ms: u64,
    #[serde(default = "default_reconnect_min_ms")]
    pub reconnect_min_ms: u64,
    /// what to do when the picture falls behind the live edge.
    #[serde(default)]
    pub catch_up: CatchUp,
}

impl Default for PreviewCfg {
    fn default() -> Self {
        Self {
            source: PreviewSource::default(),
            width: default_preview_width(),
            height: default_preview_height(),
            jpeg_quality: default_preview_jpeg_quality(),
            overlay_linger_ms: default_overlay_linger_ms(),
            overlay_same_box: default_overlay_same_box(),
            show_parked: default_show_parked(),
            crops_listed: default_crops_listed(),
            clips_listed: default_clips_listed(),
            viewer_backlog: default_viewer_backlog(),
            target_buffer_ms: default_target_buffer_ms(),
            max_buffer_ms: default_max_buffer_ms(),
            reconnect_min_ms: default_reconnect_min_ms(),
            catch_up: CatchUp::default(),
        }
    }
}

/// what the preview page should play, and how.
///
/// resolved from config once at startup so the page can ask rather than guess,
/// and so adding a source does not mean editing html.
#[derive(Debug, Clone)]
pub enum PreviewVideo {
    /// metermate re-encodes the gate frame; the page uses an `<img>`.
    ServerMjpeg,
    /// the browser fetches mjpeg from the camera itself, at this url.
    CameraMjpeg(String),
    /// metermate remuxes the main stream to fragmented mp4 as a second output
    /// of the ffmpeg already decoding it for crops; the page uses a `<video>`.
    /// nothing about the camera reaches the browser, not even its address.
    ServerRemux,
}

impl PreviewSource {
    /// resolve the configured source into something the preview can serve.
    pub fn video(&self, camera: &Camera) -> PreviewVideo {
        match self {
            // credentials stay server side: this url is only ever handed to
            // ffmpeg, never to a browser.
            PreviewSource::MainH264 => PreviewVideo::ServerRemux,
            PreviewSource::ServerSub => PreviewVideo::ServerMjpeg,
            // no credentials here, deliberately. a browser will not send them
            // for a subresource anyway, so embedding them in a page served over
            // plain http would leak them for nothing.
            PreviewSource::CameraSub => PreviewVideo::CameraMjpeg(format!(
                "http://{}/cgi-bin/mjpg/video.cgi?channel={}&subtype={}",
                camera.host, camera.channel, SUB_SUBTYPE
            )),
        }
    }
}

impl PreviewVideo {
    /// what the page should point at, and which element to use.
    pub fn page_src(&self) -> (&str, &str) {
        match self {
            PreviewVideo::ServerMjpeg => ("/stream.mjpg", "mjpeg"),
            PreviewVideo::CameraMjpeg(url) => (url, "mjpeg"),
            PreviewVideo::ServerRemux => ("/stream.mp4", "mp4"),
        }
    }

    /// true when metermate has to encode jpeg itself for this source.
    pub fn needs_jpeg_encoding(&self) -> bool {
        matches!(self, PreviewVideo::ServerMjpeg)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorCfg {
    #[serde(default = "default_model_path")]
    pub model: PathBuf,
    #[serde(default = "default_input_size")]
    pub input_size: u32,
    /// below this, a detection is not reported. deliberately low by default:
    /// the raw detector mode exists to find out what confidence a go-4 actually
    /// draws, and a high floor would hide exactly that.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    #[serde(default = "default_detector_threads")]
    pub threads: usize,
    /// coco class names worth carrying past stage 1. everything else is
    /// detected and then dropped, before the classifier or the harvest sees it.
    ///
    /// **an empty list means keep everything.** the default is the vehicle
    /// classes rather than empty, so nothing changes for anyone who does not
    /// ask; but a subject need not be a vehicle (r10.3), and this is the knob
    /// that stops that decision being compiled in.
    #[serde(default = "default_detector_classes")]
    pub classes: Vec<String>,
    /// most regions the detector may examine in one frame.
    #[serde(default = "default_max_regions_per_frame")]
    pub max_regions_per_frame: usize,
    /// always spend one inference on the whole frame when the gate fires, ahead
    /// of the rate limit.
    ///
    /// on by default because the alternative was measured missing traffic: the
    /// rate limit used to be charged against the whole-frame pass, and since
    /// motion is bursty the second's allowance ran out mid-crossing -- the
    /// detector never looked at 59% of the frames the gate fired on.
    ///
    /// **it is a knob because it costs frame rate.** the detector runs inline,
    /// so this also guarantees the gate loop stalls for one inference per motion
    /// frame. where inference does not fit inside a frame period the trade
    /// inverts: a stalled gate loses frames outright, where a skipped inference
    /// only lost a look. turn it off on a host that cannot keep up, and prefer
    /// raising `threads` first.
    #[serde(default = "default_always_inspect_whole_frame")]
    pub always_inspect_whole_frame: bool,
    /// fraction of wall-clock the detector may occupy while something is
    /// moving. the rate follows from this and the *measured* inference cost,
    /// rather than being a count -- a count tuned for one machine starves
    /// another, and r5.5 says the same binary runs on both.
    #[serde(default = "default_duty_cycle")]
    pub duty_cycle: f32,
    /// and the rate is clamped to this range however fast or slow it measures.
    #[serde(default = "default_min_per_sec")]
    pub min_per_sec: usize,
    #[serde(default = "default_max_per_sec")]
    pub max_per_sec: usize,
    /// weight of the newest measurement in the rolling average of inference
    /// time. one slow frame should not halve the next second's rate.
    #[serde(default = "default_inference_smoothing")]
    pub inference_smoothing: f32,
    /// how long before the same place is worth examining again.
    #[serde(default = "default_reinspect_ms")]
    pub reinspect_ms: u64,
    /// and how close two regions must be to count as that same place.
    #[serde(default = "default_reinspect_tolerance")]
    pub reinspect_tolerance: f32,
    /// a detection narrower than this is re-examined in a magnified crop.
    /// measured: cropping adds nothing for near vehicles and about 0.09 of
    /// confidence for small ones, so it is spent only where it pays.
    #[serde(default = "default_small_detection_px")]
    pub small_detection_px: u32,
    /// most detections the model may report in one pass. a property of the
    /// exported graph's nms rather than a preference.
    #[serde(default = "default_max_detections")]
    pub max_detections: usize,
}

impl DetectorCfg {
    pub fn class_filter(&self) -> Result<crate::detect::ClassFilter> {
        crate::detect::ClassFilter::from_names(&self.classes)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifierCfg {
    /// off until references exist. metermate is useful before then: it harvests
    /// the crops the references will be built from.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_embedder_path")]
    pub model: PathBuf,
    #[serde(default = "default_references_path")]
    pub references: PathBuf,
}

impl Default for ClassifierCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_embedder_path(),
            references: default_references_path(),
        }
    }
}

/// event clips: full-resolution video of the moments something happened.
///
/// off by default, because it writes video and nobody should discover that by
/// running out of disk. see `src/record` for why the ring buffer and the
/// decide-afterwards retention are the whole design rather than details.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_record_dir")]
    pub dir: PathBuf,
    /// where derived files live: today the still frame the events tab shows for
    /// each clip. kept out of `dir` so that holds only clips -- everything here
    /// can be deleted at any time and is rebuilt on demand.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    /// seconds of stream held before a trigger. motion fires after the subject
    /// is already in frame, so without this a clip opens on a vehicle halfway
    /// across and cannot say when it first appeared.
    #[serde(default = "default_record_preroll_secs")]
    pub preroll_secs: u64,
    /// how long after the last trigger a clip stays open, so one vehicle is one
    /// clip rather than a burst of them.
    #[serde(default = "default_record_hangover_secs")]
    pub hangover_secs: u64,
    /// ceiling on a single clip, so continuous traffic cannot become one
    /// enormous file that is neither an event nor reviewable.
    #[serde(default = "default_record_max_clip_secs")]
    pub max_clip_secs: u64,
    /// disk for clips. evicted by what was found in them and only then by age,
    /// unlike the harvest -- a clip holding a recognised subject may be the only
    /// one that exists, while a crop is one of thousands.
    #[serde(default = "default_record_max_bytes")]
    pub max_bytes: u64,
    /// which outcomes are worth a file. checked when the clip closes, because
    /// that is when what it holds is known. empty keeps everything.
    #[serde(default = "default_record_keep")]
    pub keep: Vec<String>,
    /// which streams to record: any of "main" and "sub".
    ///
    /// both by default, because a clip of main alone can be watched but not
    /// replayed -- and replay is what `[record]` exists for. the substream is
    /// what the gate and the precision scan read, and it cannot be derived from
    /// main by downscaling.
    ///
    /// empty is no filter, as in `keep`: both lists live in this section and
    /// take a list of names, so they may not disagree about what empty means.
    #[serde(default = "default_record_streams")]
    pub streams: Vec<String>,
    /// when recording is allowed at all, e.g. `"07:00-19:00"` (r11.1).
    ///
    /// local time, wrapping midnight when the start is past the end. a window
    /// decides when a clip may *open*; one already open closes on its own
    /// terms, since its pre-roll and hangover are the clip. empty (the
    /// default) records whenever motion does.
    #[serde(default)]
    pub active_hours: Hours,
}

impl RecordCfg {
    pub fn records(&self, stream: &str) -> bool {
        self.streams.is_empty() || self.streams.iter().any(|s| s == stream)
    }
}

impl Default for RecordCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: default_record_dir(),
            cache_dir: default_cache_dir(),
            preroll_secs: default_record_preroll_secs(),
            hangover_secs: default_record_hangover_secs(),
            max_clip_secs: default_record_max_clip_secs(),
            max_bytes: default_record_max_bytes(),
            keep: default_record_keep(),
            streams: default_record_streams(),
            active_hours: Hours::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HarvestCfg {
    /// on by default. data collection is bound by wall-clock, so a harvester
    /// that has to be switched on is one that will be switched on too late.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_harvest_dir")]
    pub dir: PathBuf,
    /// oldest crops are deleted past this. the harvest degrades before the disk
    /// fills, because a full disk would take the alert path with it (r6.4).
    #[serde(default = "default_harvest_budget")]
    pub max_bytes: u64,
    /// crops somebody has labelled are never deleted to meet `max_bytes`.
    ///
    /// the budget deletes oldest first, so a label outlives the crop it names --
    /// measured on this camera, 1598 of 6213 labels in one set already named
    /// crops that were gone -- and a label with no crop can never be trained on
    /// again. read from the `labels.txt` beside `dir`. labelled crops still count
    /// towards the budget, so the rest of the harvest rotates faster around them.
    #[serde(default = "default_true")]
    pub keep_labelled: bool,
    /// minimum gap before the same place is worth photographing again.
    ///
    /// this is the over- versus under-capture dial. too long and a vehicle
    /// crossing the scene yields a single crop, from whichever moment happened
    /// to win; too short and the harvest fills with near-identical frames of one
    /// car. a few views of the same vehicle at different angles and distances
    /// are worth having, so this errs short.
    #[serde(default = "default_harvest_interval")]
    pub min_interval_secs: u64,
    /// how close two crops must be, as a fraction of their size, to count as
    /// the same place for that rate limit. larger means more aggressive
    /// deduplication.
    #[serde(default = "default_same_object_tolerance")]
    pub same_object_tolerance: f32,
    /// how much of a detected vehicle's own pixels must have changed before it
    /// is harvested at all.
    ///
    /// **chosen from measurement.** logged against this street, moving vehicles
    /// sit at 0.11 to 0.17 and stationary ones at 0.00 to 0.03, so this lands
    /// between them. raise it to harvest only unambiguous movement; lower it to
    /// catch vehicles creeping or stopping, at the cost of parked ones.
    #[serde(default = "default_must_have_moved")]
    pub must_have_moved: f32,
    /// and this share of that movement must lie in the middle half of the
    /// vehicle's own box.
    ///
    /// a bounding box is not a vehicle, it is a rectangle that also holds road
    /// and whatever overlaps it. a car passing a parked one intrudes on a corner
    /// of its box and the parked car is credited with the movement -- which is
    /// where the endless crops of the same jeep came from. measured over 283
    /// crops: a vehicle's own motion sits at 0.42 of its middle (p10 0.24),
    /// motion clipping a corner at 0.00 (p90 0.33).
    #[serde(default = "default_motion_must_be_central")]
    pub motion_must_be_central: f32,
    /// context kept around a harvested vehicle, as a fraction of its box. a
    /// crop flush to the bodywork gives a classifier nothing to place the
    /// vehicle against; a little kerb and lane marking is what makes it legible.
    #[serde(default = "default_harvest_context")]
    pub context: f32,
    /// and more when the detection was clipped by its crop edge, since the box
    /// is truncated and the surrounding pixels are right there in the frame.
    #[serde(default = "default_clipped_context")]
    pub clipped_context: f32,
    /// slack when deciding whether a detection is clipped by its crop edge.
    #[serde(default = "default_edge_slack_px")]
    pub edge_slack_px: u32,
    /// jpeg quality for saved crops. these are training data, so this errs high:
    /// compression artefacts are the one kind of noise the classifier would
    /// learn as signal.
    #[serde(default = "default_harvest_jpeg_quality")]
    pub jpeg_quality: u8,
    /// most vehicles to harvest from a single frame.
    ///
    /// two cars crossing at once is ordinary, and both are worth keeping, so
    /// this is not 1. it is not unbounded either: each crop costs a jpeg encode
    /// of a few hundred thousand pixels inline in the frame loop, and a frame
    /// full of traffic would stall it. the ones kept are those that moved most.
    #[serde(default = "default_max_per_frame")]
    pub max_per_frame: usize,
    /// when harvesting is allowed at all, e.g. `"07:00-19:00"` (r11.1).
    ///
    /// enforcement is a daytime phenomenon and a night of empty road spends
    /// the same disk budget as a day of traffic. local time, wrapping
    /// midnight. empty (the default) harvests whenever vehicles do.
    ///
    /// this restricts the collection of training data, never the alerting:
    /// a confirmed subject outside the window still gets its notification and
    /// the crop that proves it, because r4.5 does not keep office hours.
    #[serde(default)]
    pub active_hours: Hours,
}

impl Default for HarvestCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: default_harvest_dir(),
            max_bytes: default_harvest_budget(),
            keep_labelled: true,
            min_interval_secs: default_harvest_interval(),
            same_object_tolerance: default_same_object_tolerance(),
            must_have_moved: default_must_have_moved(),
            motion_must_be_central: default_motion_must_be_central(),
            context: default_harvest_context(),
            clipped_context: default_clipped_context(),
            edge_slack_px: default_edge_slack_px(),
            jpeg_quality: default_harvest_jpeg_quality(),
            max_per_frame: default_max_per_frame(),
            active_hours: Hours::default(),
        }
    }
}

impl Default for DetectorCfg {
    fn default() -> Self {
        Self {
            model: default_model_path(),
            input_size: default_input_size(),
            min_confidence: default_min_confidence(),
            threads: default_detector_threads(),
            classes: default_detector_classes(),
            max_regions_per_frame: default_max_regions_per_frame(),
            always_inspect_whole_frame: default_always_inspect_whole_frame(),
            duty_cycle: default_duty_cycle(),
            min_per_sec: default_min_per_sec(),
            max_per_sec: default_max_per_sec(),
            inference_smoothing: default_inference_smoothing(),
            reinspect_ms: default_reinspect_ms(),
            reinspect_tolerance: default_reinspect_tolerance(),
            small_detection_px: default_small_detection_px(),
            max_detections: default_max_detections(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    pub host: String,
    pub username: String,
    pub password: String,
    #[serde(default = "default_channel")]
    pub channel: u8,
    /// rtsp url template, for a camera that is not dahua-shaped.
    ///
    /// `{subtype}` is replaced with the stream number, so one line names both
    /// streams; `{host}`, `{channel}`, `{username}` and `{password}` are
    /// replaced too. without `{subtype}` both feeds read the same stream, which
    /// is right for a camera with only one.
    ///
    /// the schemes are ffmpeg's, so `file:/var/clips/{subtype}.mp4` points
    /// metermate at recordings through the same code a camera takes -- which is
    /// how the e2e suite exercises a non-default resolution (r5.5).
    #[serde(default)]
    pub url: Option<String>,
    /// seconds between pan/tilt position polls, or 0 to not watch (r5.2).
    /// moving the camera invalidates the background model, the roi and every
    /// scale prior, with no error anywhere unless something notices.
    #[serde(default = "default_ptz_poll_secs")]
    pub ptz_poll_secs: u64,
    /// degrees of reported movement treated as a repoint.
    ///
    /// the cost of getting this wrong is asymmetric and both ways are bad. too
    /// high and a real move goes unnoticed, leaving every place in the pipeline
    /// pointing somewhere else. too low and reported dither resets scenery over
    /// and over, so no parked vehicle ever accumulates the 90s it needs to be
    /// recognised as parked -- and the harvest fills with the same jeep.
    #[serde(default = "default_ptz_moved_degrees")]
    pub ptz_moved_degrees: f32,
    /// how long a cgi request may take before it is abandoned, and how long to
    /// back off after one fails. a camera rebooting is the normal case (r5.1).
    #[serde(default = "default_cgi_timeout_secs")]
    pub cgi_timeout_secs: u64,
    #[serde(default = "default_cgi_retry_secs")]
    pub cgi_retry_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Stream {
    /// which camera stream feeds the gate. the substream by default, because it
    /// is five times cheaper to decode and the gate does not need detail.
    #[serde(default = "default_gate_subtype")]
    pub gate_subtype: u8,
    /// gate frame size. must match the chosen stream's native size, because
    /// metermate deliberately never rescales on this path.
    /// the gate's frame size, as declared. `metermate` asks the stream what it
    /// really sends at startup and uses that, falling back here only when the
    /// stream cannot be probed -- a camera rebooting, or one that is off.
    #[serde(default = "default_gate_width")]
    pub gate_width: u32,
    #[serde(default = "default_gate_height")]
    pub gate_height: u32,
    /// the main stream's frame size, as declared, and fallen back to the same
    /// way. crops are cut from these pixels, so getting them wrong is not an
    /// error but sheared half-frames and a harvest that writes nothing.
    #[serde(default = "default_main_width")]
    pub main_width: u32,
    #[serde(default = "default_main_height")]
    pub main_height: u32,
    #[serde(default = "default_transport")]
    pub rtsp_transport: String,
    /// decode threads handed to ffmpeg. two is enough for one 1440p stream and
    /// keeps the process from claiming every core on the box.
    #[serde(default = "default_decode_threads")]
    pub decode_threads: u32,
    /// full-resolution frames per second pulled from the main stream for crops.
    ///
    /// raising this does not cost decode -- ffmpeg decodes every frame anyway --
    /// it costs pipe bandwidth, and it buys accuracy: the crop is cut from
    /// whichever main frame is newest when the gate fires, so this sets how far
    /// out of step the two can be.
    #[serde(default = "default_crop_fps")]
    pub crop_fps: u32,
    /// decoded frames that may queue before frames start being dropped. small
    /// on purpose: this is a live stream, and a frame we are late to is worth
    /// less than the one behind it.
    #[serde(default = "default_frame_queue_depth")]
    pub frame_queue_depth: usize,
    /// when the streams are pulled at all, e.g. `"07:00-19:00"` (r11.3).
    ///
    /// the harvest and the recorder have windows for what is *kept*; this is
    /// the one for what is asked of the camera. outside it the rtsp channels
    /// are closed -- both ffmpegs stopped rather than left decoding a street
    /// nobody is looking at -- and the harvest and the recorder go quiet with
    /// them, because both are driven by frames. the process stays up: what is
    /// already on disk stays browsable (r8.5, r8.6).
    ///
    /// empty (the default) watches the street whenever the machine is running.
    #[serde(default)]
    pub active_hours: Hours,
    /// drop the models as well as the channels, for a deployment that wants the
    /// memory back while the window is shut.
    ///
    /// only ever possible alongside `active_hours`, and only ever while it is
    /// shut: a shut stream delivers no frames, so nothing is looking at a frame
    /// with the weights gone. that is what makes this an unload rather than a
    /// blind mode. the models are rebuilt when the window opens, before the
    /// first frame is looked at.
    ///
    /// off by default, because the memory is doing something useful in every
    /// deployment that has not asked otherwise.
    #[serde(default)]
    pub unload_models: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    /// per-pixel absolute difference that counts a pixel as changed.
    #[serde(default = "default_diff_threshold")]
    pub diff_threshold: u8,
    /// fraction of observed pixels that must change before motion fires. the
    /// default is deliberately small because a go-4 far down the block is only a
    /// few hundred pixels.
    #[serde(default = "default_min_changed_frac")]
    pub min_changed_frac: f32,
    /// how long motion stays latched after the last qualifying frame, so one
    /// vehicle does not produce a burst of separate events.
    #[serde(default = "default_latch_ms")]
    pub latch_ms: u64,
    /// frames to skip after a reset before trusting the gate, letting the
    /// reference frame settle. also used after the camera is panned (r5.2).
    #[serde(default = "default_warmup_frames")]
    pub warmup_frames: u32,
    /// region of interest, a polygon in **gate pixels**. pixels outside it are
    /// never compared to the background. empty means the whole frame, since an
    /// roi describes where one camera points.
    #[serde(default, deserialize_with = "roi_points")]
    pub roi: Vec<[u32; 2]>,
    /// how big a vehicle is, where, in **gate pixels**. each entry is a line
    /// drawn along a vehicle at some depth: its midpoint says where, and its
    /// length says how tall a vehicle is there.
    ///
    /// **`min_changed_frac` is a fraction of the frame, and the frame is not
    /// one distance away.** measured from 532 harvested boxes on the
    /// development camera, the same vehicle covers 50-60px of height in one
    /// corner and 153px in another -- about 9x in area -- so a single
    /// changed-pixel count is far too strict where vehicles are small and far
    /// too loose where they are large. the axis is diagonal here, which is why
    /// this is drawn rather than derived: image row alone explains R^2 0.36 of
    /// the variation, column 0.50, a plane in both 0.76.
    ///
    /// empty means no correction, which is what every deployment did before
    /// this existed.
    #[serde(default, deserialize_with = "perspective_lines")]
    pub perspective: Vec<[[u32; 2]; 2]>,
    /// how fast the background model absorbs a change, as a right-shift on 8.8
    /// fixed point. larger is slower: five means a change persists about 32
    /// frames, slow enough that a vehicle stopped at the kerb keeps triggering
    /// and fast enough to ride out lighting drift.
    #[serde(default = "default_bg_shift")]
    pub bg_shift: u32,
    /// a changed region smaller than this on **both** axes is noise.
    #[serde(default = "default_min_region_px")]
    pub min_region_px: u32,
    /// and it must be this thick on its shorter axis. the anti-wire filter: a
    /// wire or sun shimmer differences as something long and one or two pixels
    /// thick, which passes any test on the longer axis. the knob most likely to
    /// need changing on a different camera.
    #[serde(default = "default_min_region_thickness_px")]
    pub min_region_thickness_px: u32,
    /// a connected blob with fewer pixels than this never becomes an object.
    #[serde(default = "default_min_component_px")]
    pub min_component_px: u32,
    /// regions closer than this on both axes are one object. a vehicle rarely
    /// differences as a single blob -- windscreen, body and shadow fragment.
    #[serde(default = "default_merge_gap_px")]
    pub merge_gap_px: u32,
    /// most regions handed downstream from one frame. more distinct moving
    /// objects than this is a scene change, not traffic.
    #[serde(default = "default_max_regions")]
    pub max_regions: usize,
    /// vertices an roi needs before it bounds anything, and the share of the
    /// frame below which one is probably a mistake -- coordinates from a
    /// different frame size look valid and watch nothing.
    #[serde(default = "default_roi_min_vertices")]
    pub roi_min_vertices: usize,
    #[serde(default = "default_roi_suspiciously_small")]
    pub roi_suspiciously_small: f32,
}

/// how a detection becomes a saved image. these decide what the classifier
/// eventually sees, so they are worth being able to sweep without a rebuild.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CropCfg {
    /// context added around a motion region before the detector looks at it.
    #[serde(default = "default_context_margin")]
    pub context_margin: f32,
    /// crops are squared and clamped to this range. a long thin crop
    /// letterboxes to mostly black and turns a 300px car into a smudge.
    #[serde(default = "default_min_crop_px")]
    pub min_crop_px: u32,
    #[serde(default = "default_max_crop_px")]
    pub max_crop_px: u32,
    /// a region covering more of the frame than this is a scene change -- the
    /// light moved, or the camera did -- and not an object to crop.
    #[serde(default = "default_max_region_fraction")]
    pub max_region_fraction: f32,
}

impl Default for CropCfg {
    fn default() -> Self {
        Self {
            context_margin: default_context_margin(),
            min_crop_px: default_min_crop_px(),
            max_crop_px: default_max_crop_px(),
            max_region_fraction: default_max_region_fraction(),
        }
    }
}

/// telling a parked vehicle from a passing one. every one of these was a
/// hardcoded constant, and each was picked from one street on one afternoon.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SceneryCfg {
    /// seconds a vehicle must hold one place before it counts as scenery.
    /// generous, because a vehicle stopping at the kerb is what enforcement
    /// does and it must stay interesting long enough to be harvested first.
    #[serde(default = "default_parked_after_secs")]
    pub parked_after_secs: u64,
    /// seconds without a sighting before a place is forgotten.
    #[serde(default = "default_forget_after_secs")]
    pub forget_after_secs: u64,
    /// **and** this many looks that missed it. both, because the gate only
    /// looks when something moves: on a quiet night wall time alone reads "no
    /// one looked" as "it has gone", and the parked vehicle is rediscovered and
    /// re-cropped on every burst of traffic.
    #[serde(default = "default_forget_after_looks")]
    pub forget_after_looks: u64,
    /// share of looks since a place became known in which something was found
    /// there. what separates a parked car from a patch of road traffic crosses.
    #[serde(default = "default_min_occupancy")]
    pub min_occupancy: f32,
    /// how close two sightings must be to be the same place, as a fraction of
    /// the larger box.
    #[serde(default = "default_same_place")]
    pub same_place: f32,
    /// sightings before a place can be called scenery at all, so one stale
    /// detection cannot condemn it.
    #[serde(default = "default_min_sightings")]
    pub min_sightings: u32,
}

impl Default for SceneryCfg {
    fn default() -> Self {
        Self {
            parked_after_secs: default_parked_after_secs(),
            forget_after_secs: default_forget_after_secs(),
            forget_after_looks: default_forget_after_looks(),
            min_occupancy: default_min_occupancy(),
            same_place: default_same_place(),
            min_sightings: default_min_sightings(),
        }
    }
}

/// following one vehicle across frames, so dwell and confirmation have a
/// subject to be about.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrackCfg {
    /// overlap between a detection and a track's last box to count as the same
    /// vehicle. iou is right here -- two boxes around one car a frame apart --
    /// even though it was rejected for matching motion regions to detections.
    #[serde(default = "default_min_iou")]
    pub min_iou: f32,
    /// drop a track unseen for this long **and** missed in this many looks.
    /// both, because the gate only looks when something moves and a quiet
    /// stretch is not evidence anything left.
    #[serde(default = "default_max_gap_secs")]
    pub max_gap_secs: f32,
    #[serde(default = "default_max_missed_looks")]
    pub max_missed_looks: u32,
    /// gate pixels per second below which a vehicle counts as stopped, which is
    /// when dwell starts accumulating.
    #[serde(default = "default_stopped_below")]
    pub stopped_below_px_per_sec: f32,
    /// how long a vehicle must be stationary before it counts as stopped.
    #[serde(default = "default_stopped_after_secs")]
    pub stopped_after_secs: f32,
    /// a subject is confirmed after `confirm_m` of the last `confirm_n` looks
    /// agreed. one frame is not evidence (r1.4).
    #[serde(default = "default_confirm_m")]
    pub confirm_m: u32,
    #[serde(default = "default_confirm_n")]
    pub confirm_n: u32,
    /// how many recent positions the speed estimate averages over. one frame
    /// apart is mostly quantisation noise on a box that jitters by a pixel.
    #[serde(default = "default_velocity_window")]
    pub velocity_window: usize,
}

impl Default for TrackCfg {
    fn default() -> Self {
        Self {
            min_iou: default_min_iou(),
            max_gap_secs: default_max_gap_secs(),
            max_missed_looks: default_max_missed_looks(),
            stopped_below_px_per_sec: default_stopped_below(),
            stopped_after_secs: default_stopped_after_secs(),
            confirm_m: default_confirm_m(),
            confirm_n: default_confirm_n(),
            velocity_window: default_velocity_window(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Mqtt {
    pub host: String,
    #[serde(default = "default_mqtt_port")]
    pub port: u16,
    #[serde(default = "default_client_id")]
    pub client_id: String,
    #[serde(default = "default_base_topic")]
    pub base_topic: String,
    #[serde(default = "default_discovery_prefix")]
    pub discovery_prefix: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
}

/// a phone, without a broker or a bridge in between.
///
/// **mqtt is the integration surface and ntfy is the doorbell.** a rule engine
/// wants the facts -- track, box, dwell -- and a person crossing the room wants
/// a sentence and a picture. this is the second one, and it is deliberately not
/// a second copy of the payload: `server` is usually somebody else's machine.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NtfyCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_ntfy_server")]
    pub server: String,
    /// **no default, on purpose.** an ntfy topic is a password: anyone who
    /// knows it reads every alert. a guessable one would publish the street's
    /// comings and goings to whoever subscribes.
    #[serde(default)]
    pub topic: String,
    /// bearer token for a server that wants one. belongs in
    /// `metermate.local.toml`, which is gitignored, beside the camera password.
    #[serde(default)]
    pub token: String,
    /// which outcomes reach a phone. a sighting is the earliest warning and
    /// the point of the project; the rest are mostly of interest to a rule.
    #[serde(default = "default_ntfy_outcomes")]
    pub outcomes: Vec<String>,
    /// ntfy's own scale: `min`, `low`, `default`, `high`, `urgent`. `high` is
    /// what wakes a phone that is face down on a table.
    #[serde(default = "default_ntfy_priority")]
    pub priority: String,
    /// the preview's address, at whatever the phone can actually reach: the tap
    /// is aimed past it at the verdict the notification attached (r4.5 guarantees
    /// that crop is on the page), and at the verdicts themselves for an outcome
    /// with no crop of its own. empty sends no link at all.
    #[serde(default)]
    pub click: String,
}

impl Default for NtfyCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            server: default_ntfy_server(),
            topic: String::new(),
            token: String::new(),
            outcomes: default_ntfy_outcomes(),
            priority: default_ntfy_priority(),
            click: String::new(),
        }
    }
}

fn default_ntfy_server() -> String {
    "https://ntfy.sh".to_string()
}

fn default_ntfy_outcomes() -> Vec<String> {
    vec!["sighting".to_string()]
}

fn default_ntfy_priority() -> String {
    "high".to_string()
}

/// where the work between labelling and a trained file happens.
///
/// **two commands, because there are two costs.** `--prepare` reads clips with
/// the detector, which is minutes per passage; `--retrain` embeds and measures,
/// which is minutes per thousand crops. keeping them apart means a day of
/// labelling ends in one cheap command rather than waiting for both.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainCfg {
    /// where `--prepare` cuts a selection into, and where `--retrain` looks
    /// for sets besides the live harvest.
    #[serde(default = "default_sets_dir")]
    pub sets: PathBuf,
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            sets: default_sets_dir(),
        }
    }
}

fn default_sets_dir() -> PathBuf {
    PathBuf::from("sets")
}

/// whether a name may be a subject.
///
/// stricter than "a legal topic segment" on purpose. `123` and `-x` are both
/// fine in mqtt and both make poor home assistant ids, and an entity id
/// outlives the config that created it -- it stays in someone's dashboard long
/// after the subject is renamed.
///
/// **shared with the live page**, which can name a subject that is not in the
/// config yet: a name invented in a text field must not be one the config
/// would later refuse, or the labels are stranded.
pub fn is_subject_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.ends_with('-')
        && !name.contains("--")
}

/// ntfy's priority scale, as the server spells it.
pub const NTFY_PRIORITIES: [&str; 5] = ["min", "low", "default", "high", "urgent"];

/// how long a notification may take before it is abandoned.
///
/// **r2.2**: nothing here may hold up an alert. the send is off the pipeline
/// thread already, but a worker blocked forever on a dead server is a queue
/// that fills and then drops the next real sighting.
pub const NTFY_TIMEOUT_SECS: u64 = 10;

/// how many notifications may be waiting to go out.
///
/// small on purpose: these are real-time alerts, and a backlog of them is a
/// phone buzzing about vehicles that left ten minutes ago. when the queue is
/// full the newest is dropped and said so in the log.
pub const NTFY_QUEUE: usize = 8;

/// pick a config file. an explicit path always wins, including when it is
/// missing, so a typo fails loudly instead of silently loading something else.
pub fn resolve_path(explicit: Option<PathBuf>, dir: &Path) -> PathBuf {
    if let Some(p) = explicit {
        return p;
    }
    let local = dir.join(LOCAL_CONFIG);
    if local.is_file() {
        return local;
    }
    dir.join(DEFAULT_CONFIG)
}

impl Config {
    /// parse and validate without a file, so the rules can be tested without
    /// writing one per case.
    pub fn parse_str(raw: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(raw).context("parsing config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        Self::parse_str(&raw).with_context(|| format!("in config {}", path.display()))
    }

    /// **every ntfy setting is checked before anything is sent**, because the
    /// failure mode is silence: a priority the server does not recognise, or an
    /// outcome that never happens, produces a deployment that looks configured
    /// and never buzzes. the one place that is not true is the token, which
    /// only the server can judge -- `--notify-test` is for that.
    fn validate_ntfy(&self) -> Result<()> {
        let n = &self.ntfy;
        if !n.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            !n.topic.trim().is_empty(),
            "[ntfy] topic is required when ntfy is enabled: it is the only thing standing \
             between these alerts and anyone who guesses it, so there is no default"
        );
        anyhow::ensure!(
            n.server.starts_with("http://") || n.server.starts_with("https://"),
            "[ntfy] server {:?} must start with http:// or https://",
            n.server
        );
        anyhow::ensure!(
            NTFY_PRIORITIES.contains(&n.priority.as_str()),
            "[ntfy] priority {:?} is not one ntfy knows: {}",
            n.priority,
            NTFY_PRIORITIES.join(", ")
        );
        let known = crate::alert::Outcome::slugs();
        for want in &n.outcomes {
            anyhow::ensure!(
                known.contains(&want.as_str()),
                "[ntfy] outcomes names {want:?}, which is not an outcome: {}",
                known.join(", ")
            );
        }
        Ok(())
    }

    /// what to watch for, with `[classifier]` standing in when nothing is named.
    ///
    /// one code path whether or not anyone has written a `[[subject]]` block:
    /// the go-4 is the first subject, not a special case (r10).
    pub fn subjects(&self) -> Vec<SubjectCfg> {
        if self.subjects.is_empty() {
            return vec![SubjectCfg {
                name: DEFAULT_SUBJECT.to_string(),
                detector_classes: Vec::new(),
            }];
        }
        self.subjects.clone()
    }

    /// just the names, for generating topics and discovery entities.
    pub fn subject_names(&self) -> Vec<String> {
        self.subjects().into_iter().map(|s| s.name).collect()
    }

    fn validate(&self) -> Result<()> {
        self.validate_ntfy()?;
        // the name becomes a topic segment and a home assistant unique_id, so a
        // space in it would produce `metermate/street sweeper/sighting`. the
        // harvester learned this with crop filenames and sanitises on write;
        // here it is refused, because a subject name is written by a person and
        // silently rewriting it would make the config disagree with the topic.
        for s in &self.subjects {
            anyhow::ensure!(
                is_subject_name(&s.name),
                "subject name {:?} must start with a letter and hold only lowercase letters, \
                 digits and single dashes: it becomes an mqtt topic segment and a home \
                 assistant id, and the id outlives the config",
                s.name
            );
            anyhow::ensure!(
                s.name != crate::label::NEGATIVE && s.name != crate::label::UNCLEAR,
                "{:?} is reserved: it names the shared negative set rather than a subject",
                s.name
            );
            crate::detect::ClassFilter::from_names(&s.detector_classes)
                .with_context(|| format!("subject {:?}", s.name))?;
        }
        let mut seen: Vec<&str> = self.subjects.iter().map(|s| s.name.as_str()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        anyhow::ensure!(seen.len() == before, "two subjects share a name");

        // refused rather than ignored: a typo here silently keeps nothing, and
        // an events directory that stays empty looks exactly like a quiet street.
        for name in &self.record.keep {
            anyhow::ensure!(
                crate::record::Worth::parse(name).is_some(),
                "record.keep has {name:?}, which is not an outcome. it must be one of \
                 motion, detection or subject, or the list must be empty to keep every clip"
            );
        }

        // the same reasoning, and the cost of a typo is quieter still: recording
        // main but not sub leaves clips that play perfectly and cannot be
        // replayed, which is not noticed until an eval wants one.
        for name in &self.record.streams {
            anyhow::ensure!(
                name == crate::record::MAIN_STREAM || name == crate::record::SUB_STREAM,
                "record.streams has {name:?}, which is not a stream. it must be one of \
                 {:?} or {:?}",
                crate::record::MAIN_STREAM,
                crate::record::SUB_STREAM
            );
        }

        // a line of no length says a vehicle there is zero pixels tall, which
        // is not a scale -- it is a click that was never dragged, and taking it
        // literally would divide by it.
        for [a, b] in &self.gate.perspective {
            anyhow::ensure!(
                a != b,
                "a perspective line at {},{} has no length; it should be drawn \
                 along a vehicle, so that how long it is says how big one is there",
                a[0],
                a[1]
            );
        }

        anyhow::ensure!(
            self.gate.min_changed_frac > 0.0 && self.gate.min_changed_frac <= 1.0,
            "gate.min_changed_frac must be in (0, 1], got {}",
            self.gate.min_changed_frac
        );
        anyhow::ensure!(
            self.stream.gate_width > 0 && self.stream.gate_height > 0,
            "stream gate dimensions must be non-zero"
        );
        anyhow::ensure!(
            self.stream.main_width > 0 && self.stream.main_height > 0,
            "stream main dimensions must be non-zero"
        );
        Ok(())
    }
}

impl Camera {
    /// rtsp url for one of this camera's streams. the password is
    /// percent-encoded because dahua passwords routinely contain characters
    /// that are reserved in a url.
    pub fn rtsp_url(&self, subtype: u8) -> String {
        match &self.url {
            Some(template) => self.expand(template, subtype),
            None => self.expand(
                "rtsp://{user}:{password}@{host}:554/cam/realmonitor?channel={channel}&subtype={subtype}",
                subtype,
            ),
        }
    }

    fn expand(&self, template: &str, subtype: u8) -> String {
        template
            .replace("{subtype}", &subtype.to_string())
            .replace("{channel}", &self.channel.to_string())
            .replace("{host}", &self.host)
            .replace("{user}", &percent_encode(&self.username))
            .replace("{username}", &percent_encode(&self.username))
            .replace("{password}", &percent_encode(&self.password))
    }

    /// same url with the credentials masked, for logs.
    ///
    /// masked by replacing the userinfo rather than by rebuilding the url, so an
    /// `[camera] url` template with its own credentials in it cannot leak: the
    /// password is redacted wherever it appears, and a template that spells it
    /// out twice still comes out with both hidden.
    pub fn rtsp_url_redacted(&self, subtype: u8) -> String {
        let url = self.rtsp_url(subtype);
        match (url.find("://"), url.rfind('@')) {
            (Some(scheme), Some(at)) if scheme < at => {
                let userinfo = &url[scheme + 3..at];
                let name = userinfo.split(':').next().unwrap_or("");
                format!("{}{name}:***@{}", &url[..scheme + 3], &url[at + 1..])
            }
            _ => url,
        }
    }
}

/// percent-encode everything outside the unreserved set from rfc 3986. narrow on
/// purpose: over-encoding is harmless in a userinfo field, under-encoding is not.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

impl Default for Stream {
    fn default() -> Self {
        Self {
            gate_subtype: default_gate_subtype(),
            gate_width: GATE_WIDTH,
            gate_height: GATE_HEIGHT,
            main_width: MAIN_WIDTH,
            main_height: MAIN_HEIGHT,
            rtsp_transport: default_transport(),
            decode_threads: default_decode_threads(),
            crop_fps: default_crop_fps(),
            frame_queue_depth: default_frame_queue_depth(),
            active_hours: Hours::default(),
            unload_models: false,
        }
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            diff_threshold: default_diff_threshold(),
            min_changed_frac: default_min_changed_frac(),
            latch_ms: default_latch_ms(),
            warmup_frames: default_warmup_frames(),
            roi: Vec::new(),
            perspective: Vec::new(),
            bg_shift: default_bg_shift(),
            min_region_px: default_min_region_px(),
            min_region_thickness_px: default_min_region_thickness_px(),
            min_component_px: default_min_component_px(),
            merge_gap_px: default_merge_gap_px(),
            max_regions: default_max_regions(),
            roi_min_vertices: default_roi_min_vertices(),
            roi_suspiciously_small: default_roi_suspiciously_small(),
        }
    }
}

/// roi vertices, tolerating ones outside the frame.
///
/// the editor draws over a live view, so a point dragged onto the very edge
/// lands a pixel or two past it -- `[640, -3]` on a 640x480 gate. refusing to
/// start over that is hostile and, worse, the message points at the toml rather
/// than at the editor that wrote it.
///
/// a polygon is a region, not a list of exact pixels: a vertex beyond the
/// boundary means "right up to the boundary", so negatives clamp to zero. the
/// far side needs nothing, because the mask only ever asks about pixels that
/// are inside the frame to begin with.
fn roi_points<'de, D>(d: D) -> Result<Vec<[u32; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Vec::<[i64; 2]>::deserialize(d)?
        .into_iter()
        .map(|[x, y]| [x.max(0) as u32, y.max(0) as u32])
        .collect())
}

/// the same clamping as `roi_points`, for the same reason: the editor draws
/// these by dragging, and a drag past the edge of the frame produced negative
/// coordinates that stopped the config loading outright.
fn perspective_lines<'de, D>(d: D) -> Result<Vec<[[u32; 2]; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Vec::<[[i64; 2]; 2]>::deserialize(d)?
        .into_iter()
        .map(|[[ax, ay], [bx, by]]| {
            [
                [ax.max(0) as u32, ay.max(0) as u32],
                [bx.max(0) as u32, by.max(0) as u32],
            ]
        })
        .collect())
}

fn default_record_dir() -> PathBuf {
    PathBuf::from("data/events")
}
fn default_cache_dir() -> PathBuf {
    PathBuf::from("data/cache")
}
fn default_record_preroll_secs() -> u64 {
    crate::record::PREROLL.as_secs()
}
fn default_record_hangover_secs() -> u64 {
    crate::record::HANGOVER.as_secs()
}
fn default_record_max_clip_secs() -> u64 {
    crate::record::MAX_CLIP.as_secs()
}
fn default_record_max_bytes() -> u64 {
    // 20 GB, and the larger of the two budgets on purpose.
    //
    // **a clip can regenerate a crop; no crop can reconstruct a clip.** both
    // streams are recorded, so an event replays through the whole pipeline --
    // `tools/endtoend.py` takes the substream as its clip and the main stream
    // as `--crop-source` -- and every crop of that passage can be cut again
    // under whatever the harvest rules are that day. the harvest is derived
    // data; the clips are the source. so when the disk is tight it is the
    // derived half that should be evicted first, which means giving it the
    // smaller share.
    20 * 1024 * 1024 * 1024
}

fn default_channel() -> u8 {
    DEFAULT_CHANNEL
}
fn default_ptz_poll_secs() -> u64 {
    PTZ_POLL_SECS
}
fn default_ptz_moved_degrees() -> f32 {
    PTZ_MOVED_DEGREES
}
fn default_bg_shift() -> u32 {
    BG_SHIFT
}
fn default_min_region_px() -> u32 {
    MIN_REGION_PX
}
fn default_min_region_thickness_px() -> u32 {
    MIN_REGION_THICKNESS_PX
}
fn default_min_component_px() -> u32 {
    MIN_COMPONENT_PX
}
fn default_merge_gap_px() -> u32 {
    MERGE_GAP_PX
}
fn default_max_regions() -> usize {
    MAX_REGIONS
}
fn default_context_margin() -> f32 {
    CONTEXT_MARGIN
}
fn default_min_crop_px() -> u32 {
    MIN_CROP_PX
}
fn default_max_crop_px() -> u32 {
    MAX_CROP_PX
}
fn default_max_region_fraction() -> f32 {
    MAX_REGION_FRACTION
}
fn default_harvest_context() -> f32 {
    HARVEST_CONTEXT
}
fn default_clipped_context() -> f32 {
    CLIPPED_CONTEXT
}
fn default_edge_slack_px() -> u32 {
    EDGE_SLACK_PX
}
fn default_always_inspect_whole_frame() -> bool {
    true
}

fn default_max_regions_per_frame() -> usize {
    MAX_REGIONS_PER_FRAME
}
fn default_duty_cycle() -> f32 {
    DETECTOR_DUTY_CYCLE
}
fn default_min_per_sec() -> usize {
    MIN_DETECTIONS_PER_SEC
}
fn default_max_per_sec() -> usize {
    MAX_DETECTIONS_PER_SEC
}
fn default_inference_smoothing() -> f32 {
    INFERENCE_SMOOTHING
}
fn default_reinspect_ms() -> u64 {
    REINSPECT_MS
}
fn default_reinspect_tolerance() -> f32 {
    REINSPECT_TOLERANCE
}
fn default_small_detection_px() -> u32 {
    SMALL_DETECTION_PX
}
fn default_min_sightings() -> u32 {
    MIN_SIGHTINGS
}
fn default_velocity_window() -> usize {
    VELOCITY_WINDOW
}
fn default_preview_width() -> u32 {
    PREVIEW_WIDTH
}
fn default_preview_height() -> u32 {
    PREVIEW_HEIGHT
}
fn default_preview_jpeg_quality() -> u8 {
    PREVIEW_JPEG_QUALITY
}
fn default_overlay_linger_ms() -> u64 {
    OVERLAY_LINGER_MS
}
fn default_overlay_same_box() -> f32 {
    OVERLAY_SAME_BOX
}
fn default_show_parked() -> bool {
    true
}
fn default_crops_listed() -> usize {
    CROPS_LISTED
}
fn default_clips_listed() -> usize {
    CLIPS_LISTED
}
/// **not motion.** on a street with parked cars permanently in frame, a
/// motion-only clip is one where the gate fired and nothing was ever
/// established -- light moving, a pedestrian, a wire. measured on the deployed
/// box, those are the clips with no vehicle in them at all. `subject` is in the
/// default as well as `detection` because it is a separate rung rather than a
/// stronger form of it, and a clip holding a recognised go-4 is the one clip
/// this project exists to keep.
fn default_record_streams() -> Vec<String> {
    vec![
        crate::record::MAIN_STREAM.to_string(),
        crate::record::SUB_STREAM.to_string(),
    ]
}
fn default_record_keep() -> Vec<String> {
    vec![
        crate::record::Worth::Detection.slug().to_string(),
        crate::record::Worth::Subject.slug().to_string(),
    ]
}
fn default_viewer_backlog() -> usize {
    VIEWER_BACKLOG
}
fn default_target_buffer_ms() -> u64 {
    TARGET_BUFFER_MS
}
fn default_max_buffer_ms() -> u64 {
    MAX_BUFFER_MS
}
fn default_reconnect_min_ms() -> u64 {
    RECONNECT_MIN_MS
}
fn default_harvest_jpeg_quality() -> u8 {
    HARVEST_JPEG_QUALITY
}
fn default_roi_min_vertices() -> usize {
    ROI_MIN_VERTICES
}
fn default_roi_suspiciously_small() -> f32 {
    ROI_SUSPICIOUSLY_SMALL
}
fn default_max_detections() -> usize {
    MAX_DETECTIONS
}
fn default_frame_queue_depth() -> usize {
    FRAME_QUEUE_DEPTH
}
fn default_cgi_timeout_secs() -> u64 {
    CGI_TIMEOUT_SECS
}
fn default_cgi_retry_secs() -> u64 {
    CGI_RETRY_SECS
}
fn default_min_iou() -> f32 {
    TRACK_MIN_IOU
}
fn default_max_gap_secs() -> f32 {
    TRACK_MAX_GAP_SECS
}
fn default_max_missed_looks() -> u32 {
    TRACK_MAX_MISSED_LOOKS
}
fn default_stopped_below() -> f32 {
    TRACK_STOPPED_BELOW
}
fn default_stopped_after_secs() -> f32 {
    TRACK_STOPPED_AFTER_SECS
}
fn default_confirm_m() -> u32 {
    TRACK_CONFIRM_M
}
fn default_confirm_n() -> u32 {
    TRACK_CONFIRM_N
}
fn default_parked_after_secs() -> u64 {
    PARKED_AFTER_SECS
}
fn default_forget_after_secs() -> u64 {
    FORGET_AFTER_SECS
}
fn default_forget_after_looks() -> u64 {
    FORGET_AFTER_LOOKS
}
fn default_min_occupancy() -> f32 {
    MIN_OCCUPANCY
}
fn default_same_place() -> f32 {
    SAME_PLACE
}
fn default_gate_subtype() -> u8 {
    SUB_SUBTYPE
}
fn default_main_width() -> u32 {
    MAIN_WIDTH
}

fn default_main_height() -> u32 {
    MAIN_HEIGHT
}

fn default_gate_width() -> u32 {
    GATE_WIDTH
}
fn default_gate_height() -> u32 {
    GATE_HEIGHT
}
fn default_transport() -> String {
    "tcp".to_string()
}
fn default_decode_threads() -> u32 {
    2
}
fn default_crop_fps() -> u32 {
    CROP_FPS
}
fn default_diff_threshold() -> u8 {
    25
}
fn default_min_changed_frac() -> f32 {
    0.002
}
fn default_latch_ms() -> u64 {
    2_000
}
fn default_warmup_frames() -> u32 {
    15
}
fn default_embedder_path() -> PathBuf {
    PathBuf::from("models/embedder.onnx")
}
fn default_references_path() -> PathBuf {
    // a directory of `<subject>/{references,negatives}.txt`, not a file: the
    // unit of training is a subject, so one is written, shipped and erased on
    // its own.
    PathBuf::from("trained")
}
fn default_true() -> bool {
    true
}
fn default_harvest_dir() -> PathBuf {
    PathBuf::from("data/crops")
}
fn default_harvest_budget() -> u64 {
    // 10 GB, the smaller of the two: see `default_record_max_bytes`. crops are
    // cut from clips, so this is the half that can be rebuilt. still weeks of a
    // busy street at the rates this camera produces.
    10 * 1024 * 1024 * 1024
}
fn default_harvest_interval() -> u64 {
    // five seconds, not twenty. a vehicle takes a few seconds to cross, so
    // twenty meant one crop per passage and sometimes the wrong one. this
    // yields several views of the same vehicle, which is the preferred failure.
    5
}
fn default_same_object_tolerance() -> f32 {
    0.5
}
fn default_motion_must_be_central() -> f32 {
    MOTION_MUST_BE_CENTRAL
}
fn default_must_have_moved() -> f32 {
    0.08
}
fn default_max_per_frame() -> usize {
    4
}
fn default_model_path() -> PathBuf {
    PathBuf::from("models/detector.onnx")
}
fn default_input_size() -> u32 {
    DETECTOR_INPUT_SIZE
}
fn default_min_confidence() -> f32 {
    0.25
}
fn default_detector_threads() -> usize {
    2
}
fn default_detector_classes() -> Vec<String> {
    crate::detect::DEFAULT_CLASSES
        .iter()
        .map(|id| crate::detect::class_name(*id).to_string())
        .collect()
}
fn default_mqtt_port() -> u16 {
    DEFAULT_MQTT_PORT
}
fn default_client_id() -> String {
    "metermate".to_string()
}
fn default_base_topic() -> String {
    "metermate".to_string()
}
fn default_discovery_prefix() -> String {
    "homeassistant".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera {
            host: "10.1.150.205".into(),
            username: "metermate".into(),
            password: "m3t3rm4!".into(),
            url: None,
            channel: 1,
            ptz_poll_secs: 0,
            ptz_moved_degrees: 1.0,
            cgi_timeout_secs: 5,
            cgi_retry_secs: 30,
        }
    }

    /// the default must keep the browser talking only to metermate. anything
    /// else needs the camera reachable from wherever the page is open, which is
    /// not true over a vpn.
    /// the default is the mjpeg the server encodes. it is the least capable
    /// option and the only one with no buffer to mismanage: an `<img>` either
    /// shows the newest frame or it does not. `main-h264` is better in every way
    /// except that, and worse in practice until it is driven through media
    /// source extensions rather than a plain `<video>`.
    #[test]
    fn the_preview_defaults_to_the_reliable_mjpeg_path() {
        assert_eq!(PreviewCfg::default().source, PreviewSource::ServerSub);
        let video = PreviewSource::default().video(&camera());
        assert!(matches!(video, PreviewVideo::ServerMjpeg));
        assert!(video.needs_jpeg_encoding());
        assert_eq!(video.page_src(), ("/stream.mjpg", "mjpeg"));
    }

    /// the default reads the **main** stream, which the crop ingest is already
    /// pulling. the camera has one substream encoder and the gate is using it,
    /// so a preview on the substream changes what the detector sees; this one
    /// cannot, and it opens no session of its own either.
    #[test]
    fn the_default_preview_does_not_touch_the_gates_stream() {
        assert!(matches!(
            PreviewSource::MainH264.video(&camera()),
            PreviewVideo::ServerRemux
        ));
    }

    #[test]
    fn the_camera_source_points_at_the_substream_mjpeg_endpoint() {
        let PreviewVideo::CameraMjpeg(url) = PreviewSource::CameraSub.video(&camera()) else {
            panic!("expected a camera url");
        };
        assert!(url.contains("/cgi-bin/mjpg/video.cgi"), "{url}");
        assert!(url.contains("subtype=1"), "{url}");
    }

    /// the url is handed to a browser and embedded in a page served over plain
    /// http. a browser will not send credentials for a subresource anyway, so
    /// putting them there would leak them for no benefit.
    #[test]
    fn the_camera_url_carries_no_credentials() {
        let PreviewVideo::CameraMjpeg(url) = PreviewSource::CameraSub.video(&camera()) else {
            panic!("expected a camera url");
        };
        let c = camera();
        assert!(!url.contains(&c.password), "password leaked into {url}");
        assert!(!url.contains(&c.username), "username leaked into {url}");
        assert!(!url.contains('@'), "credentials leaked into {url}");
    }

    #[test]
    fn the_preview_source_is_selected_by_name_in_config() {
        let cfg: PreviewCfg = toml::from_str(r#"source = "camera-sub""#).unwrap();
        assert_eq!(cfg.source, PreviewSource::CameraSub);
        let cfg: PreviewCfg = toml::from_str(r#"source = "server-sub""#).unwrap();
        assert_eq!(cfg.source, PreviewSource::ServerSub);
        let cfg: PreviewCfg = toml::from_str(r#"source = "main-h264""#).unwrap();
        assert_eq!(cfg.source, PreviewSource::MainH264);
    }

    /// the page is told a local path. the camera's address and credentials stay
    /// on the server, which is what lets this mode work from outside the lan
    /// where `camera-sub` cannot.
    #[test]
    fn the_default_tells_the_page_nothing_about_the_camera() {
        let video = PreviewSource::MainH264.video(&camera());
        let (src, kind) = video.page_src();
        assert_eq!((src, kind), ("/stream.mp4", "mp4"));
        assert!(!src.contains(&camera().host), "{src}");
        assert!(!src.contains(&camera().password), "{src}");
    }

    /// serialise a defaults struct into a toml table, so the example can be
    /// checked key by key against the type rather than against a list someone
    /// has to remember to update.
    fn to_table<T: Serialize>(value: &T) -> toml::Table {
        toml::Value::try_from(value)
            .expect("config sections serialise")
            .as_table()
            .expect("a config section is a table")
            .clone()
    }

    /// the shipped example is the documentation. if it stops parsing, or names a
    /// key that no longer exists, the docs are wrong and nothing else would say
    /// so -- serde ignores unknown keys, so a renamed setting would silently
    /// become a comment.
    #[test]
    fn the_documented_example_config_still_loads() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/metermate.toml"))
            .expect("metermate.toml is missing");
        let cfg: Config = toml::from_str(&text).expect("the documented example does not parse");

        // **every** settable key must appear, derived from the types rather than
        // listed here. an earlier version of this test enumerated field names by
        // hand and so sailed straight past `crop_fps` being added to the code and
        // not to the file -- which is the exact drift it existed to catch.
        //
        // the *sections* were still listed by hand though, which is the same bug
        // one level up: `[crop]`, `[scenery]`, `[track]` and `[record]` were all
        // added to the code and to the file while this test watched none of
        // them. a section missing from here is a section whose documented values
        // are free to drift from its defaults.
        let documented: toml::Table = toml::from_str(&text).unwrap();
        let expected: Vec<(&str, toml::Table)> = vec![
            ("stream", to_table(&Stream::default())),
            ("gate", to_table(&Gate::default())),
            ("crop", to_table(&CropCfg::default())),
            ("detector", to_table(&DetectorCfg::default())),
            ("harvest", to_table(&HarvestCfg::default())),
            ("record", to_table(&RecordCfg::default())),
            ("scenery", to_table(&SceneryCfg::default())),
            ("track", to_table(&TrackCfg::default())),
            ("classifier", to_table(&ClassifierCfg::default())),
            ("preview", to_table(&PreviewCfg::default())),
        ];
        for (section, defaults) in expected {
            let shown = documented
                .get(section)
                .and_then(|v| v.as_table())
                .unwrap_or_else(|| panic!("metermate.toml has no [{section}] section"));
            for (key, value) in &defaults {
                let found = shown.get(key).unwrap_or_else(|| {
                    panic!("metermate.toml does not document [{section}] {key}")
                });
                // floats round-trip through f32, so `0.002` comes back as
                // 0.0020000000949949026. compare those by value, not by text.
                let same = match (found.as_float(), value.as_float()) {
                    (Some(a), Some(b)) => (a - b).abs() < 1e-6,
                    _ => found == value,
                };
                assert!(
                    same,
                    "[{section}] {key} is documented as {found} but defaults to {value}"
                );
            }
        }

        // the file states every default outright rather than commenting them
        // out, so each one is checked against the code. a default changed
        // without the example following is then a failing test rather than a
        // quietly wrong document.
        let (stream, gate, det, harv, cls) = (
            Stream::default(),
            Gate::default(),
            DetectorCfg::default(),
            HarvestCfg::default(),
            ClassifierCfg::default(),
        );

        assert_eq!(cfg.camera.channel, DEFAULT_CHANNEL);

        assert_eq!(cfg.stream.gate_subtype, stream.gate_subtype);
        assert_eq!(cfg.stream.gate_width, stream.gate_width);
        assert_eq!(cfg.stream.gate_height, stream.gate_height);
        assert_eq!(cfg.stream.rtsp_transport, stream.rtsp_transport);
        assert_eq!(cfg.stream.decode_threads, stream.decode_threads);

        assert_eq!(cfg.gate.diff_threshold, gate.diff_threshold);
        assert_eq!(cfg.gate.min_changed_frac, gate.min_changed_frac);
        assert_eq!(cfg.gate.latch_ms, gate.latch_ms);
        assert_eq!(cfg.gate.warmup_frames, gate.warmup_frames);

        assert_eq!(cfg.detector.model, det.model);
        assert_eq!(cfg.detector.input_size, det.input_size);
        assert_eq!(cfg.detector.min_confidence, det.min_confidence);
        assert_eq!(cfg.detector.threads, det.threads);

        assert_eq!(cfg.harvest.enabled, harv.enabled);
        assert_eq!(cfg.harvest.dir, harv.dir);
        assert_eq!(cfg.harvest.max_bytes, harv.max_bytes);
        assert_eq!(cfg.harvest.min_interval_secs, harv.min_interval_secs);
        assert_eq!(
            cfg.harvest.same_object_tolerance,
            harv.same_object_tolerance
        );
        assert_eq!(cfg.harvest.must_have_moved, harv.must_have_moved);
        assert_eq!(cfg.harvest.max_per_frame, harv.max_per_frame);

        assert_eq!(cfg.classifier.enabled, cls.enabled);
        assert_eq!(cfg.classifier.model, cls.model);
        assert_eq!(cfg.classifier.references, cls.references);

        assert_eq!(cfg.preview.source, PreviewCfg::default().source);

        // off by default, deliberately and unlike the harvest: a day of crops
        // not collected is gone, while a day of event clips not recorded costs
        // one replay fixture that tomorrow supplies again. writing video is not
        // something anyone should discover by running out of disk.
        let rec = RecordCfg::default();
        assert!(!cfg.record.enabled, "event recording must default to off");
        assert!(!rec.enabled);
        assert_eq!(cfg.record.dir, rec.dir);
        assert_eq!(cfg.record.preroll_secs, rec.preroll_secs);
        assert_eq!(cfg.record.hangover_secs, rec.hangover_secs);
        assert_eq!(cfg.record.max_clip_secs, rec.max_clip_secs);
        assert_eq!(cfg.record.max_bytes, rec.max_bytes);

        let mqtt = cfg.mqtt.expect("the example documents an mqtt section");
        assert_eq!(mqtt.port, DEFAULT_MQTT_PORT);
        assert_eq!(mqtt.client_id, "metermate");
        assert_eq!(mqtt.base_topic, "metermate");
        assert_eq!(mqtt.discovery_prefix, "homeassistant");
    }

    #[test]
    fn rtsp_url_encodes_reserved_password_characters() {
        // '!' is legal in a url but ambiguous in shells and some rtsp stacks, and
        // '@' or ':' would break userinfo parsing outright.
        let url = camera().rtsp_url(MAIN_SUBTYPE);
        assert!(url.contains("m3t3rm4%21"), "got {url}");
        assert!(url.contains("channel=1&subtype=0"), "got {url}");
    }

    #[test]
    fn rtsp_url_selects_the_requested_stream() {
        assert!(camera().rtsp_url(SUB_SUBTYPE).contains("subtype=1"));
        assert!(camera().rtsp_url(MAIN_SUBTYPE).contains("subtype=0"));
    }

    #[test]
    fn redacted_url_hides_password() {
        let url = camera().rtsp_url_redacted(MAIN_SUBTYPE);
        assert!(!url.contains("m3t3rm4"), "password leaked: {url}");
        assert!(url.contains("metermate"));
    }

    /// a camera whose url is not `cam/realmonitor` is not a second-class camera
    /// (r5.6). the template is also how the e2e suite reaches the rtsp path with
    /// a `file:` url, which is the only way a stream whose size is not the one
    /// compiled in gets tested at all.
    #[test]
    fn an_explicit_url_template_selects_the_stream() {
        let mut c = camera();
        c.url = Some("file:/var/clips/{subtype}.mp4".into());
        assert_eq!(c.rtsp_url(MAIN_SUBTYPE), "file:/var/clips/0.mp4");
        assert_eq!(c.rtsp_url(SUB_SUBTYPE), "file:/var/clips/1.mp4");
    }

    #[test]
    fn an_explicit_url_template_fills_in_credentials_and_masks_them() {
        let mut c = camera();
        c.url = Some("rtsp://{username}:{password}@{host}:{channel}".into());
        assert_eq!(
            c.rtsp_url(SUB_SUBTYPE),
            "rtsp://metermate:m3t3rm4%21@10.1.150.205:1"
        );

        let redacted = c.rtsp_url_redacted(SUB_SUBTYPE);
        assert!(!redacted.contains("m3t3rm4"), "password leaked: {redacted}");
        assert_eq!(redacted, "rtsp://metermate:***@10.1.150.205:1");

        // a url with no userinfo at all has nothing to mask, and must survive
        // the attempt unchanged rather than lose everything before the last `@`.
        c.url = Some("file:/var/clips/{subtype}.mp4".into());
        assert_eq!(
            c.rtsp_url_redacted(SUB_SUBTYPE),
            "file:/var/clips/1.mp4",
            "a url with no credentials was mangled"
        );
    }

    #[test]
    fn local_config_is_preferred_when_present() {
        let dir = std::env::temp_dir().join(format!("metermate-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (local, default) = (dir.join(LOCAL_CONFIG), dir.join(DEFAULT_CONFIG));
        let _ = std::fs::remove_file(&local);

        std::fs::write(&default, "").unwrap();
        assert_eq!(resolve_path(None, &dir), default, "should fall back");

        std::fs::write(&local, "").unwrap();
        assert_eq!(resolve_path(None, &dir), local, "local should win");

        // an explicit path wins even over a present local file, and even when it
        // does not exist, so a typo fails loudly rather than loading something else.
        let explicit = dir.join("elsewhere.toml");
        assert_eq!(resolve_path(Some(explicit.clone()), &dir), explicit);

        std::fs::remove_dir_all(&dir).ok();
    }

    fn with_subjects(body: &str) -> Result<Config> {
        let base = "[camera]\nhost = \"127.0.0.1\"\nusername = \"u\"\npassword = \"p\"\n";
        Config::parse_str(&format!("{base}{body}"))
    }

    /// **a key nobody reads is a setting that looks applied and is not.**
    ///
    /// serde drops an unknown field without a word, so a misspelled
    /// `min_changed_frac` reads as configured and does nothing -- the same shape
    /// as every other failure this project keeps meeting. it also lets a key
    /// that has since been *removed* sit in a deployed config pretending to
    /// still have an effect, which is why the margin could not simply be
    /// deleted from the struct and left alone in the file.
    #[test]
    fn a_misspelled_key_is_refused_rather_than_ignored() {
        let good = with_subjects("[gate]\nmin_changed_frac = 0.01\n");
        assert!(good.is_ok(), "the correctly spelled key must still parse");

        let typo = with_subjects("[gate]\nmin_changed_fraction = 0.01\n");
        let err = typo.expect_err("a misspelled key parsed as though it were read");
        let text = format!("{err:#}");
        assert!(
            text.contains("min_changed_fraction"),
            "the error must name the key: {text}"
        );
    }

    /// the subject array is `rename`d, so the accepted spelling is `[[subject]]`
    /// and the plural is a mistake worth catching rather than dropping.
    #[test]
    fn a_table_nobody_reads_is_refused() {
        assert!(with_subjects("[[subject]]\nname = \"waymo\"\n").is_ok());
        assert!(
            with_subjects("[[subjects]]\nname = \"waymo\"\n").is_err(),
            "a stray table parsed as though it configured something"
        );
        assert!(
            with_subjects("[classifer]\nenabled = true\n").is_err(),
            "a misspelled section parsed as though it configured something"
        );
    }

    /// `keep` and `streams` sit in one section and both take a list of names,
    /// so they may not disagree about what an empty one means. keep's is "no
    /// filter"; anything else here would make `streams = []` read as "record
    /// everything" and do the opposite.
    #[test]
    fn an_empty_stream_list_records_every_stream() {
        let cfg = with_subjects("[record]\nstreams = []\n").unwrap();
        assert!(cfg.record.records(crate::record::MAIN_STREAM));
        assert!(cfg.record.records(crate::record::SUB_STREAM));

        let cfg = with_subjects("[record]\nstreams = [\"main\"]\n").unwrap();
        assert!(cfg.record.records(crate::record::MAIN_STREAM));
        assert!(!cfg.record.records(crate::record::SUB_STREAM));
    }

    /// **the clips budget is the larger one, and that is not arbitrary.** both
    /// streams are recorded, so an event replays through the whole pipeline and
    /// every crop of that passage can be cut again. no crop reconstructs a clip.
    /// the harvest is derived data and the clips are the source, so when the
    /// disk is tight it is the derived half that should go first -- which means
    /// giving it the smaller share.
    #[test]
    fn the_source_gets_more_disk_than_what_is_derived_from_it() {
        let cfg = with_subjects("").expect("the defaults do not parse");
        assert!(
            cfg.record.max_bytes >= cfg.harvest.max_bytes,
            "clips {} are the source of crops {}, so they may not be evicted first",
            cfg.record.max_bytes,
            cfg.harvest.max_bytes
        );
    }

    /// the same rule `record.keep` learned: a name that matches nothing records
    /// nothing, and an events directory holding half of what was asked for looks
    /// exactly like one on a quiet street. the substream especially -- it is the
    /// half an eval replay needs, so losing it to a typo is not noticed until a
    /// clip is replayed weeks later.
    #[test]
    fn a_stream_name_that_records_nothing_is_refused_rather_than_ignored() {
        for good in [r#"["main"]"#, r#"["sub"]"#, r#"["main", "sub"]"#, "[]"] {
            with_subjects(&format!("[record]\nstreams = {good}\n"))
                .unwrap_or_else(|e| panic!("{good} rejected: {e:#}"));
        }
        for bad in ["Main", "substream", "sub-stream", "camera-sub", "both"] {
            let e = with_subjects(&format!("[record]\nstreams = [\"{bad}\"]\n"))
                .expect_err(&format!("{bad} was accepted as a stream name"));
            assert!(
                format!("{e:#}").contains(bad),
                "the error does not say which name was wrong: {e:#}"
            );
        }
    }

    /// a subject name is an mqtt topic segment and a home assistant id, and the
    /// id outlives the config that made it.
    #[test]
    fn a_subject_name_has_to_be_usable_as_a_topic_and_an_entity_id() {
        for good in ["go4", "street-sweeper", "waymo", "tow-truck2"] {
            with_subjects(&format!("[[subject]]\nname = \"{good}\"\n"))
                .unwrap_or_else(|e| panic!("{good} rejected: {e:#}"));
        }
        for bad in [
            "Go4",            // uppercase is not a topic segment anyone wants
            "street sweeper", // a space would split the topic
            "123",            // legal mqtt, poor entity id
            "-go4",           // ditto
            "go4-",
            "go--4",
            "",
        ] {
            assert!(
                with_subjects(&format!("[[subject]]\nname = \"{bad}\"\n")).is_err(),
                "{bad:?} was accepted as a subject name"
            );
        }
    }

    /// `other` names the shared negative set. a subject called that would be
    /// asking to be compared against itself.
    #[test]
    fn the_reserved_names_cannot_be_subjects() {
        for reserved in [crate::label::NEGATIVE, crate::label::UNCLEAR] {
            assert!(
                with_subjects(&format!("[[subject]]\nname = \"{reserved}\"\n")).is_err(),
                "{reserved} was accepted as a subject"
            );
        }
    }

    #[test]
    fn two_subjects_cannot_share_a_name() {
        let two = "[[subject]]\nname = \"go4\"\n\n[[subject]]\nname = \"go4\"\n";
        assert!(with_subjects(two).is_err(), "a duplicate name was accepted");
    }

    /// a config written before `[[subject]]` existed has to keep working, and
    /// there has to be one code path rather than two.
    #[test]
    fn a_config_naming_no_subject_still_has_one() {
        let cfg = with_subjects("").unwrap();
        assert_eq!(cfg.subject_names(), vec![DEFAULT_SUBJECT]);

        let named = with_subjects("[[subject]]\nname = \"waymo\"\n").unwrap();
        assert_eq!(named.subject_names(), vec!["waymo"]);

        // the margin is not a setting any more: it is measured by `--train` and
        // written beside the vectors it was measured against, so a config that
        // still names one is a config describing a rule nobody ran.
        assert!(with_subjects("[[subject]]\nname = \"waymo\"\nmargin = 0.02\n").is_err());
    }

    #[test]
    fn a_subject_cannot_name_a_detector_class_that_does_not_exist() {
        let ok = "[[subject]]\nname = \"go4\"\ndetector_classes = [\"car\", \"truck\"]\n";
        assert!(with_subjects(ok).is_ok());
        let bad = "[[subject]]\nname = \"go4\"\ndetector_classes = [\"go4\"]\n";
        let err = with_subjects(bad).unwrap_err();
        assert!(format!("{err:#}").contains("go4"), "{err:#}");
    }

    /// the roi editor draws over a live view, so a point dragged onto the very
    /// edge lands just past it. refusing to start over `[640, -3]` on a 640x480
    /// gate is hostile, and the error points at the toml rather than at the
    /// editor that wrote it.
    #[test]
    fn an_roi_vertex_outside_the_frame_does_not_stop_the_config_loading() {
        let toml = r#"
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[gate]
roi = [[571, 320], [640, -3], [-9, 88], [324, 345]]
"#;
        let cfg: Config = toml::from_str(toml).expect("a vertex past the edge was refused");
        assert_eq!(
            cfg.gate.roi,
            vec![[571, 320], [640, 0], [0, 88], [324, 345]],
            "negatives should clamp to the boundary they overshot"
        );
    }

    /// fields read via a helper inside this file, so they never appear as
    /// `.field` anywhere else. keep this list short and justified: every entry
    /// is a field the check below cannot see.
    const READ_INSIDE_CONFIG: &[&str] = &["classes", "detector_classes"];

    /// strip `tracing::` macro calls, so a field that is only ever *logged*
    /// does not count as used.
    ///
    /// this is not hypothetical. `[record] preroll_secs` was printed in the
    /// startup line and read nowhere else, so metermate announced the configured
    /// pre-roll and then recorded the compiled-in one. a check that only asked
    /// "does this name appear" would have called that field read.
    fn without_logging(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut rest = src;
        while let Some(at) = rest.find("tracing::") {
            out.push_str(&rest[..at]);
            // skip to the macro's opening paren, then past its matching close.
            let after = &rest[at..];
            let Some(open) = after.find('(') else { break };
            let (mut depth, mut end) = (0usize, None);
            for (i, c) in after[open..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(open + i + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            match end {
                Some(e) => rest = &after[e..],
                None => break,
            }
        }
        out.push_str(rest);
        out
    }

    fn rust_sources(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                rust_sources(&path, into);
            } else if path.extension().is_some_and(|x| x == "rs") {
                into.push(path);
            }
        }
    }

    /// **every field in the config has to be read by something.**
    ///
    /// a knob that is documented, parsed, defaulted and then ignored is worse
    /// than no knob: it answers "can i change this" with yes and then does not.
    /// one pass promoting constants to config left six of these at once --
    /// `crops_listed`, `viewer_backlog`, `frame_queue_depth`, `max_detections`
    /// and both `jpeg_quality` fields -- each of them a call site still reading
    /// the constant the field was supposed to replace. that is a missing check
    /// rather than six slips, so here is the check.
    #[test]
    fn every_config_field_is_read_somewhere() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mine = std::fs::read_to_string(root.join("config.rs")).unwrap();
        let fields: Vec<&str> = mine
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub "))
            .filter_map(|l| l.split_once(':'))
            .map(|(name, _)| name.trim())
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
            .collect();
        assert!(
            fields.len() > 50,
            "found only {} fields to check",
            fields.len()
        );

        let mut files = Vec::new();
        rust_sources(&root, &mut files);
        let mut body = String::new();
        for f in files {
            if f.file_name().is_some_and(|n| n == "config.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&f).unwrap();
            // tests may legitimately name a field they do not wire up.
            body.push_str(&without_logging(
                text.split("mod tests").next().unwrap_or(""),
            ));
        }

        let unread: Vec<&str> = fields
            .iter()
            .filter(|f| !READ_INSIDE_CONFIG.contains(f))
            .filter(|f| !body.contains(&format!(".{f}")))
            .copied()
            .collect();
        assert!(
            unread.is_empty(),
            "config fields that nothing reads (or that are only logged): {unread:?}"
        );
    }
}
