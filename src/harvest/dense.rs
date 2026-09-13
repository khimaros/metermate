//! every detection in a window of an event clip, cropped into a set.
//!
//! the live harvest keeps a few crops a passage by design: it rate-limits by
//! place and drops what did not move, so a recorded go-4 passage leaves a dozen
//! crops where there were hundreds of looks. a set built for labelling wants
//! the whole passage, so this crops every detection of the configured classes in
//! every frame of the window -- parked ones included, since a go-4 stopped at the
//! kerb writing a ticket is a positive worth having.

use super::{Crop, Harvester};
use crate::config::Config;
use crate::crop;
use crate::detect::{Detector, OnnxDetector};
use crate::gate::Rect;
use crate::ingest::{self, Ingest, Source};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// what an extraction wrote.
pub struct Extracted {
    pub frames: usize,
    pub crops: usize,
    pub dir: PathBuf,
}

/// `START-END` seconds into a clip, each written `SS`, `MM:SS` or `HH:MM:SS`.
///
/// offsets rather than times of day, because that is what the preview's player
/// shows, and a time of day would need the recording machine's time zone.
pub fn parse_window(text: &str) -> Result<(f64, f64)> {
    let (a, b) = text
        .split_once('-')
        .with_context(|| format!("window {text:?} is not START-END"))?;
    // **either end may be left off.** where a vehicle leaves is often "it is
    // still there when the clip ends", and counting that out of the recording to
    // type it in is work for nothing. not both, though: that is the whole clip,
    // and a window saying nothing should say so rather than look deliberate.
    anyhow::ensure!(
        !(a.trim().is_empty() && b.trim().is_empty()),
        "window {text:?} gives neither end"
    );
    let start = if a.trim().is_empty() { 0.0 } else { offset(a)? };
    let end = if b.trim().is_empty() {
        f64::INFINITY
    } else {
        offset(b)?
    };
    anyhow::ensure!(start < end, "window {text:?} ends before it starts");
    Ok((start, end))
}

fn offset(text: &str) -> Result<f64> {
    let parts: Vec<&str> = text.trim().split(':').collect();
    anyhow::ensure!(parts.len() <= 3, "{text:?} is not SS, MM:SS or HH:MM:SS");
    parts.iter().try_fold(0.0, |acc, part| {
        let value: f64 = part
            .parse()
            .with_context(|| format!("{text:?} is not SS, MM:SS or HH:MM:SS"))?;
        Ok(acc * 60.0 + value)
    })
}

/// crop every detection in `window` of `clip` into the set at `into`.
///
/// **named by when each frame was taken.** the recorder names a clip for the
/// moment it was kept and opens it `preroll_secs` earlier, so a frame's time is
/// that opening plus its offset. passages are grouped by time, and a crop named
/// by the clock of this run would squeeze the whole event into a few
/// milliseconds. crops of one frame are a millisecond apart, so a re-run writes
/// the same names rather than a second copy.
pub fn extract(cfg: &Config, clip: &Path, window: (f64, f64), into: &Path) -> Result<Extracted> {
    let name = clip
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{} has no file name", clip.display()))?;
    let (stamp, _, stream) = crate::record::parse_name(name).with_context(|| {
        format!("{name} is not an event clip: its start is read from the recorder's name")
    })?;
    anyhow::ensure!(
        stream == crate::record::MAIN_STREAM,
        "{name} is the {stream} stream; crops are cut from the main stream, at full resolution"
    );
    let opened = stamp.saturating_sub(cfg.record.preroll_secs as u128 * 1000);
    // the rate the live harvest decodes the main stream at, and never faster than
    // the clip's own. a recording's container claims 100 fps for a camera that
    // sends 15, and taken literally that crops each real frame several times.
    let harvest_fps = cfg.stream.crop_fps as f64;
    let fps = ingest::probe_fps(clip)
        .map_or(harvest_fps, |own| own.min(harvest_fps))
        .round()
        .max(1.0);

    let mut detector = OnnxDetector::load(
        &cfg.detector.model,
        cfg.detector.input_size,
        cfg.detector.min_confidence,
        cfg.detector.threads,
        cfg.detector.max_detections,
    )
    .with_context(|| format!("loading {}", cfg.detector.model.display()))?;
    let size = detector.input_size();
    let classes = cfg.detector.class_filter()?;

    // the clip goes with the set: recordings rotate out under their own budget,
    // and a set should outlive the clip it was cut from.
    let clips = into.join("clips");
    std::fs::create_dir_all(&clips).with_context(|| format!("creating {}", clips.display()))?;
    std::fs::copy(clip, clips.join(name)).with_context(|| format!("copying {name}"))?;

    // a set is kept whole, so no disk budget may evict from it.
    let keep = crate::config::HarvestCfg {
        max_bytes: u64::MAX,
        keep_labelled: false,
        ..cfg.harvest.clone()
    };
    let dir = into.join("crops");
    let mut harvester = Harvester::new(&dir, &keep)?;

    // a clip's own pixels decide the crop geometry, exactly as a live stream's
    // do: the set has to look like the harvest when they are trained together.
    let source = Source::File(clip.to_path_buf());
    let pixels = crate::ingest::resolve_size(
        "crop",
        (cfg.stream.main_width, cfg.stream.main_height),
        &source,
    );
    let frames = Ingest::crops(cfg, source, fps as u32)
        .at_size(pixels)
        .offline()
        .run();
    let now = std::time::Instant::now();
    let (mut seen, mut written) = (0usize, 0usize);
    while let Some(frame) = ingest::next(&frames) {
        if frame.is_probe() {
            continue;
        }
        let at = frame.video_secs(fps);
        if at < window.0 {
            continue;
        }
        if at >= window.1 {
            break;
        }
        seen += 1;
        let whole = Rect {
            x: 0,
            y: 0,
            w: frame.width,
            h: frame.height,
        };
        let input = crop::letterbox(&frame.data, frame.width, frame.height, size);
        let found = detector.detect(&input, size)?;
        let taken = opened + (at * 1000.0).round() as u128;
        for (i, d) in found.iter().filter(|d| classes.keeps(d)).enumerate() {
            let m = crop::detection_to_main(
                (d.x1, d.y1, d.x2, d.y2),
                whole,
                size,
                frame.width,
                frame.height,
            );
            // the harvest's own margins, so a set's crops look like the crops the
            // classifier is shown live.
            let margin = if crop::touches_edge(m, whole, cfg.harvest.edge_slack_px) {
                cfg.harvest.clipped_context
            } else {
                cfg.harvest.context
            };
            let bbox = crop::expand(m, margin, frame.width, frame.height);
            let pixels = crop::extract(&frame.data, frame.width, frame.height, bbox);
            harvester.save(
                Crop {
                    rgb: &pixels,
                    width: bbox.w,
                    height: bbox.h,
                    region: bbox,
                    label: d.label(),
                    confidence: d.confidence,
                    subject: None,
                    margin: None,
                    at_millis: Some(taken + i as u128),
                },
                now,
            )?;
            written += 1;
        }
    }
    let until = match window.1.is_finite() {
        true => format!("{:.1}s", window.1),
        false => "the end".to_string(),
    };
    anyhow::ensure!(
        seen > 0,
        "no frames of {name} fall in {:.1}s-{until}",
        window.0
    );
    Ok(Extracted {
        frames: seen,
        crops: written,
        dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_reads_seconds_minutes_and_hours() {
        assert_eq!(parse_window("12-48").unwrap(), (12.0, 48.0));
        assert_eq!(parse_window("1:05-1:40.5").unwrap(), (65.0, 100.5));
        assert_eq!(parse_window("1:00:00-1:00:30").unwrap(), (3600.0, 3630.0));
    }

    /// **either end may be left off.** where the vehicle leaves is often "it is
    /// still there when the clip ends", and counting that out of the recording
    /// to type it in is work for nothing.
    #[test]
    fn a_window_may_be_open_at_either_end() {
        assert_eq!(parse_window("12-").unwrap(), (12.0, f64::INFINITY));
        assert_eq!(parse_window("1:05-").unwrap(), (65.0, f64::INFINITY));
        assert_eq!(parse_window("-48").unwrap(), (0.0, 48.0));
        // but not both: that is the whole clip, and `--window` would say nothing.
        assert!(parse_window("-").is_err());
    }

    #[test]
    fn a_window_that_does_not_read_is_refused() {
        assert!(parse_window("12").is_err());
        assert!(parse_window("48-12").is_err());
        assert!(parse_window("a-b").is_err());
        assert!(parse_window("1:2:3:4-5").is_err());
    }
}
