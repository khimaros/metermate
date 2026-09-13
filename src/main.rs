//! metermate: watch a block for parking enforcement and say so quickly.

mod alert;
mod camera;
mod classify;
mod config;
mod crop;
mod detect;
mod gate;
mod harvest;
mod hours;
mod ingest;
mod label;
mod preview;
mod record;
mod scenery;
mod stats;
mod track;

use alert::{Alerter, Outcome};
use anyhow::{Context, Result};
use clap::Parser;
use config::Config;
use gate::MotionGate;
use hours::Hours;
use ingest::{Ingest, Source};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "metermate",
    about = "spot parking enforcement from an ip camera"
)]
struct Args {
    /// config file [default: metermate.local.toml, else metermate.toml]
    #[arg(short, long)]
    config: Option<PathBuf>,

    // a file source is how the e2e suite runs deterministically and how the
    // binary is exercised on a laptop with no camera in reach (r5.5).
    /// read from a video file instead of the camera
    #[arg(short, long)]
    source: Option<PathBuf>,

    // without this a file source feeds *both* the gate and the crops, so crops
    // are cut from 640x480 rather than 2560x1440 and every vehicle reaches the
    // detector a quarter of its real size. an eval run that way measures a
    // pipeline nobody deploys.
    /// main-stream file for crops, when --source is the substream
    #[arg(long)]
    crop_source: Option<PathBuf>,

    /// log decisions without publishing to mqtt
    #[arg(long)]
    dry_run: bool,

    // ungated and unfiltered on purpose: this is how we learn what a go-4
    // registers as before any project-specific model exists. it ignores the
    // resource budget by design (r8.4).
    /// run the detector on every frame and show everything it sees
    #[arg(long)]
    debug_detector: bool,

    /// preview server address, or "off"
    #[arg(long, default_value = "0.0.0.0:8420")]
    preview: String,

    // an eval that drops frames measures the machine it ran on. with this, a
    // replay examines every frame and ignores the detector's rate limit, so the
    // same clip gives the same answer anywhere.
    /// examine every frame of a file source, ignoring real time
    #[arg(long)]
    offline: bool,

    // `[detector] classes` takes coco class names, and there are eighty of them
    // with no obvious way to find out what they are called. guessing produces a
    // config that looks fine and silently matches nothing.
    /// print the detector class names `[detector] classes` accepts, and exit
    #[arg(long)]
    list_classes: bool,

    // labelling and measuring are the same loop, so they live in the binary
    // rather than in a second toolchain: they run wherever metermate runs, on
    // the harvest that is already there, with nothing to mirror (r5.5).
    /// label the harvest for this subject in a browser, and train from it
    #[arg(long, value_name = "SUBJECT")]
    label: Option<String>,

    /// address for `--label` [default: the preview address, port + 1]
    #[arg(long)]
    label_addr: Option<String>,

    /// seconds between looks at the harvest by `--label`; 0 stops looking
    /// [default: 60]
    #[arg(long, value_name = "SECONDS", default_value_t = 60)]
    label_ingest: u64,

    /// measure how well the harvest's labels separate this subject, and exit
    #[arg(long, value_name = "SUBJECT")]
    measure: Option<String>,

    /// subtract the harvest's mean direction before comparing, for `--measure`
    #[arg(long)]
    centre: bool,

    // one command writes the artifact the classifier loads, from the harvest
    // already there, printing the measurement before it writes anything -- so
    // what ships is what was measured, by the same toolchain.
    /// write the reference file for this subject from the harvest's labels
    #[arg(long, value_name = "SUBJECT")]
    train: Option<String>,

    /// directory `--train` writes `<subject>/` into [default: `[classifier] references`]
    #[arg(long, value_name = "DIR")]
    train_out: Option<PathBuf>,

    // the harvest deletes oldest first under a disk budget, so a label outlives
    // the crop it describes and a set silently stops being reproducible. this
    // copies what a set's labels name into the set, and reports what is already
    // beyond recovering rather than training on the remainder.
    /// copy the crops this set's labels name into it, from `--harvest`
    #[arg(long, value_name = "DIR")]
    gather: Option<PathBuf>,

    // labelling, measuring and training all read a directory of crops, and it
    // is not always the configured one: an eval set staged from recorded clips
    // is a harvest too. measuring and training take more than one, so a
    // deployment's own harvest and the sets cut from video train together.
    /// crops to label, measure, train on or gather from [default: `[harvest] dir`]
    #[arg(long, value_name = "DIR")]
    harvest: Vec<PathBuf>,

    // the harvest keeps a few crops a passage by design, and a set built for
    // labelling wants every one of them.
    /// crop every detection in a window of an event clip into a set
    #[arg(long, value_name = "CLIP", requires_all = ["window", "into"])]
    dense: Option<PathBuf>,

    /// the part of the `--dense` clip to crop: START-END as SS, MM:SS or HH:MM:SS
    #[arg(long, value_name = "START-END", requires = "dense")]
    window: Option<String>,

    /// the set `--dense` writes its crops and the clip into
    #[arg(long, value_name = "DIR", requires = "dense")]
    into: Option<PathBuf>,

    // which row of the curve to ship. a margin is an output of a measurement
    // and *which* output is a choice: an operator who knows what nuisance rate
    // they tolerate says so here rather than reading it off the table and
    // copying a number back in by hand.
    // the two commands the labelling loop ends in. everything else about a
    // day's labelling happens in the browser.
    /// cut every passage marked in the preview into a set under `[train] sets`
    #[arg(long)]
    prepare: bool,

    /// embed and train every configured subject over the harvest and the sets
    #[arg(long)]
    retrain: bool,

    // proving the phone works is a deployment step, and the alternative is
    // finding out the token was wrong on the evening a go-4 parks outside.
    /// send one ntfy notification and exit, reporting what the server said
    #[arg(long)]
    notify_test: bool,

    /// a crop to attach to `--notify-test`, for checking evidence arrives
    #[arg(long, value_name = "JPEG", requires = "notify_test")]
    notify_crop: Option<PathBuf>,

    /// most of the street that may fire, as `1%` or `0.01`: takes the most recall under it
    #[arg(long, value_name = "RATE", value_parser = fraction)]
    max_fpr: Option<f32>,

    /// least share of passages to keep, as `80%` or `0.8`: takes the cleanest margin above it
    #[arg(long, value_name = "RATE", value_parser = fraction)]
    min_recall: Option<f32>,

    // the flag wins over the config, both ways: `--record-events` turns it on
    // for one run without editing a file, and `--record-events=false` turns it
    // off for one run without losing the setting. `Option` is what makes the
    // second possible -- a bare `bool` cannot tell "asked for off" from "did
    // not ask", so it could only ever override in one direction.
    /// record full-resolution clips of events [default: [record] enabled]
    #[arg(long, num_args = 0..=1, default_missing_value = "true", value_name = "BOOL")]
    record_events: Option<bool>,
}

/// a rate written either way a person would write one: `1%` or `0.01`.
///
/// both, because the report prints these as percentages and the config carries
/// fractions, so whichever one is in front of you is the one you will type. a
/// bare `1` is refused rather than read as 100%: it is far more likely to mean
/// one percent, and silently shipping a rule at a rate a hundred times what was
/// asked for is the worst way to resolve that.
fn fraction(text: &str) -> Result<f32, String> {
    let (number, scale) = match text.strip_suffix('%') {
        Some(rest) => (rest, 100.0),
        None => (text, 1.0),
    };
    let value: f32 = number
        .trim()
        .parse()
        .map_err(|_| format!("{text:?} is not a rate; write it as `1%` or `0.01`"))?;
    let rate = value / scale;
    if !(0.0..=1.0).contains(&rate) || (scale == 1.0 && value > 1.0) {
        return Err(format!(
            "{text:?} is not a share of anything; write it as `1%` or `0.01`"
        ));
    }
    Ok(rate)
}

/// what was asked for, so the chosen row reads as an answer to a question.
///
/// without it the line is a number with no account of why that row and not the
/// one above it, which is the whole question anybody has about a margin.
fn asked_for(max_fpr: &Option<f32>, min_recall: &Option<f32>) -> String {
    match (max_fpr, min_recall) {
        (Some(f), Some(r)) => format!(
            " (asked for {:.0}% of passages under {:.2}%)",
            r * 100.0,
            f * 100.0
        ),
        (Some(f), None) => format!(" (asked for the most recall under {:.2}%)", f * 100.0),
        (None, Some(r)) => format!(
            " (asked for the cleanest margin keeping {:.0}% of passages)",
            r * 100.0
        ),
        (None, None) => String::new(),
    }
}

/// cut every passage marked in the preview into a set.
///
/// **the expensive half of the loop, run when it suits.** each selection is a
/// detector pass over the seconds of a clip somebody marked, which is why the
/// page files it instead of doing it. idempotent by directory: a set that is
/// already there is a passage already cut, and cutting it again would both
/// cost the minutes twice and leave two copies of one passage for everything
/// downstream to count as two examples.
fn prepare(cfg: &config::Config) -> Result<()> {
    let events = &cfg.record.dir;
    let waiting = record::select::load(&record::select::path(events))?;
    if waiting.is_empty() {
        println!(
            "nothing selected. mark a clip and a window on the events tab of the preview, \
             then run this again"
        );
        return Ok(());
    }
    // the first subject is what a set is filed under: sets are per-subject
    // directories, and the crops in one are mostly not the subject anyway --
    // what makes it that subject's set is that somebody went looking for one
    // in it.
    let subject = cfg
        .subject_names()
        .first()
        .cloned()
        .unwrap_or_else(|| config::DEFAULT_SUBJECT.to_string());
    let marks = record::select::path(events);
    let (mut cut, mut already) = (0, 0);
    for want in &waiting {
        let into = record::select::set_dir(&cfg.train.sets, &subject, want);
        // **the file is a queue and this drains it.** a mark that has been cut
        // is work done, and leaving it would have the events tab go on saying
        // the passage is waiting. what was cut is recorded by the set itself,
        // whose name carries the clip and the window.
        if into.exists() {
            already += 1;
            record::select::remove(&marks, &want.clip, &want.window)?;
            continue;
        }
        let window = harvest::dense::parse_window(&want.window)?;
        let clip = events.join(&want.clip);
        println!(
            "cutting {} of {} into {}",
            want.window,
            want.clip,
            into.display()
        );
        let done = harvest::dense::extract(cfg, &clip, window, &into)?;
        println!("  {} crops from {} frames", done.crops, done.frames);
        // only once it worked: a clip that could not be read is a mark still
        // waiting, not one quietly dropped.
        record::select::remove(&marks, &want.clip, &want.window)?;
        cut += 1;
    }
    println!(
        "prepared {cut} selection{}, {already} already cut. label them at \
         `metermate --label {subject} --harvest {}`",
        if cut == 1 { "" } else { "s" },
        cfg.train.sets.display()
    );
    Ok(())
}

/// train every configured subject on everything labelled since.
///
/// the hand version is a `--train <subject> --harvest data/crops --harvest
/// sets/` per subject, and the subject list is in the config already. the
/// harvest is where the deployment's own rejected verdicts are and the sets
/// are where the dense passages are; both halves are needed and forgetting
/// one is how a retrain quietly loses the negatives that stopped the last
/// false positive.
fn retrain(cfg: &config::Config, harvests: &[PathBuf], centre: bool) -> Result<()> {
    let mut roots = harvests.to_vec();
    if cfg.train.sets.is_dir() {
        roots.push(cfg.train.sets.clone());
    }
    println!(
        "training {} over {}",
        cfg.subject_names().join(", "),
        roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    for subject in cfg.subject_names() {
        let opts = label::eval::Options {
            subject: subject.clone(),
            centre,
            confirm_m: cfg.track.confirm_m,
            confirm_n: cfg.track.confirm_n,
            ..Default::default()
        };
        println!("\n=== {subject} ===");
        // one subject failing is not the others failing: a subject with no
        // negatives yet is the ordinary state of a new one, and it must not
        // stop the go-4 being retrained.
        match label::train(
            &roots,
            &cfg.classifier.model,
            &opts,
            &cfg.classifier.references,
        ) {
            Ok(report) => {
                report.print();
                match report.operating_point() {
                    Some(row) => println!(
                        "chose margin {:+.3}: {:.0}% of passages, {:.2}% of the street",
                        row.margin,
                        row.passage_recall * 100.0,
                        row.fpr * 100.0
                    ),
                    None => println!("no margin keeps enough passages to ship"),
                }
            }
            Err(e) => println!("{subject} was not trained: {e:#}"),
        }
    }
    println!(
        "\nwrote {}. a running pipeline reads it at startup, so restart it",
        cfg.classifier.references.display()
    );
    Ok(())
}

/// send one notification and say what happened, so a phone can be proved
/// before there is anything to be told about.
///
/// **synchronous, and it fails loudly.** the running pipeline drops a
/// notification it cannot send, because holding up detection for a phone is
/// the wrong trade -- but this command exists to answer "does this work", and
/// an exit code of zero on a refused token would answer it wrongly.
fn notify_test(cfg: &config::NtfyCfg, crop: Option<&Path>) -> Result<()> {
    anyhow::ensure!(
        cfg.enabled,
        "[ntfy] enabled is false, so there is nothing to test"
    );
    // built the way a live alert builds it, so what is printed below is what the
    // phone would be given: an address it cannot reach is a notification that
    // goes nowhere once it is tapped.
    let opens = alert::ntfy::link(&cfg.click, crop);
    let note = alert::ntfy::Note {
        title: "metermate test".to_string(),
        body: format!(
            "metermate would notify {} here, for {}",
            cfg.topic,
            cfg.outcomes.join(", ")
        ),
        priority: cfg.priority.clone(),
        tags: "white_check_mark".to_string(),
        // built the way a live alert builds it, so `--notify-test` proves the
        // link on the phone as well as the picture on it.
        click: opens.clone(),
        crop: crop.map(Path::to_path_buf),
    };
    alert::ntfy::Post::new(cfg).send(&note)?;
    println!(
        "sent a test notification to {}/{}{}",
        cfg.server.trim_end_matches('/'),
        cfg.topic,
        match crop {
            Some(p) => format!(", carrying {}", p.display()),
            None => String::new(),
        }
    );
    println!("outcomes that will notify: {}", cfg.outcomes.join(", "));
    if !opens.is_empty() {
        println!("a tap opens: {opens}");
    }
    Ok(())
}

/// print every class the detector can report, marking the default selection.
///
/// stdout rather than the log, because this is output someone asked for rather
/// than something that happened.
fn list_classes() {
    let default = detect::ClassFilter::from_ids(detect::DEFAULT_CLASSES);
    println!("detector classes, for `[detector] classes` in the config.");
    println!("an empty list keeps everything; * marks the default selection.\n");
    for (id, name) in detect::COCO_CLASSES.iter().enumerate() {
        let mark = if default.names().contains(name) {
            "*"
        } else {
            " "
        };
        println!("  {mark} {id:>2}  {name}");
    }
}

/// the labelling server sits one port above the preview.
///
/// running both at once is the normal case -- the preview shows what is
/// arriving, the labelling page shows what has arrived -- so they must not
/// collide, and deriving one from the other keeps that true when either moves.
fn default_label_addr(preview: &str) -> String {
    let (host, port) = preview.rsplit_once(':').unwrap_or(("0.0.0.0", "8420"));
    let next = port.parse::<u16>().unwrap_or(8420).saturating_add(1);
    format!("{host}:{next}")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.list_classes {
        list_classes();
        return Ok(());
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "metermate=info".into()),
        )
        .with_target(false)
        .init();

    let config_path = config::resolve_path(args.config, Path::new("."));
    let cfg = Config::load(&config_path)?;

    // labelling and measuring read crops already on disk. no camera, no stream
    // and no pipeline, which is also why they load the embedder: the live path
    // must never be made to hold 351 MB in case somebody opens a page.
    let harvests = if args.harvest.is_empty() {
        vec![cfg.harvest.dir.clone()]
    } else {
        args.harvest
    };

    if let Some(clip) = args.dense {
        let (Some(window), Some(into)) = (args.window, args.into) else {
            anyhow::bail!("--dense needs --window START-END and --into DIR");
        };
        let window = harvest::dense::parse_window(&window)?;
        let done = harvest::dense::extract(&cfg, &clip, window, &into)?;
        println!(
            "cropped {} detections from {} frames into {}",
            done.crops,
            done.frames,
            done.dir.display()
        );
        return Ok(());
    }

    if let Some(subject) = args.label {
        anyhow::ensure!(
            harvests.len() == 1,
            "--label works on one harvest; give --harvest once"
        );
        let addr = args
            .label_addr
            .unwrap_or_else(|| default_label_addr(&args.preview));
        let session = label::server::Session::open(
            &harvests[0],
            &cfg.classifier.model,
            &subject,
            &cfg.classifier.references,
            (cfg.track.confirm_m, cfg.track.confirm_n),
        )?;
        return session.serve(&addr, std::time::Duration::from_secs(args.label_ingest));
    }

    if let Some(set) = args.gather {
        anyhow::ensure!(
            harvests.len() == 1,
            "--gather copies from one harvest; give --harvest once"
        );
        let harvest = &harvests[0];
        let (copied, held, lost) = label::gather(&set, harvest)?;
        println!(
            "gathered {copied} crops into {} ({held} already there)",
            set.join("crops").display()
        );
        // said plainly rather than left to be counted later: these are labels a
        // person made for crops the budget has already deleted, and no amount
        // of later effort brings them back.
        if lost > 0 {
            println!(
                "{lost} labels name crops that no longer exist in {} -- \
                 that much of this set is beyond recovering",
                harvest.display()
            );
        }
        return Ok(());
    }

    if args.notify_test {
        return notify_test(&cfg.ntfy, args.notify_crop.as_deref());
    }

    if args.prepare {
        return prepare(&cfg);
    }

    if args.retrain {
        return retrain(&cfg, &harvests, args.centre);
    }

    if let Some(subject) = args.measure {
        let opts = label::eval::Options {
            subject,
            centre: args.centre,
            confirm_m: cfg.track.confirm_m,
            confirm_n: cfg.track.confirm_n,
            min_passage_recall: args.min_recall,
            max_fpr: args.max_fpr,
            ..Default::default()
        };
        let (report, _) = label::measure(&harvests, &cfg.classifier.model, &opts)?;
        report.print();
        return Ok(());
    }

    if let Some(subject) = args.train {
        let out = args
            .train_out
            .unwrap_or_else(|| cfg.classifier.references.clone());
        let opts = label::eval::Options {
            subject: subject.clone(),
            centre: args.centre,
            confirm_m: cfg.track.confirm_m,
            confirm_n: cfg.track.confirm_n,
            min_passage_recall: args.min_recall,
            max_fpr: args.max_fpr,
            ..Default::default()
        };
        // printed before the file is mentioned, so what was measured is on
        // screen whether or not it turned out to be worth writing.
        let report = label::train(&harvests, &cfg.classifier.model, &opts, &out)?;
        report.print();
        println!(
            "\nwrote {} references and {} negatives to {}",
            report.reference_crops.len(),
            report.negative_reference_crops.len(),
            out.join(&subject).display()
        );
        match report.operating_point() {
            Some(row) => println!(
                "chose margin {:+.3}: {:.0}% of passages, {:.2}% of the street{} -- \
                 written beside the vectors, nothing to copy into a config",
                row.margin,
                row.passage_recall * 100.0,
                row.fpr * 100.0,
                asked_for(&args.max_fpr, &args.min_recall)
            ),
            None => println!(
                "no margin keeps {:.0}% of the passages; `[classifier] enabled` is not earned yet",
                label::eval::OPERATING_RECALL * 100.0
            ),
        }
        return Ok(());
    }

    let source = build_source(&cfg, args.source);
    tracing::info!(
        "metermate starting on {} (config {})",
        source_label(&source),
        config_path.display()
    );

    if args.debug_detector {
        // the raw detector draws boxes over the frame it read, so it has to know
        // the scene's shape to undistort the substream at all -- and the scene's
        // shape is the main stream's, the two streams sharing a field of view.
        let scene = ingest::resolve_size(
            "scene",
            (cfg.stream.main_width, cfg.stream.main_height),
            &crop_source(&cfg, &source, &None),
        );
        return run_debug_detector(&cfg, source, &args.preview, scene);
    }

    // started before the broker is even looked at, because it is an alerting
    // channel in its own right: a deployment can have a phone and no broker.
    let ntfy = (cfg.ntfy.enabled && !args.dry_run).then(|| {
        tracing::info!(
            "notifying {} on {} for {}",
            cfg.ntfy.topic,
            cfg.ntfy.server,
            cfg.ntfy.outcomes.join(", ")
        );
        alert::ntfy::Ntfy::start(&cfg.ntfy)
    });
    let alerter = match (&cfg.mqtt, args.dry_run) {
        (_, true) => {
            tracing::info!("dry run: gate decisions will be logged, nothing published or notified");
            None
        }
        (None, _) => match ntfy {
            Some(n) => {
                tracing::info!("no [mqtt] section in config: alerting over ntfy alone");
                Some(alert::notifier_only(n))
            }
            None => {
                tracing::info!("no [mqtt] section in config: running without alerting");
                None
            }
        },
        (Some(mqtt), false) => {
            alert::check_broker_reachable(mqtt)
                .context("cannot reach the mqtt broker; use --dry-run to run without one")?;
            Some(Alerter::connect(mqtt, &cfg.subject_names(), ntfy)?)
        }
    };

    // the flag wins over the config, in both directions.
    let record_events = args.record_events.unwrap_or(cfg.record.enabled);
    run(
        &cfg,
        source,
        args.crop_source,
        alerter,
        &args.preview,
        args.offline,
        record_events,
    )
}

/// what the gate reads and what the crops are cut from, asked of the streams.
#[derive(Clone, Copy)]
struct Geometry {
    gate: (u32, u32),
    main: (u32, u32),
}

/// the geometry this run will really see.
///
/// neither size is a constant any more. the main stream's was, and a camera
/// sending 4096x1856 was then read in 2560x1440 chunks, which is not an error
/// but half a frame of sheared pixels per frame: the detector found nothing, the
/// harvest wrote nothing, and every log line reported a healthy pipeline (r5.6).
///
/// the two feeds are reported together because the pair is the thing worth
/// seeing: gate regions become crops through the ratio between them, and a
/// mapping that is off by a factor puts a box above a vehicle rather than on it.
fn resolve_geometry(cfg: &Config, gate: &Source, main: &Source) -> Geometry {
    let gate_size = ingest::resolve_size(
        "gate",
        (cfg.stream.gate_width, cfg.stream.gate_height),
        gate,
    );
    let main_size = ingest::resolve_size(
        "crop",
        (cfg.stream.main_width, cfg.stream.main_height),
        main,
    );
    let (sx, sy) = crop::scale(gate_size, main_size);
    tracing::info!(
        "gate {}x{} on subtype {} -> main stream {}, scale {:.2}x{:.2}",
        gate_size.0,
        gate_size.1,
        cfg.stream.gate_subtype,
        config::MAIN_SUBTYPE,
        sx,
        sy
    );
    // the roi and the perspective priors are hand-drawn in gate pixels, so a
    // gate that changed size leaves them pointing at the wrong part of the
    // street -- which nothing downstream notices, because a polygon is a
    // polygon. scaled silently they would be wrong by a rounding of someone's
    // careful drawing, so this is a loud complaint instead.
    if gate_size != (cfg.stream.gate_width, cfg.stream.gate_height)
        && !(cfg.gate.roi.is_empty() && cfg.gate.perspective.is_empty())
    {
        tracing::warn!(
            "[gate] roi and {} perspective lines were drawn at {}x{} and the gate is now \
             {}x{}: draw them again in the preview, or empty them",
            cfg.gate.perspective.len(),
            cfg.stream.gate_width,
            cfg.stream.gate_height,
            gate_size.0,
            gate_size.1
        );
    }
    Geometry {
        gate: gate_size,
        main: main_size,
    }
}

/// where the full-resolution pixels come from: the camera's main stream, or the
/// main-stream recording an eval is replaying. a file stands in for both streams
/// when there is no camera, which is the only way to exercise the whole cascade
/// without one (r5.5).
fn crop_source(cfg: &Config, source: &Source, crop_file: &Option<PathBuf>) -> Source {
    match (source, crop_file) {
        // an eval replaying recorded streams gets both, which is the only way a
        // replay matches production: crops are cut from main-stream pixels, and
        // feeding the substream to both makes every vehicle a quarter size.
        (_, Some(main)) => Source::File(main.clone()),
        (Source::Rtsp { .. }, None) => Source::Rtsp {
            url: cfg.camera.rtsp_url(config::MAIN_SUBTYPE),
            redacted: cfg.camera.rtsp_url_redacted(config::MAIN_SUBTYPE),
        },
        (Source::File(p), None) => Source::File(p.clone()),
    }
}

fn build_source(cfg: &Config, file: Option<PathBuf>) -> Source {
    match file {
        Some(path) => Source::File(path),
        None => Source::Rtsp {
            url: cfg.camera.rtsp_url(cfg.stream.gate_subtype),
            redacted: cfg.camera.rtsp_url_redacted(cfg.stream.gate_subtype),
        },
    }
}

fn source_label(source: &Source) -> String {
    match source {
        Source::Rtsp { redacted, .. } => redacted.clone(),
        Source::File(p) => p.display().to_string(),
    }
}

/// bind a local port and start a relay on it, returning the url ffmpeg should
/// write its remux to.
///
/// the listener is bound before ffmpeg is spawned so the url is known in advance
/// and metermate is already accepting when it connects.
fn open_relay(backlog: usize, what: &str) -> Option<(preview::Relay, String)> {
    match std::net::TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => {
            let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
            let relay = preview::Relay::new(backlog);
            relay.serve(listener);
            tracing::info!("{what} tees off on port {port}");
            Some((relay, format!("tcp://127.0.0.1:{port}")))
        }
        // losing a remux is not worth taking detection down for.
        Err(e) => {
            tracing::warn!("{what} disabled: cannot bind a local port: {e}");
            None
        }
    }
}

/// publish what each tracked vehicle is doing, on transitions only.
///
/// **outcomes, not severity** (r10): moving and stopped are states, and which
/// of them matters is a home assistant question. departures are published too,
/// because a retained ON cannot be told from "still here" without one.
fn report_tracks(
    tracker: &mut track::Tracker,
    ids: &[u64],
    now: std::time::Instant,
    alerter: Option<&Alerter>,
    cfg: &Config,
) {
    let stopped_after = std::time::Duration::from_secs_f32(cfg.track.stopped_after_secs);
    for id in ids {
        let Some(t) = tracker.tracks().iter().find(|t| t.id == *id) else {
            continue;
        };
        let (stopped, dwell, rect) = (t.dwell(now) >= stopped_after, t.dwell(now), t.rect);
        // the track now carries which subject it was confirmed as, so these no
        // longer have to assume one. unrecognised vehicles still move and stop,
        // and reporting those under a subject name would be a lie -- so they
        // are simply not published.
        let subject = t.confirmed_subject(cfg.track.confirm_m).map(str::to_string);
        // a track just met is neither yet. saying `moving` the instant one
        // appears announced every parked car on the street as moving, since
        // dwell starts at zero -- so wait until it has either actually moved or
        // held still long enough to mean it.
        let decided = stopped || t.speed() >= cfg.track.stopped_below_px_per_sec;
        let Some(subject) = subject else {
            continue;
        };
        if !decided || !tracker.transition(*id, stopped) {
            continue;
        }
        let outcome = if stopped {
            Outcome::Stopped
        } else {
            Outcome::Moving
        };
        tracing::debug!(
            "track {id} {} at {},{} {}x{} after {:.0}s",
            outcome.slug(),
            rect.x,
            rect.y,
            rect.w,
            rect.h,
            dwell.as_secs_f32()
        );
        if let Some(a) = alerter {
            a.publish(
                &subject,
                outcome,
                &alert::event_payload(&alert::Facts {
                    subject: &subject,
                    on: true,
                    track: *id,
                    dwell_s: dwell.as_secs_f32(),
                    confidence: 0.0,
                    protected: alert::Protected::Unknown,
                    box_: rect,
                }),
                // the crop that named this vehicle went out with the sighting
                // it was recognised on; this is the same vehicle doing
                // something later, so the words carry it.
                alert::Told {
                    note: alert::ntfy::describe(&subject, dwell.as_secs_f32(), None),
                    crop: None,
                },
            );
        }
    }
    for t in tracker.departed() {
        tracing::debug!(
            "track {} departed, seen {:.0}s, stopped {:.0}s",
            t.id,
            now.saturating_duration_since(t.first).as_secs_f32(),
            t.dwell(now).as_secs_f32()
        );
        // a departure is only meaningful for something that was published
        // about, and that means something with a name.
        let Some(subject) = t.confirmed_subject(cfg.track.confirm_m) else {
            continue;
        };
        if let Some(a) = alerter {
            a.publish(
                subject,
                Outcome::Departed,
                &alert::event_payload(&alert::Facts {
                    subject,
                    on: false,
                    track: t.id,
                    dwell_s: t.dwell(now).as_secs_f32(),
                    confidence: 0.0,
                    protected: alert::Protected::Unknown,
                    box_: t.rect,
                }),
                // closes the pair the sighting opened, so a rule can tell
                // "still here" from "was here". rarely worth a phone buzzing,
                // which is why `outcomes` defaults to the sighting alone.
                alert::Told {
                    note: alert::ntfy::describe(subject, t.dwell(now).as_secs_f32(), None),
                    crop: None,
                },
            );
        }
    }
}

/// how many inferences a second the measured cost allows.
///
/// called with a rolling average rather than the last measurement: inference
/// time is noisy, and one slow frame should not halve the rate for the next
/// second.
fn detections_per_sec(avg_inference: std::time::Duration, cfg: &config::DetectorCfg) -> usize {
    let ms = avg_inference.as_secs_f32() * 1000.0;
    if ms <= 0.0 {
        return cfg.max_per_sec;
    }
    ((cfg.duty_cycle * 1000.0 / ms) as usize).clamp(cfg.min_per_sec, cfg.max_per_sec)
}

/// whether the expensive stage should run on this frame.
///
/// normally only when nothing newer is waiting: the detector runs inline, so an
/// inference spent on an already-stale frame is paid for by the frames behind
/// it, and a vehicle is in shot for dozens of them.
///
/// **but never skipped indefinitely.** if the stream permanently outruns the
/// loop there is always something newer waiting, and a rule of "only when caught
/// up" would stop detecting entirely and say nothing -- the failure would look
/// like a quiet street. so `min_per_sec` is honoured as a floor on when to look,
/// not only as a clamp on the rate: falling behind may cost the detector its
/// cadence, never its existence.
fn should_inspect(
    caught_up: bool,
    since_last_look: std::time::Duration,
    cfg: &config::DetectorCfg,
) -> bool {
    caught_up
        || since_last_look
            >= std::time::Duration::from_secs_f32(1.0 / cfg.min_per_sec.max(1) as f32)
}

/// inferences one frame may spend, the first of which is the look at the whole
/// frame. never zero.
///
/// the rate limit exists so the detector cannot eat the machine, and it is
/// derived from measured cost rather than chosen (r8.2). but it was being
/// charged against the whole-frame pass, which is the only look that covers the
/// street -- and because motion is bursty, the second's allowance ran out
/// exactly while a vehicle was crossing. measured live: the gate fired and a
/// main frame was paired on 292 frames, and on 171 of them the detector never
/// looked at all.
///
/// so the rate limit now governs follow-ups, which are magnification and can be
/// given up, rather than the pass that decides whether a vehicle is seen. the
/// worst case is one inference per frame the gate fires on, which is the cost of
/// not missing traffic.
fn inspections_allowed(
    offline: bool,
    cfg: &config::DetectorCfg,
    avg_inference: std::time::Duration,
    runs_this_second: usize,
) -> usize {
    // an offline replay is not racing a clock: throttling against time it is not
    // spending would make the result depend on the machine again.
    if offline {
        return cfg.max_regions_per_frame;
    }
    let allowed = cfg
        .max_regions_per_frame
        .min(detections_per_sec(avg_inference, cfg).saturating_sub(runs_this_second));
    if cfg.always_inspect_whole_frame {
        allowed.max(1)
    } else {
        allowed
    }
}

// the movement threshold now lives in `[harvest] must_have_moved`, where it can
// be retuned without a rebuild (r5.3). the measurements behind its default are
// recorded there.
//
// two earlier attempts at this question failed and are worth keeping. excluding
// the protected vehicle by a hard-coded rectangle broke the moment the camera
// was panned. and requiring the detection to match the motion region by
// intersection over union rejected *every* real vehicle: a motion region is the
// bounding box of changed pixels, which for a car is often just its leading and
// trailing edges, so the region is a subset of the vehicle rather than a match
// for it. every measured iou was below 0.25.

/// detections as json for the preview overlay. short keys because this is sent
/// once per frame: c=class, p=probability.
fn detections_json(detections: &[detect::Detection], size: u32, ms: f64) -> String {
    let items: Vec<String> = detections
        .iter()
        .map(|d| {
            format!(
                r#"{{"c":"{}","p":{:.3},"x1":{:.1},"y1":{:.1},"x2":{:.1},"y2":{:.1}}}"#,
                d.label(),
                d.confidence,
                d.x1,
                d.y1,
                d.x2,
                d.y2
            )
        })
        .collect();
    format!(
        r#"{{"size":{size},"ms":{ms:.1},"detections":[{}]}}"#,
        items.join(",")
    )
}

/// one box on the preview, and why it is there.
///
/// the verdict fields are the point. the detector runs on the whole frame, so
/// every gated look returns every vehicle in view, parked ones included, and an
/// overlay showing only boxes says "here is a vehicle" about all of them while
/// the pipeline acts on at most one. publishing what was decided, rather than
/// only what was seen, is the whole of r8.2.
#[derive(Clone, Debug)]
struct OverlayBox {
    rect: gate::Rect,
    label: &'static str,
    confidence: f32,
    /// fraction of this vehicle's own pixels that changed.
    moved: f32,
    /// stood in one place long enough to be scenery.
    parked: bool,
    at: std::time::Instant,
    /// how fast this vehicle was moving when last seen, in gate pixels per
    /// second. a lingering box is carried forward by it, so the box sits where
    /// the vehicle is rather than where it was detected up to a linger-window
    /// ago -- which is what made the green box trail the turquoise one.
    vx: f32,
    vy: f32,
}

/// a detected vehicle and the two things the harvest judges it on.
///
/// kept apart rather than folded into one score, because the two ways of being
/// uninteresting need opposite fixes and used to be indistinguishable once
/// combined: a vehicle scoring zero was either standing still or was standing
/// *somewhere* long enough that scenery vetoed it whatever its pixels did.
pub struct Candidate {
    pub rect: gate::Rect,
    /// fraction of this vehicle's own pixels that the gate saw change.
    pub changed: f32,
    /// and how much of that change sits in the middle of its box rather than
    /// clipping a corner, which is what a passing car does to a parked one.
    pub central: f32,
    /// and whether this place has been occupied long enough to be scenery.
    pub parked: bool,
}

impl Candidate {
    /// the score the harvest ranks on, or zero if the movement is not this
    /// vehicle's. scenery is worth nothing however much moving road its box
    /// contains, and neither is a corner clipped by somebody else driving past.
    fn moved(&self, min_central: f32) -> f32 {
        if self.parked || self.central < min_central {
            0.0
        } else {
            self.changed
        }
    }
}

/// which detected vehicles the harvest should act on, most-moved first.
///
/// **all of them that moved, not just the one that moved most.** this used to
/// take the maximum, which silently discarded every other moving vehicle in the
/// frame: two cars passing at once meant one crop, and a go-4 arriving while a
/// car drove by would lose to whichever happened to move more. the second
/// vehicle was not deferred, it was dropped.
///
/// ordered by how much each moved so that `max_per_frame`, when it bites, keeps
/// the clearest movement rather than whatever the detector listed first.
fn moving_vehicles(
    vehicles: &[Candidate],
    threshold: f32,
    min_central: f32,
    cap: usize,
) -> Vec<usize> {
    let mut chosen: Vec<usize> = vehicles
        .iter()
        .enumerate()
        .filter(|(_, v)| v.moved(min_central) >= threshold)
        .map(|(i, _)| i)
        .collect();
    chosen.sort_by(|&a, &b| {
        vehicles[b]
            .moved(min_central)
            .total_cmp(&vehicles[a].moved(min_central))
    });
    chosen.truncate(cap);
    chosen
}

/// record a detection for the preview, replacing any box already drawn for the
/// same vehicle.
///
/// without this the overlay is append-only over `std::time::Duration::from_millis(cfg.preview.overlay_linger_ms)`. the detector
/// runs about four times a second and the linger is 700ms, so a single car
/// accumulates three boxes at slightly different positions, and a car covered
/// by two motion regions gets two more. the result on a busy frame is a thicket
/// of overlapping rectangles that hides the thing it is meant to show.
fn remember_detection(overlay: &mut Vec<OverlayBox>, mut found: OverlayBox, same_box: f32) {
    // the box this replaces is the same vehicle a moment ago, which is where its
    // drift comes from -- no need to ask the tracker, which does not know about
    // the overlay and would have to be threaded through to say the same thing.
    if let Some(prev) = overlay
        .iter()
        .find(|b| crop::same_object(b.rect, found.rect, same_box))
    {
        let secs = found.at.saturating_duration_since(prev.at).as_secs_f32();
        if secs > 0.0 {
            let centre =
                |r: gate::Rect| (r.x as f32 + r.w as f32 / 2.0, r.y as f32 + r.h as f32 / 2.0);
            let (px, py) = centre(prev.rect);
            let (fx, fy) = centre(found.rect);
            found.vx = (fx - px) / secs;
            found.vy = (fy - py) / secs;
        }
    }
    overlay.retain(|b| !crop::same_object(b.rect, found.rect, same_box));
    overlay.push(found);
}

/// where a box has got to, `age` seconds after it was last seen.
///
/// clamped to the frame rather than allowed to run off it: a vehicle that leaves
/// stops being detected, so its last box would otherwise sail away for the whole
/// linger window.
fn drifted(b: &OverlayBox, age: f32) -> gate::Rect {
    let dx = b.vx * age;
    let dy = b.vy * age;
    gate::Rect {
        x: (b.rect.x as f32 + dx).max(0.0) as u32,
        y: (b.rect.y as f32 + dy).max(0.0) as u32,
        w: b.rect.w,
        h: b.rect.h,
    }
}

/// preview overlay for the gated pipeline: detections plus the motion regions
/// that produced them, all in gate coordinates.
///
/// motion regions are included deliberately. seeing *why* the detector was
/// asked about a patch of street is most of the value when tuning the gate.
fn overlay_json(
    boxes: &[OverlayBox],
    regions: &[gate::Rect],
    gate_w: u32,
    gate_h: u32,
    cfg: &crate::config::PreviewCfg,
    now: std::time::Instant,
) -> String {
    let (out_w, out_h) = (cfg.width, cfg.height);
    // boxes are in gate coordinates, but the preview image may be a different
    // size and aspect. rescale here rather than in the browser, so the page
    // only ever has to draw what it is given.
    let to_out = |r: &gate::Rect| crop::main_to_gate(*r, out_w, out_h, gate_w, gate_h);
    let dets: Vec<String> = boxes
        .iter()
        // `[preview] show_parked = false` asks for the vehicles a person might
        // still disagree with. they are not sent at all rather than sent and not
        // drawn: the feed goes out every frame to every viewer, and on this street
        // most of what it carries is kerb.
        .filter(|b| !b.parked || cfg.show_parked)
        .map(|b| {
            // **carried forward to now.** a box lingers so it does not flicker
            // between detector runs, and is therefore stale by up to that whole
            // window -- while the motion region drawn beside it is from this
            // frame. on a crossing vehicle the two separate visibly. moving the
            // box by the drift measured when it was last refreshed puts it back
            // where the vehicle is.
            let age = now.saturating_duration_since(b.at).as_secs_f32();
            let o = to_out(&drifted(b, age));
            // `m` and `parked` are the verdict: they are what decides whether
            // this vehicle is harvested, so the page can show *why* a box is
            // drawn rather than only that something was seen.
            format!(
                r#"{{"c":"{}","p":{:.3},"m":{:.3},"parked":{},"x1":{},"y1":{},"x2":{},"y2":{}}}"#,
                b.label,
                b.confidence,
                b.moved,
                b.parked,
                o.x,
                o.y,
                o.x + o.w,
                o.y + o.h
            )
        })
        .collect();
    let motion: Vec<String> = regions
        .iter()
        .map(|r| {
            let o = to_out(r);
            format!(r#"[{},{},{},{}]"#, o.x, o.y, o.x + o.w, o.y + o.h)
        })
        .collect();
    format!(
        r#"{{"w":{out_w},"h":{out_h},"detections":[{}],"motion":[{}]}}"#,
        dets.join(","),
        motion.join(",")
    )
}

/// raw detector mode: no gate, no classifier, no filtering. draw nothing away.
///
/// the point is to learn what this street actually looks like to a stock coco
/// model -- which classes appear, at what confidence, at what box size -- so
/// that later thresholds are chosen from evidence instead of guessed.
/// the ungated detector view (r8.4). `scene` is the main stream's size, which is
/// the scene's shape, and the only thing that makes a box land on the vehicle it
/// is reporting rather than above it.
fn run_debug_detector(
    cfg: &Config,
    source: Source,
    preview_addr: &str,
    scene: (u32, u32),
) -> Result<()> {
    use detect::{Detector, OnnxDetector};
    use std::collections::BTreeMap;

    // the counts below are split by whether stage 1 would keep the class, so
    // the filter has to be the configured one rather than a fixed vehicle list:
    // reporting "vehicles" against a config watching for people would be a lie.
    let classes = cfg.detector.class_filter()?;

    // the debug mode runs its own detector ingest and has no crops ffmpeg to tee
    // from, so it always serves its own mjpeg whatever the config says. offering
    // a source it cannot supply would point the page at a 404.
    let preview = match (preview_addr != "off").then(|| {
        preview::Preview::start(
            preview_addr,
            preview::Crops {
                dir: cfg.harvest.dir.clone(),
                budget_bytes: cfg.harvest.max_bytes,
                // the debug mode runs no classifier either.
                classifying: false,
            },
            // the debug mode runs no recorder, so there are no event clips of
            // its own to offer and the tab is not shown.
            None,
            &cfg.preview,
            config::PreviewVideo::ServerMjpeg,
            None,
            &cfg.gate.roi,
            &cfg.gate.perspective,
            (cfg.stream.gate_width, cfg.stream.gate_height),
            scene,
            &cfg.subject_names(),
            std::sync::Arc::new(stats::Stats::default()),
        )
    }) {
        Some(Ok(p)) => Some(p),
        Some(Err(e)) => {
            tracing::warn!("preview disabled: {e:#}");
            None
        }
        None => None,
    };
    let mut detector = OnnxDetector::load(
        &cfg.detector.model,
        cfg.detector.input_size,
        cfg.detector.min_confidence,
        cfg.detector.threads,
        cfg.detector.max_detections,
    )
    .with_context(|| {
        format!(
            "loading {}; run `make models` if it is missing",
            cfg.detector.model.display()
        )
    })?;
    // the model decides the geometry, so the ingest filter follows it rather
    // than config guessing at it a second time.
    let size = detector.input_size();
    tracing::info!(
        "raw detector mode: {size}x{size} input, min confidence {:.2}, {} threads",
        cfg.detector.min_confidence,
        cfg.detector.threads
    );

    let frames = Ingest::detector(cfg, source, size, scene).run();
    let mut seen: BTreeMap<&'static str, (u64, f32)> = BTreeMap::new();
    let mut frames_done = 0u64;
    let mut total_ms = 0f64;
    let (mut vehicles, mut people) = (0u64, 0u64);

    // always take the newest frame. the detector is slower than the stream, so
    // consuming in order would mean falling steadily further behind.
    while let Some(frame) = ingest::latest(&frames) {
        if frame.is_probe() {
            continue;
        }
        let started = std::time::Instant::now();
        let detections = detector.detect(&frame.data, size)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;
        frames_done += 1;

        // encoding is skipped entirely when no browser is attached (r8.3), and
        // happens on the preview's own thread when one is.
        if let Some(p) = preview.as_ref() {
            p.offer(
                frame.data.clone(),
                size,
                size,
                detections_json(&detections, size, elapsed_ms),
            );
        }

        for d in &detections {
            let e = seen.entry(d.label()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 = e.1.max(d.confidence);
            if classes.keeps(d) {
                vehicles += 1;
            } else if d.class_id == detect::CLASS_PERSON {
                people += 1;
            }
            tracing::debug!(
                "{:>12} {:.2} {:>3.0}x{:<3.0} at ({:.0},{:.0})",
                d.label(),
                d.confidence,
                d.width(),
                d.height(),
                d.x1,
                d.y1
            );
        }
        if frames_done.is_multiple_of(config::DEBUG_SUMMARY_EVERY) {
            let tally: Vec<String> = seen
                .iter()
                .map(|(k, (n, best))| format!("{k}={n}@{best:.2}"))
                .collect();
            tracing::info!(
                "{frames_done} frames, {:.0}ms/frame, {vehicles} vehicle {people} person: {}",
                total_ms / frames_done as f64,
                if tally.is_empty() {
                    "nothing".into()
                } else {
                    tally.join(" ")
                }
            );
        }
    }
    Ok(())
}

/// the models the pipeline runs on, and the stream window's to give back.
///
/// `[stream] unload_models` is for the box that does something else at night:
/// several hundred megabytes of weights held for a street that is not being
/// watched is the same argument the channels close over (r3.1).
///
/// the whole safety of it is the window. a shut stream delivers no frames, so
/// there is no look in progress and no crop mid-cut when the weights go, and
/// nothing downstream has to be told to be careful. the moment an unload could
/// land between a frame and the look at it, this would be a pipeline that is
/// quietly blind rather than one that is properly asleep -- which is the failure
/// mode the rest of this file is written against.
///
/// both halves are `Option`: the detector's because the window can take it, and
/// the classifier's also because a deployment without references has none.
struct Models<'a> {
    cfg: &'a Config,
    detector: Option<detect::OnnxDetector>,
    classifier: Option<classify::EmbeddingClassifier>,
}

impl<'a> Models<'a> {
    /// what the config asks for, loaded now rather than built at the window's
    /// edge: the subjects have not changed overnight, and neither has the
    /// margin, so there is nothing new for `report` to say a second time.
    fn load(cfg: &'a Config, report: bool) -> Result<Self> {
        Ok(Self {
            cfg,
            detector: Some(load_detector(cfg)?),
            classifier: load_classifier(cfg, report)?,
        })
    }

    /// a window that was already shut when the process started: nothing loaded,
    /// and nothing to unload.
    fn asleep(cfg: &'a Config) -> Self {
        Self {
            cfg,
            detector: None,
            classifier: None,
        }
    }

    /// hand the weights back. called after the channels have gone down, which
    /// is the only reason there is nothing looking at them.
    ///
    /// the megabytes are measured here, on either side of the drop, rather than
    /// watched from outside: a process that has been harvesting all evening grows
    /// by more than a detector is worth, and an operator asking "did the switch
    /// buy anything" wants the size of the drop, not the size of the heap.
    fn unload(&mut self, stats: &stats::Stats) {
        if !self.cfg.stream.unload_models || self.detector.is_none() {
            return;
        }
        let before = stats::resident_mb();
        let mut what = Vec::new();
        if self.detector.take().is_some() {
            what.push("detector");
        }
        if self.classifier.take().is_some() {
            what.push("classifier");
        }
        // measured after the drop rather than from the model files: what the
        // machine gets back is what it stopped paying for.
        let freed = before.saturating_sub(stats::resident_mb());
        stats.models_released(freed);
        tracing::info!(
            "models unloaded for the window: {} released{}",
            what.join(", "),
            // no `/proc`, no number: "0 MB back" would be a claim about a thing
            // this platform cannot measure.
            if freed > 0 {
                format!(", {freed} MB back")
            } else {
                String::new()
            }
        );
    }

    /// build them again, ahead of the channels coming back up: the first
    /// vehicle through a window that opened at seven is not the one to pay a
    /// model load for.
    fn reload(&mut self) -> Result<()> {
        if !self.cfg.stream.unload_models || self.detector.is_some() {
            return Ok(());
        }
        tracing::info!("stream window open: loading the models back");
        *self = Self::load(self.cfg, false)?;
        Ok(())
    }
}

/// the detector, loaded from the file the config names.
fn load_detector(cfg: &Config) -> Result<detect::OnnxDetector> {
    detect::OnnxDetector::load(
        &cfg.detector.model,
        cfg.detector.input_size,
        cfg.detector.min_confidence,
        cfg.detector.threads,
        cfg.detector.max_detections,
    )
    .with_context(|| {
        format!(
            "loading {}; run `make models`",
            cfg.detector.model.display()
        )
    })
}

/// the classifier, or nothing when none is configured -- which is the state a
/// deployment is in until it has references to classify against (r6).
///
/// `report` says what was found, once, at startup: what the config asks for
/// against what the reference directory actually holds. a subject named in the
/// config with no references can never fire, and looks exactly like a street it
/// never passed -- and so can one with no negatives, since `judge` skips a
/// subject missing either half rather than letting an empty set score infinitely
/// well.
fn load_classifier(cfg: &Config, report: bool) -> Result<Option<classify::EmbeddingClassifier>> {
    let subjects = cfg.subjects();
    if !cfg.classifier.enabled {
        if report {
            tracing::info!(
                "classifier off: harvesting only, nothing will be called {}",
                subjects
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(" or ")
            );
        }
        return Ok(None);
    }
    let refs = classify::References::load(&cfg.classifier.references).with_context(|| {
        format!(
            "loading {}; build it with `--train <subject>`",
            cfg.classifier.references.display()
        )
    })?;
    // the margin comes from the directory, not the config: it is what the sweep
    // measured against these very vectors, so the two cannot drift.
    let mut margins = classify::Margins::uniform(classify::DEFAULT_MARGIN);
    for s in &subjects {
        if let Some(measured) = refs.margin(&s.name) {
            margins.per_subject.insert(s.name.clone(), measured);
        }
        if report {
            report_subject(&refs, s, &margins);
        }
    }
    if report {
        tracing::info!("classifier on: {:?}", refs.names());
    }
    Ok(Some(classify::EmbeddingClassifier::load(
        &cfg.classifier.model,
        refs,
        margins,
    )?))
}

/// one subject as the reference directory describes it, including the two cases
/// in which it can never fire.
fn report_subject(
    refs: &classify::References,
    subject: &config::SubjectCfg,
    margins: &classify::Margins,
) {
    let name = &subject.name;
    let (positives, negatives) = (refs.count(name), refs.negatives(name));
    tracing::info!(
        "subject {}: {} references, {} negatives, margin {:.3}{}{}",
        name,
        positives,
        negatives,
        margins.get(name),
        // a measured margin and the fallback print the same number when they
        // happen to agree, and they mean entirely different things.
        if refs.margin(name).is_some() {
            " (measured)"
        } else {
            " (default: this set was trained before the margin travelled with it)"
        },
        match (positives, negatives) {
            (0, _) => "  -- none in the reference directory, so it can never fire",
            (_, 0) => "  -- no negatives, so it is skipped: nothing to compare against",
            _ => "",
        }
    );
}

/// the line that says nothing is being asked of the camera, and for how long.
///
/// printed whenever the channels are shut, because a directory that stays
/// empty and a preview that never refreshes both read as a quiet street. the
/// time it reopens is what makes the difference between those and a machine
/// that stopped working.
fn shut_line(hours: &Hours) -> String {
    let until = hours
        .opens_at(std::time::SystemTime::now())
        .map(|at| format!(" until {at} local"))
        .unwrap_or_default();
    format!("stream shut{until}: rtsp channels closed, harvest and recording paused")
}

/// close the rtsp channels, and finish what was riding on them.
///
/// three things do not look after themselves when the streams stop: a clip
/// open at the time keeps writing under the name the listing refuses until it
/// closes, the newest main-stream frame stops being new but keeps being handed
/// out as though it were, and the stats would report a stream that stalled
/// rather than one that stopped on purpose.
fn shut_stream(
    hours: &Hours,
    channels: &ingest::Channels,
    clip: Option<&mut record::Recorder>,
    crops: &mut ingest::Crops,
    stats: &crate::stats::Stats,
) {
    channels.close();
    crops.discard_stale();
    // the minute is taken now, once, rather than per request: it is the time the
    // channels were shut against, and nothing else gets to rename it.
    stats.stream(true, hours.opens_at(std::time::SystemTime::now()));
    if let Some(rec) = clip
        && let Err(e) = rec.close()
    {
        tracing::warn!("closing the open event clip: {e:#}");
    }
    tracing::info!("{}", shut_line(hours));
}

/// open the rtsp channels again after a window shut them.
///
/// the feeds restart themselves; what is left here is what the loop was
/// holding on to from before they stopped. the background model is the one
/// that matters: the street at 07:00 is not the street of the previous
/// evening, and a model carried across the gap fires on the light.
fn open_stream(
    channels: &ingest::Channels,
    crops: &mut ingest::Crops,
    gate: &mut MotionGate,
    stats: &crate::stats::Stats,
) {
    channels.reopen();
    crops.discard_stale();
    gate.reset();
    stats.stream(false, None);
    tracing::info!("stream window open: rtsp channels starting, background model reset");
}

/// the gated pipeline: motion decides where to look, the detector looks there
/// in main-stream pixels, the classifier decides what it is, and the harvest
/// keeps the pixels either way.
#[allow(clippy::too_many_arguments)]
fn run(
    cfg: &Config,
    source: Source,
    crop_file: Option<PathBuf>,
    alerter: Option<Alerter>,
    preview_addr: &str,
    offline: bool,
    record_events: bool,
) -> Result<()> {
    use classify::Classifier;

    // one report for the run, shared with the preview server. every stall this
    // program can have shows up in here as an age that stops moving.
    let stats = std::sync::Arc::new(stats::Stats::default());
    let classes = cfg.detector.class_filter()?;
    tracing::info!(
        "stage 1 keeps {}",
        if classes.is_any() {
            "every class the detector reports".to_string()
        } else {
            classes.names().join(", ")
        }
    );
    // the pixels that arrive decide everything downstream, so both streams are
    // asked before anything is built at them.
    let main_source = crop_source(cfg, &source, &crop_file);
    let geometry = resolve_geometry(cfg, &source, &main_source);
    let (gate_w, gate_h) = geometry.gate;
    let mut gate = MotionGate::new(gate_w, gate_h, &cfg.gate);
    gate.apply_roi(&cfg.gate.roi);
    // only against a real camera: a file source has no ptz to ask, and an
    // offline replay has no wall clock for a poll interval to mean anything.
    let ptz = match (&source, cfg.camera.ptz_poll_secs) {
        (Source::Rtsp { .. }, secs) if secs > 0 => {
            camera::watch(&cfg.camera, std::time::Duration::from_secs(secs))
        }
        _ => None,
    };
    // **the substream is recorded too, when asked.** an end-to-end replay needs
    // it: the gate and the precision scan both read the substream, and it cannot
    // be derived from main by downscaling because the gate sees the camera's own
    // encode. so it rides along as a second output of the ffmpeg already pulling
    // it, exactly as the main stream's remux does.
    let record_sub = record_events && cfg.record.records(record::SUB_STREAM);
    // **the streams keep hours of their own** (r11.3). the harvest and the
    // recorder already have windows, but both sit downstream of the streams:
    // a shut harvest still leaves two ffmpegs pulling a street nobody is
    // looking at, which is r3.1 spent on nothing. closing the channels takes
    // them with it, because both are driven by frames -- while the process,
    // and everything already on disk, stay where they are (r8.5, r8.6).
    let hours = &cfg.stream.active_hours;
    let stream_open = hours.open_at(std::time::SystemTime::now());
    let channels = if stream_open {
        ingest::Channels::open()
    } else {
        ingest::Channels::shut()
    };
    // said before the first frame, because a run that starts outside its window
    // never sees an edge to report: the page has to be able to say so on load.
    stats.stream(!stream_open, hours.opens_at(std::time::SystemTime::now()));
    if !hours.is_always() {
        // echoed whether or not the window happens to be open now: a run that
        // is watching nothing has to say which of the two it is doing, and
        // this is the line that survives into a log where the shut moment was
        // twenty lines back.
        tracing::info!("stream window: only {hours} local");
    }
    if !stream_open {
        tracing::info!("{}", shut_line(hours));
    }
    if cfg.stream.unload_models && hours.is_always() {
        // a knob that is doing nothing should say so: the memory it was set for
        // is still being held, and the reason is a window that was never set.
        tracing::warn!(
            "[stream] unload_models has no window to unload with: active_hours is empty, so the stream never shuts"
        );
    }
    let mut gate_ingest = Ingest::gate(cfg, source.clone())
        .at_size(geometry.gate)
        .over(channels.clone())
        .watched_by(stats.clone(), true);
    let gate_relay = match record_sub {
        true => match open_relay(cfg.preview.viewer_backlog, "substream clips") {
            Some((relay, url)) => {
                gate_ingest = gate_ingest.remux_to(url);
                Some(relay)
            }
            None => None,
        },
        false => None,
    };
    let frames = if offline {
        gate_ingest.offline()
    } else {
        gate_ingest
    }
    .run();

    // `Some((origin, fps))` when replaying offline, where elapsed time has to be
    // derived from the frame number rather than read off the wall. it is also
    // what lets the crop feed be pulled to the gate's position, so it is settled
    // before either is built.
    let video_time = match (offline, &source) {
        (true, Source::File(p)) => match ingest::probe_fps(p) {
            Some(fps) => {
                tracing::info!("offline replay: clock driven by video time at {fps:.2} fps");
                Some((std::time::Instant::now(), fps))
            }
            None => {
                tracing::warn!(
                    "offline replay: cannot read the clip's frame rate, using wall clock"
                );
                None
            }
        },
        _ => None,
    };
    let clip_fps = video_time.map_or(1.0, |(_, fps)| fps);

    // a file stands in for both streams, so the same clip feeds the crop path
    // and the whole cascade can be exercised without a camera (r5.5).
    // the preview's video, when it is the remuxed main stream, rides along as a
    // second output of this same ffmpeg rather than a second rtsp session.
    // pulling the main stream twice cost another 4 mbit/s off the camera and
    // starved the gate over wifi, dropping detection from fifteen frames a
    // second to one. the listener is bound first so the url is known before
    // ffmpeg is spawned, and metermate is already accepting when it connects.
    let mut crop_ingest = Ingest::crops(cfg, main_source, cfg.stream.crop_fps)
        .at_size(geometry.main)
        .over(channels.clone())
        .watched_by(stats.clone(), false);
    let wants_remux = matches!(
        cfg.preview.source.video(&cfg.camera),
        config::PreviewVideo::ServerRemux
    ) && preview_addr != "off";
    // the recorder reads the same remuxed stream the preview does, so it needs
    // that output even with the preview off -- which is the normal way an
    // unattended deployment runs.
    let relay = match wants_remux || record_events {
        true => match open_relay(cfg.preview.viewer_backlog, "preview video and main clips") {
            Some((relay, url)) => {
                crop_ingest = crop_ingest.remux_to(url);
                Some(relay)
            }
            None => None,
        },
        false => None,
    };
    // the recorder is a second subscriber to the relay's fragments. it is
    // deliberately not fed from the crop decoder: those are decoded rgb frames,
    // and re-encoding them would cost more than everything else in the loop,
    // while the remux is a byte copy of what the camera already sent.
    // one feed per stream being recorded, each from the relay riding the ffmpeg
    // that already pulls it. a stream whose relay could not be opened is simply
    // absent: the clip then holds whatever did work, which is worth more than
    // refusing to record at all.
    let feeds: Vec<(&str, preview::Relay)> = [
        (record::MAIN_STREAM, relay.clone()),
        (record::SUB_STREAM, gate_relay.clone()),
    ]
    .into_iter()
    .filter(|(name, _)| cfg.record.records(name))
    .filter_map(|(name, r)| r.map(|r| (name, r)))
    .collect();
    let names: Vec<&str> = feeds.iter().map(|(n, _)| *n).collect();
    let mut recorder = match (record_events, feeds.is_empty()) {
        (true, false) => match record::Recorder::new(&cfg.record, &names) {
            Ok(rec) => {
                tracing::info!(
                    "recording event clips to {} ({} s pre-roll, {:.0} GB budget){}",
                    cfg.record.dir.display(),
                    cfg.record.preroll_secs,
                    cfg.record.max_bytes as f64 / 1e9,
                    // a directory that stays empty inside a window says the
                    // window, not the street, said no.
                    Hours::note(&cfg.record.active_hours),
                );
                tracing::info!("recording streams: {}", names.join(", "));
                let subs: Vec<(&str, Option<_>, preview::Relay)> = feeds
                    .iter()
                    .map(|(n, r)| (*n, r.subscribe(), r.clone()))
                    .collect();
                Some((rec, subs))
            }
            Err(e) => {
                tracing::warn!("event recording disabled: {e:#}");
                None
            }
        },
        (true, true) => {
            tracing::warn!(
                "event recording needs the remuxed main stream, which is not available; \
                 set [preview] source = \"main-h264\""
            );
            None
        }
        (false, _) => None,
    };

    // a replay has no wall clock for the two decoders to share, so the crop feed
    // is pulled to the gate's position rather than left to race ahead of it. a
    // clip whose rate could not be probed has no position to be pulled to, and
    // falls back to being paced at wall-clock like a camera.
    let mut crops = match video_time {
        Some(_) => ingest::Crops::replay(crop_ingest.offline(), cfg.stream.crop_fps as f64),
        None => ingest::Crops::live(crop_ingest),
    };

    // the models, which the stream window can take away with it (r11.3). a run
    // that starts outside its window and wants the memory back does not load
    // them at all: there is nothing to look at yet, and the reopening builds
    // them before the first frame is looked at.
    let mut models = if cfg.stream.unload_models && !stream_open {
        tracing::info!("stream window shut: models not loaded; they come up when it opens");
        Models::asleep(cfg)
    } else {
        Models::load(cfg, true)?
    };
    // the detector's input size is the one the config asked for -- it is what
    // `OnnxDetector::load` is handed -- so the shape of a frame is known even
    // while the session itself is not resident.
    let size = cfg.detector.input_size;
    let subjects = cfg.subjects();

    // a preview that cannot bind must not stop detection. the usual cause is a
    // second metermate already running, and losing the alert path over a
    // convenience feature would be absurd.
    // the events tab is offered when there is something to show *or* something
    // being recorded.
    //
    // not simply "is recording on". clips outlive the setting that wrote them,
    // and turning recording off -- which is the first thing anyone does when it
    // misbehaves -- must not also hide the clips already on disk. nor is it
    // "is `[record] enabled` set": that would offer an empty tab in exactly the
    // case needing explanation, recording configured on with no remuxed main
    // stream for it to read.
    let kept = record::list(&cfg.record.dir);
    let clips = (recorder.is_some() || !kept.is_empty()).then(|| preview::Clips {
        dir: cfg.record.dir.clone(),
        cache_dir: cfg.record.cache_dir.clone(),
        budget_bytes: cfg.record.max_bytes,
        preroll_ms: cfg.record.preroll_secs as u128 * 1000,
        max_clip_ms: cfg.record.max_clip_secs as u128 * 1000,
        recording: recorder.is_some(),
    });
    if recorder.is_none() && !kept.is_empty() {
        tracing::info!(
            "not recording, but {} event clips are on disk and still browsable",
            kept.len()
        );
    }
    let preview = match (preview_addr != "off").then(|| {
        preview::Preview::start(
            preview_addr,
            preview::Crops {
                dir: cfg.harvest.dir.clone(),
                budget_bytes: cfg.harvest.max_bytes,
                // what the config says, not what is resident: the crops on disk
                // were cut by the classifier whether or not it is loaded right
                // now, and the page is describing them.
                classifying: cfg.classifier.enabled,
            },
            clips,
            &cfg.preview,
            cfg.preview.source.video(&cfg.camera),
            relay.clone(),
            &cfg.gate.roi,
            &cfg.gate.perspective,
            geometry.gate,
            geometry.main,
            &cfg.subject_names(),
            stats.clone(),
        )
    }) {
        Some(Ok(p)) => Some(p),
        Some(Err(e)) => {
            tracing::warn!("preview disabled: {e:#}");
            None
        }
        None => None,
    };

    let mut alerting = harvest::Alerting::default();
    let mut harvester = if cfg.harvest.enabled {
        let h = harvest::Harvester::new(&cfg.harvest.dir, &cfg.harvest)?;
        // from the start, not from the first crop. how large the harvest has
        // grown is exactly the number that was invisible when it mattered, and
        // waiting for a vehicle to pass before reporting it is no use at 3am.
        let (files, bytes) = h.on_disk();
        stats.harvest_on_disk(files, bytes);
        tracing::info!(
            "harvesting crops to {} (budget {} MB, one per place per {}s, moved >= {:.2}){}",
            cfg.harvest.dir.display(),
            cfg.harvest.max_bytes / (1024 * 1024),
            cfg.harvest.min_interval_secs,
            cfg.harvest.must_have_moved,
            Hours::note(&cfg.harvest.active_hours),
        );
        Some(h)
    } else {
        None
    };

    let mut motion_on = false;
    let mut luma: Vec<u8> = Vec::new();
    // places the detector has looked at recently, so it does not re-examine the
    // same vehicle on every frame.
    let mut inspected: Vec<(gate::Rect, std::time::Instant)> = Vec::new();
    // vehicles that have stood in one place long enough to be scenery.
    let mut scenery = scenery::Scenery::new(&cfg.scenery);
    // identity across frames, so dwell and n-of-m confirmation have a subject
    // to be about. tracks whatever the detector reports, not only vehicles.
    let mut tracker = track::Tracker::new(&cfg.track);
    // boxes to draw, kept briefly so they do not flicker between detector runs.
    let mut overlay: Vec<OverlayBox> = Vec::new();
    // detector invocations in the last second, to cap throughput rather than
    // just per-frame count.
    let mut detector_runs: Vec<std::time::Instant> = Vec::new();
    // rolling cost of one inference, which is what that cap is derived from.
    // zero means "not measured yet": the first frame runs at the maximum rate
    // and the measurement immediately corrects it.
    let mut avg_inference = std::time::Duration::ZERO;
    // frames the decoder produced that this loop never looked at, counted from
    // gaps in the sequence number. the detector runs inline, so a slow machine
    // silently watches less of the street than a fast one, and that is worth
    // knowing rather than inferring (r8.2).
    let (mut seen, mut skipped, mut last_seq) = (0u64, 0u64, 0u64);

    // a frame already decoded and waiting, taken while deciding whether this
    // loop is behind, and used on the next turn rather than dropped.
    let mut queued: Option<ingest::Frame> = None;
    // frames the expensive stage ran on, against the frames the gate saw. the
    // gate now sees all of them, so the old skipped count is always zero and no
    // longer says anything; this is the number that does.
    let mut inspected_frames = 0u64;
    // frames where the gate fired and no main-stream frame existed to examine.
    let mut blind_looks = 0u64;
    // when the expensive stage last ran, so that being behind cannot starve it
    // completely. see `should_inspect`.
    let mut last_look: Option<std::time::Instant> = None;

    // **every frame, in order.** this used to take only the newest decoded frame
    // and discard the rest, which was right when the loop was cheap: a frame you
    // are late to is worth less than the one behind it. once the detector moved
    // inline that stopped being true of the *gate*, which is the part that has
    // to see every frame -- its background model is built from them, and a model
    // fed one frame in four adapts four times too slowly. measured on a clip
    // with continuous motion: the gate saw 26% of the stream.
    //
    // so the cheap stage runs on everything and the expensive stage runs only
    // when nothing newer is waiting.
    //
    // the window is asked about on every turn of the loop, and on a poll when
    // nothing is arriving at all -- which outside it is the normal state, and
    // a loop parked in `recv` would notice neither edge of it.
    let mut window_open = stream_open;
    loop {
        let open = hours.open_at(std::time::SystemTime::now());
        if open != window_open {
            window_open = open;
            // the open clip, if there is one, is the only piece of running
            // state the channels leaving takes with them.
            let clip = recorder.as_mut().map(|(rec, _)| rec);
            if open {
                // the models come back before the channels do, so the first
                // vehicle through a window that opened at seven is looked at
                // rather than waited for. a model that cannot be built again is
                // the pipeline having no way to do its job, and the run ends
                // saying so rather than watching a street it cannot read.
                models.reload()?;
                open_stream(&channels, &mut crops, &mut gate, &stats);
            } else {
                shut_stream(hours, &channels, clip, &mut crops, &stats);
                // after the channels are down: the weights are only let go once
                // nothing is looking at a frame (r11.3).
                models.unload(&stats);
            }
        }
        let frame = match queued.take() {
            Some(frame) => frame,
            None => {
                let poll = std::time::Duration::from_millis(config::STREAM_POLL_MS);
                match ingest::wait(&frames, poll) {
                    ingest::Wait::Frame(frame) => frame,
                    ingest::Wait::Idle => continue,
                    ingest::Wait::Ended => break,
                }
            }
        };
        if frame.is_probe() {
            continue;
        }
        seen += 1;
        let missed = frame.seq.saturating_sub(last_seq).saturating_sub(1);
        skipped += missed;
        last_seq = frame.seq;
        stats.gate_frame(missed);
        if seen.is_multiple_of(config::FRAME_REPORT_EVERY) {
            tracing::debug!(
                "{seen} frames examined, {skipped} skipped ({}% of the stream seen), \
                 {inspected_frames} inspected | {} places, {} tracks, {} boxes",
                seen * 100 / (seen + skipped).max(1),
                scenery.tracked(),
                tracker.tracks().len(),
                overlay.len()
            );
            stats.carrying(scenery.tracked(), tracker.tracks().len(), overlay.len());
        }
        // the camera moved, so the background model, the roi and every place
        // scenery is tracking now refer to somewhere else (r5.2).
        if camera::repointed(&ptz) {
            tracing::warn!("camera repointed: resetting the background model and scenery");
            gate.reset();
            scenery = scenery::Scenery::new(&cfg.scenery);
            inspected.clear();
            overlay.clear();
        }
        // the ingest delivers colour so the preview can use it; the gate wants
        // luma, which is a millisecond to derive and avoids a second decode.
        crop::rgb_to_luma(&frame.data, &mut luma);
        let gate_frame = ingest::Frame {
            width: frame.width,
            height: frame.height,
            data: std::sync::Arc::new(std::mem::take(&mut luma)),
            seq: frame.seq,
            received: frame.received,
        };
        let result = gate.update(&gate_frame);
        // every frame's motion, not only the frames that cleared the threshold.
        //
        // the same argument the harvest's `declined` line rests on: a
        // distribution truncated at the threshold cannot be used to choose the
        // threshold. `motion on`/`motion off` report the transitions, so the
        // only changed fractions visible were ones already above
        // `min_changed_frac`, and picking it from those is circular.
        // each region carries the share of its own pixels that changed, not
        // only its bounding box. a box is not an object: it contains road and
        // sky and whatever else it spans, so summing box areas overstates the
        // motion by however empty the boxes are -- which is exactly the mistake
        // to avoid when choosing a threshold that acts on changed pixels.
        tracing::debug!(
            "gate frame {} frac={:.5} motion={} regions={}",
            frame.seq,
            result.changed_frac,
            result.motion,
            result
                .regions
                .iter()
                .map(|r| format!(
                    "{},{},{}x{}@{:.3}",
                    r.x,
                    r.y,
                    r.w,
                    r.h,
                    gate.changed_fraction(*r)
                ))
                .collect::<Vec<_>>()
                .join(" ")
        );
        luma = std::sync::Arc::try_unwrap(gate_frame.data).unwrap_or_default();

        // an offline replay examines everything however long it takes, which is
        // what makes a replay independent of the machine (r5.5).
        if !offline {
            queued = ingest::pending(&frames);
        }

        // every judgement below about elapsed time -- has this vehicle been
        // standing here long enough to be scenery, was this place looked at
        // recently, is this crop too soon after the last -- is asked of *this*
        // clock. offline it runs on video time, because an offline replay takes
        // longer than the clip it is replaying, and answering "has it been
        // parked for 90 seconds" with wall-clock made every vehicle scenery and
        // suppressed the harvest entirely.
        let frame_at = video_time.map_or_else(std::time::Instant::now, |(start, fps)| {
            start + std::time::Duration::from_secs_f64(frame.seq as f64 / fps)
        });
        let caught_up = should_inspect(
            queued.is_none(),
            last_look.map_or(std::time::Duration::MAX, |t: std::time::Instant| {
                frame_at.saturating_duration_since(t)
            }),
            &cfg.detector,
        );

        overlay.retain(|b| {
            frame_at.duration_since(b.at)
                < std::time::Duration::from_millis(cfg.preview.overlay_linger_ms)
        });

        // feed the recorder before anything else looks at the frame, so the
        // ring already holds this moment if the gate is about to fire on it.
        if let Some((rec, subs)) = recorder.as_mut() {
            for (name, rx, relay) in subs.iter_mut() {
                // `init_needed` first: `relay.init()` clones the header under a
                // lock the relay thread also wants, and asking every frame for
                // something needed once is a lock the reader has to queue behind.
                if rec.init_needed()
                    && let Some(init) = relay.init()
                {
                    rec.set_init(name, init);
                }
                // **being dropped by the relay used to be permanent and
                // silent.** the subscription is bounded and the relay drops
                // whoever stops draining it, which is right -- the recorder must
                // never push back on the socket ffmpeg shares with the crop
                // feed. but the frame loop then read `Disconnected` forever
                // without noticing, so one slow stretch ended event recording
                // for the lifetime of the process while the log said nothing and
                // the events tab simply stopped filling. recovering is a
                // resubscribe.
                let mut lost = false;
                if let Some(chan) = rx.as_ref() {
                    // non-blocking: the fragments arrive on ffmpeg's schedule,
                    // not the gate's, and the frame loop must never wait.
                    loop {
                        match chan.try_recv() {
                            Ok(bytes) => {
                                if let Err(e) = rec.push(name, &bytes, frame_at) {
                                    tracing::warn!("event clip ({name}): {e:#}");
                                }
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => break,
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                lost = true;
                                break;
                            }
                        }
                    }
                }
                if lost {
                    tracing::warn!(
                        "the recorder fell behind the {name} stream and was dropped; resubscribing"
                    );
                    *rx = relay.subscribe();
                }
            }
            // the window opens and closes *clips*, not recording: the ring
            // keeps turning because the relay is already feeding it, and a
            // clip open when the window shuts finishes on its own terms.
            if result.motion
                && rec.ready()
                && cfg
                    .record
                    .active_hours
                    .open_at(std::time::SystemTime::now())
            {
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis());
                let already = rec.recording();
                match rec.trigger(frame_at, stamp) {
                    // one line per clip, not per frame of a passage: a vehicle
                    // crossing re-triggers on nearly every frame it is in.
                    Ok(()) if !already => tracing::info!("event clip {stamp} opened"),
                    Err(e) => tracing::warn!("event clip: {e:#}"),
                    Ok(()) => {}
                }
            }
            if let Err(e) = rec.tick(frame_at) {
                tracing::warn!("event clip: {e:#}");
            }
        }

        // the gate said something moved; look at it properly, in main-stream
        // pixels, where a distant vehicle is large enough to detect at all.
        //
        // **only when nothing newer is waiting.** the detector runs inline, so
        // this is also the stall: one inference is about two frame periods at
        // 15fps. running it on a frame that is already stale costs the gate the
        // frames behind it, and a vehicle is in shot for dozens of them -- the
        // next frame is a better place to spend an inference than this one.
        let main_frame = result
            .motion
            .then(|| crops.at(frame.video_secs(clip_fps)))
            .flatten();
        // **say so when the street is moving and there is nothing to look at.**
        //
        // the detector reads main-stream pixels, so with no main frame the whole
        // stage is skipped -- no detection, no classification, no crop -- and
        // this used to happen in silence. the preview goes on drawing motion
        // from the gate, which is a different stream on a different ffmpeg, so
        // the symptom is a street full of turquoise boxes with nothing in them
        // and a machine doing no work at all. that read as a detection bug for
        // an afternoon when the crop feed was simply not running.
        if result.motion {
            stats.motion();
        }
        if result.motion && main_frame.is_none() {
            blind_looks += 1;
            stats.blind_look();
            if blind_looks.is_multiple_of(config::FRAME_REPORT_EVERY) {
                tracing::warn!(
                    "motion {blind_looks} times with no main-stream frame to look at: \
                     the crop feed is not delivering, so nothing is being detected"
                );
            }
        }
        if caught_up && let Some(full) = main_frame {
            // the models are only ever absent while the window is shut, and a
            // shut stream delivers no frames, so a look reached with nothing
            // loaded is a broken invariant rather than a case to handle: saying
            // so is what keeps a pipeline that cannot see from looking like a
            // street that is empty.
            let detector = models
                .detector
                .as_mut()
                .expect("a look has the detector loaded");
            inspected_frames += 1;
            stats.inspected();
            stats.crop_frame(full.received.elapsed());
            last_look = Some(frame_at);
            // which two frames were paired. the detector's own line carries this
            // too, but only when it finds a vehicle, and a feed that has stopped
            // advancing is exactly the case where it does not.
            tracing::debug!("frame {} cropping from main frame {}", frame.seq, full.seq);
            let now = frame_at;
            inspected.retain(|(_, at)| {
                now.duration_since(*at)
                    < std::time::Duration::from_millis(cfg.detector.reinspect_ms)
            });
            detector_runs.retain(|at| now.duration_since(*at) < std::time::Duration::from_secs(1));
            let mut budget =
                inspections_allowed(offline, &cfg.detector, avg_inference, detector_runs.len());

            // the whole frame first, once. it finds near and mid vehicles on
            // this street at 0.80-0.92, which is as well as cropping them does,
            // so most frames need nothing further. `queue` then grows by
            // whatever that pass could not settle.
            let whole_gate = gate::Rect {
                x: 0,
                y: 0,
                w: frame.width,
                h: frame.height,
            };
            let whole_main = gate::Rect {
                x: 0,
                y: 0,
                w: full.width,
                h: full.height,
            };
            let mut queue: Vec<(gate::Rect, gate::Rect)> = vec![(whole_gate, whole_main)];
            let mut step = 0;

            while step < queue.len() {
                let (region, region_main) = queue[step];
                let whole_frame_pass = step == 0;
                step += 1;

                // the whole scene changed: the camera moved, or the light did.
                // there is no object here to detect, classify, or keep. the
                // whole-frame pass is exempt: covering the scene is its job.
                if !whole_frame_pass
                    && crop::covers_most_of(
                        region,
                        frame.width,
                        frame.height,
                        cfg.crop.max_region_fraction,
                    )
                {
                    continue;
                }
                // already looked here a moment ago, or out of budget for this
                // frame. either way, spending the time would cost a frame, and
                // a dropped frame is how a vehicle gets missed entirely.
                //
                // both of these draw a grey outline and nothing else, which is
                // indistinguishable in the preview from "looked, saw nothing".
                // say which, or a starved region looks like a detector failure
                // (r8.2).
                if budget == 0 {
                    tracing::debug!(
                        "region {region:?} skipped: spent this second's {} inferences ({:.0}ms each)",
                        detections_per_sec(avg_inference, &cfg.detector),
                        avg_inference.as_secs_f32() * 1000.0
                    );
                    continue;
                }
                if !whole_frame_pass
                    && inspected.iter().any(|(r, _)| {
                        crop::same_object(*r, region, cfg.detector.reinspect_tolerance)
                    })
                {
                    tracing::debug!("region {region:?} skipped: inspected within reinspect window");
                    continue;
                }
                budget -= 1;
                inspected.push((region, now));
                detector_runs.push(now);

                // one look at the scene. scenery counts occupancy against this,
                // so it has to be marked whether or not anything is found: a
                // place seen in one look out of a hundred is traffic, and a
                // place seen in ninety-nine is parked.
                scenery.tick();
                let began = std::time::Instant::now();
                let found = inspect_main(detector, &full, region_main, size)?;
                // the detector runs inline, so this is also how long the gate
                // loop stalled. at 15fps anything over 67ms costs frames, and it
                // is what the next second's budget is derived from.
                let took = began.elapsed();
                stats.inference(took, found.len());
                avg_inference = if avg_inference.is_zero() {
                    took
                } else {
                    avg_inference.mul_f32(1.0 - cfg.detector.inference_smoothing)
                        + took.mul_f32(cfg.detector.inference_smoothing)
                };

                // the whole-frame pass decides what still needs magnifying.
                if whole_frame_pass {
                    let in_gate: Vec<gate::Rect> = found
                        .iter()
                        .filter(|d| classes.keeps(d))
                        .map(|d| {
                            let m = crop::detection_to_main(
                                (d.x1, d.y1, d.x2, d.y2),
                                region_main,
                                size,
                                full.width,
                                full.height,
                            );
                            in_gate_of(m, &frame, &full)
                        })
                        .collect();
                    queue.extend(
                        follow_ups(&in_gate, &result.regions, cfg.detector.small_detection_px)
                            .into_iter()
                            .map(|r| {
                                (
                                    r,
                                    crop::to_main(
                                        r,
                                        frame.width,
                                        frame.height,
                                        full.width,
                                        full.height,
                                        &cfg.crop,
                                    ),
                                )
                            }),
                    );
                }
                if found.iter().any(|d| classes.keeps(d)) {
                    // iou and moved are the two numbers the harvest decides on,
                    // so report them: thresholds should be picked from what
                    // this street actually produces, not guessed (r8.2).
                    // the frame number, not just the timestamp: an offline
                    // replay's wall clock bears no relation to the clip's, so
                    // this is the only ordering an eval can trust.
                    // and which main-stream frame it was read from. the gate and
                    // the crop feed are separate decoders, so a skew between
                    // them is invisible in every other line of output -- and a
                    // skew is exactly what makes a detection land on the wrong
                    // moment of the street.
                    let (mut moving, mut parked) = (Vec::new(), 0usize);
                    for d in found.iter().filter(|d| classes.keeps(d)) {
                        let m = crop::detection_to_main(
                            (d.x1, d.y1, d.x2, d.y2),
                            region_main,
                            size,
                            full.width,
                            full.height,
                        );
                        let g = in_gate_of(m, &frame, &full);
                        if scenery.is_scenery(g, now) {
                            parked += 1;
                            continue;
                        }
                        moving.push(format!(
                            "{} {:.2} moved={:.2} box={},{} {}x{}",
                            d.label(),
                            d.confidence,
                            gate.changed_fraction(g),
                            g.x,
                            g.y,
                            g.w,
                            g.h
                        ));
                    }
                    // **debug, because it is a line per gated frame.** on a
                    // street with traffic that is fifteen a second, and the
                    // question it answers -- what did the detector see -- is
                    // asked of a replay rather than of a deployment. the
                    // tools that ask it (`tools/eval.py`, `tools/endtoend.py`)
                    // run the binary themselves and turn debug on.
                    match look_line(moving, parked) {
                        Some(said) => tracing::debug!(
                            "frame {} main {} region {:?} -> {said}",
                            frame.seq,
                            full.seq,
                            region
                        ),
                        // the whole look, parked vehicles included, is still
                        // one `RUST_LOG=metermate=debug` away: which of them
                        // the scenery model is holding is exactly what a
                        // "why was this never harvested" question needs.
                        None => tracing::debug!(
                            "frame {} main {} region {:?} -> {parked} parked, nothing moving",
                            frame.seq,
                            full.seq,
                            region
                        ),
                    }
                } else {
                    // the case the preview cannot show: motion was inspected and
                    // the detector genuinely saw no vehicle there. report the
                    // best non-vehicle guess, because "saw a bench at 0.31" and
                    // "saw nothing at all" point at different problems.
                    let best = found
                        .iter()
                        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                        .map(|d| format!("{} {:.2}", d.label(), d.confidence))
                        .unwrap_or_else(|| "nothing".into());
                    tracing::debug!(
                        "region {region:?} -> no vehicle, best was {best} ({}ms)",
                        took.as_millis()
                    );
                }

                // choose the vehicle the motion is about, then crop its own
                // box rather than the motion region. the region is "these
                // pixels changed", which routinely includes a garage door and
                // a shadow; and the most confident detection is often a parked
                // car that happens to be in shot.
                // score each vehicle by how much of *it* changed, per the
                // gate's mask, and let that decide which one the motion is
                // about. a parked car beside a passing one scores near zero.
                let vehicles: Vec<Candidate> = found
                    .iter()
                    .filter(|d| classes.keeps(d))
                    .map(|d| {
                        let m = crop::detection_to_main(
                            (d.x1, d.y1, d.x2, d.y2),
                            region_main,
                            size,
                            full.width,
                            full.height,
                        );
                        let rect = in_gate_of(m, &frame, &full);
                        // two questions, neither naming any object. did this
                        // vehicle's own pixels change, and has it been standing
                        // in this spot long enough to be scenery? the first
                        // catches a parked car that motion passed beside; the
                        // second catches one whose box is large enough to
                        // contain moving road, which no pixel test can.
                        Candidate {
                            parked: scenery.observe(rect, now),
                            changed: gate.changed_fraction(rect),
                            central: gate.central_share(rect),
                            rect,
                        }
                    })
                    .collect();
                // every vehicle that moved, not only the one that moved most.
                // two cars crossing at once is ordinary here, and taking the
                // maximum discarded the other outright rather than deferring it.
                let vehicle_boxes: Vec<&detect::Detection> =
                    found.iter().filter(|d| classes.keeps(d)).collect();
                // follow every vehicle this look found, whether or not it is
                // harvested: dwell is about the ones that stop, and those are
                // exactly the ones the harvest declines.
                let ids = tracker.update(now, &vehicles.iter().map(|v| v.rect).collect::<Vec<_>>());
                alerting.retain(|id| tracker.tracks().iter().any(|t| t.id == id));
                report_tracks(&mut tracker, &ids, now, alerter.as_ref(), cfg);

                let moving = moving_vehicles(
                    &vehicles,
                    cfg.harvest.must_have_moved,
                    cfg.harvest.motion_must_be_central,
                    cfg.harvest.max_per_frame,
                );

                // the window restricts the *harvest*, read as collecting
                // training data: the classifier above and the alerting below
                // run whatever the clock says, and a confirmation outside the
                // window still writes its own evidence crop -- r4.5 does not
                // keep office hours.
                let harvest_hours_open = cfg
                    .harvest
                    .active_hours
                    .open_at(std::time::SystemTime::now());

                // **a clip is worth what the pipeline acted on, not what was in
                // frame.** this street has parked cars permanently in view, so
                // "the detector found a class we watch" is true on nearly every
                // look: twenty of twenty-three clips on the deployed box were
                // named `detection`, which empties both the label and the
                // eviction order that depends on it. the harvest's own test is
                // the honest one -- it moved, and it is not scenery.
                if !moving.is_empty()
                    && let Some((rec, _)) = recorder.as_mut()
                {
                    rec.saw(record::Worth::Detection);
                }

                // what the threshold turned away, and its score.
                //
                // every harvested crop reports its own `moved`, so the smallest
                // number the eval can ever see is the threshold itself: the
                // distribution arrives pre-truncated, and picking a threshold
                // from it is circular. measured on three clips, the minimum
                // harvested `moved` was exactly 0.080 on all of them, which says
                // only that 0.08 is what was configured. this is the other side.
                for (i, v) in vehicles.iter().enumerate() {
                    if moving.contains(&i) {
                        continue;
                    }
                    let d = vehicle_boxes[i];
                    // `changed` rather than the ranking score, and the veto
                    // reported separately: a scenery veto and a genuinely still
                    // vehicle both rank zero, and pooling them would put the
                    // parked row at the bottom of the distribution this exists
                    // to show.
                    tracing::debug!(
                        "frame {} declined {} {:.2} changed={:.4} central={:.2} parked={} box={},{} {}x{} {}",
                        frame.seq,
                        d.label(),
                        d.confidence,
                        v.changed,
                        v.central,
                        v.parked,
                        v.rect.x,
                        v.rect.y,
                        v.rect.w,
                        v.rect.h,
                        if v.moved(cfg.harvest.motion_must_be_central) < cfg.harvest.must_have_moved
                        {
                            "below must_have_moved"
                        } else {
                            "over max_per_frame"
                        }
                    );
                    stats.declined();
                }

                for i in moving {
                    let Some(best) = vehicle_boxes.get(i) else {
                        continue;
                    };
                    let m = crop::detection_to_main(
                        (best.x1, best.y1, best.x2, best.y2),
                        region_main,
                        size,
                        full.width,
                        full.height,
                    );
                    // a clipped detection gets more surrounding context rather
                    // than being thrown away: the pixels are in the frame.
                    let margin = if crop::touches_edge(m, region_main, cfg.harvest.edge_slack_px) {
                        cfg.harvest.clipped_context
                    } else {
                        cfg.harvest.context
                    };
                    let bbox = crop::expand(m, margin, full.width, full.height);

                    // what stage two made of this crop, carried down to the
                    // harvest so the name records it.
                    //
                    // **the frame's own verdict, not the track's.** a crop is
                    // one look at one moment, and confirmation is a property of
                    // the track over five of them -- gating the name on it would
                    // mark the first crops of a passage differently from the
                    // last for a reason that is not about their pixels. the
                    // alert is the stricter question and stays where it is.
                    let mut verdict: Option<String> = None;
                    // and how far clear of the street that verdict was, which the
                    // crop's name carries off to the verdict page and whatever is
                    // written from these pixels.
                    let mut verdict_margin: Option<f32> = None;

                    // is this an enforcement vehicle? the detector only knows
                    // "vehicle"; this is the stage that knows "go-4".
                    if let Some(c) = models.classifier.as_mut() {
                        let pixels = crop::extract(&full.data, full.width, full.height, bbox);
                        // timed because it runs inline here, exactly as the
                        // detector does: what it costs is frame rate, and a
                        // stage nobody measures is a stage nobody can acquit.
                        let began = std::time::Instant::now();
                        let judged = c.classify(&pixels, bbox.w, bbox.h);
                        stats.classified(
                            began.elapsed(),
                            judged.as_ref().is_ok_and(|j| j.subject.is_some()),
                        );
                        // every verdict, not only the positive ones: n-of-m is a
                        // ratio, and feeding it just the hits would confirm on
                        // the first frame that guessed right (r1.4).
                        if let Ok(j) = &judged {
                            verdict = j.subject.clone();
                            verdict_margin = j.subject.is_some().then_some(j.margin());
                            if let Some(id) = ids.get(i) {
                                // the *name*, not a flag. a track that alternated
                                // between two subjects would otherwise confirm on
                                // five looks that never agreed, then publish under
                                // whichever happened to win the last one.
                                tracker.saw_subject(*id, j.subject.as_deref());
                            }
                        }
                        match judged {
                            // configured, not merely present in the reference
                            // file: a subject with references but no `[[subject]]`
                            // block has no topic to publish on and no discovery
                            // entity, so firing on it would be a sighting nobody
                            // can subscribe to.
                            Ok(j)
                                if subjects.iter().any(|s| j.is(&s.name))
                                    && ids.get(i).is_some_and(|id| {
                                        tracker.tracks().iter().find(|t| t.id == *id).is_some_and(
                                            |t| {
                                                j.subject.as_deref().is_some_and(|name| {
                                                    t.confirmed(name, cfg.track.confirm_m)
                                                })
                                            },
                                        )
                                    }) =>
                            {
                                if let Some((rec, _)) = recorder.as_mut() {
                                    rec.saw(record::Worth::Subject);
                                }
                                let subject = j.subject.clone().unwrap_or_default();
                                // whether this is the look that publishes, and the
                                // crops already on disk naming the subject.
                                let firing = ids
                                    .get(i)
                                    .map(|id| alerting.confirmed(*id, &subject))
                                    .unwrap_or_else(harvest::Confirmation::repeat);
                                let mut fired = firing.crops;
                                // **a confirmation with no crop behind it writes
                                // its own.** the harvest keeps one crop per place
                                // per `min_interval_secs`, so a track that sat
                                // still while it agreed could confirm with every
                                // agreeing look deduplicated away -- leaving a
                                // notification to describe something the verdict
                                // page has never seen (r4.5). `wants` is a question
                                // about variety, and evidence is not a variety
                                // question, so this goes straight to `save`.
                                if firing.first && fired.is_empty() {
                                    let evidence = harvest::Crop {
                                        rgb: &pixels,
                                        width: bbox.w,
                                        height: bbox.h,
                                        region: in_gate_of(bbox, &frame, &full),
                                        label: best.label(),
                                        confidence: best.confidence,
                                        subject: Some(subject.as_str()),
                                        margin: Some(j.margin()),
                                        at_millis: None,
                                    };
                                    fired = keep_evidence(
                                        harvester.as_mut(),
                                        &stats,
                                        evidence,
                                        frame.seq,
                                        now,
                                    );
                                }
                                if let Some(h) = harvester.as_ref()
                                    && let Err(e) = h.mark_alerted(&fired)
                                {
                                    tracing::warn!("{e:#}");
                                }
                                // **one alert per vehicle, and only with the crop
                                // that proved it** (r4.5). a track stays confirmed
                                // for the rest of its dwell, so every look of a
                                // parked go-4 used to publish the same message and
                                // buzz the same phone, and only a look that found
                                // the place unphotographed had a picture to attach
                                // -- which is what a notification the verdict page
                                // had never heard of looked like. a repeat says
                                // nothing new and is sent to nobody.
                                if firing.first {
                                    tracing::info!(
                                        "{} at {:?} (detector said {} {:.2}, {:.3} vs other {:.3})",
                                        subject.to_uppercase(),
                                        region,
                                        best.label(),
                                        best.confidence,
                                        j.score,
                                        j.other_score
                                    );
                                    match (&alerter, fired.first()) {
                                        // **the crop that named it goes to the phone**
                                        // (r4.3): a buzz saying "go4" is a claim, and
                                        // the picture is what makes moving the car a
                                        // decision rather than a walk to a laptop.
                                        (Some(a), Some(name)) => {
                                            let t = ids.get(i).and_then(|id| {
                                                tracker.tracks().iter().find(|t| t.id == *id)
                                            });
                                            let dwell =
                                                t.map_or(0.0, |t| t.dwell(now).as_secs_f32());
                                            a.publish(
                                                &subject,
                                                Outcome::Sighting,
                                                &alert::event_payload(&alert::Facts {
                                                    subject: &subject,
                                                    on: true,
                                                    track: t.map_or(0, |t| t.id),
                                                    dwell_s: dwell,
                                                    confidence: j.margin(),
                                                    // r9.1 once the protected vehicle
                                                    // exists; a fact, never a tier.
                                                    protected: alert::Protected::Unknown,
                                                    box_: in_gate_of(bbox, &frame, &full),
                                                }),
                                                alert::Told {
                                                    note: alert::ntfy::describe(
                                                        &subject,
                                                        dwell,
                                                        Some(j.margin()),
                                                    ),
                                                    crop: Some(cfg.harvest.dir.join(name)),
                                                },
                                            );
                                        }
                                        // nothing on disk, so nothing the verdict page
                                        // could show: a claim, and r4.5 refuses it.
                                        (Some(_), None) => tracing::warn!(
                                            "{subject} confirmed with nothing in {} to show for \
                                             it: not alerting",
                                            cfg.harvest.dir.display()
                                        ),
                                        (None, _) => {}
                                    }
                                }
                            }
                            Ok(_) => {}
                            // a classifier failure must not stop the harvest.
                            Err(e) => tracing::warn!("classify failed: {e:#}"),
                        }
                    }

                    // keep the pixels while we have them. even once the
                    // classifier works, the harvest is what improves it (r6.1).
                    let in_gate = in_gate_of(bbox, &frame, &full);
                    if let Some(h) = harvester.as_mut()
                        && harvest_hours_open
                        && h.wants(
                            in_gate,
                            frame.width,
                            frame.height,
                            cfg.crop.max_region_fraction,
                            now,
                        )
                    {
                        let pixels = crop::extract(&full.data, full.width, full.height, bbox);
                        let record = harvest::Crop {
                            rgb: &pixels,
                            width: bbox.w,
                            height: bbox.h,
                            region: in_gate,
                            label: best.label(),
                            confidence: best.confidence,
                            subject: verdict.as_deref(),
                            margin: verdict_margin,
                            at_millis: None,
                        };
                        match h.save(record, now) {
                            Ok(path) => {
                                if let (Some(v), Some(id), Some(name)) = (
                                    verdict.as_deref(),
                                    ids.get(i),
                                    path.file_name().and_then(|n| n.to_str()),
                                ) && let Err(e) =
                                    h.mark_alerted(&alerting.saved(*id, name.to_string(), v))
                                {
                                    tracing::warn!("{e:#}");
                                }
                                stats.harvested();
                                let (files, bytes) = h.on_disk();
                                stats.harvest_on_disk(files, bytes);
                                // the frame number, so an eval can ask whether a
                                // *particular* vehicle was harvested rather than
                                // only how many crops came out. crop filenames
                                // carry wall-clock, which means nothing in an
                                // offline replay running on video time.
                                tracing::info!(
                                    "harvested frame {} {} {:.2} at {},{} {}x{} moved={:.3}",
                                    frame.seq,
                                    best.label(),
                                    best.confidence,
                                    in_gate.x,
                                    in_gate.y,
                                    in_gate.w,
                                    in_gate.h,
                                    vehicles[i].moved(cfg.harvest.motion_must_be_central)
                                );
                                if h.written().is_multiple_of(config::HARVEST_REPORT_EVERY) {
                                    tracing::info!("{} crops harvested", h.written());
                                }
                            }
                            // a failed write must never stop detection.
                            Err(e) => tracing::warn!("harvest write failed: {e:#}"),
                        }
                    }
                }

                // collect boxes in gate coordinates so the preview can draw
                // them on the scene rather than on a disembodied crop.
                //
                // each carries the two numbers that decide its fate, because the
                // whole-frame pass returns every vehicle in view and the harvest
                // acts on at most one of them. without these the overlay draws
                // six identical boxes and says nothing about which mattered.
                if preview.is_some() {
                    for d in &found {
                        let m = crop::detection_to_main(
                            (d.x1, d.y1, d.x2, d.y2),
                            region_main,
                            size,
                            full.width,
                            full.height,
                        );
                        let in_gate = in_gate_of(m, &frame, &full);
                        remember_detection(
                            &mut overlay,
                            OverlayBox {
                                rect: in_gate,
                                label: d.label(),
                                confidence: d.confidence,
                                moved: gate.changed_fraction(in_gate),
                                parked: scenery.is_scenery(in_gate, now),
                                at: now,
                                vx: 0.0,
                                vy: 0.0,
                            },
                            cfg.preview.overlay_same_box,
                        );
                    }
                }
            }
        }

        // the scene, every frame, in colour. a preview that only updated during
        // motion showed nothing on a quiet street, and one fed from the crop
        // stream updated only a few times a second.
        // the downscale and the jpeg encode happen on the preview's thread. the
        // handover is an `Arc` clone and a few hundred bytes of json, so a
        // browser -- however many, in whatever state -- cannot cost the pipeline
        // a frame.
        if let Some(p) = preview.as_ref() {
            p.offer(
                frame.data.clone(),
                frame.width,
                frame.height,
                overlay_json(
                    &overlay,
                    &result.regions,
                    frame.width,
                    frame.height,
                    &cfg.preview,
                    frame_at,
                ),
            );
        }

        // publish transitions only. a vehicle crossing the frame is one event,
        // not one per frame.
        if result.motion == motion_on {
            continue;
        }
        motion_on = result.motion;
        let payload = alert::motion_payload(motion_on, result.changed_frac, &result.regions);
        tracing::info!(
            "motion {} frac={:.4} regions={:?}",
            if motion_on { "on" } else { "off" },
            result.changed_frac,
            result.regions
        );
        if let Some(a) = &alerter {
            a.publish_motion(&payload);
        }
    }
    // a clip still open when the stream runs out had its worth decided and then
    // thrown away. it is written under a provisional name until it closes, so
    // leaving it is not untidiness: nothing lists it, nothing evicts it, and
    // nothing can play it. a process killed outright still leaves one, which is
    // what `adopt_partials` picks up at the next start; this is the case we can
    // finish honestly rather than guess at.
    if let Some((rec, _)) = recorder.as_mut()
        && let Err(e) = rec.close()
    {
        tracing::warn!("closing the open event clip: {e:#}");
    }
    // **the whole report, once, as json.** an eval reads this back: a change
    // that keeps recall and halves the frame rate is a regression, and nothing
    // before this could see it.
    tracing::info!("stats {}", stats.json(0));

    // the same accounting as the periodic line, once at the end.
    //
    // the periodic one is keyed on frames *examined*, so a loop that has fallen
    // far enough behind never reaches the interval and reports nothing -- the
    // worse it is doing, the quieter it gets. a run against a short clip could
    // therefore drop three frames in four and say so nowhere.
    tracing::info!(
        "stream ended: {seen} frames examined, {skipped} skipped ({}% of the stream seen), \
         {inspected_frames} inspected",
        seen * 100 / (seen + skipped).max(1)
    );
    Ok(())
}

/// a detection box expressed in gate coordinates.
///
/// the harvest rate-limits on this rather than on the motion region. the region
/// drifts as shadows and light change, which reads as a new place and let the
/// same parked car through repeatedly; the vehicle's own box is stable.
/// what one look found, for the log: what moved, and how many parked vehicles
/// were standing there as usual.
///
/// **a parked car is not an event.** this street has several permanently in
/// frame and every gated look rediscovers all of them, so spelling each one out
/// made the info log mostly a list of things that had not changed since the
/// last frame -- which is what buries the one line about something that did.
/// they are still counted, because "nothing moved" and "nothing is there" are
/// different, and a detector that has gone blind looks like the second.
fn look_line(moving: Vec<String>, parked: usize) -> Option<String> {
    if moving.is_empty() {
        return None;
    }
    let mut said = moving.join(" | ");
    if parked > 0 {
        said += &format!(" | {parked} parked");
    }
    Some(said)
}

fn in_gate_of(bbox: gate::Rect, frame: &ingest::Frame, full: &ingest::Frame) -> gate::Rect {
    crop::main_to_gate(bbox, frame.width, frame.height, full.width, full.height)
}

/// write the crop a confirmation rests on, and name it for the phone and the page.
///
/// the look that confirmed is the one stage two agreed on, so these are the pixels
/// the notification attaches and the verdict page lists under the subject in its
/// name. everything else about the harvest -- one crop per place, the region
/// guards, the budget -- is about variety, and evidence is not a variety question
/// (r4.5), which is why this goes to `save` without asking `wants`.
fn keep_evidence(
    harvester: Option<&mut harvest::Harvester>,
    stats: &stats::Stats,
    record: harvest::Crop,
    seq: u64,
    now: std::time::Instant,
) -> Vec<String> {
    let Some(h) = harvester else {
        tracing::warn!("nothing harvesting, so nothing to show for the alert");
        return Vec::new();
    };
    let subject = record.subject.unwrap_or_default().to_string();
    let confidence = record.confidence;
    match h.save(record, now) {
        Ok(path) => {
            stats.harvested();
            let (files, bytes) = h.on_disk();
            stats.harvest_on_disk(files, bytes);
            tracing::info!(
                "harvested frame {seq} {subject} {:.2} as the evidence behind the alert",
                confidence
            );
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| vec![n.to_string()])
                .unwrap_or_default()
        }
        // a failed write costs the notification, which is the whole point (r4.5).
        Err(e) => {
            tracing::warn!("harvest write failed: {e:#}");
            Vec::new()
        }
    }
}

/// run the detector over one rectangle of the main-stream frame.
///
/// the rectangle is given in main-stream pixels rather than derived from a gate
/// region, because the same call serves both halves of the cascade: the whole
/// frame, and a magnified crop of one part of it.
fn inspect_main(
    detector: &mut dyn detect::Detector,
    full: &ingest::Frame,
    box_main: gate::Rect,
    size: u32,
) -> Result<Vec<detect::Detection>> {
    let pixels = crop::extract(&full.data, full.width, full.height, box_main);
    let input = crop::letterbox(&pixels, box_main.w, box_main.h, size);
    detector.detect(&input, size)
}

/// do two rectangles share any pixel?
fn overlaps(a: gate::Rect, b: gate::Rect) -> bool {
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
}

/// where to look closely, after looking at the whole frame once.
///
/// the whole-frame pass finds near and mid vehicles at 0.80-0.92 on this street,
/// so cropping them again buys nothing. two cases still need magnification:
///
/// - **a motion region the whole-frame pass found nothing in.** something moved
///   and the detector did not see it, which is exactly the far-side case: a go-4
///   across the street is ~50px in a 640 whole-frame input, under the floor, and
///   ~200px in a crop.
/// - **a detection that came back small.** it was found, but near enough to the
///   floor that its box and class are not to be trusted.
///
/// takes and returns gate coordinates, so it can be reasoned about against the
/// motion regions without any mapping.
fn follow_ups(detections: &[gate::Rect], regions: &[gate::Rect], small_px: u32) -> Vec<gate::Rect> {
    let mut out: Vec<gate::Rect> = regions
        .iter()
        .filter(|r| !detections.iter().any(|d| overlaps(*d, **r)))
        .copied()
        .collect();
    out.extend(
        detections
            .iter()
            .filter(|d| d.w.max(d.h) < small_px)
            .copied(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// **a parked car is not an event.** this street has them permanently in
    /// frame, so a line per look per parked vehicle is most of the log, and it
    /// says the same thing every time -- which buries the one line about
    /// something that actually moved.
    #[test]
    fn a_look_reports_what_moved_and_counts_what_did_not() {
        let said = look_line(vec!["car 0.91 moved=0.31".into()], 3);
        assert_eq!(said.as_deref(), Some("car 0.91 moved=0.31 | 3 parked"));
    }

    /// counted rather than dropped: "nothing moved here" and "nothing is here"
    /// are different, and a detector that has stopped seeing the street looks
    /// like the second one.
    #[test]
    fn a_look_that_only_found_parked_vehicles_says_nothing() {
        assert_eq!(look_line(Vec::new(), 4), None);
        assert_eq!(look_line(Vec::new(), 0), None);
    }

    #[test]
    fn nothing_is_said_about_parked_vehicles_when_there_are_none() {
        let said = look_line(vec!["truck 0.80 moved=0.42".into()], 0);
        assert_eq!(said.as_deref(), Some("truck 0.80 moved=0.42"));
    }

    const SMALL_DETECTION_PX: u32 = config::SMALL_DETECTION_PX;
    const OVERLAY_SAME_BOX: f32 = config::OVERLAY_SAME_BOX;
    use std::time::Duration;

    const MIN_DETECTIONS_PER_SEC: usize = config::MIN_DETECTIONS_PER_SEC;
    const MAX_DETECTIONS_PER_SEC: usize = config::MAX_DETECTIONS_PER_SEC;

    fn det() -> config::DetectorCfg {
        config::DetectorCfg::default()
    }

    fn at(x: u32, y: u32, w: u32, h: u32) -> gate::Rect {
        gate::Rect { x, y, w, h }
    }

    /// the reported symptom: "tons of overlapping bounding boxes". one car,
    /// inspected repeatedly inside the linger window, must be one box.
    /// **a box is drawn where the vehicle is, not where it was detected.**
    ///
    /// detections linger for `overlay_linger_ms` so boxes do not flicker between
    /// detector runs, which is right -- the detector looks a few times a second
    /// and the preview draws at fifteen. but a lingering box is stale by up to
    /// that whole window, and the motion region beside it is from the current
    /// frame, so on a crossing vehicle the two visibly separate: the turquoise
    /// tracks the car and the green trails it.
    ///
    /// the drift is known without asking the tracker. the box a detection
    /// replaces *is* the same vehicle a moment earlier, so the two together give
    /// a velocity, and a stale box can be carried forward by it.
    #[test]
    fn a_lingering_box_is_carried_forward_by_its_own_drift() {
        let mut overlay: Vec<OverlayBox> = Vec::new();
        let t = Instant::now();
        let car = |rect, at| OverlayBox {
            rect,
            label: "car",
            confidence: 0.8,
            moved: 0.2,
            parked: false,
            at,
            vx: 0.0,
            vy: 0.0,
        };
        remember_detection(&mut overlay, car(at(100, 100, 80, 60), t), OVERLAY_SAME_BOX);
        // 40 gate pixels right in a quarter second: 160 px/s.
        remember_detection(
            &mut overlay,
            car(at(140, 100, 80, 60), t + Duration::from_millis(250)),
            OVERLAY_SAME_BOX,
        );
        assert_eq!(overlay.len(), 1, "{overlay:?}");
        assert!(
            (overlay[0].vx - 160.0).abs() < 1.0,
            "drift not measured: {:?}",
            overlay[0].vx
        );
        assert!(overlay[0].vy.abs() < 1.0, "invented vertical drift");

        // drawn a quarter second after that detection, it belongs 40px further
        // on again.
        let drawn = overlay_json(
            &overlay,
            &[],
            640,
            480,
            &config::PreviewCfg {
                width: 640,
                height: 480,
                ..Default::default()
            },
            t + Duration::from_millis(500),
        );
        let x1 = drawn
            .split(r#""x1":"#)
            .nth(1)
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(f32::NAN);
        assert!(
            (x1 - 180.0).abs() < 2.0,
            "box drawn at {x1}, expected ~180 after carrying 160px/s for 250ms"
        );
    }

    /// **the option takes boxes away, not marks off them.** the feed goes out
    /// every frame to every viewer, and on a street where most of what the
    /// detector reports is kerb, `show_parked = false` is worth the bytes too.
    #[test]
    fn scenery_is_left_out_of_the_overlay_when_asked_that_way() {
        let t = Instant::now();
        let one = |parked| OverlayBox {
            rect: at(if parked { 300 } else { 100 }, 100, 80, 60),
            label: "car",
            confidence: 0.8,
            moved: if parked { 0.0 } else { 0.4 },
            parked,
            at: t,
            vx: 0.0,
            vy: 0.0,
        };
        let boxes = vec![one(false), one(true)];
        let size = config::PreviewCfg {
            width: 640,
            height: 480,
            ..Default::default()
        };

        let drawn = overlay_json(&boxes, &[], 640, 480, &size, t);
        assert!(drawn.contains(r#""parked":true"#), "{drawn}");

        let quiet = config::PreviewCfg {
            show_parked: false,
            ..size
        };
        let drawn = overlay_json(&boxes, &[], 640, 480, &quiet, t);
        assert!(!drawn.contains(r#""parked":true"#), "{drawn}");
        assert!(
            drawn.contains(r#""parked":false"#),
            "the moving one went too: {drawn}"
        );
    }

    #[test]
    fn re_detecting_one_car_does_not_stack_up_boxes() {
        let mut overlay: Vec<OverlayBox> = Vec::new();
        let t = Instant::now();

        // the same car, drifting a little between inspections a quarter of a
        // second apart, as it actually does.
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(100, 100, 80, 60),
                label: "car",
                confidence: 0.81,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(108, 103, 82, 61),
                label: "car",
                confidence: 0.84,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(115, 106, 79, 58),
                label: "car",
                confidence: 0.79,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );

        assert_eq!(overlay.len(), 1, "boxes stacked up: {overlay:?}");
        assert_eq!(overlay[0].rect.x, 115, "the newest box should have won");
        assert!((overlay[0].confidence - 0.79).abs() < 1e-6);
    }

    /// the counterweight: dedup must not swallow a second vehicle. two cars in
    /// frame have to stay two boxes, or the preview lies in the other direction.
    #[test]
    fn two_separate_vehicles_keep_their_own_boxes() {
        let mut overlay: Vec<OverlayBox> = Vec::new();
        let t = Instant::now();

        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(100, 100, 80, 60),
                label: "car",
                confidence: 0.81,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(400, 300, 80, 60),
                label: "truck",
                confidence: 0.77,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        assert_eq!(overlay.len(), 2, "a second vehicle was swallowed");

        // and re-detecting the first still leaves the second alone.
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(104, 102, 80, 60),
                label: "car",
                confidence: 0.83,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        assert_eq!(overlay.len(), 2, "{overlay:?}");
    }

    /// a car covered by two overlapping motion regions is inspected twice in the
    /// same frame and reported twice. that is the other half of the thicket.
    #[test]
    fn one_car_found_by_two_regions_is_drawn_once() {
        let mut overlay: Vec<OverlayBox> = Vec::new();
        let t = Instant::now();
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(200, 150, 90, 70),
                label: "car",
                confidence: 0.66,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        remember_detection(
            &mut overlay,
            OverlayBox {
                rect: at(205, 148, 88, 72),
                label: "car",
                confidence: 0.71,
                moved: 0.0,
                parked: false,
                at: t,
                vx: 0.0,
                vy: 0.0,
            },
            OVERLAY_SAME_BOX,
        );
        assert_eq!(overlay.len(), 1, "{overlay:?}");
    }

    /// the point of the hybrid: a vehicle the whole-frame pass already found at
    /// a comfortable size does not need cropping again.
    #[test]
    fn a_big_whole_frame_detection_needs_no_second_look() {
        let car = at(100, 100, 180, 120);
        let region = at(110, 105, 150, 100);
        assert!(
            follow_ups(&[car], &[region], SMALL_DETECTION_PX).is_empty(),
            "a well-found car was re-inspected for nothing"
        );
    }

    /// the far-side case, and the reason slicing is kept at all: something moved
    /// and the whole-frame pass saw nothing there.
    #[test]
    fn motion_the_whole_frame_pass_missed_is_looked_at_closely() {
        let car = at(100, 100, 180, 120);
        let far = at(500, 60, 40, 20);
        let got = follow_ups(&[car], &[at(110, 105, 150, 100), far], SMALL_DETECTION_PX);
        assert_eq!(got, vec![far], "the missed region was not followed up");
    }

    /// found, but only just: near the measured floor its box and class are not
    /// worth trusting without a closer look.
    #[test]
    fn a_small_detection_is_looked_at_again() {
        let small = at(500, 60, 50, 30);
        let got = follow_ups(&[small], &[at(495, 55, 60, 40)], SMALL_DETECTION_PX);
        assert_eq!(got, vec![small], "a near-floor detection was left alone");
    }

    /// and the boundary is the constant, not a coincidence of the examples.
    #[test]
    fn the_small_threshold_is_where_it_says_it_is() {
        let just_under = at(0, 0, SMALL_DETECTION_PX - 1, 20);
        let just_over = at(0, 0, SMALL_DETECTION_PX, 20);
        assert_eq!(follow_ups(&[just_under], &[], SMALL_DETECTION_PX).len(), 1);
        assert!(follow_ups(&[just_over], &[], SMALL_DETECTION_PX).is_empty());
    }

    /// **the detector was blind on 59% of the frames the gate had just fired on.**
    ///
    /// measured over 120 seconds of this camera: 292 frames where motion fired
    /// and a main frame was paired, 171 of which spent the second's allowance
    /// before the whole-frame pass ran. motion is bursty -- a car crossing fires
    /// the gate on ~30 consecutive frames -- so the allowance runs out exactly
    /// while a vehicle is in front of the camera, and the preview draws a motion
    /// box with no detection in it.
    ///
    /// the look at the whole frame is the only one that covers the street, so it
    /// is not optional. follow-ups are magnification: giving one up costs
    /// confidence on a distant vehicle rather than the vehicle itself.
    #[test]
    fn a_frame_the_gate_fired_on_always_gets_its_look_at_the_street() {
        let cfg = det();
        let slow = std::time::Duration::from_millis(500);
        let spent = detections_per_sec(slow, &cfg);
        assert!(
            inspections_allowed(false, &cfg, slow, spent * 2) >= 1,
            "a frame with motion in it was given no inference at all"
        );
        assert_eq!(
            inspections_allowed(false, &cfg, slow, spent * 2),
            1,
            "a starved second should buy the whole-frame pass and nothing more"
        );
    }

    /// **and it has an off switch, because it costs frame rate.**
    ///
    /// the detector runs inline, so guaranteeing a look on every motion frame
    /// also guarantees the gate loop stalls for one inference on every motion
    /// frame -- 130ms against a 67ms frame period at 15fps and two onnx threads.
    /// on a host where inference does not fit inside a frame that is a worse
    /// trade than the bug it fixes, because a stalled gate loses frames outright
    /// where a skipped inference only lost a look. it showed up in production as
    /// the preview falling seconds behind.
    ///
    /// so the reservation is configurable, and turning it off restores exactly
    /// the old arithmetic rather than something near it.
    #[test]
    fn the_whole_frame_reservation_can_be_turned_off() {
        let mut cfg = det();
        cfg.always_inspect_whole_frame = false;
        let slow = std::time::Duration::from_millis(500);
        let spent = detections_per_sec(slow, &cfg);
        assert_eq!(
            inspections_allowed(false, &cfg, slow, spent * 2),
            0,
            "with the reservation off, a spent second buys nothing"
        );
        // and it is the reservation doing it, not the clamp.
        cfg.always_inspect_whole_frame = true;
        assert_eq!(inspections_allowed(false, &cfg, slow, spent * 2), 1);
    }

    /// and with the second's allowance untouched, follow-ups are still capped by
    /// the per-frame limit rather than by whatever is left of the rate.
    #[test]
    fn a_quiet_second_still_respects_the_per_frame_cap() {
        let cfg = det();
        let quick = std::time::Duration::from_millis(10);
        assert_eq!(
            inspections_allowed(false, &cfg, quick, 0),
            cfg.max_regions_per_frame
        );
        assert_eq!(
            inspections_allowed(true, &cfg, quick, 9_999),
            cfg.max_regions_per_frame,
            "an offline replay is not racing a clock"
        );
    }

    /// the rate has to follow the machine, not a constant tuned on one of them.
    #[test]
    fn the_detector_rate_follows_measured_inference_cost() {
        use std::time::Duration;
        // the development laptop: 148ms an inference.
        assert_eq!(detections_per_sec(Duration::from_millis(148), &det()), 4);
        // a faster host should be allowed to look more often, which is what
        // "caught late" was about.
        assert_eq!(detections_per_sec(Duration::from_millis(60), &det()), 10);
        // and a slow one must still look sometimes.
        assert_eq!(
            detections_per_sec(Duration::from_millis(2000), &det()),
            MIN_DETECTIONS_PER_SEC
        );
        // an absurdly fast one must not be allowed to run flat out.
        assert_eq!(
            detections_per_sec(Duration::from_millis(1), &det()),
            MAX_DETECTIONS_PER_SEC
        );
    }

    /// before anything has been measured, do not stall waiting for a number.
    #[test]
    fn an_unmeasured_detector_starts_at_full_rate() {
        assert_eq!(
            detections_per_sec(std::time::Duration::ZERO, &det()),
            MAX_DETECTIONS_PER_SEC
        );
    }

    /// a vehicle whose motion is its own: fully central, nothing clipping it.
    fn moved(rect: gate::Rect, changed: f32) -> Candidate {
        Candidate {
            rect,
            changed,
            central: 1.0,
            parked: false,
        }
    }

    /// two vehicles crossing at the same time is not unusual on this street, and
    /// both are worth having. taking only the one that moved most meant the
    /// other was discarded outright.
    #[test]
    fn two_vehicles_moving_at_once_are_both_harvested() {
        let vehicles = [
            moved(at(100, 300, 160, 120), 0.21), // a car in the foreground
            moved(at(520, 60, 90, 70), 0.14),    // another crossing behind it
            moved(at(300, 200, 140, 100), 0.01), // parked, beside the moving one
        ];
        let got = moving_vehicles(&vehicles, 0.08, 0.2, 4);
        assert_eq!(got.len(), 2, "a simultaneous vehicle was dropped: {got:?}");
        assert!(got.contains(&0) && got.contains(&1));
        assert!(!got.contains(&2), "a stationary vehicle was harvested");
    }

    /// ordered by movement, so that when the per-frame cap bites it keeps the
    /// clearest movement rather than whatever the detector happened to list
    /// first.
    #[test]
    fn the_cap_keeps_the_most_movement() {
        let vehicles = [
            moved(at(0, 0, 50, 50), 0.10),
            moved(at(100, 0, 50, 50), 0.40),
            moved(at(200, 0, 50, 50), 0.25),
        ];
        assert_eq!(moving_vehicles(&vehicles, 0.08, 0.2, 2), vec![1, 2]);
        assert_eq!(moving_vehicles(&vehicles, 0.08, 0.2, 1), vec![1]);
    }

    #[test]
    fn a_frame_with_nothing_moving_harvests_nothing() {
        let vehicles = [
            moved(at(0, 0, 50, 50), 0.0),
            moved(at(100, 0, 50, 50), 0.03),
        ];
        assert!(moving_vehicles(&vehicles, 0.08, 0.2, 4).is_empty());
    }

    /// scenery's veto outranks the pixels. a box large enough to contain moving
    /// road -- the van in the foreground -- clears any threshold that also
    /// admits real vehicles, which is the whole reason the veto exists.
    #[test]
    fn a_vehicle_vetoed_as_scenery_is_not_harvested_however_much_changed() {
        let vehicles = [Candidate {
            rect: at(100, 300, 160, 120),
            changed: 0.62,
            central: 1.0,
            parked: true,
        }];
        assert!(
            moving_vehicles(&vehicles, 0.08, 0.2, 4).is_empty(),
            "scenery was harvested because traffic crossed its box"
        );
        // and the raw number survives for the log, so the two ways of scoring
        // zero stay distinguishable in the output.
        assert_eq!(vehicles[0].changed, 0.62);
        assert_eq!(vehicles[0].moved(0.2), 0.0);
    }

    #[test]
    fn overlap_is_not_confused_by_touching_edges() {
        assert!(overlaps(at(0, 0, 10, 10), at(5, 5, 10, 10)));
        assert!(!overlaps(at(0, 0, 10, 10), at(10, 0, 10, 10)), "touching");
        assert!(!overlaps(at(0, 0, 10, 10), at(20, 20, 5, 5)));
    }
}
