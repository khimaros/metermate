//! save full-resolution vehicle crops, so a training set accumulates (r6.1).
//!
//! this is the critical path of the whole project, and it is bound by
//! wall-clock rather than by work: enforcement passes a few times a day, so
//! every day the harvester is not running is a day of data that cannot be
//! recovered later. it therefore runs from the earliest phase, long before
//! anything can classify what it collects.
//!
//! two things stop it being useless. it must not fill the disk (r6.4), and it
//! must not write two thousand near-identical crops of one parked car.

pub mod dense;

use crate::crop::{covers_most_of, same_object};
use crate::gate::Rect;
use anyhow::{Context, Result};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// what is known about one harvest directory: how many times this process has
/// changed it, and the last listing taken of it.
///
/// keyed by directory rather than global. a global counter would have one
/// harvest's writes invalidate another's listing, which is wrong in principle
/// and observable in the tests, where several harvests exist at once.
#[derive(Default)]
struct DirState {
    generation: u64,
    listing: Option<(u64, Instant, Vec<Saved>)>,
    /// walks of *this* directory. per directory rather than global because the
    /// tests run in parallel over several harvests at once, and a global count
    /// makes each one's assertion depend on what the others happened to do.
    walks: u64,
}

static DIRS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<PathBuf, DirState>>> =
    std::sync::LazyLock::new(Default::default);

/// how long a listing is trusted when nothing in this process has changed the
/// harvest. only for changes made behind our back -- somebody archiving crops
/// by hand -- since anything metermate does bumps the generation.
const LISTING_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// forget the cached listing of `dir`, so the next ask walks it.
///
/// a listing is trusted for `LISTING_TTL` because crops usually arrive from this very
/// process, and thirty seconds of a stale answer to a page redrawing itself costs
/// nobody anything. somebody asking "is there more?" is the exception: that is a
/// question about the directory, and a cached answer says there is nothing new while
/// the pipeline has been harvesting for half a minute.
pub fn forget(dir: &Path) {
    changed(dir);
}

/// note that this process has changed `dir`, so any cached listing is stale.
fn changed(dir: &Path) {
    if let Ok(mut dirs) = DIRS.lock() {
        let e = dirs.entry(dir.to_path_buf()).or_default();
        e.generation = e.generation.wrapping_add(1);
        e.listing = None;
    }
}

/// walks of `dir` so far. asserted on, because "it only reads the directory
/// when it has to" is otherwise a claim nothing checks.
#[cfg(test)]
fn listing_walks(dir: &Path) -> u64 {
    DIRS.lock()
        .ok()
        .and_then(|d| d.get(dir).map(|e| e.walks))
        .unwrap_or(0)
}

/// a record of something recently saved, used to avoid saving it again.
struct Recent {
    region: Rect,
    at: Instant,
}

/// one crop and everything known about it at capture time.
pub struct Crop<'a> {
    pub rgb: &'a [u8],
    pub width: u32,
    pub height: u32,
    /// where it came from in gate coordinates, for rate limiting.
    pub region: Rect,
    pub label: &'a str,
    pub confidence: f32,
    /// what stage two made of these very pixels, when it ran and recognised
    /// something. the detector class is what it looks like generically; this is
    /// the answer the project exists for, and it is known here or nowhere.
    pub subject: Option<&'a str>,
    /// how far clear of its negatives stage two was on these pixels, signed, and
    /// `None` when there was no verdict to go with it. it is the number a
    /// notification quotes and the number a person tunes the bar against, so it
    /// outlives the frame that measured it.
    pub margin: Option<f32>,
    /// when the frame was taken, for a crop cut from a recording. `None` names
    /// the crop for the moment it is written, which is the live harvest.
    pub at_millis: Option<u128>,
}

/// the crops a `labels.txt` names, re-read only when the file changes.
///
/// a person labels in another process while this one harvests, so a label has
/// to reach eviction without a restart -- and eviction runs on the frame loop,
/// so it must not cost a read per crop either.
struct Kept {
    path: PathBuf,
    seen: Option<SystemTime>,
    names: std::collections::BTreeSet<String>,
}

impl Kept {
    fn at(path: PathBuf) -> Self {
        Self {
            path,
            seen: None,
            names: Default::default(),
        }
    }

    /// re-read the file if it changed since last time, and say whether it did.
    fn refresh(&mut self) -> bool {
        let modified = fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        if modified == self.seen {
            return false;
        }
        match crate::label::Labels::load(&self.path) {
            Ok(labels) => {
                self.names = labels.entries.into_keys().collect();
                self.seen = modified;
            }
            // what was known stays known: forgetting every label because one
            // read failed would hand all of them to the next eviction.
            Err(e) => tracing::warn!("reading {}: {e}", self.path.display()),
        }
        true
    }

    fn holds(&self, path: &Path) -> bool {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| self.names.contains(n))
    }
}

/// the file beside the harvest naming the crops of a vehicle that alerted.
///
/// beside the crop rather than in its name: the name is written the moment the
/// crop is, and whether its vehicle alerts is only known looks later.
pub const ALERTED: &str = "alerted.txt";

fn alerted_path(dir: &Path) -> PathBuf {
    dir.parent().unwrap_or(dir).join(ALERTED)
}

/// the crops recorded as belonging to a vehicle that alerted.
///
/// read per call: the pipeline appends to it while a page is listing, and it
/// grows by a handful of names an alert.
pub fn alerted(dir: &Path) -> std::collections::BTreeSet<String> {
    fs::read_to_string(alerted_path(dir))
        .map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

/// what a confirmed subject hands back: whether this is the look that
/// published, and the crops on disk that name the subject.
///
/// **the two can be empty together**, and that is the case that matters. a track
/// confirms on its `confirm_m`th agreeing look, and the harvest keeps one crop per
/// place per `[harvest] min_interval_secs`, so every agreeing look before this one
/// can have been deduplicated away -- leaving a phone buzzing about something the
/// verdict page cannot show. the caller writes this look anyway (r4.5).
#[derive(Debug)]
pub struct Confirmation {
    pub first: bool,
    pub crops: Vec<String>,
}

impl Confirmation {
    /// a look that agreed with one already published: nothing to name, nothing
    /// new to write, and nothing to tell the phone.
    pub fn repeat() -> Self {
        Self {
            first: false,
            crops: Vec::new(),
        }
    }
}

/// which harvested crops belong to a vehicle that alerted.
///
/// **most of what shows a vehicle is saved before anything is known.** a crop
/// is written on the look it was taken, and a track only confirms on the
/// `confirm_m`th look that agrees, so its verdict crops are held per track until
/// it confirms -- and are named all at once -- or until it leaves.
#[derive(Default)]
pub struct Alerting {
    /// crops saved with a verdict, by track, not yet known to have alerted.
    waiting: std::collections::HashMap<u64, Vec<(String, String)>>,
    /// the subject each track has confirmed.
    confirmed: std::collections::HashMap<u64, String>,
}

impl Alerting {
    /// a crop carrying `verdict` was saved for `track`. returned straight back
    /// when the track has already confirmed that subject.
    pub fn saved(&mut self, track: u64, crop: String, verdict: &str) -> Vec<String> {
        if self.confirmed.get(&track).is_some_and(|s| s == verdict) {
            return vec![crop];
        }
        self.waiting
            .entry(track)
            .or_default()
            .push((crop, verdict.to_string()));
        Vec::new()
    }

    /// `track` has confirmed `subject`: which look that was, and what it left
    /// behind on disk. see [`Confirmation`].
    pub fn confirmed(&mut self, track: u64, subject: &str) -> Confirmation {
        if self.confirmed.get(&track).is_some_and(|s| s == subject) {
            return Confirmation::repeat();
        }
        self.confirmed.insert(track, subject.to_string());
        let mut named = Vec::new();
        if let Some(held) = self.waiting.get_mut(&track) {
            held.retain(|(crop, verdict)| {
                let alerted = verdict == subject;
                if alerted {
                    named.push(crop.clone());
                }
                !alerted
            });
        }
        Confirmation {
            first: true,
            crops: named,
        }
    }

    /// forget every track the tracker no longer follows.
    pub fn retain(&mut self, live: impl Fn(u64) -> bool) {
        self.waiting.retain(|id, _| live(*id));
        self.confirmed.retain(|id, _| live(*id));
    }
}

pub struct Harvester {
    dir: PathBuf,
    budget_bytes: u64,
    min_interval: std::time::Duration,
    same_object_tolerance: f32,
    jpeg_quality: u8,
    recent: Vec<Recent>,
    written: u64,
    /// the millisecond the last crop was named for. see `stamp`.
    last_stamp: u128,
    /// bytes on disk, carried rather than recomputed. see `enforce_budget`.
    total_bytes: u64,
    /// oldest first, what eviction draws from. refilled by a walk of the
    /// directory only when it runs dry.
    evictable: std::collections::VecDeque<(PathBuf, u64)>,
    /// crops the budget may not delete, when `keep_labelled` is on.
    kept: Option<Kept>,
    /// labelled crops on disk: in the total, never in `evictable`.
    held: u64,
    /// the last walk found only labelled crops left to delete. nothing but a
    /// label changing can alter that, so until one does, running dry is not a
    /// reason to walk the harvest again.
    only_kept: bool,
    /// how many times the directory has been walked. the point of the design is
    /// that this stays small however many crops are written, so it is worth
    /// being able to assert on.
    scans: u64,
}

impl Harvester {
    pub fn new(dir: &Path, cfg: &crate::config::HarvestCfg) -> Result<Self> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut h = Self {
            dir: dir.to_path_buf(),
            budget_bytes: cfg.max_bytes,
            min_interval: std::time::Duration::from_secs(cfg.min_interval_secs),
            same_object_tolerance: cfg.same_object_tolerance,
            jpeg_quality: cfg.jpeg_quality,
            recent: Vec::new(),
            written: 0,
            last_stamp: 0,
            total_bytes: 0,
            evictable: std::collections::VecDeque::new(),
            kept: cfg
                .keep_labelled
                .then(|| Kept::at(crate::label::paths(dir).0)),
            held: 0,
            only_kept: false,
            scans: 0,
        };
        // once, at startup, so the running total starts from the truth. also the
        // only place the size of an existing harvest is ever reported: a
        // directory that has grown enough to be slow looks like nothing at all
        // from the inside.
        let began = Instant::now();
        h.rescan();

        tracing::info!(
            "harvest holds {} crops, {:.1} MB (walked in {}ms)",
            h.on_disk().0,
            h.total_bytes as f64 / 1e6,
            began.elapsed().as_millis()
        );
        // said at startup because the failure is silent: labels kept anywhere
        // but beside the harvest protect nothing, and look exactly like labels
        // that do until the crops are gone.
        if let Some(k) = &h.kept {
            tracing::info!(
                "{} labelled crops are kept from the budget ({})",
                h.held,
                k.path.display()
            );
        }
        Ok(h)
    }

    /// how many times the directory has been walked. the whole point of the
    /// design is that this stays small however many crops are written, and an
    /// invariant nothing checks is a comment.
    #[cfg(test)]
    pub fn scans(&self) -> u64 {
        self.scans
    }

    /// rebuild the eviction queue and the byte total from the directory.
    ///
    /// oldest first by modification time, which is the order eviction wants and
    /// the only thing the walk is for.
    fn rescan(&mut self) {
        self.scans += 1;
        if let Some(k) = self.kept.as_mut() {
            k.refresh();
        }
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut files: Vec<(PathBuf, u64, SystemTime)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                m.is_file().then_some(())?;
                Some((e.path(), m.len(), m.modified().ok()?))
            })
            .collect();
        files.sort_by_key(|(_, _, t)| *t);
        self.total_bytes = files.iter().map(|(_, n, _)| n).sum();
        // a restart must not reuse a millisecond the previous run already named
        // a crop for. the stamp is monotonic within a process; this is what
        // makes it monotonic across one, and the walk is already here.
        self.last_stamp = files
            .iter()
            .filter_map(|(p, _, _)| parse_name(p.file_name()?.to_str()?))
            .map(|s| s.at_millis)
            .max()
            .unwrap_or(0)
            .max(self.last_stamp);
        let count = files.len() as u64;
        let kept = self.kept.as_ref();
        self.evictable = files
            .into_iter()
            .filter(|(p, _, _)| !kept.is_some_and(|k| k.holds(p)))
            .map(|(p, n, _)| (p, n))
            .collect();
        self.held = count - self.evictable.len() as u64;
    }

    pub fn written(&self) -> u64 {
        self.written
    }

    /// crops on disk and what they weigh, carried rather than counted. a
    /// harvest large enough to be slow looks like nothing from the inside, which
    /// is how a directory of four hundred thousand files went unnoticed.
    pub fn on_disk(&self) -> (u64, u64) {
        (self.evictable.len() as u64 + self.held, self.total_bytes)
    }

    /// record crops of a vehicle that alerted, in `alerted.txt` beside the harvest.
    pub fn mark_alerted(&self, names: &[String]) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        let path = alerted_path(&self.dir);
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let body: String = names.iter().map(|n| format!("{n}\n")).collect();
        std::io::Write::write_all(&mut file, body.as_bytes())
            .with_context(|| format!("writing {}", path.display()))
    }

    /// should this region be saved, or is it the same thing we just saved?
    ///
    /// a parked car sitting in the gate's output would otherwise produce a crop
    /// every frame. the aim is variety, not volume: a hundred different vehicles
    /// is worth far more than a thousand frames of one.
    pub fn wants(
        &mut self,
        region: Rect,
        frame_w: u32,
        frame_h: u32,
        max_fraction: f32,
        now: Instant,
    ) -> bool {
        if covers_most_of(region, frame_w, frame_h, max_fraction) {
            return false;
        }
        self.recent
            .retain(|r| now.duration_since(r.at) < self.min_interval);
        !self
            .recent
            .iter()
            .any(|r| same_object(r.region, region, self.same_object_tolerance))
    }

    /// encode and write a crop, then enforce the disk budget.
    pub fn save(&mut self, crop: Crop<'_>, now: Instant) -> Result<PathBuf> {
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(Cursor::new(&mut jpeg), self.jpeg_quality)
            .encode(
                crop.rgb,
                crop.width as u16,
                crop.height as u16,
                jpeg_encoder::ColorType::Rgb,
            )
            .map_err(|e| anyhow::anyhow!("encoding crop: {e}"))?;

        // the name carries what is known at capture time, so the directory is
        // still sortable and searchable without a sidecar database.
        //
        // the verdict is a field of its own after the class rather than a
        // replacement for it: a crop the detector called a truck and stage two
        // called a go-4 is the case worth finding again. an underscore cannot
        // occur in either -- `url_safe` leaves only letters, digits and dashes
        // -- so it is unambiguously the separator when reading back.
        // the margin rides only beside a verdict: it is the gap that made the
        // claim, and a crop stage two said nothing about has no claim to measure.
        // thousandths, signed, because a shipped margin may be negative.
        let verdict = match crop.subject {
            Some(s) => match crop.margin.filter(|m| m.is_finite()) {
                Some(m) => format!("_{}_{:03}", url_safe(s), (m * 1000.0).round() as i32),
                None => format!("_{}", url_safe(s)),
            },
            None => String::new(),
        };
        let at = match crop.at_millis {
            Some(taken) => taken,
            None => self.stamp(),
        };
        let name = format!(
            "{}_{}{}_{:03}_{}x{}.jpg",
            at,
            url_safe(crop.label),
            verdict,
            (crop.confidence * 100.0) as u32,
            crop.width,
            crop.height
        );
        let path = self.dir.join(&name);
        // **written through a temporary and renamed.** `fs::write` creates the
        // file and then fills it, so a complete-looking name sits over empty
        // bytes for as long as that takes. the viewer lists this directory every
        // few seconds and serves crops `immutable` -- correctly, since a name
        // carries the millisecond it was taken -- so a browser that caught the
        // gap cached the empty answer permanently. rename is atomic within a
        // filesystem, so a crop is either absent or whole.
        let tmp = self.dir.join(format!(".{name}.part"));
        fs::write(&tmp, &jpeg).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("renaming into {}", path.display()))?;

        self.recent.push(Recent {
            region: crop.region,
            at: now,
        });
        self.written += 1;
        changed(&self.dir);
        // the crop just written is the newest, so it is the last thing eviction
        // should reach for.
        self.evictable.push_back((path.clone(), jpeg.len() as u64));
        self.enforce_budget(jpeg.len() as u64)?;
        Ok(path)
    }

    /// a millisecond no crop has been named for yet.
    ///
    /// the name is the only index the harvest has, so two crops sharing one is
    /// two crops that were one: the rename puts the second where the first was
    /// and nothing says the first existed. `max_per_frame` is 4, so four crops
    /// are written in the same millisecond routinely, and two cars of the same
    /// class at the same rounded confidence cropped to the same size collide.
    /// two cars crossing together is precisely the case worth keeping.
    ///
    /// so the stamp only ever moves forwards. under a burst it runs at most a
    /// few milliseconds ahead of the clock, which costs nothing: it is a label,
    /// and what it has to be is unique and ordered.
    fn stamp(&mut self) -> u128 {
        let now = epoch_millis();
        self.last_stamp = if now > self.last_stamp {
            now
        } else {
            self.last_stamp + 1
        };
        self.last_stamp
    }

    /// delete oldest first until the directory fits the budget.
    ///
    /// the harvest degrades before the disk fills. a full disk would take the
    /// alert path down with it, and an alert matters more than a training crop.
    ///
    /// **the total is carried, not recomputed.** this used to walk the whole
    /// directory on every save, which made writing one crop cost time
    /// proportional to how many crops already existed -- on the frame loop, so
    /// the pipeline sat in uninterruptible sleep while it happened. at a 20 GB
    /// budget that directory holds several hundred thousand files and a cold
    /// walk of it measured 2.9 seconds.
    ///
    /// so a walk happens at startup and then only when the eviction queue runs
    /// dry, which is once per as many crops as the queue held.
    fn enforce_budget(&mut self, added: u64) -> Result<()> {
        self.total_bytes = self.total_bytes.saturating_add(added);
        // labels are written by another process while this one runs, so they are
        // re-read here -- once over budget, and only if the file has moved.
        if self.total_bytes > self.budget_bytes && self.kept.as_mut().is_some_and(|k| k.refresh()) {
            self.only_kept = false;
        }
        while self.total_bytes > self.budget_bytes {
            let Some((path, size)) = self.evictable.pop_front() else {
                // nothing left to evict that we know of. either the harvest was
                // written to behind our back or the budget cannot be met by
                // deleting crops, and a walk settles which.
                if self.only_kept {
                    return Ok(());
                }
                let before = self.total_bytes;
                self.rescan();
                if self.evictable.is_empty() && self.held > 0 {
                    self.only_kept = true;
                    tracing::warn!(
                        "{} labelled crops alone exceed [harvest] max_bytes; \
                         every other crop is deleted as soon as it is written",
                        self.held
                    );
                }
                if self.evictable.is_empty() || self.total_bytes >= before {
                    return Ok(());
                }
                continue;
            };
            // labelled since the queue was built.
            if self.kept.as_ref().is_some_and(|k| k.holds(&path)) {
                self.held += 1;
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => {
                    changed(&self.dir);
                    self.total_bytes = self.total_bytes.saturating_sub(size);
                }
                // already gone: it is not on disk, so it is not in the total.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    self.total_bytes = self.total_bytes.saturating_sub(size);
                }
                Err(e) => {
                    tracing::warn!("evicting {}: {e}", path.display());
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

fn epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// reduce a label to characters that survive a url path untouched.
///
/// several coco classes contain a space -- `traffic light`, `parking meter` --
/// and the preview serves crops by name. a space would be percent-encoded by
/// the browser and arrive at a server that refuses `%` in a filename, so the
/// crop would silently 404 in the viewer. fixing it at the point the name is
/// made is better than teaching the server to decode.
fn url_safe(label: &str) -> String {
    label
        .chars()
        .map(|c| match c {
            'a'..='z' | '0'..='9' | '-' => c,
            'A'..='Z' => c.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

/// a crop on disk, described by its own filename.
///
/// `save` deliberately encodes everything it knows into the name so the
/// directory stays searchable without a sidecar database. that only stays true
/// if something reads it back, so the parser lives next to the writer: change
/// one and the tests for the other fail.
#[derive(Debug, Clone, PartialEq)]
pub struct Saved {
    pub name: String,
    pub label: String,
    /// what stage two called it, when it was run and recognised something.
    pub subject: Option<String>,
    /// how far that verdict cleared its negatives, signed. `None` for every crop
    /// written before it was recorded, and for every crop with no verdict.
    pub margin: Option<f32>,
    pub confidence: f32,
    pub width: u32,
    pub height: u32,
    pub at_millis: u128,
}

/// read `{millis}_{label}[_{subject}[_{margin}]]_{conf}_{w}x{h}.jpg` back into its
/// parts. the margin is thousandths, so `20` is `+0.020` and `-07` is `-0.007`.
///
/// parsed from the right, because the label is the only field that can contain
/// a separator: several coco class names have a space in them.
///
/// the verdict is optional, and absent from every crop harvested before it was
/// recorded and from every crop stage two did not recognise -- which is nearly
/// all of them.
pub fn parse_name(name: &str) -> Option<Saved> {
    let stem = name.strip_suffix(".jpg")?;
    let (rest, dims) = stem.rsplit_once('_')?;
    let (w, h) = dims.split_once('x')?;
    let (millis, label) = rest.split_once('_')?;
    let (label, conf) = label.rsplit_once('_')?;
    // a class name holds no underscore, so the ones after it separate the verdict
    // and its margin. an older name carrying a space -- written before labels were
    // sanitised -- has none and reads as it always did. three fields means a
    // margin was recorded, since only a verdict can be followed by one.
    let parts: Vec<&str> = label.split('_').collect();
    let (label, subject, margin) = match parts.as_slice() {
        [class, subject, gap, ..] => (
            *class,
            Some(subject.to_string()),
            gap.parse::<i32>().ok().map(|v| v as f32 / 1000.0),
        ),
        [class, subject] => (*class, Some(subject.to_string()), None),
        [class] => (*class, None, None),
        _ => return None,
    };
    Some(Saved {
        name: name.to_string(),
        label: label.to_string(),
        subject,
        margin,
        // written as hundredths so the name sorts sensibly.
        confidence: conf.parse::<u32>().ok()? as f32 / 100.0,
        width: w.parse().ok()?,
        height: h.parse().ok()?,
        at_millis: millis.parse().ok()?,
    })
}

/// which crops a listing is asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want<'a> {
    Everything,
    /// only what stage two named. the subject is well under a percent of the
    /// harvest, so this has to be applied before the page is cut: a page of the
    /// raw harvest holds no verdict at all on almost every scroll.
    Recognised,
    /// exactly the crops named here, which is how the live verdicts tab asks.
    ///
    /// **a verdict somebody has settled is no longer a verdict, and a crop
    /// somebody called the subject is one.** the live page lists what stands:
    /// on the first evening the classifier ran every crop it named was wrong,
    /// so leaving the judged ones in makes the tab mostly a record of what has
    /// already been dealt with, and a subject it missed is nowhere on it.
    ///
    /// **the labelling page does not use this.** it is where a verdict gets
    /// rejected, so it has to keep showing the ones that were -- otherwise a
    /// crop vanishes the moment it is judged and a misclick cannot be undone.
    ///
    /// which crops stand is decided by the caller, because it is a fact about
    /// labels and this module knows only filenames.
    Named(&'a std::collections::BTreeSet<String>),
}

impl Want<'_> {
    fn keeps(&self, s: &Saved) -> bool {
        match self {
            Want::Everything => true,
            Want::Recognised => s.subject.is_some(),
            Want::Named(names) => names.contains(&s.name),
        }
    }
}

/// one page of crops, newest first, older than `before` when given.
///
/// paged because the harvest runs to tens of thousands of files and the viewer
/// scrolls: sending the lot would mean a huge json body and a browser holding
/// thousands of image elements, to show the dozen on screen.
///
/// reads the directory on demand rather than keeping an index: the harvester
/// deletes oldest-first to stay inside its budget, so any cache would go stale
/// silently. a directory listing is cheap next to the inference already running.
pub fn page(dir: &Path, before: Option<u128>, limit: usize, want: Want) -> Vec<Saved> {
    let mut found = all(dir);
    // strictly older, so paging cannot hand back the crop it was given as the
    // cursor and stall on it forever.
    found.retain(|s| before.is_none_or(|b| s.at_millis < b));
    found.retain(|s| want.keeps(s));
    found.truncate(limit);
    found
}

/// the whole harvest, newest first.
///
/// for callers matching many timestamps at once, where paging would mean one
/// directory read per timestamp. the events tab is the case: it asks what was
/// harvested during each of its clips, and doing that a clip at a time turns
/// one listing into twenty.
pub fn all(dir: &Path) -> Vec<Saved> {
    // the harvest only changes when this process changes it, so the previous
    // answer stands until it does. the crops tab asks for this every few
    // seconds for as long as the page is open, and on a directory holding a few
    // hundred thousand crops a cold walk of it measured 2.9 seconds.
    let generation = {
        let Ok(mut dirs) = DIRS.lock() else {
            return Vec::new();
        };
        let e = dirs.entry(dir.to_path_buf()).or_default();
        if let Some((seen, taken, crops)) = e.listing.as_ref()
            && *seen == e.generation
            && taken.elapsed() < LISTING_TTL
        {
            return crops.clone();
        }
        e.generation
    };

    if let Ok(mut dirs) = DIRS.lock() {
        dirs.entry(dir.to_path_buf()).or_default().walks += 1;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<Saved> = entries
        .filter_map(|e| {
            let e = e.ok()?;
            // belt and braces against the above: an empty file is one being
            // written, or one a crash left behind, and either way there is
            // nothing to show.
            (e.metadata().ok()?.len() > 0).then_some(())?;
            parse_name(e.file_name().to_str()?)
        })
        .collect();
    found.sort_by_key(|s| std::cmp::Reverse(s.at_millis));
    if let Ok(mut dirs) = DIRS.lock() {
        let e = dirs.entry(dir.to_path_buf()).or_default();
        // only cache against the generation the walk actually saw: a save that
        // landed while it ran must not be hidden by it.
        if e.generation == generation {
            e.listing = Some((generation, Instant::now(), found.clone()));
        }
    }
    found
}

// the events tab used to illustrate a clip with the clearest crop harvested
// during it, which lived here. it is gone rather than kept: which crops belong
// to a clip has to be inferred from timestamps, and measured on the deployment
// every way of inferring it put a vehicle on a card it does not appear in. the
// card shows a frame of the clip itself now, which cannot be wrong about that.

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// **the crop that showed as black until the page was refreshed.**
    ///
    /// `fs::write` creates the file and then fills it, so between those two a
    /// complete-looking name sits in the directory over empty bytes. the viewer
    /// lists the directory every few seconds, so it offers that name, the
    /// browser fetches nothing -- and because a crop is served
    /// `immutable, max-age=1y` on the entirely correct grounds that its name
    /// carries the millisecond it was taken, the browser then keeps the empty
    /// answer forever. clicking it re-requests the same url and gets the same
    /// cached nothing; only a reload revalidates.
    ///
    /// so a half-written crop must never be reachable by its final name.
    #[test]
    fn a_half_written_crop_is_not_offered() {
        let d = tmpdir("partial");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        h.save(
            crop_of(
                &rgb(20, 10),
                20,
                10,
                Rect {
                    x: 0,
                    y: 0,
                    w: 1,
                    h: 1,
                },
                "car",
                0.9,
            ),
            Instant::now(),
        )
        .unwrap();
        // what a file caught mid-write looks like from the outside.
        fs::write(d.join("1789260000000_car_090_20x10.jpg"), b"").unwrap();

        let listed = page(&d, None, 100, Want::Everything);
        assert_eq!(listed.len(), 1, "an empty crop was offered: {listed:?}");
        for s in &listed {
            let n = fs::metadata(d.join(&s.name)).unwrap().len();
            assert!(n > 0, "{} is empty", s.name);
        }
        fs::remove_dir_all(&d).ok();
    }

    fn harvest_cfg(max_bytes: u64, min_interval_secs: u64) -> crate::config::HarvestCfg {
        crate::config::HarvestCfg {
            max_bytes,
            min_interval_secs,
            ..Default::default()
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("metermate-harvest-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn crop_of<'a>(
        rgb: &'a [u8],
        width: u32,
        height: u32,
        region: Rect,
        label: &'a str,
        confidence: f32,
    ) -> Crop<'a> {
        Crop {
            rgb,
            width,
            height,
            region,
            label,
            confidence,
            subject: None,
            margin: None,
            at_millis: None,
        }
    }

    fn rgb(w: u32, h: u32) -> Vec<u8> {
        vec![128; (w * h * 3) as usize]
    }

    /// the name is the only record of what a crop is, so the writer and the
    /// parser have to agree. round-tripping through a real save is the only way
    /// to catch one of them changing.
    #[test]
    fn a_saved_crop_can_be_read_back_from_its_name() {
        let d = tmpdir("roundtrip");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 60)).unwrap();
        let region = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let path = h
            .save(
                crop_of(&rgb(48, 32), 48, 32, region, "truck", 0.86),
                Instant::now(),
            )
            .unwrap();

        let got = parse_name(path.file_name().unwrap().to_str().unwrap())
            .expect("the harvester wrote a name its own parser rejects");
        assert_eq!(got.label, "truck");
        assert_eq!(got.width, 48);
        assert_eq!(got.height, 32);
        assert!(
            (got.confidence - 0.86).abs() < 0.01,
            "confidence {} != 0.86",
            got.confidence
        );
        assert!(got.at_millis > 0, "no timestamp in {}", got.name);
    }

    /// **a crop the classifier recognised looked like every other crop.**
    ///
    /// the verdict is reached on these very pixels a moment before they are
    /// written, and was then dropped: so the one crop in two hundred that is a
    /// go-4 sat in the grid captioned `car 0.83`, indistinguishable from the
    /// cars around it. the name is the only index the harvest has, so the
    /// verdict goes there -- beside the detector class rather than instead of
    /// it, since which of the two disagreed is the interesting case.
    #[test]
    fn a_crop_the_classifier_recognised_says_so_in_its_name() {
        let d = tmpdir("verdict");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let pixels = rgb(20, 10);
        let mut crop = crop_of(
            &pixels,
            20,
            10,
            Rect {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
            "car",
            0.83,
        );
        crop.subject = Some("go4");
        let path = h.save(crop, Instant::now()).unwrap();

        let name = path.file_name().unwrap().to_str().unwrap();
        let got = parse_name(name).expect("the harvester wrote a name its parser rejects");
        assert_eq!(got.subject.as_deref(), Some("go4"), "{name}");
        assert_eq!(got.label, "car", "the detector class was lost: {name}");
        assert!((got.confidence - 0.83).abs() < 0.01, "{name}");
        // the preview fetches a crop by this name, and a server that refuses
        // `%` in a filename would 404 anything the browser had to encode.
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')),
            "the verdict introduced a character a url would encode: {name}"
        );
        fs::remove_dir_all(&d).ok();
    }

    /// **a verdict is a claim about a margin, and the margin is the only number
    /// worth tuning.** the notification says `margin +0.020` and the bar it
    /// cleared lives in `trained.toml`; with neither number on the crop the two
    /// cannot be compared from the verdict page, which is where a person decides
    /// whether the bar should move.
    #[test]
    fn a_verdict_records_the_margin_that_named_it() {
        let d = tmpdir("margin");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let pixels = rgb(20, 10);
        let mut crop = crop_of(
            &pixels,
            20,
            10,
            Rect {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
            "car",
            0.83,
        );
        crop.subject = Some("go4");
        crop.margin = Some(0.020);
        let path = h.save(crop, Instant::now()).unwrap();

        let name = path.file_name().unwrap().to_str().unwrap();
        let got = parse_name(name).expect("a margin wrote a name its parser rejects");
        assert!(
            got.margin.is_some_and(|m| (m - 0.020).abs() < 0.0005),
            "{name} did not carry the margin back: {:?}",
            got.margin
        );
        assert_eq!(got.subject.as_deref(), Some("go4"), "{name}");
        assert_eq!(got.label, "car", "the detector class was lost: {name}");
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')),
            "the margin introduced a character a url would encode: {name}"
        );
        fs::remove_dir_all(&d).ok();
    }

    /// **the sign is the point.** a shipped margin is allowed to be negative --
    /// this deployment runs at -0.005 -- so a crop can be a verdict while sitting
    /// nearer the street than its own class. a number that hid which side it was
    /// on would hide the one case worth looking at.
    #[test]
    fn a_margin_on_the_wrong_side_of_the_street_stays_on_that_side() {
        let d = tmpdir("negative");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let pixels = rgb(20, 10);
        let mut crop = crop_of(
            &pixels,
            20,
            10,
            Rect {
                x: 0,
                y: 0,
                w: 10,
                h: 10,
            },
            "car",
            0.83,
        );
        crop.subject = Some("go4");
        crop.margin = Some(-0.007);
        let path = h.save(crop, Instant::now()).unwrap();

        let name = path.file_name().unwrap().to_str().unwrap();
        let got = parse_name(name).unwrap().margin;
        assert!(
            got.is_some_and(|m| (m + 0.007).abs() < 0.0005),
            "{name} lost the sign: {got:?}"
        );
        fs::remove_dir_all(&d).ok();
    }

    /// every crop written before the verdict was recorded, and every crop the
    /// classifier was not run on or did not recognise, still has to read back.
    #[test]
    fn a_crop_with_no_verdict_parses_as_having_none() {
        for name in [
            "1757700000123_car_085_64x64.jpg",
            "1757700000123_traffic light_042_64x64.jpg",
        ] {
            assert_eq!(parse_name(name).unwrap().subject, None, "{name}");
        }
    }

    /// a verdict written before the margin was recorded is not a verdict with a
    /// margin of zero, and the page has to be able to tell the two apart.
    #[test]
    fn a_name_with_no_margin_reads_as_having_none() {
        for name in [
            "1757700000123_car_085_64x64.jpg",
            "1757700000123_truck_go4_075_240x200.jpg",
        ] {
            assert_eq!(parse_name(name).unwrap().margin, None, "{name}");
        }
    }

    /// several coco class names contain a space. the name must still be usable
    /// as a url path segment, because that is how the preview fetches crops.
    #[test]
    fn a_label_with_a_space_is_written_url_safe() {
        let d = tmpdir("spacey");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 60)).unwrap();
        let region = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let path = h
            .save(
                crop_of(&rgb(16, 16), 16, 16, region, "traffic light", 0.42),
                Instant::now(),
            )
            .unwrap();

        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(!name.contains(' '), "space survived into {name}");
        assert_eq!(parse_name(name).unwrap().label, "traffic-light");
    }

    /// the parser stays tolerant of names written before labels were sanitised.
    #[test]
    fn an_older_name_containing_a_space_still_parses() {
        let got = parse_name("1757700000123_traffic light_042_64x64.jpg").expect("should parse");
        assert_eq!(got.label, "traffic light");
        assert_eq!(got.width, 64);
    }

    #[test]
    fn things_that_are_not_crops_are_ignored() {
        for name in ["notes.txt", "1757700000123_car_086.jpg", "car.jpg", ".jpg"] {
            assert!(parse_name(name).is_none(), "{name} should not parse");
        }
    }

    /// the viewer shows the newest first, and the directory gives no order.
    #[test]
    fn recent_crops_come_back_newest_first() {
        let d = tmpdir("recent");
        fs::create_dir_all(&d).unwrap();
        for millis in [1757700000100u64, 1757700000300, 1757700000200] {
            fs::write(d.join(format!("{millis}_car_050_64x64.jpg")), b"x").unwrap();
        }
        let got = page(&d, None, 10, Want::Everything);
        let order: Vec<u128> = got.iter().map(|s| s.at_millis).collect();
        assert_eq!(order, vec![1757700000300, 1757700000200, 1757700000100]);

        assert_eq!(
            page(&d, None, 2, Want::Everything).len(),
            2,
            "limit not applied"
        );

        // paging: the cursor is exclusive, so scrolling cannot stick on the
        // crop it was given and re-serve it forever.
        let older = page(&d, Some(1757700000300), 10, Want::Everything);
        assert_eq!(
            older.iter().map(|s| s.at_millis).collect::<Vec<_>>(),
            vec![1757700000200, 1757700000100]
        );
        assert!(
            page(&d, Some(1757700000100), 10, Want::Everything).is_empty(),
            "nothing is older than the oldest"
        );
        assert!(
            page(
                Path::new("/nonexistent/metermate"),
                None,
                10,
                Want::Everything
            )
            .is_empty(),
            "a missing harvest directory must not be an error"
        );
    }

    /// **a verdict is well under a percent of the harvest.**
    ///
    /// so the filter has to run before the page is cut, not after: a page of the
    /// raw harvest holds no verdict at all on almost every scroll, and a viewer
    /// paging for one would walk thousands of crops to find nothing.
    #[test]
    fn a_listing_can_ask_for_only_what_was_recognised() {
        let d = tmpdir("recognised");
        fs::create_dir_all(&d).unwrap();
        for millis in 1757700000100u64..1757700000160 {
            fs::write(d.join(format!("{millis}_car_050_64x64.jpg")), b"x").unwrap();
        }
        fs::write(d.join("1757700000105_car_go4_091_64x64.jpg"), b"x").unwrap();

        let got = page(&d, None, 10, Want::Recognised);
        assert_eq!(got.len(), 1, "a page of cars came back: {got:?}");
        assert_eq!(got[0].subject.as_deref(), Some("go4"));
        assert_eq!(page(&d, None, 10, Want::Everything).len(), 10);
        fs::remove_dir_all(&d).ok();
    }

    /// **the verdicts tab lists what stands, and the caller says what does.**
    ///
    /// a verdict a person has settled drops out and a crop a person called the
    /// subject comes in, whatever its name says -- both facts about labels,
    /// which this module does not read.
    ///
    /// the filter runs before the page is cut, for the same reason the
    /// recognised filter does: cutting first would hand back a short page and
    /// stall the scroll.
    #[test]
    fn a_listing_can_ask_for_crops_by_name() {
        let d = tmpdir("standing");
        fs::create_dir_all(&d).unwrap();
        let kept = "1757700000105_car_go4_091_64x64.jpg";
        let thrown = "1757700000106_truck_go4_077_64x64.jpg";
        let by_hand = "1757700000104_car_088_64x64.jpg";
        for name in [kept, thrown, by_hand] {
            fs::write(d.join(name), b"x").unwrap();
        }

        let standing: std::collections::BTreeSet<String> =
            [kept.to_string(), by_hand.to_string()].into();
        let got = page(&d, None, 1, Want::Named(&standing));
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].name, kept);
        let got = page(&d, None, 10, Want::Named(&standing));
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[1].name, by_hand);

        // and the labelling page is deliberately not filtered: it is where a
        // verdict is rejected, so it has to keep showing the ones that were.
        assert_eq!(page(&d, None, 10, Want::Recognised).len(), 2);
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_same_parked_vehicle_is_not_saved_repeatedly() {
        let d = tmpdir("dedup");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 60)).unwrap();
        let now = Instant::now();
        let region = Rect {
            x: 100,
            y: 100,
            w: 80,
            h: 60,
        };

        assert!(
            h.wants(region, 640, 480, crate::config::MAX_REGION_FRACTION, now),
            "first sighting should be wanted"
        );
        h.save(crop_of(&rgb(32, 32), 32, 32, region, "truck", 0.8), now)
            .unwrap();

        // the same car, a moment later, drifting slightly: not new information.
        let nudged = Rect {
            x: 104,
            y: 102,
            w: 80,
            h: 60,
        };
        assert!(!h.wants(
            nudged,
            640,
            480,
            crate::config::MAX_REGION_FRACTION,
            now + Duration::from_secs(1)
        ));

        // a different vehicle elsewhere in the frame is.
        let elsewhere = Rect {
            x: 400,
            y: 300,
            w: 80,
            h: 60,
        };
        assert!(h.wants(
            elsewhere,
            640,
            480,
            crate::config::MAX_REGION_FRACTION,
            now + Duration::from_secs(1)
        ));

        fs::remove_dir_all(&d).ok();
    }

    /// on panning news footage the gate correctly reports one region covering
    /// the whole scene. cropping that yields a photograph of a street, which is
    /// worthless training data and silently eats the disk budget.
    #[test]
    fn a_region_covering_the_scene_is_not_an_object() {
        let d = tmpdir("scene");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let now = Instant::now();

        let whole_frame = Rect {
            x: 0,
            y: 0,
            w: 640,
            h: 480,
        };
        assert!(
            !h.wants(
                whole_frame,
                640,
                480,
                crate::config::MAX_REGION_FRACTION,
                now
            ),
            "whole frame accepted"
        );

        let most_of_it = Rect {
            x: 0,
            y: 0,
            w: 500,
            h: 400,
        };
        assert!(
            !h.wants(
                most_of_it,
                640,
                480,
                crate::config::MAX_REGION_FRACTION,
                now
            ),
            "most of frame accepted"
        );

        // a vehicle-sized region in the same frame is still fine.
        let vehicle = Rect {
            x: 100,
            y: 100,
            w: 160,
            h: 120,
        };
        assert!(
            h.wants(vehicle, 640, 480, crate::config::MAX_REGION_FRACTION, now),
            "a real object was rejected"
        );

        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_same_place_becomes_interesting_again_after_the_interval() {
        let d = tmpdir("interval");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 10)).unwrap();
        let now = Instant::now();
        let region = Rect {
            x: 100,
            y: 100,
            w: 80,
            h: 60,
        };

        h.save(crop_of(&rgb(16, 16), 16, 16, region, "car", 0.5), now)
            .unwrap();
        assert!(
            !h.wants(
                region,
                640,
                480,
                crate::config::MAX_REGION_FRACTION,
                now + Duration::from_secs(5)
            ),
            "too soon"
        );
        assert!(
            h.wants(
                region,
                640,
                480,
                crate::config::MAX_REGION_FRACTION,
                now + Duration::from_secs(11)
            ),
            "interval elapsed"
        );

        fs::remove_dir_all(&d).ok();
    }

    /// r6.4: the harvest must degrade before the disk does.
    /// **the harvest was walking its whole directory on every crop.**
    ///
    /// `enforce_budget` ran per save and rebuilt the file list from scratch, so
    /// the cost of writing one crop was proportional to how many crops already
    /// existed. at a 20 GB budget and ~40 KB a crop that is a directory of
    /// several hundred thousand entries, read and stat-ed once per save, on the
    /// frame loop.
    ///
    /// measured on the deployment: a cold listing of that directory took 2.9
    /// seconds. the loop sat in uninterruptible sleep -- 5% of one core, load
    /// average 2.15 with every core idle -- and missed frames, inspections and
    /// crops the whole time. it presented as a detection bug for an afternoon.
    ///
    /// the budget still has to hold. it just may not be recomputed from nothing
    /// each time.
    /// **and neither does listing it.**
    ///
    /// `all` is the other walk of the same directory, and the crops tab asks for
    /// it every few seconds for as long as somebody has the page open. measured
    /// on the deployment, a cold walk of that directory took 2.9 seconds.
    ///
    /// the harvest only changes when this process changes it, so a listing is
    /// reusable until it does. the ttl is for the case that is not true --
    /// somebody archiving crops by hand while it runs.
    /// **two crops from one frame could land on the same name.**
    ///
    /// the name is `<millisecond>_<label>_<confidence>_<w>x<h>.jpg`, and
    /// `max_per_frame` is 4 -- so four crops are written in the same
    /// millisecond. two cars of the same class, at the same rounded confidence,
    /// cropped to the same size, produce the same name, and the rename that
    /// puts the second in place silently destroys the first.
    ///
    /// two cars crossing together is the case the harvest most wants: it is
    /// exactly when a go-4 might be the one that is lost, and nothing would say
    /// so. found by a test of something else, which had written two crops in a
    /// millisecond by accident.
    /// and the same across a restart, where the counter starts from nothing.
    #[test]
    fn a_restart_does_not_reuse_the_millisecond_it_stopped_on() {
        let d = tmpdir("collide-restart");
        let region = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let now = Instant::now();
        {
            let mut first = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
            first
                .save(crop_of(&rgb(20, 10), 20, 10, region, "car", 0.9), now)
                .unwrap();
        }
        // a fresh harvester over the same directory, as a restart gives.
        let mut second = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        second
            .save(crop_of(&rgb(20, 10), 20, 10, region, "car", 0.9), now)
            .unwrap();

        let n = fs::read_dir(&d).unwrap().filter_map(|e| e.ok()).count();
        assert_eq!(n, 2, "the restart overwrote the crop before it");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn two_crops_in_one_millisecond_do_not_overwrite_each_other() {
        let d = tmpdir("collide");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let now = Instant::now();
        // identical in every way the name records.
        for x in [0u32, 400] {
            h.save(
                crop_of(
                    &rgb(20, 10),
                    20,
                    10,
                    Rect {
                        x,
                        y: 0,
                        w: 10,
                        h: 10,
                    },
                    "car",
                    0.9,
                ),
                now,
            )
            .unwrap();
        }
        let on_disk: Vec<_> = fs::read_dir(&d)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            on_disk.len(),
            2,
            "a crop was overwritten by the one beside it: {on_disk:?}"
        );
        // and both are still readable as crops, so the fix did not corrupt the
        // name the whole harvest is indexed by.
        for name in &on_disk {
            assert!(parse_name(name).is_some(), "{name} no longer parses");
        }
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn listing_the_harvest_twice_reads_the_directory_once() {
        let d = tmpdir("listing-cache");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let region = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        h.save(
            crop_of(&rgb(20, 10), 20, 10, region, "car", 0.9),
            Instant::now(),
        )
        .unwrap();

        let before = listing_walks(&d);
        let first = all(&d);
        let second = all(&d);
        let third = page(&d, None, 10, Want::Everything);
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(third.len(), 1);
        assert_eq!(
            listing_walks(&d) - before,
            1,
            "the directory was read {} times for three listings",
            listing_walks(&d) - before
        );

        // a new crop has to be visible immediately: a viewer watching the
        // harvest fill is the whole point of the page.
        //
        // a different size, so the name differs. two crops saved in the same
        // millisecond with the same label, confidence and dimensions collide on
        // the name and one overwrites the other -- which this test hit.
        h.save(
            crop_of(
                &rgb(22, 12),
                22,
                12,
                Rect {
                    x: 300,
                    y: 300,
                    w: 10,
                    h: 10,
                },
                "car",
                0.9,
            ),
            Instant::now(),
        )
        .unwrap();
        assert_eq!(all(&d).len(), 2, "a crop just saved was not listed");
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_budget_does_not_walk_the_harvest_on_every_save() {
        let d = tmpdir("budget-scan");
        // room for a handful of crops, so eviction runs on most saves.
        let mut h = Harvester::new(&d, &harvest_cfg(16 * 1024, 0)).unwrap();
        let mut now = Instant::now();
        const SAVES: u32 = 40;
        for i in 0..SAVES {
            let region = Rect {
                x: (i % 6) * 100,
                y: 0,
                w: 40,
                h: 40,
            };
            h.save(crop_of(&rgb(64, 64), 64, 64, region, "car", 0.5), now)
                .unwrap();
            now += Duration::from_secs(1);
        }
        assert!(
            h.scans() <= 2,
            "walked the harvest {} times for {SAVES} crops",
            h.scans()
        );
        let total: u64 = fs::read_dir(&d)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        assert!(
            total <= 16 * 1024,
            "budget not held: {total} bytes over a 16384 byte budget"
        );
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn oldest_crops_are_dropped_when_the_budget_is_exceeded() {
        let d = tmpdir("budget");
        // a budget of a few kilobytes: each crop below is larger than that.
        let mut h = Harvester::new(&d, &harvest_cfg(4096, 0)).unwrap();
        let mut now = Instant::now();

        for i in 0..6 {
            let region = Rect {
                x: i * 100,
                y: 0,
                w: 40,
                h: 40,
            };
            h.save(crop_of(&rgb(64, 64), 64, 64, region, "car", 0.5), now)
                .unwrap();
            now += Duration::from_secs(1);
            // filesystem mtime resolution is coarse; keep the ordering distinct.
            std::thread::sleep(Duration::from_millis(12));
        }

        let total: u64 = fs::read_dir(&d)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        assert!(total <= 4096, "budget exceeded: {total} bytes left on disk");
        assert!(
            fs::read_dir(&d).unwrap().count() >= 1,
            "budget enforcement deleted everything"
        );

        fs::remove_dir_all(&d).ok();
    }

    fn disk_bytes(dir: &Path) -> u64 {
        fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }

    /// saves one crop, labels it the way the labelling page would -- from outside,
    /// after the crop is already queued -- then saves enough to force eviction.
    fn label_the_first_then_fill(tag: &str, keep_labelled: bool) -> (PathBuf, PathBuf) {
        let d = tmpdir(tag);
        let dir = d.join("crops");
        let cfg = crate::config::HarvestCfg {
            keep_labelled,
            ..harvest_cfg(16 * 1024, 0)
        };
        let mut h = Harvester::new(&dir, &cfg).unwrap();
        let mut now = Instant::now();
        let at = |i: u32| Rect {
            x: (i % 6) * 100,
            y: 0,
            w: 40,
            h: 40,
        };
        let first = h
            .save(crop_of(&rgb(64, 64), 64, 64, at(0), "car", 0.5), now)
            .unwrap();
        let name = first.file_name().unwrap().to_str().unwrap();
        fs::write(d.join("labels.txt"), format!("{name} other seed\n")).unwrap();
        for i in 1..40 {
            now += Duration::from_secs(1);
            h.save(crop_of(&rgb(64, 64), 64, 64, at(i), "car", 0.5), now)
                .unwrap();
        }
        (d, first)
    }

    #[test]
    fn a_crop_labelled_while_harvesting_is_never_evicted() {
        let (d, first) = label_the_first_then_fill("keep", true);
        let dir = d.join("crops");
        assert!(first.exists(), "a labelled crop was evicted");
        let total = disk_bytes(&dir);
        assert!(
            total <= 16 * 1024,
            "budget not held around it: {total} bytes"
        );
        assert!(
            fs::read_dir(&dir).unwrap().count() < 40,
            "nothing was evicted at all"
        );
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn with_keeping_off_a_labelled_crop_rotates_like_any_other() {
        let (d, first) = label_the_first_then_fill("keep-off", false);
        assert!(
            !first.exists(),
            "kept a labelled crop with keep_labelled off"
        );
        fs::remove_dir_all(&d).ok();
    }

    /// **labels alone over the budget must not cost a walk per save.** nothing
    /// is left that eviction may delete, and walking the harvest to rediscover
    /// that on every crop is the stall `the_budget_does_not_walk_the_harvest_on_
    /// every_save` exists to prevent.
    #[test]
    fn labels_that_fill_the_budget_do_not_walk_the_harvest_on_every_save() {
        let d = tmpdir("keep-full");
        let dir = d.join("crops");
        let mut first = Harvester::new(&dir, &harvest_cfg(1 << 30, 0)).unwrap();
        let mut now = Instant::now();
        let mut names = Vec::new();
        for i in 0..3 {
            let region = Rect {
                x: i * 100,
                y: 0,
                w: 40,
                h: 40,
            };
            let p = first
                .save(crop_of(&rgb(64, 64), 64, 64, region, "car", 0.5), now)
                .unwrap();
            names.push(p.file_name().unwrap().to_str().unwrap().to_string());
            now += Duration::from_secs(1);
        }
        let rows: String = names.iter().map(|n| format!("{n} other seed\n")).collect();
        fs::write(d.join("labels.txt"), rows).unwrap();

        // a budget smaller than any one crop, so the labelled ones alone exceed it.
        let mut h = Harvester::new(&dir, &harvest_cfg(4096, 0)).unwrap();
        for i in 0..40 {
            let region = Rect {
                x: (i % 6) * 100,
                y: 0,
                w: 40,
                h: 40,
            };
            h.save(crop_of(&rgb(64, 64), 64, 64, region, "car", 0.5), now)
                .unwrap();
            now += Duration::from_secs(1);
        }
        for n in &names {
            assert!(dir.join(n).exists(), "{n} was labelled and evicted");
        }
        assert!(
            h.scans() <= 2,
            "walked the harvest {} times for 40 crops",
            h.scans()
        );
        fs::remove_dir_all(&d).ok();
    }

    /// **most of a vehicle's crops are saved before anything is known.** a track
    /// confirms on its `confirm_m`th agreeing look, so the crops that show what
    /// alerted were already on disk by then.
    #[test]
    fn a_confirmed_track_names_the_crops_it_saved_before_confirming() {
        let mut a = Alerting::default();
        assert!(a.saved(7, "first.jpg".into(), "go4").is_empty());
        assert!(a.saved(7, "second.jpg".into(), "go4").is_empty());
        assert!(a.saved(8, "other-car.jpg".into(), "go4").is_empty());
        let first = a.confirmed(7, "go4");
        assert!(
            first.first,
            "the look that confirms is the one that publishes"
        );
        assert_eq!(first.crops, vec!["first.jpg", "second.jpg"]);
        // every later confirmed look publishes again; the crops are named once.
        let again = a.confirmed(7, "go4");
        assert!(
            !again.first,
            "the same subject confirmed twice is one alert"
        );
        assert!(again.crops.is_empty());
        // a crop saved after confirmation is named straight away.
        assert_eq!(a.saved(7, "third.jpg".into(), "go4"), vec!["third.jpg"]);
        // a track that left without confirming is forgotten, not held forever.
        a.retain(|id| id != 8);
        assert!(
            a.confirmed(8, "go4").crops.is_empty(),
            "a departed track was kept"
        );
    }

    /// **a confirmation with no crop behind it is the bug, not a state.** the
    /// notification and the verdict page are the same fact seen twice, so the
    /// look that confirms has to know it has nothing to show for itself: it is
    /// saved anyway, past the rate limit, and only then published (r4.5).
    #[test]
    fn a_confirmation_with_nothing_saved_says_so() {
        let mut a = Alerting::default();
        let first = a.confirmed(1, "go4");
        assert!(first.first, "it was the first look to agree");
        assert!(
            first.crops.is_empty(),
            "nothing was saved for a track that had not been seen"
        );
    }

    /// a verdict for another subject on the same track is not what alerted.
    #[test]
    fn only_crops_naming_the_confirmed_subject_are_named() {
        let mut a = Alerting::default();
        a.saved(3, "go4.jpg".into(), "go4");
        a.saved(3, "waymo.jpg".into(), "waymo");
        assert_eq!(a.confirmed(3, "go4").crops, vec!["go4.jpg"]);
    }

    #[test]
    fn alerted_crops_are_recorded_beside_the_harvest_and_read_back() {
        let d = tmpdir("alerted");
        let dir = d.join("crops");
        let h = Harvester::new(&dir, &harvest_cfg(1 << 30, 0)).unwrap();
        h.mark_alerted(&["a.jpg".to_string()]).unwrap();
        h.mark_alerted(&[]).unwrap();
        h.mark_alerted(&["b.jpg".to_string()]).unwrap();
        assert!(
            d.join(ALERTED).exists(),
            "nothing written beside the harvest"
        );
        assert_eq!(
            alerted(&dir).into_iter().collect::<Vec<_>>(),
            vec!["a.jpg", "b.jpg"]
        );
        fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn saved_names_carry_what_was_known_at_capture_time() {
        let d = tmpdir("names");
        let mut h = Harvester::new(&d, &harvest_cfg(1 << 30, 0)).unwrap();
        let p = h
            .save(
                crop_of(
                    &rgb(20, 10),
                    20,
                    10,
                    Rect {
                        x: 0,
                        y: 0,
                        w: 1,
                        h: 1,
                    },
                    "truck",
                    0.87,
                ),
                Instant::now(),
            )
            .unwrap();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.contains("truck"), "{name}");
        assert!(name.contains("087"), "confidence missing: {name}");
        assert!(name.contains("20x10"), "dimensions missing: {name}");
        fs::remove_dir_all(&d).ok();
    }
}
