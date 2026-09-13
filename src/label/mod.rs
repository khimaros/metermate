//! labelling the harvest, and measuring what the labels are worth.
//!
//! the harvest is the only source of reference crops that share this camera's
//! glass, angle and light, and it arrives unlabelled. this turns it into an
//! answer without leaving the binary: no second toolchain to install, no copy
//! of the harvest to keep in sync, and it runs wherever metermate runs (r5.5).
//!
//! **nothing here knows what a go-4 is.** a subject is a name carried through
//! as a parameter (r10): the same code labels a street sweeper, and nothing in
//! it assumes the thing is a vehicle. the go-4 is the first subject, not a
//! special case.
//!
//! **a passage, not a crop, is the unit of an example.** a thing crossing this
//! block is cropped every few tenths of a second for several seconds, so eight
//! crops of one vehicle are eight pictures of one moment. counting them as
//! eight examples inflates everything that matters: a top-k mean over one
//! passage is a single reference with extra steps, and a recall over 3 passages
//! reported as over 19 crops claims an interval it has not earned.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub mod eval;
pub mod server;

/// the reserved subject name for "none of the ones we know about".
///
/// it labels the negative half of every subject's directory, but the halves are
/// not one pool: a crop labelled `other` is a negative for each subject, while a
/// crop labelled as some *other subject* is a negative only for the rest. one
/// shared pool cannot express the second case, since a go-4 sitting in it would
/// drag the go-4's own score down by exactly what it contributes to the waymo's.
pub const NEGATIVE: &str = "other";

/// and for "a person looked and could not tell".
///
/// the third answer matters. a crop clipped by the frame edge, or too small to
/// read, is neither an example nor a clean negative, and forcing it into one of
/// those is how a reference set acquires a vehicle nobody could identify.
pub const UNCLEAR: &str = "unclear";

/// crops of one subject further apart than this are separate passages.
///
/// generous on purpose. the per-place rate limit spaces crops of one transit by
/// a few tenths of a second, but something that slows, queues, or is briefly
/// occluded leaves a hole: one observed passage ran 7.5s with a 2.9s gap in the
/// middle. splitting one passage in two is the expensive error -- it puts the
/// same object in both the reference and the eval set, which is the one thing
/// the eval exists to prevent -- while merging two costs one example.
pub const PASSAGE_GAP_MS: u128 = 10_000;

/// how often a long loop says where it is.
///
/// a clock rather than a fraction of the total, because a fraction is measured in
/// steps and the steps are not alike: embedding one crop costs a jpeg decode and an
/// inference, ranking one costs a few dot products. a decile of a 65807-crop
/// harvest was minutes of silence -- which is the exact thing this exists to
/// rule out -- while a decile of a ranking pass would have been a flood.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// a count with a clock on it, for the loops long enough to look hung.
///
/// **embedding a harvest and cross-validating it are minutes each**, on a
/// machine whose only other sign of life is a fan. announcing the total at the
/// start says a wait is coming and never how much of it is left, and the
/// difference matters: one is a reason to wait and the other is a reason to
/// check whether anything is still running.
pub struct Progress {
    doing: &'static str,
    did: &'static str,
    /// owned, so a phase can name the subject it is about: "crops labelled
    /// go4" says which of three directions is running where "crops" does not.
    unit: String,
    total: usize,
    started: std::time::Instant,
    /// when the last line was printed, measured from the start, so "is a line
    /// due" is one subtraction rather than a second clock.
    last: std::time::Duration,
    /// how far the last line was, so a milestone is a comparison and not a
    /// remainder kept up with.
    at: usize,
    done: usize,
}

impl Progress {
    pub fn new(
        doing: &'static str,
        did: &'static str,
        unit: impl Into<String>,
        total: usize,
    ) -> Self {
        Self {
            doing,
            did,
            unit: unit.into(),
            total,
            started: std::time::Instant::now(),
            last: std::time::Duration::ZERO,
            at: 0,
            done: 0,
        }
    }

    /// whether the clock says a line is due at `gone`, measured from the start.
    fn due(&self, gone: std::time::Duration) -> bool {
        gone - self.last >= PROGRESS_INTERVAL
    }

    /// whether this step crossed a tenth of the total. a run too small to have
    /// tenths reports every step instead: the loops this counts are tens of
    /// milliseconds a step at worst, and one of its steps is one direction of the
    /// second look, which is worth saying out loud.
    fn milestone(&self) -> bool {
        let tenth = (self.total / 10).max(1);
        self.done / tenth > self.at / tenth
    }

    /// one step done, reported whenever the clock or a tenth of the total says
    /// so.
    ///
    /// **both, because the two loops differ by three orders of magnitude.** a
    /// decile alone is a line every two minutes when a harvest is a few thousand
    /// crops, which is the silence that looks like a hung run; a clock alone is
    /// no line at all for the fifty-crop harvest that finishes in two seconds,
    /// and a run that reports nothing until it is over has told nobody how far
    /// through it was. returns whether this step was the line.
    pub fn tick(&mut self) -> bool {
        self.done += 1;
        // the last step belongs to `finished`.
        if self.done >= self.total {
            return false;
        }
        let gone = self.started.elapsed();
        if !(self.due(gone) || self.milestone()) {
            return false;
        }
        self.last = gone;
        self.at = self.done;
        // estimated from the rate so far, which is the only estimate available
        // and is good enough for deciding whether to wait: every step costs the
        // same jpeg decode, or the same dot products, as the last one.
        let left = gone.mul_f64((self.total - self.done) as f64 / self.done as f64);
        let timing = match left.as_secs() {
            0 => format!(", {} gone", short(gone)),
            _ => format!(", {} gone, ~{} left", short(gone), short(left)),
        };
        tracing::info!(
            "{} {}: {}/{} ({}%){timing}",
            self.doing,
            self.unit,
            self.done,
            self.total,
            self.done * 100 / self.total.max(1),
        );
        true
    }

    /// what it cost, once. the rate is what says whether the next run of this
    /// is worth starting before lunch.
    ///
    /// a phase with nothing in it says nothing: "ranked 0 undecided crops" is
    /// a line about work that did not happen.
    pub fn finished(&self) {
        if self.total == 0 {
            return;
        }
        tracing::info!(
            "{} {} {} in {}",
            self.did,
            self.done,
            self.unit,
            short(self.started.elapsed())
        );
    }
}

/// a duration as a person reads one: `45s`, `2m30s`, `1h4m`.
fn short(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    match secs {
        0 => "<1s".to_string(),
        1..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m{}s", secs / 60, secs % 60),
        _ => format!("{}h{}m", secs / 3600, (secs % 3600) / 60),
    }
}

/// how a crop came to be on screen. the distinction decides what may be
/// measured on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    /// drawn by a hash of the filename, so the sample is the same set on every
    /// run and does not reshuffle as the harvest grows. **the only pool a false
    /// positive rate may be measured on.**
    Random,
    /// nearest the known examples. finds the next ones far faster than chance
    /// and is biased by construction, so a rate measured on it would be
    /// measuring the ranker.
    Ranked,
    /// entered by hand rather than offered by the queue.
    Seed,
    /// put back on screen by a measurement that found it unlike the class. it
    /// is already labelled; this asks whether that label should stand.
    Review,
    /// the classifier fires on this one at the current operating point. what an
    /// alert would actually be, and the only way to measure precision rather
    /// than assume it.
    Alert,
    /// labelled off the live page while watching the street.
    ///
    /// **chosen by whoever was looking**, which is the opposite of the random
    /// pool: the crops grid is newest-first, so what is on it is whatever just
    /// happened rather than a sample of anything. kept apart for the same
    /// reason `Ranked` is -- a rate measured over crops a person picked out
    /// measures the person.
    Preview,
}

impl Via {
    pub fn slug(self) -> &'static str {
        match self {
            Via::Random => "random",
            Via::Ranked => "ranked",
            Via::Seed => "seed",
            Via::Review => "review",
            Via::Alert => "alert",
            Via::Preview => "preview",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "random" => Some(Via::Random),
            "ranked" => Some(Via::Ranked),
            "seed" => Some(Via::Seed),
            "review" => Some(Via::Review),
            "alert" => Some(Via::Alert),
            "preview" => Some(Via::Preview),
            _ => None,
        }
    }
}

/// what a person said a crop is: a subject name, `other`, or `unclear`.
///
/// there is deliberately no record of whether a decision was made one crop at a
/// time or by marking a screenful at once. looking at a screen and saying "none
/// of these" is the same judgement as clicking each one, and on this street it
/// is the usual one: the subject is well under a percent of the harvest, so a
/// random screenful is almost all negative and a per-crop click adds nothing
/// but wear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub truth: String,
    /// which pool put it on screen. a false positive rate may only be measured
    /// on the random one.
    pub via: Via,
}

impl Entry {
    pub fn new(truth: &str, via: Via) -> Self {
        Self {
            truth: truth.to_string(),
            via,
        }
    }
}

/// one decision per crop, in a plain text file.
///
/// text for the same reasons the reference file is text: it needs no json
/// dependency, it diffs, and when something looks wrong it can be read
/// directly. one line per crop, `<name> <truth> <via>`.
///
/// the truth is a subject name rather than a boolean, so one file covers every
/// subject at once (r10.4). a crop labelled `other` is negative *for the
/// subjects known when it was labelled*, which is why the contamination screen
/// matters more as the list grows rather than less.
#[derive(Debug, Default, Clone)]
pub struct Labels {
    pub entries: BTreeMap<String, Entry>,
}

impl Labels {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading labels {}", path.display()))?;
        let mut out = Self::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(name), Some(truth)) = (parts.next(), parts.next()) else {
                anyhow::bail!(
                    "{}:{}: expected `<name> <truth> <via>`",
                    path.display(),
                    n + 1
                );
            };
            let via = parts.next().and_then(Via::parse).unwrap_or(Via::Seed);
            out.entries.insert(name.to_string(), Entry::new(truth, via));
        }
        Ok(out)
    }

    /// written through a temporary, so a crash mid-write leaves yesterday's
    /// labels rather than half of today's. this file is hours of human
    /// attention and cannot be regenerated from anything.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let mut body = String::from(
            "# one decision per crop: <name> <truth> <via>\n# truth is a subject name, \"other\", or \"unclear\"\n",
        );
        for (name, e) in &self.entries {
            body.push_str(&format!("{name} {} {}\n", e.truth, e.via.slug()));
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
        Ok(())
    }

    /// write the file back, keeping decisions another writer made meanwhile.
    ///
    /// **the live page appends to this file while a session has it open.** a
    /// plain `save` writes what this process loaded at startup plus what it has
    /// decided since, which silently drops every label made in the preview in
    /// between -- and those are labels nothing can recover, since the file is
    /// the only record. so the file is re-read first and anything new in it
    /// that this session has no opinion about is carried through.
    /// `dropped` names the crops this session deliberately un-labelled, and it
    /// has to be said rather than inferred: "absent from what I hold" is also
    /// true of every label the other writer added, so inferring it would throw
    /// away the labels this exists to keep.
    pub fn save_merged(&mut self, path: &Path, dropped: &[String]) -> Result<()> {
        let mut merged = Self::load(path)?;
        for name in dropped {
            merged.entries.remove(name);
        }
        for (name, entry) in &self.entries {
            merged.entries.insert(name.clone(), entry.clone());
        }
        merged.save(path)?;
        // and the session adopts what it found, so the page shows a label made
        // on the live page rather than re-offering the crop as undecided.
        self.entries = merged.entries;
        Ok(())
    }

    /// add one decision to the file without reading or rewriting it.
    ///
    /// **two processes write this file.** the pipeline serves the preview and
    /// a labelling session is a second `metermate` against the same harvest,
    /// which is how it is actually used. a read-modify-write from here would
    /// drop whatever the other one wrote in between, so a decision made on the
    /// live page is a line appended to the end -- `load` is last-line-wins, so
    /// appending *is* deciding, and `O_APPEND` keeps a line from interleaving
    /// with another writer's.
    pub fn append(path: &Path, name: &str, truth: &str, via: Via) -> Result<()> {
        use std::io::Write;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        writeln!(file, "{name} {truth} {}", via.slug())
            .with_context(|| format!("appending to {}", path.display()))
    }

    /// every crop a person said is this subject.
    pub fn named(&self, subject: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.truth == subject)
            .map(|(n, _)| n.clone())
            .collect();
        out.sort();
        out
    }

    /// the subjects that appear in the file, excluding the reserved names.
    pub fn subjects(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .entries
            .values()
            .map(|e| e.truth.clone())
            .filter(|t| t != NEGATIVE && t != UNCLEAR)
            .collect();
        out.sort();
        out.dedup();
        out
    }
}

/// group crop names into runs of one passage, oldest first.
///
/// the input is one subject's crops, because a gap in time only means "a
/// different one" among crops already known to be the same kind of thing.
pub fn passages(names: &[String], gap_ms: u128) -> Vec<Vec<String>> {
    let mut dated: Vec<(u128, String)> = names
        .iter()
        .filter_map(|n| Some((crate::harvest::parse_name(n)?.at_millis, n.clone())))
        .collect();
    dated.sort();
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut last = 0u128;
    for (at, name) in dated {
        match out.last_mut() {
            Some(run) if at.saturating_sub(last) <= gap_ms => run.push(name),
            _ => out.push(vec![name]),
        }
        last = at;
    }
    out
}

/// the crops of a passage most worth being a reference, biggest first.
///
/// something is largest when it is nearest and unoccluded, and the crops to
/// avoid are the slivers cut at the frame edge as it enters and leaves -- a
/// fifth of an object stretched to clip's square input, which three-nearest
/// voting must not land on.
pub fn clearest(names: &[String], n: usize) -> Vec<String> {
    let mut sized: Vec<(u64, String)> = names
        .iter()
        .filter_map(|name| {
            let s = crate::harvest::parse_name(name)?;
            Some((u64::from(s.width) * u64::from(s.height), name.clone()))
        })
        .collect();
    sized.sort_by_key(|(area, _)| std::cmp::Reverse(*area));
    sized.into_iter().take(n).map(|(_, name)| name).collect()
}

/// a stable pseudo-random order over filenames, by fnv-1a.
///
/// a seeded shuffle would reorder everything the moment the harvest grew, so
/// yesterday's random sample would not be a subset of today's and the pool a
/// false positive rate was measured on would quietly change under it. hashing
/// the name fixes each crop's place in the order for good.
///
/// fnv rather than `DefaultHasher`: the standard hasher makes no promise of
/// stability across releases, and a sample that silently reshuffles on a
/// toolchain upgrade is exactly what this exists to prevent.
pub fn hash_rank(name: &str, seed: u64) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut h = OFFSET ^ seed;
    for b in name.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// decode a harvested crop back to rgb.
///
/// the one thing that kept this out of the binary: a crop is a jpeg on disk and
/// embedding it means getting pixels back. off the hot path -- ffmpeg hands the
/// pipeline raw frames, so nothing in gate or detect decodes a jpeg.
pub fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;

    let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(bytes));
    decoder
        .decode_headers()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let info = decoder
        .info()
        .ok_or_else(|| anyhow::anyhow!("jpeg has no header"))?;
    // the embedder wants rgb24 and reads three bytes per pixel. a greyscale or
    // cmyk jpeg would decode to a different stride and be embedded as noise,
    // silently, so it is refused rather than reinterpreted.
    let space = decoder.output_colorspace();
    anyhow::ensure!(
        space == Some(ColorSpace::RGB),
        "expected an rgb jpeg, got {space:?}"
    );
    let pixels = decoder.decode().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok((pixels, u32::from(info.width), u32::from(info.height)))
}

/// embeddings for harvested crops, cached on disk.
///
/// embedding is the slow part of everything here and a crop's vector never
/// changes, so it is computed once. the cache is what makes retraining seconds
/// rather than a re-embed of the whole harvest, and what makes adding a subject
/// free: a vector does not depend on what is being looked for (r10.4).
///
/// one table, however many files it was read from: a tree of sets is several
/// caches read as one. it holds no path of its own, because a vector is always
/// written beside the crops it was computed from (`ensure`) -- a view that
/// remembered one file would collapse every set's vectors into it.
#[derive(Default)]
pub struct Vectors {
    table: BTreeMap<String, Vec<f32>>,
    dim: usize,
}

/// identifies the file and the layout, so a cache written by an earlier
/// embedder export is refused rather than silently mixed with a newer one --
/// which would look like nothing at all and quietly change every number.
const CACHE_MAGIC: &[u8; 8] = b"MMVEC001";

impl Vectors {
    pub fn load(path: &Path) -> Result<Self> {
        let mut out = Self::default();
        let Ok(bytes) = std::fs::read(path) else {
            return Ok(out);
        };
        if bytes.len() < 12 || &bytes[..8] != CACHE_MAGIC {
            tracing::warn!(
                "{} is not a vector cache; starting a new one",
                path.display()
            );
            return Ok(out);
        }
        out.dim = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let mut at = 12;
        while at + 2 <= bytes.len() {
            let n = u16::from_le_bytes(bytes[at..at + 2].try_into()?) as usize;
            at += 2;
            let end = at + n + out.dim * 4;
            if end > bytes.len() {
                tracing::warn!("{} ends mid-record; the tail is ignored", path.display());
                break;
            }
            let name = String::from_utf8(bytes[at..at + n].to_vec())?;
            let vector = bytes[at + n..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            out.table.insert(name, vector);
            at = end;
        }
        Ok(out)
    }

    pub fn get(&self, name: &str) -> Option<&Vec<f32>> {
        self.table.get(name)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// take in another cache's records, refusing one of a different width.
    fn adopt(&mut self, other: Vectors) -> Result<()> {
        if other.table.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(
            self.dim == 0 || self.dim == other.dim,
            "a cache of {} floats cannot be merged with one of {}: `cosine` truncates \
             to the shorter of the two and would compare them anyway",
            other.dim,
            self.dim
        );
        self.dim = other.dim;
        self.table.extend(other.table);
        Ok(())
    }

    /// append one record to the cache at `path`. the file is append-only so a
    /// run killed part way through keeps everything it had already embedded.
    fn append(&mut self, path: &Path, name: &str, vector: &[f32]) -> Result<()> {
        use std::io::Write;
        if self.dim == 0 {
            self.dim = vector.len();
        }
        anyhow::ensure!(
            vector.len() == self.dim,
            "embedding is {} long, cache holds {}",
            vector.len(),
            self.dim
        );
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let fresh = !path.exists();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        if fresh {
            f.write_all(CACHE_MAGIC)?;
            f.write_all(&(self.dim as u32).to_le_bytes())?;
        }
        f.write_all(&(name.len() as u16).to_le_bytes())?;
        f.write_all(name.as_bytes())?;
        for v in vector {
            f.write_all(&v.to_le_bytes())?;
        }
        self.table.insert(name.to_string(), vector.to_vec());
        Ok(())
    }

    /// embed whatever of `dir` is not cached yet, into the cache beside `dir`.
    /// returns how many were computed.
    pub fn ensure(
        &mut self,
        dir: &Path,
        names: &[String],
        embedder: &mut crate::classify::Embedder,
    ) -> Result<usize> {
        let (_, path) = paths(dir);
        let mut done = 0;
        let mut progress = Progress::new(
            "embedding",
            "embedded",
            "crops",
            names
                .iter()
                .filter(|n| !self.table.contains_key(*n))
                .count(),
        );
        for name in names {
            if self.table.contains_key(name) {
                continue;
            }
            let bytes = match std::fs::read(dir.join(name)) {
                Ok(b) => b,
                // a crop can be evicted by the disk budget at any moment. that
                // is the budget working, not a reason to abandon the batch.
                Err(e) => {
                    tracing::warn!("{name}: {e}");
                    continue;
                }
            };
            let (rgb, w, h) = match decode_jpeg(&bytes) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("{name}: {e:#}");
                    continue;
                }
            };
            let vector = embedder.embed(&rgb, w, h)?;
            self.append(&path, name, &vector)?;
            done += 1;
            // counted where the work happened, so a crop the budget evicted
            // mid-run is not reported as embedded.
            progress.tick();
        }
        progress.finished();
        Ok(done)
    }
}

/// the crop directories under `root`, each one a set with its own labels.
///
/// a set is `sets/<subject>/<id>/crops` with `labels.txt` and `embeddings.bin`
/// beside it, so training can walk a tree of them and let one subject's crops be
/// negatives for another. a directory that holds crops itself is a single
/// harvest and is returned as one, which is what the live `[harvest] dir` is.
///
/// **the subject in that path is what the set was collected for, not what its
/// crops are.** the go-4's own set is mostly negatives -- they came off the same
/// camera in the same window, which is the whole reason they are worth having --
/// so truth comes from `labels.txt` and never from the directory name.
pub fn sets(root: &Path) -> Vec<PathBuf> {
    // what counts as a crop is the harvest's own question, not a second copy of
    // it here, and the listing it memoises is the one `scan` reads next.
    if !crate::harvest::page(root, None, 1, crate::harvest::Want::Everything).is_empty() {
        return vec![root.to_path_buf()];
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut below: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    // sorted, so which set is read first does not depend on the filesystem.
    below.sort();
    below.iter().flat_map(|d| sets(d)).collect()
}

/// every set under `roots`, a set reached from two roots listed once.
pub fn sets_under(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = roots.iter().flat_map(|h| sets(h)).collect();
    dirs.sort();
    dirs.dedup();
    dirs
}

/// the caches beside `dirs` as one table to read vectors out of.
pub fn merged(dirs: &[PathBuf]) -> Result<Vectors> {
    let mut out = Vectors::default();
    for dir in dirs {
        out.adopt(Vectors::load(&paths(dir).1)?)?;
    }
    Ok(out)
}

/// what people decided about the crops in `dirs`, each set's labels read from
/// beside it.
pub fn labels_of(dirs: &[PathBuf]) -> Result<Labels> {
    let parts: Result<Vec<_>> = dirs
        .iter()
        .map(|d| Ok((d.clone(), Labels::load(&paths(d).0)?)))
        .collect();
    merge_labels(parts?)
}

/// one set of labels from several, refusing a crop two sets disagree about.
///
/// a label is a person's judgement and cannot be regenerated from anything, so
/// one set quietly overwriting another's is not a merge conflict to resolve by
/// ordering -- it is evidence being dropped.
pub fn merge_labels(parts: Vec<(PathBuf, Labels)>) -> Result<Labels> {
    let mut out = Labels::default();
    let mut from: BTreeMap<String, PathBuf> = BTreeMap::new();
    for (at, labels) in parts {
        for (name, e) in labels.entries {
            // `via` may differ -- the same crop can reach two sets by different
            // routes -- but the truth cannot. only the judgement is evidence.
            if let Some(prev) = out.entries.get(&name) {
                anyhow::ensure!(
                    prev.truth == e.truth,
                    "{name} is labelled {} in {} and {} in {}",
                    prev.truth,
                    from[&name].display(),
                    e.truth,
                    at.display()
                );
                continue;
            }
            out.entries.insert(name.clone(), e);
            from.insert(name, at.clone());
        }
    }
    Ok(out)
}

/// copy the crops a set's labels name into the set, and say what is already gone.
///
/// **a set that does not own its crops stops being reproducible without
/// anything failing.** the harvest deletes oldest first under a disk budget, so
/// a label outlives the crop it describes: measured on this camera, 1598 of 6213
/// labels in one set already name crops that no longer exist. training then runs
/// on whatever survived and reports the same confident numbers about it.
///
/// the labels are the manifest, not the harvest listing. copying whatever
/// happens to be in `from` would quietly grow the set with crops nobody judged.
///
/// returns (copied, already had, unrecoverable). idempotent, because this runs
/// again every time labelling continues.
pub fn gather(set: &Path, from: &Path) -> Result<(usize, usize, usize)> {
    let labels = Labels::load(&set.join("labels.txt"))?;
    let crops = set.join("crops");
    std::fs::create_dir_all(&crops).with_context(|| format!("creating {}", crops.display()))?;

    let (mut copied, mut held, mut lost) = (0, 0, 0);
    for name in labels.entries.keys() {
        let there = crops.join(name);
        if there.exists() {
            held += 1;
            continue;
        }
        let here = from.join(name);
        if !here.exists() {
            lost += 1;
            continue;
        }
        // through a temporary, so a run killed mid-copy leaves no half a crop
        // to be read later as a whole one.
        let tmp = crops.join(format!("{name}.tmp"));
        std::fs::copy(&here, &tmp).with_context(|| format!("copying {}", here.display()))?;
        std::fs::rename(&tmp, &there)?;
        copied += 1;
    }
    Ok((copied, held, lost))
}

/// where the labels and the vector cache live, beside the harvest.
///
/// the harvest directory is already configured and these belong with it: the
/// cache is derived from it and is thrown away with it, and the labels are
/// about the crops in it.
pub fn paths(harvest: &Path) -> (PathBuf, PathBuf) {
    let root = harvest.parent().unwrap_or(harvest);
    (root.join("labels.txt"), root.join("embeddings.bin"))
}

/// embed whatever the measurement will need and then measure it.
///
/// embedding is the slow part and the cache makes it a one-off, so this is the
/// call that turns a directory of jpegs into an answer.
pub fn measure(
    harvests: &[PathBuf],
    model: &Path,
    opts: &eval::Options,
) -> Result<(eval::Report, Labels)> {
    let (labels, cache, names) = loaded(harvests, model)?;
    Ok((eval::measure(&labels, &cache, &names, opts), labels))
}

/// the labels, the vectors and the crop names, with the cache backfilled.
///
/// shared by measuring and training rather than written twice: they must agree
/// about which crops exist and what each one embeds to, and two copies would
/// agree until one of them was edited.
fn loaded(harvests: &[PathBuf], model: &Path) -> Result<(Labels, Vectors, Vec<String>)> {
    let roots = || {
        harvests
            .iter()
            .map(|h| h.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let dirs = sets_under(harvests);
    anyhow::ensure!(!dirs.is_empty(), "no crops in {}", roots());
    let labels = labels_of(&dirs)?;
    let mut cache = merged(&dirs)?;
    let names = scan(&dirs, model, &mut cache)?;
    anyhow::ensure!(!names.is_empty(), "no crops in {}", roots());
    Ok((labels, cache, names))
}

/// every crop in `dirs`, oldest first, embedding what `cache` has never seen.
///
/// **doing it again does nothing.** a vector is written once, into the cache
/// beside the set its crop was found in, and a crop two sets share is embedded
/// and named once -- so walking a tree a second time costs the listing.
fn scan(dirs: &[PathBuf], model: &Path, cache: &mut Vectors) -> Result<Vec<String>> {
    let mut found: Vec<crate::harvest::Saved> = Vec::new();
    // loaded once however many sets there are: it is 351 MB, and a tree of sets
    // already embedded costs nothing to walk.
    let mut embedder = None;
    for dir in dirs {
        let mine = crate::harvest::page(dir, None, usize::MAX, crate::harvest::Want::Everything);
        let names: Vec<String> = mine.iter().map(|s| s.name.clone()).collect();
        let missing = names.iter().filter(|n| cache.get(n).is_none()).count();
        if missing > 0 {
            tracing::info!(
                "embedding {missing} crops not yet in {}",
                paths(dir).1.display()
            );
            if embedder.is_none() {
                embedder = Some(
                    crate::classify::Embedder::load(model)
                        .with_context(|| format!("loading embedder {}", model.display()))?,
                );
            }
            cache.ensure(dir, &names, embedder.as_mut().expect("just loaded"))?;
        }
        found.extend(mine);
    }

    // **sorted across every set, not merely within each one.** `passages`
    // groups by the gap between consecutive crops, so names concatenated
    // set-by-set would invent passage boundaries at each seam -- which is what
    // puts one vehicle on both sides of the reference and eval split.
    found.sort_by_key(|s| s.at_millis);
    // deduplicated across the whole tree rather than between neighbours: two
    // crops can share a millisecond, so an equal pair need not end up adjacent
    // and `dedup` would leave one set's copy to be counted as a second example.
    let mut seen = std::collections::BTreeSet::new();
    Ok(found
        .into_iter()
        .map(|s| s.name)
        .filter(|n| seen.insert(n.clone()))
        .collect())
}

/// write the reference file the classifier loads, from the labels.
///
/// **exactly what the measurement scored**, taken off the report rather than
/// selected again here. two selections would agree until the day they did not,
/// and nothing would say which of them the deployment was running.
///
/// **uncentred vectors, whatever the measurement ran with.** centring helps a
/// trained head and does nothing for the shipped rule, whose two-sided
/// subtraction already centres implicitly, so a centred file would leave the
/// runtime comparing against a space it never sees.
pub fn train(
    harvests: &[PathBuf],
    model: &Path,
    opts: &eval::Options,
    out: &Path,
) -> Result<eval::Report> {
    let (labels, cache, names) = loaded(harvests, model)?;
    let report = eval::measure(&labels, &cache, &names, opts);
    let point = report.operating_point();
    // **an operating point nobody can reach is not quietly replaced by one they
    // did not ask for.** the default bracket going unmet is a measurement
    // saying stage 2 is not ready yet, and it writes the vectors without a
    // margin; a rate an operator ruled out explicitly is different, and
    // shipping the closest thing to it would ship a rule they rejected.
    if point.is_none() && report.constrained() {
        let (fpr, recall) = report.best_available();
        anyhow::bail!(
            "no row of the curve meets that operating point. \
             the cleanest margin fires on {:.2}% of the street, \
             and the most generous one keeps {:.0}% of the passages -- \
             `--measure {}` prints the whole curve",
            fpr * 100.0,
            recall * 100.0,
            report.subject
        );
    }
    write_trained(&report, &cache, out, point)?;
    Ok(report)
}

/// write `trained/<subject>/` from a measurement, at one row of its curve.
///
/// `--train` passes the operating point and the labelling page passes whichever
/// row a person picked; either way the numbers saved beside the margin are that
/// row's own.
pub fn write_trained(
    report: &eval::Report,
    cache: &Vectors,
    out: &Path,
    row: Option<&eval::Row>,
) -> Result<()> {
    if let Some(e) = &report.error {
        anyhow::bail!("{e}");
    }
    // the preconditions exist to stop confident numbers about a rule nobody
    // ran. writing a file from them would ship that mistake rather than print
    // it, and the deployment would be the only place it showed up.
    anyhow::ensure!(
        report.fatal.is_empty(),
        "{} is not worth shipping:\n  {}",
        report.subject,
        report.fatal.join("\n  ")
    );

    // an existing file may name other subjects, and one file covers them all
    // (r10.2), so replacing the lot would silently retire one nobody asked
    // about.
    let mut refs = if out.exists() {
        crate::classify::References::load(out)?
    } else {
        crate::classify::References::default()
    };
    let raw = |wanted: &[String]| -> Vec<Vec<f32>> {
        wanted
            .iter()
            .filter_map(|n| cache.get(n).cloned())
            .collect()
    };
    let positives = raw(&report.reference_crops);
    let negatives = raw(&report.negative_reference_crops);
    anyhow::ensure!(
        !positives.is_empty(),
        "none of the {} chosen references are in the vector cache",
        report.reference_crops.len()
    );
    anyhow::ensure!(
        !negatives.is_empty(),
        "no negatives: \"is this a {}\" has no answer without \"compared to what\"",
        report.subject
    );
    // **only this subject's two files are written.** the others keep whatever
    // they were trained against, which is the point of a directory apiece: a
    // shared file meant training one subject rewrote every subject's rows, and
    // nothing said which measurement the deployment was then running.
    refs.subjects.insert(
        report.subject.clone(),
        crate::classify::Subject {
            positives,
            negatives,
            margin: row.map(|r| r.margin),
        },
    );
    refs.save(out, &report.subject)?;

    // **the margin goes with the vectors it was measured against.** printing it
    // and trusting a person to copy it into a config on another machine is the
    // one step of a deployment that can silently not happen, and the result is
    // a reference set and a margin that were never measured together.
    //
    // taken off the same report that chose the references, for the same reason
    // the references are: two computations agree until the day they do not, and
    // nothing would say which of them the deployment was running.
    crate::classify::Trained {
        margin: row.map(|r| r.margin),
        passages: row.map_or(0, |r| r.passages),
        passage_recall: row.map_or(0.0, |r| r.passage_recall),
        fpr: row.map_or(0.0, |r| r.fpr),
        references: report.n_references,
        negatives: report.n_negative_references,
        trained_at_millis: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
        // the curve the row came off, so the alternatives to the margin were
        // measured in the same run as the margin.
        ladder: report
            .decision_curve()
            .iter()
            .map(|r| crate::classify::Rung {
                margin: r.margin,
                passages: r.passages,
                passage_recall: r.passage_recall,
                fpr: r.fpr,
            })
            .collect(),
    }
    .save(&out.join(&report.subject))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a decile of a 65807-crop harvest is 6580 crops, which on this machine is
    /// minutes of a log saying nothing -- and a silent log is what looks hung.
    /// a clock is the right unit here anyway, because the loops this covers differ
    /// by orders of magnitude in what one step costs: a jpeg decode when embedding,
    /// a few dot products when ranking.
    #[test]
    fn progress_reports_on_a_clock_when_steps_cost_seconds() {
        let mut p = Progress::new("embedding", "embedded", "crops", 65_807);
        p.done = 300;
        assert!(
            !p.due(std::time::Duration::from_secs(1)),
            "the first line came early"
        );
        assert!(p.due(PROGRESS_INTERVAL), "the first line never came");
        // and every interval after, however cheap the steps turn out to be.
        p.last = PROGRESS_INTERVAL;
        assert!(!p.due(PROGRESS_INTERVAL + std::time::Duration::from_secs(9)));
        assert!(p.due(PROGRESS_INTERVAL * 2));
    }

    /// **a clock alone is silent about the run that finishes in two seconds**, and
    /// the fifty-crop harvest is the common case rather than the rare one: nothing
    /// is printed until the phase is over, which is the same nothing a hung run
    /// prints. a tenth of the total is the other half of the rule.
    #[test]
    fn a_run_shorter_than_the_interval_reports_every_tenth_step() {
        let mut p = Progress::new("embedding", "embedded", "crops", 52);
        let lines = (0..52).filter(|_| p.tick()).count();
        // one per decile crossed, the last step left to `finished`.
        assert_eq!(lines, 10, "a two second run printed {lines} progress lines");
    }

    /// and a run too small to have tenths reports every step, except the one that
    /// has nothing before it.
    #[test]
    fn a_run_with_no_tenths_reports_every_step() {
        let mut p = Progress::new("comparing", "compared", "crops", 3);
        assert_eq!(
            (0..3).filter(|_| p.tick()).count(),
            2,
            "the last step belongs to `finished`"
        );
        let mut one = Progress::new("comparing", "compared", "crops", 1);
        assert_eq!(
            (0..1).filter(|_| one.tick()).count(),
            0,
            "a single crop reports only that it is done"
        );
    }

    /// a loop shorter than the interval still has to be accounted for, or a phase
    /// that ran for nine seconds ends without ever having said anything.
    #[test]
    fn a_short_phase_reports_once_at_the_end() {
        let mut p = Progress::new("ranking", "ranked", "crops", 3);
        for _ in 0..3 {
            p.tick();
        }
        assert_eq!(p.done, 3);
        p.finished();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("metermate-label-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn name(millis: u128, w: u32, h: u32) -> String {
        format!("{millis}_car_085_{w}x{h}.jpg")
    }

    #[test]
    fn labels_round_trip_through_the_text_format() {
        let d = tmpdir("labels");
        let path = d.join("waymo.labels");
        let mut labels = Labels::default();
        labels.entries.insert(
            name(1_789_259_517_126, 673, 396),
            Entry::new("waymo", Via::Random),
        );
        labels.entries.insert(
            name(1_789_259_518_933, 171, 185),
            Entry::new(NEGATIVE, Via::Ranked),
        );
        labels.save(&path).unwrap();

        let back = Labels::load(&path).unwrap();
        assert_eq!(back.entries, labels.entries);
        assert_eq!(back.named("waymo").len(), 1);
        assert_eq!(back.subjects(), vec!["waymo"]);
    }

    /// a subject is a name carried through, not a compiled-in case (r10).
    #[test]
    fn one_file_holds_more_than_one_subject() {
        let d = tmpdir("subjects");
        let path = d.join("all.labels");
        let mut labels = Labels::default();
        for (millis, truth) in [
            (1_789_259_517_126u128, "go4"),
            (1_789_259_518_126, "sweeper"),
            (1_789_259_519_126, "waymo"),
            (1_789_259_520_126, NEGATIVE),
            (1_789_259_521_126, UNCLEAR),
        ] {
            labels
                .entries
                .insert(name(millis, 100, 100), Entry::new(truth, Via::Seed));
        }
        labels.save(&path).unwrap();

        let back = Labels::load(&path).unwrap();
        assert_eq!(back.subjects(), vec!["go4", "sweeper", "waymo"]);
        assert_eq!(back.named("sweeper").len(), 1);
        assert!(back.named("bicycle").is_empty());
    }

    #[test]
    fn a_missing_label_file_is_an_empty_one_rather_than_an_error() {
        assert!(
            Labels::load(Path::new("/nonexistent/metermate.labels"))
                .unwrap()
                .entries
                .is_empty()
        );
    }

    #[test]
    fn crops_seconds_apart_are_one_passage_and_minutes_apart_are_two() {
        let names: Vec<String> = [
            1_789_259_517_126u128,
            1_789_259_517_526,
            1_789_259_519_425, // a 1.9s hole: it passed behind a van
            1_789_260_855_530, // twenty minutes later
        ]
        .iter()
        .map(|m| name(*m, 100, 100))
        .collect();

        let found = passages(&names, PASSAGE_GAP_MS);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].len(), 3, "a gap inside one transit split it");
        assert_eq!(found[1].len(), 1);
    }

    #[test]
    fn the_clearest_crops_of_a_passage_are_the_biggest_ones() {
        // the sliver is taller than two of the real crops and larger in area
        // than one, so a rule keyed on either dimension alone would keep it.
        let sliver = name(1_789_259_516_524, 138, 330);
        let names = vec![
            sliver.clone(),
            name(1_789_259_517_126, 673, 396),
            name(1_789_259_517_526, 706, 426),
            name(1_789_259_519_425, 368, 258),
            name(1_789_259_521_125, 214, 197),
        ];
        let picked = clearest(&names, 2);
        assert_eq!(picked[0], name(1_789_259_517_526, 706, 426));
        assert!(!clearest(&names, 3).contains(&sliver));
    }

    /// yesterday's random sample has to stay a subset of today's, or the pool a
    /// false positive rate was measured on changes under it.
    #[test]
    fn the_random_order_does_not_reshuffle_as_the_harvest_grows() {
        let early: Vec<String> = (0..20)
            .map(|i| name(1_789_250_000_000 + i, 100, 100))
            .collect();
        let mut later = early.clone();
        later.extend((0..20).map(|i| name(1_789_260_000_000 + i, 100, 100)));

        let order = |set: &[String]| {
            let mut v = set.to_vec();
            v.sort_by_key(|n| hash_rank(n, 0));
            v
        };
        let grown: Vec<String> = order(&later)
            .into_iter()
            .filter(|n| early.contains(n))
            .collect();
        assert_eq!(
            order(&early),
            grown,
            "adding crops reordered the existing ones"
        );
    }

    #[test]
    fn the_vector_cache_survives_being_reopened() {
        let d = tmpdir("vectors");
        let path = d.join("embeddings.bin");
        let mut cache = Vectors::load(&path).unwrap();
        assert!(cache.is_empty());

        let a = vec![0.5f32, -0.25, 0.125];
        cache
            .append(&path, "1789259517126_car_085_673x396.jpg", &a)
            .unwrap();
        cache
            .append(&path, "1789259517526_car_089_706x426.jpg", &[1.0, 0.0, 0.0])
            .unwrap();

        let back = Vectors::load(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.get("1789259517126_car_085_673x396.jpg"), Some(&a));
    }

    /// a cache from an earlier embedder export must not be mixed with a newer
    /// one: the numbers would move and nothing would look wrong.
    #[test]
    fn a_foreign_cache_file_is_discarded_rather_than_misread() {
        let d = tmpdir("foreign");
        let path = d.join("embeddings.bin");
        std::fs::write(&path, b"not a metermate cache at all").unwrap();
        assert!(Vectors::load(&path).unwrap().is_empty());
    }

    #[test]
    fn a_truncated_cache_keeps_the_records_it_did_finish() {
        let d = tmpdir("truncated");
        let path = d.join("embeddings.bin");
        let mut cache = Vectors::load(&path).unwrap();
        cache.append(&path, "a.jpg", &[1.0, 2.0, 3.0]).unwrap();
        cache.append(&path, "b.jpg", &[4.0, 5.0, 6.0]).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 5);
        std::fs::write(&path, bytes).unwrap();

        let back = Vectors::load(&path).unwrap();
        assert_eq!(
            back.len(),
            1,
            "a half-written tail took the whole file with it"
        );
        assert_eq!(back.get("a.jpg"), Some(&vec![1.0, 2.0, 3.0]));
    }

    /// a tree of sets is walked; a directory that holds crops is one set.
    ///
    /// without this, training sees one set per invocation and another subject's
    /// crops are always in a different one -- so folding them into a subject's
    /// negatives can never actually happen.
    #[test]
    fn a_set_tree_is_walked_and_a_plain_harvest_is_one_set() {
        let d = tmpdir("sets");
        let root = d.join("sets");
        for (set, id) in [("go4", "001"), ("go4", "002"), ("waymo", "001")] {
            let crops = root.join(set).join(id).join("crops");
            std::fs::create_dir_all(&crops).unwrap();
            std::fs::write(crops.join(name(1_789_259_517_126, 673, 396)), b"x").unwrap();
        }
        let found = sets(&root);
        assert_eq!(found.len(), 3, "expected three sets, found {found:?}");
        assert!(
            found.iter().all(|p| p.ends_with("crops")),
            "a set is the crops directory, so labels sit beside it: {found:?}"
        );

        // the live `[harvest] dir` is crops and nothing below it.
        let flat = d.join("crops");
        std::fs::create_dir_all(&flat).unwrap();
        std::fs::write(flat.join(name(1_789_259_517_126, 673, 396)), b"x").unwrap();
        assert_eq!(sets(&flat), vec![flat.clone()], "a flat harvest is one set");
    }

    /// **`cosine` truncates to the shorter of its two arguments.** so a cache
    /// written by one embedder merged with a cache written by another would not
    /// fail -- it would compare the first few dimensions and return numbers that
    /// look entirely reasonable. the `MMVEC001` magic refuses this per file; a
    /// merge is a second way in.
    #[test]
    fn caches_of_different_width_are_refused_rather_than_silently_compared() {
        let d = tmpdir("widths");
        let sets = [d.join("a").join("crops"), d.join("b").join("crops")];
        Vectors::default()
            .append(&paths(&sets[0]).1, "a.jpg", &[1.0, 0.0, 0.0])
            .unwrap();
        Vectors::default()
            .append(&paths(&sets[1]).1, "b.jpg", &[1.0, 0.0])
            .unwrap();
        assert!(
            merged(&sets).is_err(),
            "two widths merged into one table and nothing said so"
        );
    }

    /// a merged view has no file of its own, so what it gains is written where
    /// it is told to -- beside the set the crop came from -- and the other
    /// sets' caches stay their own.
    #[test]
    fn a_merged_cache_writes_into_the_set_it_is_given() {
        let d = tmpdir("merged");
        let sets = [d.join("a").join("crops"), d.join("b").join("crops")];
        let (a, b) = (paths(&sets[0]).1, paths(&sets[1]).1);
        Vectors::default()
            .append(&a, "a.jpg", &[1.0, 0.0, 0.0])
            .unwrap();
        let mut m = merged(&sets).unwrap();
        assert_eq!(m.get("a.jpg"), Some(&vec![1.0, 0.0, 0.0]));
        m.append(&b, "b.jpg", &[0.0, 1.0, 0.0]).unwrap();

        assert_eq!(Vectors::load(&a).unwrap().len(), 1, "a gained b's vector");
        let back = Vectors::load(&b).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.get("b.jpg"), Some(&vec![0.0, 1.0, 0.0]));
    }

    /// two sets disagreeing about one crop is evidence being dropped, not an
    /// ordering question.
    #[test]
    fn the_same_crop_labelled_differently_in_two_sets_is_refused() {
        let crop = name(1_789_259_517_126, 673, 396);
        let mut one = Labels::default();
        one.entries
            .insert(crop.clone(), Entry::new("go4", Via::Seed));
        let mut two = Labels::default();
        two.entries
            .insert(crop.clone(), Entry::new(NEGATIVE, Via::Random));

        assert!(
            merge_labels(vec![
                (PathBuf::from("sets/go4/001"), one.clone()),
                (PathBuf::from("sets/waymo/001"), two),
            ])
            .is_err(),
            "a hand-made label was overwritten by another set's"
        );
        // the same judgement in both sets is agreement, not a conflict.
        assert!(
            merge_labels(vec![
                (PathBuf::from("a"), one.clone()),
                (PathBuf::from("b"), one),
            ])
            .is_ok()
        );
    }
}
