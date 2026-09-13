//! stage 2: is this vehicle an sfmta go-4?
//!
//! the only project-specific model in the pipeline, and deliberately the
//! smallest commitment that can work. a frozen clip vision tower turns a crop
//! into an embedding, and the decision is nearest-neighbour against a handful
//! of reference crops. no training, so it works the day it is installed (r6.2).
//!
//! **ranking, never an absolute threshold.** measured on real crops, go-4s
//! resemble each other at 0.716 and resemble cars at 0.694: a margin of about
//! 0.02, while cars resemble each other at 0.834. an absolute cut-off in that
//! gap would be hopelessly brittle. which reference is *nearest* is reliable
//! where the absolute number is not.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// clip's input size and channel statistics. these are properties of the
/// pretrained model, not tuning knobs: it degrades badly away from them.
pub const EMBED_INPUT: u32 = 224;
// written to full published precision even though f32 cannot hold every digit.
// these are openai's constants verbatim; truncating them to what f32 stores
// would hide where they came from and invite someone to "correct" them.
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// how many nearest references vote. a single nearest neighbour is swayed by
/// one unlucky reference; the whole set washes out the signal.
pub const VOTE_K: usize = 3;

/// how much nearer the go-4 references must be than the negative ones before a
/// crop is called a go-4. small because the margin itself is small; this is a
/// tiebreak, not a confidence threshold.
pub const DEFAULT_MARGIN: f32 = 0.005;

#[derive(Debug, Clone)]
pub struct Judgement {
    /// which subject this is, or `None` for none of them (r10.2).
    ///
    /// a name rather than an enum variant, because what metermate watches for
    /// is configuration and adding one must not need a recompile (r10.1).
    pub subject: Option<String>,
    /// mean similarity to the nearest references of the winning subject, or of
    /// the closest one when nothing cleared its margin.
    pub score: f32,
    /// mean similarity to the nearest negative references. one shared set: the
    /// street supplies a single pool of "none of these".
    pub other_score: f32,
}

impl Judgement {
    pub fn is(&self, subject: &str) -> bool {
        self.subject.as_deref() == Some(subject)
    }

    /// how much nearer this crop's subject was than the street, **signed**: a
    /// negative means it was named while sitting nearer the negatives than its own
    /// class, which a shipped margin permits -- this deployment runs at -0.005 --
    /// and which an absolute value would report as a confident sighting.
    ///
    /// small values mean "barely decided", which callers should treat as
    /// provisional rather than as a sighting.
    pub fn margin(&self) -> f32 {
        self.score - self.other_score
    }
}

pub trait Classifier {
    fn classify(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Judgement>;
}

/// reference embeddings, both what we are looking for and what we are not.
///
/// negatives matter as much as positives. asking "is this like a go-4" has no
/// answer without "compared to what", and the street supplies endless cars.
/// one subject's half of the decision: what it looks like, and what it is being
/// told apart from.
///
/// **the negatives belong to the subject rather than to the street.** another
/// subject is the most informative negative there is -- a waymo is what stops a
/// waymo being called a go-4 -- and a single shared set cannot hold one without
/// dragging down the subject it belongs to. measured the other way round on the
/// deployment: with negatives drawn from ordinary traffic, three unrelated
/// silhouettes (a work van, a box truck, a motorcycle with a hi-vis rider) all
/// cleared the bar, because sedans are not near the boundary and the vehicles
/// that are were never in the comparison.
#[derive(Debug, Default, Clone)]
pub struct Subject {
    pub positives: Vec<Vec<f32>>,
    pub negatives: Vec<Vec<f32>>,
    /// what the sweep measured against these very vectors, when it was written
    /// down. `None` for a set trained before the margin travelled with it, and
    /// the classifier then falls back to `DEFAULT_MARGIN`.
    pub margin: Option<f32>,
}

impl Subject {
    /// both halves, or the question has no answer: "is this a go-4" needs
    /// "compared to what".
    pub fn is_usable(&self) -> bool {
        !self.positives.is_empty() && !self.negatives.is_empty()
    }
}

/// what `--train` measured, written beside the vectors it measured.
///
/// **the margin is an output, not a setting.** it is chosen by sweeping against
/// held-out passages, so a margin typed into a config on another machine names
/// an operating point nobody scored -- and the two would drift with nothing to
/// say which the deployment was running. the rest is provenance: enough to
/// answer "measured against what" without keeping the whole report.
///
/// no `deny_unknown_fields` here, unlike the config: this file is written by a
/// machine, and a newer metermate adding a field must not stop an older one
/// loading the set.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Trained {
    /// absent when the sweep found no operating point worth shipping, in which
    /// case the classifier falls back to `DEFAULT_MARGIN` rather than to zero.
    pub margin: Option<f32>,
    #[serde(default)]
    pub passages: usize,
    #[serde(default)]
    pub passage_recall: f32,
    #[serde(default)]
    pub fpr: f32,
    #[serde(default)]
    pub references: usize,
    #[serde(default)]
    pub negatives: usize,
    /// milliseconds since the epoch, which is what every other timestamp in
    /// this project already speaks.
    #[serde(default)]
    pub trained_at_millis: u128,
    /// every row of the curve the margin was chosen from, lowest margin first.
    /// the margin says where the bar is; this says what moving it would cost,
    /// against the same references and the same street.
    #[serde(default)]
    pub ladder: Vec<Rung>,
}

/// one margin of the sweep and what it measured, named as `Trained` names them.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Rung {
    pub margin: f32,
    pub passages: usize,
    pub passage_recall: f32,
    pub fpr: f32,
}

/// what each half of a subject is called on disk, and what records the rest.
pub const TRAINED: &str = "trained.toml";

impl Trained {
    /// absent is the ordinary case for a directory written before this existed,
    /// so it is not an error -- the margin simply falls back.
    pub fn load(at: &Path) -> Result<Option<Self>> {
        let path = at.join(TRAINED);
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let parsed = toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(parsed))
    }

    pub fn save(&self, at: &Path) -> Result<()> {
        let path = at.join(TRAINED);
        let body = toml::to_string(self).context("serialising the trained metadata")?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming into {}", path.display()))?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct References {
    /// one entry per subject, keyed by its directory name (r10.2). ordered, so
    /// a tie between two subjects breaks the same way every run rather than by
    /// hash order.
    pub subjects: std::collections::BTreeMap<String, Subject>,
}

impl References {
    pub fn is_usable(&self) -> bool {
        self.subjects.values().any(Subject::is_usable)
    }

    /// the subject names present, for reporting what was loaded.
    pub fn names(&self) -> Vec<&str> {
        self.subjects.keys().map(|k| k.as_str()).collect()
    }

    pub fn count(&self, subject: &str) -> usize {
        self.subjects.get(subject).map_or(0, |s| s.positives.len())
    }

    pub fn negatives(&self, subject: &str) -> usize {
        self.subjects.get(subject).map_or(0, |s| s.negatives.len())
    }

    /// the margin this subject was measured at, if it was written down.
    pub fn margin(&self, subject: &str) -> Option<f32> {
        self.subjects.get(subject).and_then(|s| s.margin)
    }

    /// load `<dir>/<subject>/{references,negatives}.txt`.
    ///
    /// **a directory rather than a file**, because the unit of training is a
    /// subject: one is written, shipped and erased on its own, and a shared
    /// file forced `--train` into a read-modify-write of everyone's rows to
    /// change one subject's. a subject directory missing either half is loaded
    /// and then skipped by `judge`, so a half-built subject cannot answer.
    pub fn load(dir: &Path) -> Result<Self> {
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("reading references directory {}", dir.display()))?;
        let mut refs = References::default();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let at = entry.path();
            refs.subjects.insert(
                name.clone(),
                Subject {
                    positives: read_vectors(&at.join(POSITIVES), &name)?,
                    negatives: read_vectors(&at.join(NEGATIVES), crate::label::NEGATIVE)?,
                    margin: Trained::load(&at)?.and_then(|t| t.margin),
                },
            );
        }
        Ok(refs)
    }

    /// write one subject's two files, leaving every other subject alone.
    ///
    /// **the only writer.** a reference file built by one toolchain and read by
    /// another is how the thing that ships stops being the thing that was
    /// measured, which is invisible: the pipeline keeps classifying and the
    /// eval keeps printing confident numbers about a rule nobody runs.
    ///
    /// each file goes through a temporary, so a crash mid-write leaves the
    /// previous ones rather than half of them.
    pub fn save(&self, dir: &Path, subject: &str) -> Result<()> {
        let s = self
            .subjects
            .get(subject)
            .with_context(|| format!("no subject {subject} to write"))?;
        let at = dir.join(subject);
        std::fs::create_dir_all(&at).with_context(|| format!("creating {}", at.display()))?;
        write_vectors(&at.join(POSITIVES), subject, &s.positives)?;
        write_vectors(&at.join(NEGATIVES), crate::label::NEGATIVE, &s.negatives)?;
        Ok(())
    }
}

/// what each half of a subject is called on disk.
pub const POSITIVES: &str = "references.txt";
pub const NEGATIVES: &str = "negatives.txt";

/// `<label> <floats...>` per line, one embedding each.
///
/// a plain text format on purpose: it needs no json dependency, it diffs, and
/// when the classifier misbehaves the references can be read directly.
///
/// **the label is redundant inside a subject's own directory and kept anyway.**
/// a file that says what it is can be read on its own, and a label disagreeing
/// with the directory that holds it is a mistake worth refusing rather than a
/// detail worth ignoring -- a `negatives.txt` full of go-4s would otherwise
/// train the exact inversion of the rule and never say so.
pub fn read_labelled(path: &Path) -> Result<Vec<(String, Vec<f32>)>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let mut parts = line.split_whitespace();
        let Some(label) = parts.next() else { continue };
        let values: Vec<f32> = parts.filter_map(|v| v.parse().ok()).collect();
        anyhow::ensure!(
            !values.is_empty(),
            "{}:{}: no embedding values",
            path.display(),
            n + 1
        );
        out.push((label.to_string(), values));
    }
    Ok(out)
}

fn read_vectors(path: &Path, expect: &str) -> Result<Vec<Vec<f32>>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let rows = read_labelled(path)?;
    for (label, _) in &rows {
        anyhow::ensure!(
            label == expect,
            "{}: labelled {label}, expected {expect}",
            path.display()
        );
    }
    Ok(rows.into_iter().map(|(_, v)| v).collect())
}

fn write_vectors(path: &Path, label: &str, vectors: &[Vec<f32>]) -> Result<()> {
    let mut body = String::new();
    for v in vectors {
        body.push_str(label);
        for x in v {
            body.push_str(&format!(" {x:.6}"));
        }
        body.push('\n');
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}

/// the frozen vision tower on its own, with no opinion about what it is looking
/// for.
///
/// separate from the classifier because the classifier is not the only caller:
/// labelling and evaluating the harvest need vectors for crops long before
/// there is a reference set to compare them against, and requiring one would
/// mean inventing references in order to build references.
pub struct Embedder {
    session: ort::session::Session,
    input: Vec<f32>,
}

impl Embedder {
    pub fn load(model: &Path) -> Result<Self> {
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading embedder {}", model.display()))?;
        Ok(Self {
            session,
            input: vec![0.0; (EMBED_INPUT * EMBED_INPUT * 3) as usize],
        })
    }

    /// embed one crop. the graph l2-normalises its output, so similarity is a
    /// plain dot product downstream.
    pub fn embed(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        fill_clip_input(&mut self.input, rgb, width, height);
        let s = EMBED_INPUT as usize;
        let tensor = ort::value::Tensor::from_array(([1, 3, s, s], self.input.clone()))
            .context("building embedder input")?;
        let out = self
            .session
            .run(ort::inputs!["image" => tensor])
            .context("running embedder")?;
        let (_, data) = out[0]
            .try_extract_tensor::<f32>()
            .context("reading embedding")?;
        Ok(data.to_vec())
    }
}

pub struct EmbeddingClassifier {
    embedder: Embedder,
    references: References,
    margins: Margins,
}

impl EmbeddingClassifier {
    pub fn load(model: &Path, references: References, margins: Margins) -> Result<Self> {
        anyhow::ensure!(
            references.is_usable(),
            "classifier needs a subject with both halves; got {}",
            if references.subjects.is_empty() {
                "nothing at all".to_string()
            } else {
                references
                    .names()
                    .iter()
                    .map(|n| {
                        format!(
                            "{n} ({} references, {} negatives)",
                            references.count(n),
                            references.negatives(n)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        Ok(Self {
            embedder: Embedder::load(model)?,
            references,
            margins,
        })
    }

    pub fn embed(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        self.embedder.embed(rgb, width, height)
    }
}

impl Classifier for EmbeddingClassifier {
    fn classify(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Judgement> {
        let e = self.embed(rgb, width, height)?;
        Ok(judge(&e, &self.references, &self.margins))
    }
}

/// how much nearer each subject's references must be, by name.
///
/// a map rather than one number because a distinctive subject can afford a bar
/// an ambiguous one cannot, and they are compared against the same negatives.
#[derive(Debug, Clone, Default)]
pub struct Margins {
    pub default: f32,
    pub per_subject: std::collections::BTreeMap<String, f32>,
}

impl Margins {
    pub fn uniform(margin: f32) -> Self {
        Self {
            default: margin,
            per_subject: Default::default(),
        }
    }

    pub fn get(&self, subject: &str) -> f32 {
        *self.per_subject.get(subject).unwrap_or(&self.default)
    }
}

/// compare an embedding against every subject and the shared negatives.
///
/// one-vs-rest: each subject scores a top-k mean, and clears its bar when it
/// beats the negative score by its own margin. where several clear it, the
/// winner is the one that cleared by the most -- comparing raw scores would let
/// a subject with a generous margin outrank one that was actually more certain.
///
/// the reported score is the winner's, or the best subject's when nothing
/// cleared, so a near miss still says what it nearly was.
pub fn judge(embedding: &[f32], refs: &References, margins: &Margins) -> Judgement {
    let mut best: Option<(&str, f32, f32, f32)> = None;
    for (name, subject) in &refs.subjects {
        // **a subject missing either half is skipped, not scored.** an empty
        // negative set makes `top_k_mean` return negative infinity, so the
        // excess would be infinite and that subject would win every crop on the
        // street. refusing to answer is the only safe reading of half a rule.
        if !subject.is_usable() {
            continue;
        }
        let score = top_k_mean(embedding, &subject.positives, VOTE_K);
        let other = top_k_mean(embedding, &subject.negatives, VOTE_K);
        let excess = score - other - margins.get(name);
        if best.is_none_or(|(_, _, _, b)| excess > b) {
            best = Some((name, score, other, excess));
        }
    }
    match best {
        Some((name, score, other, excess)) => Judgement {
            subject: (excess > 0.0).then(|| name.to_string()),
            score,
            other_score: other,
        },
        None => Judgement {
            subject: None,
            score: f32::NEG_INFINITY,
            other_score: f32::NEG_INFINITY,
        },
    }
}

/// mean similarity to the k nearest references in a set.
pub fn top_k_mean(embedding: &[f32], set: &[Vec<f32>], k: usize) -> f32 {
    mean_of_top(set.iter().map(|r| cosine(embedding, r)).collect(), k)
}

/// the vote `top_k_mean` casts, over similarities already computed.
///
/// one definition for every caller that needs to choose which references count,
/// so a ranking built on it cannot drift from the rule the classifier applies.
pub fn mean_of_top(mut sims: Vec<f32>, k: usize) -> f32 {
    if sims.is_empty() {
        return f32::NEG_INFINITY;
    }
    sims.sort_by(|a, b| b.total_cmp(a));
    let take = k.min(sims.len());
    sims[..take].iter().sum::<f32>() / take as f32
}

/// both sides are l2-normalised, so a dot product is the cosine. guarded
/// anyway: a mismatched embedding length would otherwise read past the end.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum()
}

/// resize into clip's square input and apply its channel normalisation.
///
/// clip stretches rather than letterboxes, which is what it was trained on;
/// padding a crop to square instead measurably weakens the embedding.
fn fill_clip_input(dst: &mut [f32], rgb: &[u8], width: u32, height: u32) {
    let s = EMBED_INPUT as usize;
    let px = s * s;
    for y in 0..s {
        // nearest neighbour: the crop is already soft, and bilinear here costs
        // more than it recovers.
        let sy = (y as u32 * height / EMBED_INPUT).min(height.saturating_sub(1));
        for x in 0..s {
            let sx = (x as u32 * width / EMBED_INPUT).min(width.saturating_sub(1));
            let src = ((sy * width + sx) * 3) as usize;
            for c in 0..3 {
                let v = rgb.get(src + c).copied().unwrap_or(0) as f32 / 255.0;
                dst[c * px + y * s + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    /// every subject gets the same negatives, which is what these tests mean by
    /// `other`: they are about margins and tie-breaking, not about who owns the
    /// negative set. `one_subject_is_a_negative_for_another` is the one that
    /// cares, and it builds its subjects by hand.
    fn refs_of(subjects: &[(&str, Vec<Vec<f32>>)], other: Vec<Vec<f32>>) -> References {
        References {
            subjects: subjects
                .iter()
                .map(|(n, v)| {
                    (
                        n.to_string(),
                        Subject {
                            positives: v.clone(),
                            negatives: other.clone(),
                            // a hand-built fixture has no measurement behind
                            // it, so it exercises the `DEFAULT_MARGIN` fallback.
                            margin: None,
                        },
                    )
                })
                .collect(),
        }
    }

    fn flat(margin: f32) -> Margins {
        Margins::uniform(margin)
    }

    #[test]
    fn a_crop_resembling_a_subject_is_named_as_that_subject() {
        let refs = refs_of(
            &[(
                "go4",
                vec![unit(vec![1.0, 0.0, 0.0]), unit(vec![0.9, 0.1, 0.0])],
            )],
            vec![unit(vec![0.0, 1.0, 0.0]), unit(vec![0.0, 0.9, 0.1])],
        );
        let j = judge(&unit(vec![0.95, 0.05, 0.0]), &refs, &flat(DEFAULT_MARGIN));
        assert!(j.is("go4"), "{j:?}");
        assert!(j.score > j.other_score);
    }

    #[test]
    fn a_crop_resembling_the_negatives_is_no_subject_at_all() {
        let refs = refs_of(
            &[("go4", vec![unit(vec![1.0, 0.0, 0.0])])],
            vec![unit(vec![0.0, 1.0, 0.0])],
        );
        let j = judge(&unit(vec![0.1, 0.99, 0.0]), &refs, &flat(DEFAULT_MARGIN));
        assert_eq!(j.subject, None, "{j:?}");
    }

    /// the margin exists because the real gap is about 0.02. a crop sitting
    /// between the two sets must not be claimed as a sighting.
    #[test]
    fn an_ambiguous_crop_names_nothing_rather_than_guessing() {
        let refs = refs_of(
            &[("go4", vec![unit(vec![1.0, 0.0, 0.0])])],
            vec![unit(vec![0.0, 1.0, 0.0])],
        );
        // exactly between the two references.
        let j = judge(&unit(vec![1.0, 1.0, 0.0]), &refs, &flat(DEFAULT_MARGIN));
        assert_eq!(j.subject, None, "ambiguity resolved as a sighting");
        assert!(j.margin() < 0.01, "margin should be tiny: {}", j.margin());
    }

    /// **the sign is the point of the number.** the margin a crop is reported with
    /// is what a notification quotes and what the verdict page shows, and a bar
    /// below zero makes room for verdicts on the far side of it -- which are the
    /// ones worth reading twice.
    #[test]
    fn a_verdict_nearer_the_street_than_its_class_reports_a_negative_margin() {
        let refs = refs_of(
            &[("go4", vec![unit(vec![1.0, 0.0, 0.0])])],
            vec![unit(vec![0.99, 0.01, 0.0])],
        );
        let j = judge(&unit(vec![0.9, 0.1, 0.0]), &refs, &flat(-0.5));
        assert!(j.is("go4"), "{j:?}");
        assert!(j.margin() < 0.0, "the sign was lost: {}", j.margin());
    }

    #[test]
    fn voting_uses_several_references_not_just_the_closest() {
        // one unlucky reference sits right next to the query, but the rest of
        // that set is far away; the negatives are consistently closer.
        let refs = refs_of(
            &[(
                "go4",
                vec![
                    unit(vec![1.0, 0.02, 0.0]),
                    unit(vec![0.0, 0.0, 1.0]),
                    unit(vec![0.0, 0.0, 1.0]),
                ],
            )],
            vec![
                unit(vec![1.0, 0.0, 0.0]),
                unit(vec![0.99, 0.01, 0.0]),
                unit(vec![0.98, 0.02, 0.0]),
            ],
        );
        let j = judge(&unit(vec![1.0, 0.0, 0.0]), &refs, &flat(DEFAULT_MARGIN));
        assert_eq!(j.subject, None, "single neighbour dominated: {j:?}");
    }

    /// several subjects against one shared negative set (r10.2).
    #[test]
    fn the_nearest_subject_wins_and_the_others_are_not_reported() {
        let refs = refs_of(
            &[
                ("go4", vec![unit(vec![1.0, 0.0, 0.0])]),
                ("sweeper", vec![unit(vec![0.0, 0.0, 1.0])]),
            ],
            vec![unit(vec![0.0, 1.0, 0.0])],
        );
        assert!(judge(&unit(vec![1.0, 0.05, 0.0]), &refs, &flat(DEFAULT_MARGIN)).is("go4"));
        assert!(judge(&unit(vec![0.0, 0.05, 1.0]), &refs, &flat(DEFAULT_MARGIN)).is("sweeper"));
    }

    /// where two subjects both clear their bar, the winner is the one that
    /// cleared by the most. comparing raw scores would let a subject configured
    /// with a generous margin outrank one that was actually more certain.
    #[test]
    fn a_generous_margin_does_not_win_a_tie_it_did_not_earn() {
        let refs = refs_of(
            &[
                ("lenient", vec![unit(vec![1.0, 0.0, 0.02])]),
                ("strict", vec![unit(vec![1.0, 0.0, 0.0])]),
            ],
            vec![unit(vec![0.0, 1.0, 0.0])],
        );
        let mut margins = Margins::uniform(0.0);
        margins.per_subject.insert("lenient".into(), -0.5);
        margins.per_subject.insert("strict".into(), 0.0);
        // `lenient` scores slightly lower but has a far easier bar, so on raw
        // score alone it would still lose -- and on excess it wins, which is
        // the whole point of the margin being per subject.
        let j = judge(&unit(vec![1.0, 0.0, 0.0]), &refs, &margins);
        assert!(
            j.is("lenient"),
            "excess over its own bar should decide: {j:?}"
        );
    }

    #[test]
    fn a_subject_whose_margin_it_cannot_clear_is_not_reported() {
        let refs = refs_of(
            &[("go4", vec![unit(vec![1.0, 0.0, 0.0])])],
            vec![unit(vec![0.0, 1.0, 0.0])],
        );
        let close = unit(vec![1.0, 0.9, 0.0]);
        assert!(judge(&close, &refs, &flat(0.0)).is("go4"));
        assert_eq!(judge(&close, &refs, &flat(0.5)).subject, None);
    }

    /// `judge` is the whole decision, and a change to it would move every
    /// verdict and every training report without failing anything else. so it
    /// is pinned to hand-built vectors whose verdicts can be checked by eye. see
    /// `tests/e2e/fixtures/judge/README.md`.
    #[test]
    fn the_fixture_is_judged_as_written() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/fixtures/judge");
        let refs = References::load(&dir).unwrap();
        // in the fixture the label on a query is the verdict it should receive,
        // which is why it is read as labelled rows rather than as references.
        let queries = read_labelled(&dir.join("queries.txt")).unwrap();
        assert_eq!(queries.len(), 5, "fixture changed shape");
        for (expected, q) in &queries {
            let j = judge(q, &refs, &flat(DEFAULT_MARGIN));
            if expected == crate::label::NEGATIVE {
                assert_eq!(j.subject, None, "{q:?} judged {j:?}");
            } else {
                assert!(j.is(expected), "{q:?} was not called {expected}: {j:?}");
            }
        }
    }

    #[test]
    fn references_round_trip_through_the_text_format() {
        let d = std::env::temp_dir().join(format!("mm-refs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("go4")).unwrap();
        std::fs::write(
            d.join("go4").join(POSITIVES),
            "go4 1.0 0.0 0.0\ngo4 0.9 0.1 0.0\n",
        )
        .unwrap();
        std::fs::write(d.join("go4").join(NEGATIVES), "other 0.0 1.0 0.0\n").unwrap();

        let r = References::load(&d).unwrap();
        assert_eq!(r.count("go4"), 2);
        assert_eq!(r.negatives("go4"), 1);
        assert_eq!(r.subjects["go4"].positives[0], vec![1.0, 0.0, 0.0]);
        assert!(r.is_usable());
        std::fs::remove_dir_all(&d).ok();
    }

    /// a `negatives.txt` full of go-4s would train the exact inversion of the
    /// rule and never say so, so the label inside each file is checked against
    /// the directory that holds it rather than ignored as redundant.
    #[test]
    fn a_file_labelled_for_the_wrong_half_is_refused() {
        let d = std::env::temp_dir().join(format!("mm-mislabel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("go4")).unwrap();
        std::fs::write(d.join("go4").join(POSITIVES), "go4 1.0 0.0\n").unwrap();
        // the negatives claim to be go-4s.
        std::fs::write(d.join("go4").join(NEGATIVES), "go4 0.0 1.0\n").unwrap();

        let err = References::load(&d).unwrap_err().to_string();
        assert!(err.contains("expected other"), "{err}");
        std::fs::remove_dir_all(&d).ok();
    }

    /// the margin a subject was measured at, read from `trained.toml` beside
    /// its vectors.
    ///
    /// **the file is the whole point of the feature**: a margin copied into a
    /// config by hand on another machine is an operating point nobody scored,
    /// and nothing would say the two had drifted. absent is not an error -- a
    /// set trained before the margin travelled with it still loads, and falls
    /// back rather than reading as zero, which would fire on everything.
    #[test]
    fn a_subject_carries_the_margin_it_was_measured_at() {
        let d = std::env::temp_dir().join(format!("mm-margin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        for name in ["go4", "sweeper"] {
            std::fs::create_dir_all(d.join(name)).unwrap();
            std::fs::write(d.join(name).join(POSITIVES), format!("{name} 1.0 0.0\n")).unwrap();
            std::fs::write(d.join(name).join(NEGATIVES), "other 0.0 1.0\n").unwrap();
        }
        Trained {
            margin: Some(-0.005),
            passages: 3,
            ..Default::default()
        }
        .save(&d.join("go4"))
        .unwrap();

        let refs = References::load(&d).unwrap();
        assert_eq!(refs.margin("go4"), Some(-0.005));
        assert_eq!(
            refs.margin("sweeper"),
            None,
            "a subject with no trained.toml must fall back, not read as zero"
        );
    }

    /// the test above promised a round trip and only ever loaded. with a writer
    /// it can mean what it says: what `save` emits is what `load` reads back,
    /// including a second subject it was not asked about.
    #[test]
    fn a_saved_reference_file_reloads_as_itself() {
        let d = std::env::temp_dir().join(format!("mm-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let mut refs = References::default();
        refs.subjects.insert(
            "go4".into(),
            Subject {
                positives: vec![vec![1.0, 0.0, 0.5], vec![0.9, 0.1, 0.0]],
                negatives: vec![vec![0.0, 1.0, 0.0]],
                margin: Some(-0.01),
            },
        );
        refs.subjects.insert(
            "sweeper".into(),
            Subject {
                positives: vec![vec![0.25, 0.75, 0.0]],
                negatives: vec![vec![0.0, 0.0, 1.0]],
                margin: None,
            },
        );
        refs.save(&d, "go4").unwrap();
        refs.save(&d, "sweeper").unwrap();

        let back = References::load(&d).unwrap();
        assert_eq!(back.names(), vec!["go4", "sweeper"]);
        assert_eq!(back.count("go4"), 2);
        // **`save` carries vectors, not the measurement.** the margin is written
        // by `train`, which has the report that chose it; `save` has only the
        // vectors and must not invent a margin for them. stated here so the
        // round trip's promise is exact rather than a trap for the next reader.
        assert_eq!(
            back.margin("go4"),
            None,
            "save wrote a margin it had no measurement for"
        );
        assert_eq!(back.negatives("go4"), 1);
        assert_eq!(back.subjects["go4"].positives[0], vec![1.0, 0.0, 0.5]);
        assert_eq!(back.subjects["sweeper"].positives[0], vec![0.25, 0.75, 0.0]);
        assert!(back.is_usable());

        // **writing one subject leaves the others alone**, which is the whole
        // reason each has its own directory: training go-4 used to rewrite a
        // shared file holding every subject's rows.
        let before = std::fs::read_to_string(d.join("sweeper").join(POSITIVES)).unwrap();
        refs.save(&d, "go4").unwrap();
        let after = std::fs::read_to_string(d.join("sweeper").join(POSITIVES)).unwrap();
        assert_eq!(before, after, "training go4 rewrote sweeper");
        std::fs::remove_dir_all(&d).ok();
    }

    /// a **directory** is how a new subject arrives, so an unfamiliar one names
    /// a subject rather than being an error. it used to be a label in a shared
    /// file, which was right when there were exactly two of them and wrong the
    /// moment a subject had to be shipped or erased on its own.
    #[test]
    fn an_unfamiliar_label_names_a_subject() {
        let d = std::env::temp_dir().join(format!("mm-multi-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        for (name, positives) in [("go4", "go4 1.0 0.0\n"), ("sweeper", "sweeper 0.5 0.5\n")] {
            std::fs::create_dir_all(d.join(name)).unwrap();
            std::fs::write(d.join(name).join(POSITIVES), positives).unwrap();
            std::fs::write(d.join(name).join(NEGATIVES), "other 0.0 1.0\n").unwrap();
        }

        let r = References::load(&d).unwrap();
        assert_eq!(r.names(), vec!["go4", "sweeper"]);
        assert_eq!(r.count("sweeper"), 1);
        assert!(r.is_usable());
        std::fs::remove_dir_all(&d).ok();
    }

    /// **a subject is pushed back on by the other subjects' crops.**
    ///
    /// one `other_score` was computed for every subject, so the negative set
    /// could not contain another subject: a waymo sitting in go-4's negatives
    /// would also drag the waymo's own score down. `split_negatives` duly
    /// discarded crops labelled as any subject, which are the most informative
    /// negatives there are -- the street's sedans are not near the boundary and
    /// the other subjects are.
    ///
    /// here the query *is* a go-4. it must not be called a waymo, and what makes
    /// that true is a go-4 sitting in waymo's negatives -- not the go-4
    /// references, which a one-vs-rest rule never consults when scoring waymo.
    #[test]
    fn one_subject_is_a_negative_for_another() {
        let go4 = unit(vec![1.0, 0.0, 0.0]);
        let waymo = unit(vec![0.9, 0.3, 0.0]); // deliberately close to a go-4
        let street = unit(vec![0.0, 0.0, 1.0]);

        let refs = References {
            subjects: [
                (
                    "go4".to_string(),
                    Subject {
                        positives: vec![go4.clone()],
                        negatives: vec![waymo.clone(), street.clone()],
                        margin: None,
                    },
                ),
                (
                    "waymo".to_string(),
                    Subject {
                        positives: vec![waymo.clone()],
                        // the go-4 is what stops a go-4 being called a waymo.
                        negatives: vec![go4.clone(), street.clone()],
                        margin: None,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };
        assert!(refs.is_usable());

        let j = judge(&go4, &refs, &flat(DEFAULT_MARGIN));
        assert!(j.is("go4"), "a go-4 was not called one: {j:?}");

        // the control. take the go-4 back out of waymo's negatives and waymo
        // wins it: scoring waymo then consults only the street, so nothing it
        // compares against is anywhere near a go-4. asserting the failure
        // appears is what stops the half above passing for some other reason.
        let mut without = References {
            subjects: refs.subjects.clone(),
        };
        without
            .subjects
            .get_mut("waymo")
            .unwrap()
            .negatives
            .retain(|v| v != &go4);
        let j = judge(&go4, &without, &flat(DEFAULT_MARGIN));
        assert!(
            j.is("waymo"),
            "the go-4 in waymo's negatives is supposed to be what prevents this, \
             so removing it should have let waymo win: {j:?}"
        );
    }

    #[test]
    fn references_need_both_sides_to_mean_anything() {
        assert!(!References::default().is_usable());
        assert!(
            !refs_of(&[("go4", vec![vec![1.0]])], vec![]).is_usable(),
            "subjects alone cannot answer 'compared to what'"
        );
        assert!(
            !refs_of(&[], vec![vec![1.0]]).is_usable(),
            "negatives alone name nothing"
        );
    }

    #[test]
    fn clip_normalisation_centres_the_input() {
        // a mid-grey crop should land near zero once clip's statistics are
        // applied. getting this wrong shifts every embedding subtly.
        let mut dst = vec![0.0; (EMBED_INPUT * EMBED_INPUT * 3) as usize];
        let grey = vec![124u8; 4 * 4 * 3]; // close to clip's channel means
        fill_clip_input(&mut dst, &grey, 4, 4);
        let mean: f32 = dst.iter().sum::<f32>() / dst.len() as f32;
        assert!(mean.abs() < 0.35, "normalisation looks wrong: mean {mean}");
    }
}
