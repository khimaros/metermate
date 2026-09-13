//! does few-shot embedding matching actually work, on this camera's own crops.
//!
//! the stage-2 head is a frozen embedding and a nearest-neighbour vote, and
//! `classify`'s header records why that is barely working: go-4s resemble each
//! other at 0.716 and resemble cars at 0.694, while cars resemble each other at
//! 0.834. this measures the same three numbers for any subject whose crops come
//! from our own harvest, which separates "the model cannot do this" from "the
//! references are from the wrong world".
//!
//! what it refuses to do:
//!
//! - **count crops as examples.** reference and eval sets split by passage, and
//!   similarity between examples is only measured *across* passages. two crops
//!   0.4s apart score near 1.0 and report nothing but that a picture resembles
//!   itself.
//! - **measure a false positive rate on the ranked review pool**, which is
//!   ordered by the very similarity being tested.
//! - **report numbers whose preconditions failed.** when the reference and eval
//!   sets touch, or the reference set is smaller than `VOTE_K`, it says so and
//!   marks the result void.

use super::{Labels, NEGATIVE, PASSAGE_GAP_MS, Vectors, Via};
use crate::classify::{VOTE_K, cosine, mean_of_top, top_k_mean};

/// what the classifier module records for the web go-4 references, so every
/// number here has something to be read against.
pub const GO4_SELF: f32 = 0.716;
pub const GO4_OTHER: f32 = 0.694;
pub const CAR_SELF: f32 = 0.834;

/// how many crops of each reference passage are used. one is too few to survive
/// a bad frame; the whole passage would make `VOTE_K` vote three times for one
/// object.
pub const CROPS_PER_PASSAGE: usize = 2;

/// how many crops of ordinary traffic sample the street among the negatives.
///
/// a sample of the street only: what the classifier fired on and another
/// subject's crops are always negatives however many there are, see
/// `split_negatives`. `References::is_usable` is false without negatives and
/// the negative score is half the decision, so the sample is never skipped.
pub const NEGATIVE_REFERENCES: usize = 30;

/// below this many held-out passages a recall carries an interval too wide to
/// separate "works" from "does not". 25 puts 95% confidence near +/-12pp at a
/// recall of 0.9.
pub const MIN_EVAL_PASSAGES: usize = 25;

/// and below this many sampled negatives a 1% false positive rate cannot be
/// seen at all, let alone bounded.
pub const MIN_FPR_SAMPLE: usize = 500;

/// how many crops the three cosine means are estimated from. the false positive
/// rate wants every crop in the harvest; a distribution does not, and the whole
/// pool makes millions of pairs to average for no extra precision.
pub const SEPARATION_SAMPLE: usize = 400;

/// crops nearest the references that a person should look at whatever they are
/// labelled. contamination concentrates here, so this is the cheap way to check
/// an unlabelled negative pool: review 40 crops, not 3000.
pub const SCREEN_TOP: usize = 40;

/// the margins to sweep.
///
/// **it runs negative on purpose.** the rule is `subject > other + margin` and
/// nothing requires the margin to be positive: a negative one says "call it the
/// subject even when it sits slightly nearer the negatives", which is where the
/// operating point turns out to live. a sweep starting at zero reported "no
/// margin keeps 80% of passages" and meant only that it had not looked -- the
/// same truncation as reading `must_have_moved` off the crops that cleared it.
pub const MARGIN_SWEEP: [f32; 16] = [
    -0.05, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
    0.15, 0.2,
];

/// mean, median and interquartile range of a set of similarities.
#[derive(Debug, Clone, Copy, Default)]
pub struct Summary {
    pub mean: f32,
    pub median: f32,
    pub lo: f32,
    pub hi: f32,
    pub n: usize,
}

impl Summary {
    pub fn of(values: &mut [f32]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f32::total_cmp);
        let at = |q: f32| values[((values.len() - 1) as f32 * q).round() as usize];
        Some(Self {
            mean: values.iter().sum::<f32>() / values.len() as f32,
            median: at(0.5),
            lo: at(0.25),
            hi: at(0.75),
            n: values.len(),
        })
    }
}

/// one row of the margin sweep.
#[derive(Debug, Clone, Copy, Default)]
pub struct Row {
    pub margin: f32,
    pub passages: usize,
    pub passage_recall: f32,
    pub crop_recall: f32,
    pub fpr: f32,
    pub per_hour: f32,
    pub alerts_per_hour: f32,
}

#[derive(Debug, Clone)]
pub struct PassageRow {
    pub first: String,
    pub crops: usize,
    pub span_s: f32,
    pub reference: bool,
    pub best: f32,
    /// long edge of the largest crop, in main-stream pixels: how big the thing
    /// actually was. stage 1 has a measured floor in these terms already
    /// (reliable at 64px, not at 48px) and stage 2 turns out to have its own,
    /// much higher one -- which a single recall number averages away.
    pub px: u32,
}

/// apparent size above which a passage is "near" for the split below.
///
/// not tuned. it is the midpoint of the gap this harvest actually shows between
/// the passages that work and the ones that do not, and it is reported as an
/// observation rather than used as a threshold anywhere.
pub const NEAR_PX: u32 = 220;

/// the share of passages a shipped margin keeps unless a person says otherwise.
///
/// a bracket rather than a rate, because recall is the thing the labels measure
/// directly: the false positive rate is estimated off the random pool and moves
/// with how much of the street has been sampled, while "did it catch the
/// passage" is counted on passages that were actually looked at. `--max-fpr`
/// names the other axis for an operator who knows what nuisance rate they will
/// tolerate.
pub const OPERATING_RECALL: f32 = 0.8;

/// everything a measurement produced. the http layer renders this; nothing
/// renders anything it computed itself.
#[derive(Debug, Default)]
pub struct Report {
    pub subject: String,
    pub error: Option<String>,
    pub n_crops: usize,
    pub n_labelled: usize,
    pub n_negatives: usize,
    pub n_references: usize,
    pub n_negative_references: usize,
    pub hand_negatives: bool,
    pub per_hour: f32,
    pub self_self: Option<Summary>,
    pub other_other: Option<Summary>,
    pub self_other: Option<Summary>,
    pub matrix: Vec<Vec<f32>>,
    pub passages: Vec<PassageRow>,
    pub sweep: Vec<Row>,
    /// the same curve over the passages the classifier can actually see, and
    /// over negatives of the same apparent size.
    ///
    /// **both sides have to be restricted or the comparison is rigged.** a
    /// far-away civilian car cannot be a false positive for an alert nobody
    /// would trust on a far-away vehicle in the first place, so counting it
    /// charges the near operating point for the far one's mistakes.
    pub sweep_near: Vec<Row>,
    /// the confirmation both curves were counted under, so a report says which
    /// rule its passages and alerts describe.
    pub confirm_m: u32,
    pub confirm_n: u32,
    /// the operating point asked for, carried so the row the terminal prints,
    /// the row the page highlights and the margin beside the vectors are one
    /// choice made once.
    pub min_passage_recall: Option<f32>,
    pub max_fpr: Option<f32>,
    pub leave_one_out: bool,
    pub fatal: Vec<String>,
    pub caveats: Vec<String>,
    pub screen: Vec<String>,
    /// 95% interval on the passage recall at the shipped margin. a handful of
    /// passages must not be allowed to look certain.
    pub recall_interval: (f32, f32),
    pub probe: Option<Probe>,
    /// passages least like the other examples, worst first: `(first crop, mean
    /// similarity to the others, excluded from the references)`.
    pub odd: Vec<(String, f32, bool)>,
    /// what the classifier would fire on right now, strongest first, with the
    /// margin each clears by: `(crop, margin)`.
    ///
    /// the sweep says what share of the street fires; this says *which crops*.
    /// labelling them is the only way precision stops being an assumption --
    /// the pool is unlabelled by construction, so a real example hiding in it
    /// is currently counted as a false positive.
    pub alerts: Vec<(String, f32)>,
    /// the margin `alerts` was taken at.
    pub alert_margin: f32,
    /// every crop of the excluded passages, for a second look. naming a passage
    /// is not enough to act on: the decision that needs revisiting is per crop,
    /// and it is made by looking at the picture.
    pub odd_crops: Vec<String>,
    /// the crops chosen as references, and the negatives they were scored
    /// against.
    ///
    /// recorded rather than re-derived, because the file that ships has to hold
    /// exactly what was measured. a second selection written beside this one
    /// would agree until the day it did not, and nothing would say which of the
    /// two the deployment was running.
    pub reference_crops: Vec<String>,
    pub negative_reference_crops: Vec<String>,
}

/// most alerts to carry back. a page of a few hundred is already more than
/// anyone will label in a sitting, and the tail is the least informative part.
pub const ALERTS_SHOWN: usize = 300;

/// the day-one head against a trained one, on the same frozen vectors.
///
/// if the linear probe separates where nearest-neighbour does not, the backbone
/// is fine and the head is the limit. if neither separates, no head will rescue
/// the backbone.
#[derive(Debug, Clone, Default)]
pub struct Probe {
    pub nn_crop: f32,
    pub nn_passage: f32,
    pub probe_crop: f32,
    pub probe_passage: f32,
    /// what share of the street fires at thresholds keeping most of the crops.
    pub thresholds: Vec<(f32, f32)>,
}

impl Report {
    /// the most selective margin that still keeps most of the passages.
    ///
    /// computed here rather than in the page, so the row the browser
    /// highlights, the row the terminal prints and the margin the alerts are
    /// taken at cannot disagree about which one it is.
    pub fn operating_point(&self) -> Option<&Row> {
        let floor = self.min_passage_recall.unwrap_or(OPERATING_RECALL);
        let rows = self
            .decision_curve()
            .iter()
            .filter(|r| r.passage_recall >= floor)
            .filter(|r| self.max_fpr.is_none_or(|cap| r.fpr <= cap));
        match self.max_fpr {
            // **under a rate ceiling, recall is what is left to buy.** an
            // operator who has said what nuisance rate they tolerate is not
            // also asking for the quietest margin under it -- that would spend
            // the allowance they just set on nothing.
            // the most selective margin among equals, spelled out rather than
            // left to which end of the sweep `max_by` happens to keep: rows
            // that tie on both axes are the same measurement, and the tighter
            // margin is the one that has room to be wrong.
            Some(_) => rows.max_by(|a, b| {
                a.passage_recall
                    .total_cmp(&b.passage_recall)
                    .then(b.fpr.total_cmp(&a.fpr))
                    .then(a.margin.total_cmp(&b.margin))
            }),
            None => rows.min_by(|a, b| {
                a.fpr
                    .total_cmp(&b.fpr)
                    .then(b.passage_recall.total_cmp(&a.passage_recall))
            }),
        }
    }

    /// whether a person named the operating point rather than taking the
    /// default bracket. an unmet constraint they set is an error; the default
    /// going unmet is a measurement that says stage 2 is not ready.
    pub fn constrained(&self) -> bool {
        self.max_fpr.is_some() || self.min_passage_recall.is_some()
    }

    /// `<--` against the margin that will ship.
    ///
    /// matched on the margin rather than on which table this is, because the
    /// point is chosen off the near curve when there is one while the table in
    /// front of you may be the whole of it -- and the margin is the thing both
    /// tables share and the only part of the row that ships.
    fn mark(&self, row: &Row) -> &'static str {
        match self.operating_point() {
            Some(point) if point.margin == row.margin => "  <--",
            _ => "",
        }
    }

    /// which margin the mark is against, and which curve chose it.
    fn print_point(&self) {
        let Some(point) = self.operating_point() else {
            println!(
                "  no margin keeps {:.0}% of the passages, so nothing here is worth shipping yet",
                self.min_passage_recall.unwrap_or(OPERATING_RECALL) * 100.0
            );
            return;
        };
        let asked = match (self.max_fpr, self.min_passage_recall) {
            (Some(f), _) => format!("the most recall under {:.2}% of the street", f * 100.0),
            (None, r) => format!(
                "the cleanest margin keeping {:.0}% of the passages",
                r.unwrap_or(OPERATING_RECALL) * 100.0
            ),
        };
        let curve = match self.sweep_near.is_empty() {
            true => "",
            false => ", off the near curve",
        };
        println!(
            "  <-- marks the margin that would ship, {:+.3}: {asked}{curve}",
            point.margin
        );
    }

    /// the best each axis of the curve offers, for saying what was available
    /// when nothing met the constraint.
    pub fn best_available(&self) -> (f32, f32) {
        let rows = self.decision_curve();
        let fpr = rows.iter().map(|r| r.fpr).fold(f32::INFINITY, f32::min);
        let recall = rows.iter().map(|r| r.passage_recall).fold(0.0f32, f32::max);
        (fpr, recall)
    }

    /// the curve an operating point is chosen from: the near half when it could
    /// be measured, because that is the alert anybody would trust.
    pub fn decision_curve(&self) -> &[Row] {
        if self.sweep_near.is_empty() {
            &self.sweep
        } else {
            &self.sweep_near
        }
    }

    /// the row of that curve at `margin`, for a margin a person picked off it.
    pub fn row_at(&self, margin: f32) -> Option<&Row> {
        self.decision_curve()
            .iter()
            .find(|r| (r.margin - margin).abs() < 1e-4)
    }

    pub fn gap(&self) -> f32 {
        match (self.self_self, self.self_other) {
            (Some(a), Some(b)) => a.mean - b.mean,
            _ => 0.0,
        }
    }

    /// print a measurement. the browser renders the same struct, differently;
    /// neither computes anything of its own.
    pub fn print(&self) {
        if let Some(e) = &self.error {
            println!("{e}");
            return;
        }
        let refs = self.passages.iter().filter(|p| p.reference).count();
        println!(
            "{} crops in the harvest, {} labelled {}",
            self.n_crops, self.n_labelled, self.subject
        );
        println!(
            "\n{} passages, {} crops ({refs} reference, {} held out):",
            self.passages.len(),
            self.passages.iter().map(|p| p.crops).sum::<usize>(),
            self.passages.len() - refs
        );
        for (i, p) in self.passages.iter().enumerate() {
            println!(
                "  {:>2}  {}  {:>2} crops over {:>4.1}s  {:>4}px  best {:+.4}   {}",
                i + 1,
                if p.reference {
                    "reference"
                } else {
                    "held out "
                },
                p.crops,
                p.span_s,
                p.px,
                p.best,
                p.first
            );
        }
        self.print_separation();
        self.print_sweep();
        for note in &self.caveats {
            println!("\n  note: {note}");
        }
    }

    fn print_separation(&self) {
        let show = |s: Option<Summary>| match s {
            Some(s) => format!(
                "{:.3}   median {:.3}  iqr {:.3}-{:.3}  n={}",
                s.mean, s.median, s.lo, s.hi, s.n
            ),
            None => "n/a".to_string(),
        };
        println!(
            "\n=== separation: does the embedding tell a {} apart ===",
            self.subject
        );
        println!(
            "  {:<8} <-> {:<8} {}",
            self.subject,
            self.subject,
            show(self.self_self)
        );
        println!(
            "  {:<8} <-> {:<8} {}",
            "other",
            "other",
            show(self.other_other)
        );
        println!(
            "  {:<8} <-> {:<8} {}",
            self.subject,
            "other",
            show(self.self_other)
        );
        println!(
            "\n  gap                   {:+.3}     (go-4, web references: {:+.3})",
            self.gap(),
            GO4_SELF - GO4_OTHER
        );
        println!("  the same three for go-4: {GO4_SELF:.3}, {CAR_SELF:.3}, {GO4_OTHER:.3}");
        if !self.matrix.is_empty() {
            println!("\n  every pair of {} passages, mean cosine:", self.subject);
            for (i, row) in self.matrix.iter().enumerate() {
                let cells: Vec<String> = row
                    .iter()
                    .enumerate()
                    .map(|(j, v)| {
                        if i == j {
                            "     -".into()
                        } else {
                            format!("{v:>6.3}")
                        }
                    })
                    .collect();
                println!("   {:>2} {}", i + 1, cells.join(" "));
            }
        }
    }

    fn print_sweep(&self) {
        if !self.fatal.is_empty() {
            for f in &self.fatal {
                println!("\n  !! {f}");
            }
            println!("  the eval numbers are void. the separation figures above still stand.");
            return;
        }
        println!(
            "\n=== eval: {} passages, {} negatives ===",
            self.passages.len(),
            self.n_negatives
        );
        if self.leave_one_out {
            println!("  leaving one passage out at a time: too few to hold a fixed set back");
        }
        println!(
            "  the harvest runs at {:.0} crops/hour, which is what an fpr costs",
            self.per_hour
        );
        println!(
            "  passages: the share of held-out passages an alert fires on, counting {} of {}\n  \
             consecutive crops agreeing as [track] confirm_m and confirm_n do. crops: the\n  \
             share of their individual crops above the margin, always the lower of the two\n  \
             because a passage fires on its best few frames. fpr is per crop, over the street",
            self.confirm_m, self.confirm_n
        );
        println!("\n  margin   passages        crops    fpr       crops/hr  alerts/hr");
        for r in &self.sweep {
            println!(
                "  {:<+7.3} {:>2}/{:<2} = {:>3.0}%     {:>3.0}%   {:>6.2}%   {:>7.1}   {:>7.1}{}",
                r.margin,
                r.passages,
                self.passages.len(),
                r.passage_recall * 100.0,
                r.crop_recall * 100.0,
                r.fpr * 100.0,
                r.per_hour,
                r.alerts_per_hour,
                self.mark(r)
            );
        }
        self.print_point();
        println!(
            "  alerts/hr clusters the firing crops in time: one car disliked for three\n  \
             consecutive crops is one alert, not three"
        );
        if !self.screen.is_empty() {
            println!(
                "\n  nearest unlabelled crops, worth a look ({} of the top {SCREEN_TOP}):",
                self.screen.len()
            );
            for name in self.screen.iter().take(8) {
                println!("    {name}");
            }
        }
        self.print_by_size();
        self.print_odd();
        self.print_probe();
    }

    /// recall split by how big the thing actually got.
    ///
    /// **one recall number over a mixed set is the average of two regimes.**
    /// stage 1 already has a measured floor in apparent pixels; stage 2 has its
    /// own and it is much higher, because a crop 80px on its long edge is
    /// stretched to clip's 224 square and arrives as mush. reporting the two
    /// together says "half of them work" when the truth is nearer "the near
    /// ones work and the far ones do not", which is a different project.
    fn print_by_size(&self) {
        if self.passages.len() < 4 {
            return;
        }
        let (near, far): (Vec<&PassageRow>, Vec<&PassageRow>) =
            self.passages.iter().partition(|p| p.px >= NEAR_PX);
        if near.is_empty() || far.is_empty() {
            return;
        }
        let fires = |set: &[&PassageRow]| set.iter().filter(|p| p.best > 0.0).count();
        println!("\n=== by apparent size, at the margin that decides nothing else ===");
        for (label, set) in [
            (format!("{NEAR_PX}px and over"), &near),
            (format!("under {NEAR_PX}px"), &far),
        ] {
            let hit = fires(set);
            let worst = set.iter().map(|p| p.best).fold(f32::INFINITY, f32::min);
            let best = set.iter().map(|p| p.best).fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  {label:<16} {hit}/{} passages score above the negatives   \
                 ({worst:+.4} to {best:+.4})",
                set.len()
            );
        }
        println!(
            "  a crop stretched to clip's {}px square from well under it arrives as mush.\n  \
             the two halves are different questions and a single recall averages them.",
            crate::classify::EMBED_INPUT
        );
        if self.sweep_near.is_empty() {
            return;
        }
        println!(
            "\n  the curve over the near half only, negatives included -- the operating\n  \
             point for an alert nobody would trust on a far vehicle anyway:"
        );
        println!("\n  margin   passages        crops    fpr       crops/hr  alerts/hr");
        for r in &self.sweep_near {
            if r.passages == 0 && r.margin > 0.0 {
                continue;
            }
            println!(
                "  {:<+7.3} {:>2}/{:<2} = {:>3.0}%     {:>3.0}%   {:>6.2}%   {:>7.1}   {:>7.1}{}",
                r.margin,
                r.passages,
                near.len(),
                r.passage_recall * 100.0,
                r.crop_recall * 100.0,
                r.fpr * 100.0,
                r.per_hour,
                r.alerts_per_hour,
                self.mark(r)
            );
        }
    }

    fn print_odd(&self) {
        let flagged = self.odd.iter().filter(|(_, _, f)| *f).count();
        if flagged == 0 {
            return;
        }
        println!("\n  least like the other examples, worst first:");
        for (name, mean, excluded) in self.odd.iter().take(5) {
            println!(
                "    {mean:.3}  {name}{}",
                if *excluded {
                    "   [kept out of the references]"
                } else {
                    ""
                }
            );
        }
        println!(
            "  either a mislabel, or a crop where the subject is there but not what the\n  \
             crop is *of* -- another vehicle filling the frame with a sliver of the\n  \
             subject at the edge. the label can be right and the vector still wrong."
        );
    }

    fn print_probe(&self) {
        let Some(p) = &self.probe else { return };
        println!("\n=== is it the backbone or the head ===");
        println!("                                     per crop   per passage");
        println!(
            "  nearest neighbour (what ships)      {:>5.1}%      {:>5.1}%",
            p.nn_crop * 100.0,
            p.nn_passage * 100.0
        );
        println!(
            "  linear probe on the same vectors    {:>5.1}%      {:>5.1}%",
            p.probe_crop * 100.0,
            p.probe_passage * 100.0
        );
        for (keeps, fires) in &p.thresholds {
            println!(
                "    keeping {:>3.0}% of crops   {:>5.2}% of the street fires",
                keeps * 100.0,
                fires * 100.0
            );
        }
        // the per-passage column is the honest one and it is also the one
        // computed over almost no points. saying so here rather than in a
        // footnote, because this is the line somebody will quote.
        if self.passages.len() < MIN_EVAL_PASSAGES {
            println!(
                "  **on {} passages.** the per-passage column is an auc over {} points, so\n  \
                 which single passage happens to be hardest moves it more than the choice of\n  \
                 head does. suggestive, not settled.",
                self.passages.len(),
                self.passages.len()
            );
        }
    }
}

/// 95% interval for a proportion, by wilson's method.
///
/// not the textbook normal interval: at 5/5 that one is zero wide, and a
/// handful of passages must not be allowed to look certain.
pub fn wilson(hits: usize, n: usize) -> (f32, f32) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let (z, p, n_f) = (1.96f32, hits as f32 / n as f32, n as f32);
    let d = 1.0 + z * z / n_f;
    let centre = (p + z * z / (2.0 * n_f)) / d;
    let half = z * ((p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt()) / d;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// how often an example outscores a random negative. 0.5 is chance.
///
/// threshold-free on purpose: it compares two heads without giving either an
/// operating point, which at these sample sizes would be fitted to noise.
pub fn auc(positive: &[f32], negative: &[f32]) -> f32 {
    if positive.is_empty() || negative.is_empty() {
        return f32::NAN;
    }
    let mut wins = 0.0f64;
    for p in positive {
        for n in negative {
            wins += match p.partial_cmp(n) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Equal) => 0.5,
                _ => 0.0,
            };
        }
    }
    (wins / (positive.len() as f64 * negative.len() as f64)) as f32
}

/// every crop-pair cosine between two *different* passages.
///
/// the within-passage pairs it drops are the ones that would say 0.99 and mean
/// nothing, and they outnumber the useful pairs badly: eight crops of one
/// passage make 28 useless pairs on their own.
pub fn cross_passage(groups: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, a) in groups.iter().enumerate() {
        for b in &groups[i + 1..] {
            for x in a {
                for y in b {
                    out.push(cosine(x, y));
                }
            }
        }
    }
    out
}

/// pick `n` passages for spread, by farthest-point sampling.
///
/// starting from the passage least like the rest and repeatedly taking whatever
/// is furthest from everything chosen. picking at random tends to draw several
/// of the same aspect, and aspect is the axis that matters: a reference set that
/// is all side-on views has to recognise a rear view it has never seen.
pub fn choose_references(centres: &[Vec<f32>], skip: &[bool], n: usize) -> Vec<usize> {
    if n >= centres.len() {
        return (0..centres.len()).collect();
    }
    let sim: Vec<Vec<f32>> = centres
        .iter()
        .map(|a| centres.iter().map(|b| cosine(a, b)).collect())
        .collect();
    // whatever looks more like the street than like the class is dropped before
    // spread is considered at all. farthest-point maximises unlikeness, so a
    // mislabel -- the thing least like the class by construction -- is exactly
    // what it reaches for, and the reference set ends up built from the cases
    // the class is worst represented by.
    let pool: Vec<usize> = (0..centres.len())
        .filter(|i| !skip.get(*i).copied().unwrap_or(false))
        .collect();
    let pool = if pool.len() >= n.min(VOTE_K) {
        pool
    } else {
        (0..centres.len()).collect()
    };

    let mean = |i: usize| pool.iter().map(|j| sim[i][*j]).sum::<f32>() / pool.len() as f32;
    let mut chosen = vec![
        *pool
            .iter()
            .min_by(|a, b| mean(**a).total_cmp(&mean(**b)))
            .unwrap_or(&0),
    ];
    while chosen.len() < n.min(pool.len()) {
        let next = pool
            .iter()
            .filter(|i| !chosen.contains(i))
            .min_by(|a, b| nearest(&sim, **a, &chosen).total_cmp(&nearest(&sim, **b, &chosen)));
        match next {
            Some(i) => chosen.push(*i),
            None => break,
        }
    }
    chosen.sort_unstable();
    chosen
}

/// which examples look more like the street than like the class.
///
/// **this is the contamination screen pointed inwards.** the outward one asks
/// which negatives resemble examples; this asks which examples resemble
/// negatives. two quite different things show up, and both make poor
/// references:
///
/// - a genuine mislabel -- a dark sedan among white i-paces.
/// - a correctly labelled crop where the subject is **incidental**: another
///   vehicle filling the frame with a sliver of the subject at the edge. the
///   label is right, the subject really did pass, but the embedding describes
///   the whole crop and this one describes the other vehicle.
///
/// the second is not a labelling error and must not be reported as one.
///
/// **the comparison is against the negatives, not against the other
/// examples**, and that distinction is the whole of it. an unusual *aspect* --
/// the one rear view among side views -- is also unlike the other examples,
/// and a rule keyed on that would throw away precisely the coverage the
/// reference set exists to get. what separates the two is that a rear view is
/// still unlike the street, while a mislabel and an incidental crop are not.
pub fn odd_ones_out(to_examples: &[f32], to_street: &[f32]) -> Vec<(usize, f32, bool)> {
    if to_examples.len() < 3 {
        return Vec::new();
    }
    let mut out: Vec<(usize, f32, bool)> = to_examples
        .iter()
        .zip(to_street)
        .enumerate()
        .map(|(i, (own, street))| (i, *own, own <= street))
        .collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out
}

/// each passage's mean similarity to the other passages, and to the street.
///
/// **centred, whatever the measurement itself is doing.** raw cosines on this
/// harvest are useless for the comparison: every crop shares a camera, a street
/// and a lens, so unrelated cars sit at 0.83 and the whole distance between
/// "belongs to the class" and "is just traffic" is about 0.01. subtracting the
/// common direction costs nothing, needs no labels, and turns that into a gap
/// wide enough to decide on -- car-to-car falls from 0.83 to 0.00 while
/// class-to-class stays well above it.
pub fn belonging(groups: &[Vec<Vec<f32>>], pool: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    // a sample of the pool: this is a distribution rather than a rate, and the
    // whole harvest would be millions of pairs for no extra precision.
    let step = (pool.len() / SEPARATION_SAMPLE).max(1);
    let sampled: Vec<Vec<f32>> = pool.iter().step_by(step).cloned().collect();

    let everything: Vec<&Vec<f32>> = pool.iter().chain(groups.iter().flatten()).collect();
    let centre = centre_of(&everything);
    let groups: Vec<Vec<Vec<f32>>> = groups
        .iter()
        .map(|g| g.iter().map(|v| centred(v, &centre)).collect())
        .collect();
    let sampled: Vec<Vec<f32>> = sampled.iter().map(|v| centred(v, &centre)).collect();
    let groups = &groups;
    let street: Vec<&Vec<f32>> = sampled.iter().collect();
    let mean = |xs: Vec<f32>| {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<f32>() / xs.len() as f32
        }
    };
    let to_examples = (0..groups.len())
        .map(|i| {
            mean(
                groups
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .flat_map(|(_, other)| {
                        groups[i]
                            .iter()
                            .flat_map(move |a| other.iter().map(move |b| cosine(a, b)))
                    })
                    .collect(),
            )
        })
        .collect();
    let to_street = groups
        .iter()
        .map(|g| {
            mean(
                g.iter()
                    .flat_map(|a| street.iter().map(move |b| cosine(a, b)))
                    .collect(),
            )
        })
        .collect();
    (to_examples, to_street)
}

fn nearest(sim: &[Vec<f32>], i: usize, chosen: &[usize]) -> f32 {
    chosen
        .iter()
        .map(|&c| sim[i][c])
        .fold(f32::NEG_INFINITY, f32::max)
}

/// the mean direction of a set of embeddings, l2-normalised.
///
/// clip's space is anisotropic: everything shares a large common component, and
/// on crops that all come from one camera pointed at one street that component
/// is most of what a cosine measures. two unrelated cars sit at 0.84 for that
/// reason, not because they are alike. computed over the whole harvest, so it
/// needs no labels.
pub fn centre_of(vectors: &[&Vec<f32>]) -> Vec<f32> {
    let Some(dim) = vectors.first().map(|v| v.len()) else {
        return Vec::new();
    };
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v.iter()) {
            *m += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= vectors.len() as f32;
    }
    unit(&mean)
}

/// labelled crops their own class would vote against, the most disputed first.
///
/// scored the way the classifier votes: the mean of a crop's `VOTE_K` nearest
/// in the opposite class, less the same in its own. above zero, the rule would
/// call it the other way if its label were taken back, which is what a
/// mislabel looks like.
///
/// **its own passage is left out of its own class.** the frames of a passage
/// are near-copies, so leaving out only the crop itself lets the rest of the
/// passage vouch for it -- and labels are made a passage at a time, so a
/// mislabel is usually a whole passage of them. a class seen in one passage has
/// nothing outside it, so there only the crop itself is left out.
///
/// every crop that can be scored is ranked; how many to show is the caller's.
pub fn disputed(
    ranked: &[(String, Vec<f32>)],
    own: &[(String, Vec<f32>)],
    opposite: &[Vec<f32>],
    what: &str,
) -> Vec<(String, f32)> {
    let names: Vec<String> = own.iter().map(|(n, _)| n.clone()).collect();
    let passage_of: std::collections::HashMap<String, usize> =
        super::passages(&names, PASSAGE_GAP_MS)
            .into_iter()
            .enumerate()
            .flat_map(|(i, run)| run.into_iter().map(move |n| (n, i)))
            .collect();
    // **every crop against every crop of the other class**, which is the whole
    // cost of opening a session and grows with how much labelling has been done
    // -- so it is smallest on the day this is least useful to watch.
    let mut progress = super::Progress::new("comparing", "compared", what, ranked.len());
    let mut scored: Vec<(String, f32)> = ranked
        .iter()
        .filter_map(|(name, v)| {
            progress.tick();
            let passage = passage_of.get(name);
            let outside: Vec<f32> = own
                .iter()
                .filter(|(n, _)| passage_of.get(n) != passage)
                .map(|(_, o)| cosine(v, o))
                .collect();
            let to_own = if outside.is_empty() {
                own.iter()
                    .filter(|(n, _)| n != name)
                    .map(|(_, o)| cosine(v, o))
                    .collect()
            } else {
                outside
            };
            let to_opposite = opposite.iter().map(|o| cosine(v, o)).collect();
            let score = mean_of_top(to_opposite, VOTE_K) - mean_of_top(to_own, VOTE_K);
            // a class of one crop, or no opposite class at all, casts no vote.
            score.is_finite().then(|| (name.clone(), score))
        })
        .collect();
    progress.finished();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// remove the common direction and renormalise.
pub fn centred(v: &[f32], centre: &[f32]) -> Vec<f32> {
    if centre.is_empty() {
        return v.to_vec();
    }
    let u = unit(v);
    unit(
        &u.iter()
            .zip(centre)
            .map(|(a, b)| a - b)
            .collect::<Vec<f32>>(),
    )
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// the two scores the shipped rule compares, for one crop.
fn scores(v: &[f32], refs: &[Vec<f32>], negatives: &[Vec<f32>]) -> (f32, f32) {
    (
        top_k_mean(v, refs, VOTE_K),
        top_k_mean(v, negatives, VOTE_K),
    )
}

/// false *alerts* an hour, not false crops an hour.
///
/// a thing is cropped several times as it crosses, so one civilian car the
/// classifier dislikes produces a handful of firing crops within a couple of
/// seconds. downstream those are one outcome, not five -- the same
/// passages-not-crops argument applied to the negatives.
///
/// only meaningful because the pool is the whole harvest: clustering a random
/// sample in time would count almost every crop as its own event.
///
/// **counted under the deployment's confirmation.** a crop fires an alert only
/// when `m` of the last `n` crops agree, the window the tracker keeps, and the
/// window is cleared after a passage gap because the tracker would have lost
/// the vehicle. an alert within a passage gap of the last one is the same alert.
/// two approximations stay: the harvest saves fewer crops than the tracker has
/// looks, and on a busy street one window can span two vehicles.
pub fn alert_events(names: &[String], fired: &[bool], span_h: f32, m: u32, n: u32) -> f32 {
    if span_h <= 0.0 {
        return 0.0;
    }
    let mut dated: Vec<(u128, bool)> = names
        .iter()
        .zip(fired)
        .filter_map(|(name, f)| Some((crate::harvest::parse_name(name)?.at_millis, *f)))
        .collect();
    dated.sort_by_key(|(at, _)| *at);
    let n = n.max(1) as usize;
    let mut window: std::collections::VecDeque<bool> = std::collections::VecDeque::new();
    let (mut last, mut last_alert) = (None::<u128>, None::<u128>);
    let mut events = 0usize;
    for (at, f) in dated {
        if last.is_some_and(|t| at.saturating_sub(t) > PASSAGE_GAP_MS) {
            window.clear();
        }
        last = Some(at);
        window.push_back(f);
        if window.len() > n {
            window.pop_front();
        }
        if f && window.iter().filter(|x| **x).count() as u32 >= m {
            if !last_alert.is_some_and(|t| at.saturating_sub(t) <= PASSAGE_GAP_MS) {
                events += 1;
            }
            last_alert = Some(at);
        }
    }
    events as f32 / span_h
}

/// whether a run of looks in time order confirms: some look that agrees has at
/// least `m` agreeing among the last `n`, counting itself. the tracker's own
/// window, so a passage's first looks count with fewer than `n` behind them.
fn confirms(fired: &[bool], m: u32, n: u32) -> bool {
    let n = n.max(1) as usize;
    (0..fired.len()).any(|i| {
        fired[i]
            && fired[i.saturating_sub(n - 1)..=i]
                .iter()
                .filter(|f| **f)
                .count() as u32
                >= m
    })
}

/// a linear probe on the frozen embedding, by plain gradient descent.
///
/// forty lines rather than a dependency, and the shape DESIGN.md already plans
/// for: "frozen embedding, swappable head", where the day-one head is a cosine
/// match and the next one is a logistic regression over the same cached
/// vectors. fitting it costs nothing and answers which of the two is holding
/// accuracy back.
///
/// the classes are wildly imbalanced -- tens of examples against thousands of
/// negatives -- so each is weighted to count the same in total, or the fit would
/// do best by calling everything a negative.
pub fn fit_logistic(positive: &[Vec<f32>], negative: &[Vec<f32>]) -> (Vec<f32>, f32) {
    const STEPS: usize = 600;
    const LR: f32 = 4.0;
    const L2: f32 = 1e-3;
    let dim = positive.first().or(negative.first()).map_or(0, |v| v.len());
    let (mut w, mut b) = (vec![0.0f32; dim], 0.0f32);
    if positive.is_empty() || negative.is_empty() {
        return (w, b);
    }
    let pos_weight = negative.len() as f32 / positive.len() as f32;
    let total = positive.len() as f32 * pos_weight + negative.len() as f32;
    for _ in 0..STEPS {
        let mut grad = vec![0.0f32; dim];
        let mut bias = 0.0f32;
        for (set, target, weight) in [(positive, 1.0f32, pos_weight), (negative, 0.0, 1.0)] {
            for v in set {
                let z: f32 = v.iter().zip(&w).map(|(x, k)| x * k).sum::<f32>() + b;
                let g = (1.0 / (1.0 + (-z).exp()) - target) * weight;
                for (gr, x) in grad.iter_mut().zip(v) {
                    *gr += g * x;
                }
                bias += g;
            }
        }
        for (k, gr) in w.iter_mut().zip(&grad) {
            *k -= LR * (gr / total + L2 * *k);
        }
        b -= LR * bias / total;
    }
    (w, b)
}

pub fn probe_score(v: &[f32], w: &[f32], b: f32) -> f32 {
    v.iter().zip(w).map(|(x, k)| x * k).sum::<f32>() + b
}

/// what would make every number meaningless. checked, not assumed.
pub fn preconditions(
    reference_crops: &[String],
    eval_crops: &[String],
    n_refs: usize,
    n_negatives: usize,
) -> Vec<String> {
    let mut fatal = Vec::new();
    if let Some(shared) = reference_crops.iter().find(|n| eval_crops.contains(n)) {
        fatal.push(format!(
            "{shared} is in both the reference and eval sets: recall is 100% by construction"
        ));
    }
    if n_refs < VOTE_K {
        fatal.push(format!(
            "{n_refs} reference crops is fewer than VOTE_K={VOTE_K}, so the top-k mean \
             degenerates to a single neighbour"
        ));
    }
    if n_negatives < VOTE_K {
        fatal.push(format!(
            "{n_negatives} negative references is fewer than VOTE_K={VOTE_K}"
        ));
    }
    fatal
}

/// true but not fatal: the numbers mean something, just less than they look.
pub fn caveats(eval_passages: usize, random_negatives: usize, unscreened: usize) -> Vec<String> {
    let mut out = Vec::new();
    if eval_passages < MIN_EVAL_PASSAGES {
        out.push(format!(
            "{eval_passages} held-out passages, not {MIN_EVAL_PASSAGES}. recall carries an \
             interval too wide to call this working or not; the separation figures do not \
             depend on it"
        ));
    }
    if random_negatives < MIN_FPR_SAMPLE {
        out.push(format!(
            "{random_negatives} negatives is below {MIN_FPR_SAMPLE}, so a 1% false positive \
             rate cannot be seen"
        ));
    }
    if unscreened > 0 {
        out.push(format!(
            "{unscreened} of the {SCREEN_TOP} crops nearest the references are unlabelled. if \
             any is a real example it counts as a false positive, which understates precision"
        ));
    }
    out
}

/// below this share of the harvest's span, the examples are a narrow slice of it.
const NARROW_COVERAGE: f32 = 0.6;

/// are the examples drawn from the same stretch of time as the negatives.
///
/// this is not pedantry about sampling. the negatives come from the whole
/// harvest and the examples from whenever they happened to pass, so a subject
/// that only crossed in one afternoon is being compared against crops from
/// every light the street has had. some of the measured separation is then
/// time of day rather than vehicle.
///
/// it also catches the case that matters when the pipeline changes underneath
/// the harvest. crop framing, the frame a crop is cut from, and what is
/// harvested at all have each moved at least once; crops from either side of
/// such a change are different populations, and a labelled set that sits
/// entirely on one side of it will not say so on its own.
pub fn coverage_caveat(example_names: &[String], all: &[String]) -> Option<String> {
    let span = |names: &[String]| -> Option<(u128, u128)> {
        let times: Vec<u128> = names
            .iter()
            .filter_map(|n| Some(crate::harvest::parse_name(n)?.at_millis))
            .collect();
        Some((*times.iter().min()?, *times.iter().max()?))
    };
    let ((lo, hi), (all_lo, all_hi)) = (span(example_names)?, span(all)?);
    let whole = (all_hi - all_lo) as f32;
    if whole <= 0.0 {
        return None;
    }
    let share = (hi - lo) as f32 / whole;
    if share >= NARROW_COVERAGE {
        return None;
    }
    Some(format!(
        "the examples span {:.1}h of a {:.1}h harvest ({:.0}% of it), while the negatives \
         come from all of it. some of the separation above is time of day rather than \
         subject -- and if the pipeline changed during the gap, the two sides are \
         different populations",
        (hi - lo) as f32 / 3_600_000.0,
        whole / 3_600_000.0,
        share * 100.0
    ))
}

/// which crops a person should look at, whatever they are labelled.
pub fn screen(pool: &[(String, Vec<f32>)], refs: &[Vec<f32>], labels: &Labels) -> Vec<String> {
    let mut ranked: Vec<(f32, &String)> = pool
        .iter()
        .map(|(n, v)| (top_k_mean(v, refs, VOTE_K), n))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked
        .into_iter()
        .take(SCREEN_TOP)
        .filter(|(_, n)| !labels.entries.contains_key(*n))
        .map(|(_, n)| n.clone())
        .collect()
}

/// the negative pool a rate may be measured on.
///
/// only the random review pool counts. the ranked pool is ordered by the very
/// similarity being tested, so a rate measured on it measures the ranker.
pub fn random_negatives(pool: &[(String, Vec<f32>)], labels: &Labels) -> usize {
    pool.iter()
        .filter(|(n, _)| {
            labels
                .entries
                .get(n)
                .is_some_and(|e| e.via == Via::Random && e.truth == NEGATIVE)
        })
        .count()
}

/// judge each passage against references built from all the others.
///
/// the honest answer when there are too few passages to hold a fixed set back,
/// which is the normal state for weeks. every passage is scored exactly once by
/// a reference set that has never seen it, so there is no leakage, and all of
/// the data is used for both jobs instead of half for each.
///
/// the negatives are rescored in every fold too, because the reference set they
/// are compared against changes with it. that makes the folds' false positives
/// correlated -- the same crops, several times -- so the rate is averaged across
/// folds rather than pooled, which would claim several times the evidence.
pub struct Fold {
    pub gaps: Vec<f32>,
    pub negative_gaps: Vec<f32>,
}

pub fn leave_one_out(
    groups: &[Vec<Vec<f32>>],
    names: &[Vec<String>],
    negatives: &[Vec<f32>],
    pool: &[Vec<f32>],
) -> Vec<Fold> {
    // the negatives are the same in every fold, so each crop's score against
    // them is taken once. recomputing it per fold was most of what training
    // costs once every rejected verdict is a negative.
    let against = |v: &Vec<f32>| top_k_mean(v, negatives, VOTE_K);
    tracing::info!(
        "cross-validating {} passages against {} negatives over {} street crops",
        groups.len(),
        negatives.len(),
        pool.len()
    );
    let pool_scores: Vec<f32> = pool.iter().map(against).collect();
    let group_scores: Vec<Vec<f32>> = groups
        .iter()
        .map(|g| g.iter().map(against).collect())
        .collect();
    let mut folds = Vec::new();
    let mut progress = super::Progress::new("scoring", "scored", "passages", groups.len());
    for out in 0..groups.len() {
        progress.tick();
        let mut refs: Vec<Vec<f32>> = Vec::new();
        for (i, group) in groups.iter().enumerate() {
            if i == out {
                continue;
            }
            for name in super::clearest(&names[i], CROPS_PER_PASSAGE) {
                if let Some(pos) = names[i].iter().position(|n| *n == name) {
                    refs.push(group[pos].clone());
                }
            }
        }
        if refs.len() < VOTE_K {
            continue;
        }
        let gap = |(v, b): (&Vec<f32>, &f32)| top_k_mean(v, &refs, VOTE_K) - b;
        folds.push(Fold {
            gaps: groups[out]
                .iter()
                .zip(&group_scores[out])
                .map(gap)
                .collect(),
            negative_gaps: pool.iter().zip(&pool_scores).map(gap).collect(),
        });
    }
    progress.finished();
    folds
}

/// recall and false positives across the margin sweep.
///
/// the scores do not depend on the margin, so they are computed once and the
/// sweep is arithmetic. one number would be an assertion; the curve is what
/// lets an operating point be chosen.
pub fn sweep(
    folds: &[Fold],
    pool_names: &[String],
    per_hour: f32,
    span_h: f32,
    m: u32,
    n: u32,
) -> Vec<Row> {
    MARGIN_SWEEP
        .iter()
        .map(|&margin| {
            // caught the way the pipeline would catch it: `m` of `n` consecutive
            // crops above the margin, not any one of them.
            let kept = folds
                .iter()
                .filter(|f| {
                    let fired: Vec<bool> = f.gaps.iter().map(|g| *g > margin).collect();
                    confirms(&fired, m, n)
                })
                .count();
            let (hit, total) = folds.iter().fold((0usize, 0usize), |(h, t), f| {
                (
                    h + f.gaps.iter().filter(|g| **g > margin).count(),
                    t + f.gaps.len(),
                )
            });
            let rates: Vec<f32> = folds
                .iter()
                .map(|f| {
                    f.negative_gaps.iter().filter(|g| **g > margin).count() as f32
                        / f.negative_gaps.len().max(1) as f32
                })
                .collect();
            let fpr = if rates.is_empty() {
                0.0
            } else {
                rates.iter().sum::<f32>() / rates.len() as f32
            };
            let alerts: Vec<f32> = folds
                .iter()
                .map(|f| {
                    let fired: Vec<bool> = f.negative_gaps.iter().map(|g| *g > margin).collect();
                    alert_events(pool_names, &fired, span_h, m, n)
                })
                .collect();
            Row {
                margin,
                passages: kept,
                passage_recall: kept as f32 / folds.len().max(1) as f32,
                crop_recall: hit as f32 / total.max(1) as f32,
                fpr,
                per_hour: fpr * per_hour,
                alerts_per_hour: if alerts.is_empty() {
                    0.0
                } else {
                    alerts.iter().sum::<f32>() / alerts.len() as f32
                },
            }
        })
        .collect()
}

/// the crops the harvest collects per hour, measured rather than assumed.
///
/// it decides what a false positive rate costs: a rate is a fraction until it is
/// multiplied by how often the classifier is asked.
pub fn crops_per_hour(names: &[String]) -> (f32, f32) {
    let times: Vec<u128> = names
        .iter()
        .filter_map(|n| Some(crate::harvest::parse_name(n)?.at_millis))
        .collect();
    let (Some(lo), Some(hi)) = (times.iter().min(), times.iter().max()) else {
        return (0.0, 0.0);
    };
    let hours = (hi - lo) as f32 / 3_600_000.0;
    if hours < 0.1 {
        return (0.0, hours);
    }
    (names.len() as f32 / hours, hours)
}

/// look up the cached vectors for a list of crops, dropping any that are missing.
pub fn gather(cache: &Vectors, names: &[String]) -> (Vec<String>, Vec<Vec<f32>>) {
    let mut kept = Vec::new();
    let mut vectors = Vec::new();
    for name in names {
        if let Some(v) = cache.get(name) {
            kept.push(name.clone());
            vectors.push(v.clone());
        }
    }
    (kept, vectors)
}

/// everything a measurement needs to know. a subject is a parameter, never a
/// compiled-in case (r10).
#[derive(Debug, Clone)]
pub struct Options {
    pub subject: String,
    pub reference_passages: usize,
    pub negative_references: usize,
    pub centre: bool,
    pub include_unclear: bool,
    pub seed: u64,
    /// `[track] confirm_m` and `confirm_n`: how many of how many consecutive
    /// looks must agree before the pipeline alerts.
    pub confirm_m: u32,
    pub confirm_n: u32,
    /// which row of the curve to ship, when a person named one. `None` is the
    /// default bracket: the cleanest margin keeping `OPERATING_RECALL` of the
    /// passages.
    pub min_passage_recall: Option<f32>,
    pub max_fpr: Option<f32>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            subject: String::new(),
            reference_passages: 10,
            negative_references: NEGATIVE_REFERENCES,
            centre: false,
            include_unclear: false,
            seed: 0,
            min_passage_recall: None,
            max_fpr: None,
            confirm_m: crate::config::TRACK_CONFIRM_M,
            confirm_n: crate::config::TRACK_CONFIRM_N,
        }
    }
}

/// split the harvest into negative references and a rate pool.
///
/// **crops a person labelled `other` are the best negatives there are**, and an
/// earlier version threw all of them away: it excluded every labelled crop, so
/// hours of review went into the file and then straight past the measurement,
/// which drew its negatives from unlabelled crops instead. verified negatives
/// fill the reference slot first and the rest join the pool.
///
/// what is excluded is only what cannot be a negative: the subject's own crops,
/// and the ones a person could not call either way. an `unclear` crop in the
/// pool would be counted as a false positive if it fired, which is precisely
/// the judgement nobody was able to make.
///
/// both halves are drawn by the stable hash order, so the reference draw is not
/// the oldest crops -- which is what taking them off a name-sorted list gave,
/// thirty crops from one hour of one afternoon, making the negative half of the
/// decision out of a single lighting condition.
fn split_negatives(
    all: &[String],
    labels: &Labels,
    subject: &str,
    n: usize,
    seed: u64,
) -> (Vec<String>, Vec<String>, bool) {
    let truth = |name: &String| labels.entries.get(name).map(|e| e.truth.as_str());
    let usable =
        |name: &&String| !matches!(truth(name), Some(t) if t == subject || t == super::UNCLEAR);

    let via = |name: &String| labels.entries.get(name).map(|e| e.via);
    // **what the classifier fired on goes in first.** the draw used to be hash
    // order across every verified negative, so a vehicle a person had just
    // rejected had no better chance of being among the thirty than any sedan.
    // measured on the deployment: five crops the classifier called a go-4 were
    // labelled `other`, none were drawn, and all five still scored as the
    // subject by the same margin to four decimals. a negative set that excludes
    // what actually confuses the classifier reports the rate at which sedans
    // fire, which is not the question anyone is asking.
    let (mut fired, mut quiet): (Vec<String>, Vec<String>) = all
        .iter()
        .filter(usable)
        .filter(|name| truth(name) == Some(NEGATIVE))
        .cloned()
        .partition(|name| via(name) == Some(Via::Alert));
    // **another subject is a negative for this one.** a waymo is not a go-4,
    // and it is a far better negative than any sedan: the street's ordinary
    // traffic sits nowhere near the boundary and another distinctive vehicle
    // does. these used to fall through -- past `usable`, matching neither
    // `other` nor unlabelled -- so hand-labelled evidence was discarded without
    // a word. they can only be used at all because each subject now has its own
    // negative set; one shared set could not hold a waymo without dragging the
    // waymo's own score down with it.
    let mut others: Vec<String> = all
        .iter()
        .filter(usable)
        .filter(|name| matches!(truth(name), Some(t) if t != NEGATIVE))
        .cloned()
        .collect();
    let mut assumed: Vec<String> = all
        .iter()
        .filter(usable)
        .filter(|name| truth(name).is_none())
        .cloned()
        .collect();
    // within each tier the order is still the stable hash, so the draw does not
    // reshuffle as the harvest grows.
    fired.sort_by_key(|name| super::hash_rank(name, seed));
    others.sort_by_key(|name| super::hash_rank(name, seed));
    quiet.sort_by_key(|name| super::hash_rank(name, seed));
    assumed.sort_by_key(|name| super::hash_rank(name, seed));

    // the tiers are ordered by how much each one tells us: what the classifier
    // actually fired on, then the other subjects, then ordinary traffic a
    // person confirmed, then crops nobody has looked at.
    //
    // **every confuser, however many.** what the classifier fired on and
    // another subject's crops are the negatives that decide a vote, and drawing
    // them into the same thirty as the street dropped exactly those: measured
    // on the deployment, 658 rejected verdicts of which the thirty drawn held no
    // cargo bike, and cargo bikes then fired. only ordinary traffic -- confirmed
    // first, then crops nobody has looked at -- is a sample of `n`.
    let hand = quiet.len() >= n;
    let mut refs: Vec<String> = fired;
    refs.extend(others);
    let confirmed = n.min(quiet.len());
    refs.extend(quiet.drain(..confirmed));
    let top_up = (n - confirmed).min(assumed.len());
    refs.extend(assumed.drain(..top_up));
    let mut pool = quiet;
    pool.extend(assumed);
    pool.sort();
    (refs, pool, hand)
}

/// the whole measurement, as data rather than as printing.
///
/// one code path for however it is asked for. two would be worse than it
/// sounds: the browser's numbers would drift from anything else's and nobody
/// would notice which was wrong.
pub fn measure(labels: &Labels, cache: &Vectors, all: &[String], opts: &Options) -> Report {
    let mut report = Report {
        subject: opts.subject.clone(),
        confirm_m: opts.confirm_m,
        min_passage_recall: opts.min_passage_recall,
        max_fpr: opts.max_fpr,
        confirm_n: opts.confirm_n,
        n_crops: all.len(),
        ..Default::default()
    };
    let mut wanted = labels.named(&opts.subject);
    if opts.include_unclear {
        wanted.extend(labels.named(super::UNCLEAR));
    }
    let groups = super::passages(&wanted, PASSAGE_GAP_MS);
    report.n_labelled = wanted.len();
    if groups.is_empty() {
        report.error = Some(format!("nothing is labelled {}", opts.subject));
        return report;
    }

    // said before the long quiet part rather than after it, so the shape of the
    // job is on screen while it runs: how much of the harvest is labelled is
    // what decides whether this takes seconds or minutes.
    tracing::info!(
        "measuring {} crops: {} labelled {} in {} passages",
        all.len(),
        wanted.len(),
        opts.subject,
        groups.len()
    );

    // the common direction is a property of the camera and the street rather
    // than of either class, so it comes off the whole harvest and needs no
    // labels at all.
    let centre = if opts.centre {
        let cached: Vec<&Vec<f32>> = all.iter().filter_map(|n| cache.get(n)).collect();
        centre_of(&cached)
    } else {
        Vec::new()
    };
    let prepare = |names: &[String]| -> (Vec<String>, Vec<Vec<f32>>) {
        let (kept, vectors) = gather(cache, names);
        (kept, vectors.iter().map(|v| centred(v, &centre)).collect())
    };

    let (negative_names, pool_names, hand) = split_negatives(
        all,
        labels,
        &opts.subject,
        opts.negative_references,
        opts.seed,
    );
    // the names as well as the vectors: what ships has to be the crops that
    // were scored, and `prepare` drops any the cache does not hold.
    let (negative_kept, negatives) = prepare(&negative_names);
    report.negative_reference_crops = negative_kept;
    let (pool_names, pool) = prepare(&pool_names);
    report.n_negatives = pool.len();
    report.n_negative_references = negatives.len();
    report.hand_negatives = hand;

    let prepared: Vec<(Vec<String>, Vec<Vec<f32>>)> = groups.iter().map(|g| prepare(g)).collect();
    let names: Vec<Vec<String>> = prepared.iter().map(|(n, _)| n.clone()).collect();
    let vectors: Vec<Vec<Vec<f32>>> = prepared.iter().map(|(_, v)| v.clone()).collect();
    if vectors.iter().all(|g| g.is_empty()) {
        report.error = Some(format!(
            "none of the {} labelled crops have been embedded yet",
            wanted.len()
        ));
        return report;
    }

    fill_separation(&mut report, &vectors, &pool, &pool_names);
    fill_eval(
        &mut report,
        &vectors,
        &names,
        &negatives,
        &pool,
        &pool_names,
        labels,
        all,
        opts,
    );
    report
}

/// the three cosine means, and the pairwise passage matrix behind them.
fn fill_separation(
    report: &mut Report,
    groups: &[Vec<Vec<f32>>],
    pool: &[Vec<f32>],
    pool_names: &[String],
) {
    report.self_self = Summary::of(&mut cross_passage(groups));

    // a sample of the pool: the false positive rate wants every crop, a
    // distribution does not, and the whole harvest makes millions of pairs to
    // average over for no extra precision.
    let step = (pool.len() / SEPARATION_SAMPLE).max(1);
    let sample: Vec<Vec<f32>> = pool.iter().step_by(step).cloned().collect();
    let sample_names: Vec<String> = pool_names.iter().step_by(step).cloned().collect();
    let civ_groups: Vec<Vec<Vec<f32>>> = super::passages(&sample_names, PASSAGE_GAP_MS)
        .iter()
        .map(|g| {
            g.iter()
                .filter_map(|n| {
                    sample_names
                        .iter()
                        .position(|s| s == n)
                        .map(|i| sample[i].clone())
                })
                .collect()
        })
        .collect();
    report.other_other = Summary::of(&mut cross_passage(&civ_groups));

    let mut across: Vec<f32> = groups
        .iter()
        .flatten()
        .flat_map(|a| sample.iter().map(move |b| cosine(a, b)))
        .collect();
    report.self_other = Summary::of(&mut across);

    report.matrix = groups
        .iter()
        .map(|a| {
            groups
                .iter()
                .map(|b| {
                    let n = (a.len() * b.len()).max(1) as f32;
                    a.iter()
                        .flat_map(|x| b.iter().map(move |y| cosine(x, y)))
                        .sum::<f32>()
                        / n
                })
                .collect()
        })
        .collect();
}

/// recall, false positives, the probe, and everything that would void them.
#[allow(clippy::too_many_arguments)]
fn fill_eval(
    report: &mut Report,
    groups: &[Vec<Vec<f32>>],
    names: &[Vec<String>],
    negatives: &[Vec<f32>],
    pool: &[Vec<f32>],
    pool_names: &[String],
    labels: &Labels,
    all: &[String],
    opts: &Options,
) {
    let centres: Vec<Vec<f32>> = groups
        .iter()
        .map(|g| centre_of(&g.iter().collect::<Vec<_>>()))
        .collect();
    let (to_examples, to_street) = belonging(groups, pool);
    let odd = odd_ones_out(&to_examples, &to_street);
    let mut skip = vec![false; groups.len()];
    for (i, _, flagged) in &odd {
        skip[*i] = *flagged;
    }
    let chosen = choose_references(&centres, &skip, opts.reference_passages);
    report.odd_crops = odd
        .iter()
        .filter(|(_, _, flagged)| *flagged)
        .flat_map(|(i, _, _)| names[*i].clone())
        .collect();
    report.odd = odd
        .into_iter()
        .map(|(i, m, flagged)| (names[i].first().cloned().unwrap_or_default(), m, flagged))
        .collect();
    let reference_crops: Vec<String> = chosen
        .iter()
        .flat_map(|i| super::clearest(&names[*i], CROPS_PER_PASSAGE))
        .collect();
    let eval_crops: Vec<String> = names
        .iter()
        .enumerate()
        .filter(|(i, _)| !chosen.contains(i))
        .flat_map(|(_, g)| g.clone())
        .collect();
    report.n_references = reference_crops.len();
    report.reference_crops = reference_crops.clone();
    report.fatal = preconditions(
        &reference_crops,
        &eval_crops,
        reference_crops.len(),
        negatives.len(),
    );

    let reference_vectors: Vec<Vec<f32>> = chosen
        .iter()
        .flat_map(|i| {
            super::clearest(&names[*i], CROPS_PER_PASSAGE)
                .into_iter()
                .filter_map(|n| {
                    names[*i]
                        .iter()
                        .position(|x| *x == n)
                        .map(|p| groups[*i][p].clone())
                })
        })
        .collect();
    let paired: Vec<(String, Vec<f32>)> = pool_names
        .iter()
        .cloned()
        .zip(pool.iter().cloned())
        .collect();
    report.screen = screen(&paired, &reference_vectors, labels);

    let (per_hour, span_h) = crops_per_hour(all);
    report.per_hour = per_hour;

    // too few passages to hold a fixed set back is the normal state for weeks.
    // leaving one out uses every passage as both a reference and a query
    // without ever doing both at once.
    report.leave_one_out = true;
    let folds = leave_one_out(groups, names, negatives, pool);
    report.sweep = sweep(
        &folds,
        pool_names,
        per_hour,
        span_h,
        opts.confirm_m,
        opts.confirm_n,
    );

    // and again over only what the classifier can see, on both sides.
    let near_pool: Vec<usize> = (0..pool.len())
        .filter(|i| {
            pool_names
                .get(*i)
                .and_then(|n| crate::harvest::parse_name(n))
                .is_some_and(|s| s.width.max(s.height) >= NEAR_PX)
        })
        .collect();
    let near_groups: Vec<usize> = (0..groups.len())
        .filter(|i| apparent_px(&names[*i]) >= NEAR_PX)
        .collect();
    if near_groups.len() >= 2 && near_pool.len() >= MIN_FPR_SAMPLE {
        let near_names: Vec<String> = near_pool.iter().map(|i| pool_names[*i].clone()).collect();
        let near_rate = near_pool.len() as f32 / pool.len().max(1) as f32;
        let folds_near: Vec<Fold> = near_groups
            .iter()
            .filter_map(|i| folds.get(*i))
            .map(|f| Fold {
                gaps: f.gaps.clone(),
                negative_gaps: near_pool.iter().map(|i| f.negative_gaps[*i]).collect(),
            })
            .collect();
        report.sweep_near = sweep(
            &folds_near,
            &near_names,
            per_hour * near_rate,
            span_h,
            opts.confirm_m,
            opts.confirm_n,
        );
    }
    report.passages = groups
        .iter()
        .enumerate()
        .map(|(i, g)| PassageRow {
            first: names[i].first().cloned().unwrap_or_default(),
            crops: g.len(),
            span_s: passage_span_s(&names[i]),
            reference: chosen.contains(&i),
            best: folds
                .get(i)
                .and_then(|f| f.gaps.iter().cloned().reduce(f32::max))
                .unwrap_or(f32::NAN),
            px: apparent_px(&names[i]),
        })
        .collect();
    report.caveats = caveats(
        groups.len(),
        random_negatives(&paired, labels).max(pool.len()),
        report.screen.len(),
    );
    let examples: Vec<String> = names.iter().flatten().cloned().collect();
    report.caveats.extend(coverage_caveat(&examples, all));
    // what would fire right now, against the reference set as chosen -- not
    // leave-one-out. this is the shipped decision on every crop in the pool,
    // which is what an alert would actually be.
    report.alert_margin = report
        .operating_point()
        .map(|r| r.margin)
        .unwrap_or(crate::classify::DEFAULT_MARGIN);
    //
    // **over the examples as well as the pool.** the pool excludes crops
    // already labelled as the subject, so alerts drawn from it alone contain no
    // true positives by construction and the precision computed on them is
    // structurally zero -- a number that looks like a catastrophic result and
    // is actually an artefact of which crops were in the bag.
    tracing::info!(
        "scoring {} crops at margin {:+.3}, for what would fire right now",
        pool.len() + groups.iter().flatten().count(),
        report.alert_margin
    );
    let everything = pool
        .iter()
        .zip(pool_names.iter())
        .chain(groups.iter().flatten().zip(names.iter().flatten()));
    let mut fired: Vec<(String, f32)> = everything
        .filter_map(|(v, name)| {
            let (a, b) = scores(v, &reference_vectors, negatives);
            (a - b > report.alert_margin).then(|| (name.clone(), a - b))
        })
        .collect();
    fired.sort_by(|a, b| b.1.total_cmp(&a.1));
    fired.truncate(ALERTS_SHOWN);
    report.alerts = fired;

    let at_default = report
        .sweep
        .iter()
        .find(|r| r.margin == crate::classify::DEFAULT_MARGIN)
        .or(report.sweep.first());
    report.recall_interval = at_default
        .map(|r| wilson(r.passages, groups.len()))
        .unwrap_or((0.0, 1.0));
    report.probe = fit_probe(groups, names, negatives, pool, &folds);
}

/// the two heads, scored the same leave-one-out way.
///
/// **per passage is the honest column.** an alert fires on one crop of a
/// passage rather than all of them, so a passage's score is its best crop's;
/// scoring per crop lets a passage that happened to be cropped seven times
/// outvote four that were cropped once.
fn fit_probe(
    groups: &[Vec<Vec<f32>>],
    names: &[Vec<String>],
    negatives: &[Vec<f32>],
    pool: &[Vec<f32>],
    folds: &[Fold],
) -> Option<Probe> {
    if folds.len() < 2 || pool.is_empty() {
        return None;
    }
    let nn_crop: Vec<f32> = folds.iter().flat_map(|f| f.gaps.clone()).collect();
    let nn_neg: Vec<f32> = folds.iter().flat_map(|f| f.negative_gaps.clone()).collect();
    let nn_passage: Vec<f32> = folds
        .iter()
        .filter_map(|f| f.gaps.iter().cloned().reduce(f32::max))
        .collect();

    // **the longest quiet stretch of a measurement.** `fit_logistic` is a
    // full-batch gradient descent over every negative in the harvest, run once
    // per passage, and it happens after the last thing that printed anything.
    tracing::info!(
        "fitting a linear probe per passage over {} street crops -- a diagnostic, \
         and the slowest part of this",
        pool.len()
    );
    let mut progress = super::Progress::new("fitting", "fitted", "probe folds", groups.len());
    let (mut lp_crop, mut lp_neg, mut lp_passage) = (Vec::new(), Vec::new(), Vec::new());
    for out in 0..groups.len() {
        progress.tick();
        let train: Vec<Vec<f32>> = groups
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != out)
            .flat_map(|(_, g)| g.clone())
            .collect();
        if train.is_empty() || groups[out].is_empty() {
            continue;
        }
        let (w, b) = fit_logistic(&train, pool);
        let scores: Vec<f32> = groups[out].iter().map(|v| probe_score(v, &w, b)).collect();
        if let Some(best) = scores.iter().cloned().reduce(f32::max) {
            lp_passage.push(best);
        }
        lp_crop.extend(scores);
        lp_neg.extend(pool.iter().map(|v| probe_score(v, &w, b)));
    }
    progress.finished();
    let _ = (names, negatives);
    Some(Probe {
        nn_crop: auc(&nn_crop, &nn_neg),
        nn_passage: auc(&nn_passage, &nn_neg),
        probe_crop: auc(&lp_crop, &lp_neg),
        probe_passage: auc(&lp_passage, &lp_neg),
        thresholds: probe_thresholds(&lp_crop, &lp_neg),
    })
}

/// what share of the street fires, at thresholds keeping most of the examples.
///
/// auc is threshold-free, which is what makes it fair between two heads, but it
/// hides how a handful of bad crops drag an operating point down. so the curve
/// is reported at a few recalls rather than summarised at one.
fn probe_thresholds(positive: &[f32], negative: &[f32]) -> Vec<(f32, f32)> {
    if positive.is_empty() || negative.is_empty() {
        return Vec::new();
    }
    let mut ranked = positive.to_vec();
    ranked.sort_by(f32::total_cmp);
    [1.0f32, 0.9, 0.8]
        .iter()
        .map(|want| {
            let i = (((1.0 - want) * ranked.len() as f32).round() as usize).min(ranked.len() - 1);
            let cut = ranked[i];
            let keeps = ranked.iter().filter(|v| **v >= cut).count() as f32 / ranked.len() as f32;
            let fires =
                negative.iter().filter(|v| **v >= cut).count() as f32 / negative.len() as f32;
            (keeps, fires)
        })
        .collect()
}

/// the long edge of the largest crop in a passage: how big the thing got.
fn apparent_px(names: &[String]) -> u32 {
    names
        .iter()
        .filter_map(|n| crate::harvest::parse_name(n))
        .map(|s| s.width.max(s.height))
        .max()
        .unwrap_or(0)
}

fn passage_span_s(names: &[String]) -> f32 {
    let times: Vec<u128> = names
        .iter()
        .filter_map(|n| Some(crate::harvest::parse_name(n)?.at_millis))
        .collect();
    match (times.iter().min(), times.iter().max()) {
        (Some(lo), Some(hi)) => (hi - lo) as f32 / 1000.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(v: Vec<f32>) -> Vec<f32> {
        super::unit(&v)
    }

    #[test]
    fn recall_reports_the_interval_a_handful_of_passages_earns() {
        let (lo, hi) = wilson(5, 5);
        assert!(hi <= 1.0 && lo < 0.7, "5/5 claimed as {lo:.2}-{hi:.2}");
        let (lo, hi) = wilson(90, 100);
        assert!(
            (0.82..0.85).contains(&lo) && (0.94..0.96).contains(&hi),
            "{lo} {hi}"
        );
        assert_eq!(wilson(0, 0), (0.0, 1.0));
    }

    #[test]
    fn auc_counts_a_tie_as_half_or_two_identical_heads_would_score_zero() {
        assert_eq!(auc(&[1.0, 2.0], &[0.0, 0.5]), 1.0);
        assert_eq!(auc(&[0.0], &[1.0]), 0.0);
        assert_eq!(auc(&[1.0, 1.0], &[1.0, 1.0]), 0.5);
        assert!(auc(&[], &[1.0]).is_nan());
    }

    #[test]
    fn similarity_between_examples_never_compares_a_passage_with_itself() {
        // two tight clusters, one per passage. every within-passage pair is
        // ~1.0 and every cross-passage pair ~0.0, so they cannot be confused.
        let groups = vec![
            vec![
                unit_vec(vec![1.0, 0.0]),
                unit_vec(vec![1.0, 0.01]),
                unit_vec(vec![1.0, 0.0]),
            ],
            vec![unit_vec(vec![0.0, 1.0]), unit_vec(vec![0.0, 1.0])],
        ];
        let pairs = cross_passage(&groups);
        assert_eq!(pairs.len(), 6, "expected 3x2 cross-passage pairs only");
        assert!(
            pairs.iter().cloned().fold(f32::MIN, f32::max) < 0.1,
            "a within-passage pair leaked in"
        );
    }

    #[test]
    fn the_margin_sweep_looks_below_zero() {
        let lowest = MARGIN_SWEEP.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            lowest < 0.0,
            "the useful region is negative, and it is not swept"
        );
        assert!(MARGIN_SWEEP.contains(&0.0));
        assert!(MARGIN_SWEEP.windows(2).all(|w| w[0] < w[1]), "not sorted");
    }

    #[test]
    fn a_reference_set_that_is_also_the_eval_set_voids_the_numbers() {
        let shared = vec!["a.jpg".to_string()];
        let fatal = preconditions(&shared, &shared, 10, 10);
        assert!(
            fatal[0].contains("both the reference and eval sets"),
            "{fatal:?}"
        );
        assert!(preconditions(&shared, &["b.jpg".to_string()], 10, 10).is_empty());
    }

    #[test]
    fn a_reference_set_below_the_vote_size_is_refused() {
        let fatal = preconditions(&[], &[], VOTE_K - 1, 10);
        assert!(fatal.iter().any(|f| f.contains("VOTE_K")), "{fatal:?}");
    }

    /// the examples come from whenever the subject happened to pass; the
    /// negatives come from the whole harvest. a subject that only crossed in
    /// one afternoon is being compared against every light the street has had.
    #[test]
    fn examples_from_a_narrow_slice_of_the_harvest_are_flagged() {
        let at = |h: u128| {
            format!(
                "{}_car_085_200x200.jpg",
                1_789_200_000_000u128 + h * 3_600_000
            )
        };
        let all: Vec<String> = (0..10).map(at).collect();

        let narrow: Vec<String> = (0..2).map(at).collect();
        let note = coverage_caveat(&narrow, &all).expect("a 2h slice of 9h was not flagged");
        assert!(note.contains("time of day"), "{note}");

        // and the counterweight: examples spread across the harvest are fine.
        let wide: Vec<String> = [0u128, 4, 9].iter().map(|h| at(*h)).collect();
        assert!(coverage_caveat(&wide, &all).is_none());
    }

    #[test]
    fn a_coverage_caveat_needs_a_harvest_that_spans_any_time_at_all() {
        let one = vec!["1789200000000_car_085_200x200.jpg".to_string()];
        assert!(coverage_caveat(&one, &one).is_none());
        assert!(coverage_caveat(&[], &one).is_none());
    }

    fn minute(i: u128) -> String {
        format!("{}_car_085_200x200.jpg", 1_789_200_000_000u128 + i * 60_000)
    }

    fn entry(truth: &str, via: Via) -> super::super::Entry {
        super::super::Entry::new(truth, via)
    }

    /// the bug that threw away hours of review: every labelled crop was
    /// excluded from the negatives, so crops a person had confirmed were
    /// ordinary traffic went into the file and straight past the measurement.
    #[test]
    fn hand_labelled_negatives_are_used_rather_than_discarded() {
        let all: Vec<String> = (0..40).map(minute).collect();
        let mut labels = Labels::default();
        for i in 0..20 {
            labels
                .entries
                .insert(minute(i), entry(NEGATIVE, Via::Random));
        }
        labels.entries.insert(minute(30), entry("waymo", Via::Seed));

        let (refs, pool, hand) = split_negatives(&all, &labels, "waymo", 10, 0);
        assert!(hand, "twenty verified negatives did not fill a slot of ten");
        assert_eq!(refs.len(), 10);
        assert!(
            refs.iter().all(|n| labels.entries.contains_key(n)),
            "a verified negative was passed over for an unlabelled crop"
        );
        // the subject's own crop is in neither half, and everything else is
        // available to measure a rate on.
        assert!(!refs.contains(&minute(30)) && !pool.contains(&minute(30)));
        assert_eq!(refs.len() + pool.len(), all.len() - 1);
    }

    /// with too few verified ones the slot is topped up rather than left short:
    /// below `VOTE_K` the negative half of the decision stops working at all.
    #[test]
    fn a_short_verified_set_is_topped_up_from_the_unlabelled_harvest() {
        let all: Vec<String> = (0..40).map(minute).collect();
        let mut labels = Labels::default();
        labels
            .entries
            .insert(minute(0), entry(NEGATIVE, Via::Random));

        let (refs, pool, hand) = split_negatives(&all, &labels, "waymo", 10, 0);
        assert!(
            !hand,
            "one verified negative should not count as a hand-built set"
        );
        assert_eq!(refs.len(), 10);
        assert_eq!(refs.len() + pool.len(), all.len());
    }

    /// **the crops the classifier fired on are the ones worth comparing
    /// against**, and the draw passed them over.
    ///
    /// the thirty were taken by hash order across every verified negative, so a
    /// vehicle a person had just rejected had no better chance of being in them
    /// than any sedan on the street. measured on the deployment: the classifier
    /// fired on a white work van and a white box truck, both were labelled
    /// `other`, and adding them changed nothing -- none were drawn, and all five
    /// crops still scored as the subject by the same margin to four decimals.
    ///
    /// a negative set that excludes what actually confuses the classifier is
    /// measuring the wrong question: it reports the rate at which sedans fire.
    #[test]
    fn negatives_the_classifier_fired_on_are_drawn_first() {
        let all: Vec<String> = (0..40).map(minute).collect();
        let mut labels = Labels::default();
        for i in 0..30 {
            labels
                .entries
                .insert(minute(i), entry(NEGATIVE, Via::Random));
        }
        // three a person rejected *after* the classifier fired on them.
        let hard: Vec<String> = (30..33).map(minute).collect();
        for n in &hard {
            labels
                .entries
                .insert(n.clone(), entry(NEGATIVE, Via::Alert));
        }

        let (refs, pool, _) = split_negatives(&all, &labels, "waymo", 5, 0);
        for n in &hard {
            assert!(
                refs.contains(n),
                "{n} fired on the street and was passed over for a crop nobody disputed"
            );
            assert!(
                !pool.contains(n),
                "{n} is both a reference and in the rate pool"
            );
        }
        assert_eq!(
            refs.len(),
            hard.len() + 5,
            "ordinary traffic is still a sample of five"
        );
    }

    /// **the cap dropped exactly the negatives that mattered.** measured on the
    /// deployment: 658 verdicts rejected by hand, 69 of them cargo bikes the
    /// detector called motorcycles, and the thirty drawn by hash order held no
    /// bike at all. two bikes then fired at a +0.03 margin; with every rejected
    /// verdict among the negatives they scored -0.09 and -0.10.
    #[test]
    fn every_rejected_verdict_is_a_negative_reference_however_many() {
        let all: Vec<String> = (0..200).map(minute).collect();
        let mut labels = Labels::default();
        for i in 0..60 {
            labels
                .entries
                .insert(minute(i), entry(NEGATIVE, Via::Alert));
        }
        for i in 60..100 {
            labels
                .entries
                .insert(minute(i), entry(NEGATIVE, Via::Random));
        }
        for i in 100..110 {
            labels.entries.insert(minute(i), entry("waymo", Via::Seed));
        }

        let (refs, pool, _) = split_negatives(&all, &labels, "go4", 30, 0);
        for i in (0..60).chain(100..110) {
            assert!(refs.contains(&minute(i)), "confuser {i} was capped out");
        }
        // ordinary traffic is still a sample, not the whole street.
        assert_eq!(refs.len(), 60 + 10 + 30);
        assert!(refs.iter().all(|n| !pool.contains(n)));
        assert_eq!(refs.len() + pool.len(), all.len());
    }

    /// **another subject is a negative, and used to be thrown away.**
    ///
    /// a waymo is not a go-4, so a labelled waymo is a verified negative for
    /// go-4 -- and the most informative one available, since the street's sedans
    /// sit nowhere near the boundary and another distinctive vehicle does. the
    /// old filter let them past `usable` and then matched neither `other` nor
    /// unlabelled, so they fell through into nothing: hand-labelled evidence,
    /// discarded silently.
    ///
    /// they rank below the crops the classifier actually fired on and above
    /// ordinary traffic, which is the order of how much each one tells us.
    #[test]
    fn another_subjects_crops_are_negatives_for_this_one() {
        let all: Vec<String> = (0..40).map(minute).collect();
        let mut labels = Labels::default();
        for i in 0..20 {
            labels
                .entries
                .insert(minute(i), entry(NEGATIVE, Via::Random));
        }
        let waymos: Vec<String> = (30..34).map(minute).collect();
        for n in &waymos {
            labels.entries.insert(n.clone(), entry("waymo", Via::Seed));
        }

        let (refs, pool, _) = split_negatives(&all, &labels, "go4", 6, 0);
        for n in &waymos {
            assert!(
                refs.contains(n),
                "{n} is a labelled waymo and was not drawn as a go-4 negative"
            );
        }
        // and the subject's own crops are still in neither half.
        labels.entries.insert(minute(5), entry("go4", Via::Seed));
        let (refs, pool2, _) = split_negatives(&all, &labels, "go4", 6, 0);
        assert!(!refs.contains(&minute(5)) && !pool2.contains(&minute(5)));
        assert!(!pool.contains(&minute(30)), "drawn and pooled at once");
    }

    /// an `unclear` crop firing would be counted as a false positive, which is
    /// exactly the judgement nobody was able to make.
    #[test]
    fn crops_nobody_could_call_are_in_neither_half() {
        let all: Vec<String> = (0..10).map(minute).collect();
        let mut labels = Labels::default();
        labels
            .entries
            .insert(minute(3), entry(super::super::UNCLEAR, Via::Seed));

        let (refs, pool, _) = split_negatives(&all, &labels, "waymo", 2, 0);
        assert!(!refs.contains(&minute(3)) && !pool.contains(&minute(3)));
        assert_eq!(refs.len() + pool.len(), all.len() - 1);
    }

    #[test]
    fn too_few_passages_and_too_few_negatives_are_both_reported() {
        let notes = caveats(3, 100, 0);
        assert!(notes.iter().any(|n| n.contains("held-out passages")));
        assert!(notes.iter().any(|n| n.contains("1% false positive")));
        assert!(caveats(MIN_EVAL_PASSAGES, MIN_FPR_SAMPLE, 0).is_empty());
    }

    #[test]
    fn false_alerts_are_counted_per_passage_not_per_crop() {
        let names: Vec<String> = [
            1_789_259_517_126u128,
            1_789_259_517_526,
            1_789_259_519_425,
            1_789_260_855_530,
        ]
        .iter()
        .map(|m| format!("{m}_car_085_100x100.jpg"))
        .collect();
        assert_eq!(
            alert_events(&names, &[true, true, true, true], 1.0, 1, 1),
            2.0
        );
        assert_eq!(alert_events(&names, &[false; 4], 1.0, 1, 1), 0.0);
    }

    fn frames(n: u128, apart_ms: u128) -> Vec<String> {
        (0..n)
            .map(|i| format!("{}_car_085_100x100.jpg", 1_789_259_517_126 + i * apart_ms))
            .collect()
    }

    /// **a passage is caught only the way the pipeline would catch it.** one
    /// crop scoring above the margin used to be enough, which is a one-of-one
    /// rule the deployment never runs: it alerts on `confirm_m` of the last
    /// `confirm_n` looks.
    #[test]
    fn a_passage_counts_only_when_m_of_n_crops_agree() {
        let fold = Fold {
            gaps: vec![0.1, -0.1, 0.1, -0.1, -0.1],
            negative_gaps: Vec::new(),
        };
        let caught = |m, n| {
            sweep(std::slice::from_ref(&fold), &[], 1.0, 1.0, m, n)
                .into_iter()
                .find(|r| r.margin == 0.0)
                .unwrap()
                .passages
        };
        assert_eq!(caught(1, 1), 1);
        assert_eq!(caught(2, 3), 1, "two of the first three agreed");
        assert_eq!(
            caught(3, 5),
            0,
            "two agreeing crops confirmed three of five"
        );
    }

    #[test]
    fn a_false_alert_needs_m_of_n_consecutive_crops() {
        let names = frames(5, 400);
        let fired = [true, false, true, false, false];
        assert_eq!(alert_events(&names, &fired, 1.0, 1, 1), 1.0);
        assert_eq!(alert_events(&names, &fired, 1.0, 2, 3), 1.0);
        assert_eq!(alert_events(&names, &fired, 1.0, 3, 5), 0.0);
    }

    /// the tracker loses a vehicle it has not seen for a passage gap, and what
    /// it had counted towards confirming goes with it.
    #[test]
    fn crops_further_apart_than_a_passage_do_not_add_up() {
        let names = frames(2, 60_000);
        assert_eq!(alert_events(&names, &[true, true], 1.0, 2, 5), 0.0);
        assert_eq!(alert_events(&names, &[true, true], 1.0, 1, 1), 2.0);
    }

    #[test]
    fn centring_removes_what_every_crop_shares() {
        // vectors that are a shared direction plus a little noise: their
        // similarity must collapse once the common part is taken out.
        let raw: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let j = i as f32 / 20.0;
                unit_vec(vec![3.0, j * 0.3, (1.0 - j) * 0.2, j * j * 0.1])
            })
            .collect();
        let before = cross_passage(&raw.iter().map(|v| vec![v.clone()]).collect::<Vec<_>>());
        let mean_before = before.iter().sum::<f32>() / before.len() as f32;
        assert!(
            mean_before > 0.9,
            "fixture is not anisotropic: {mean_before}"
        );

        let centre = centre_of(&raw.iter().collect::<Vec<_>>());
        let flat: Vec<Vec<Vec<f32>>> = raw.iter().map(|v| vec![centred(v, &centre)]).collect();
        let after = cross_passage(&flat);
        let mean_after = after.iter().sum::<f32>() / after.len() as f32;
        assert!(
            mean_after < mean_before - 0.5,
            "common direction survived: {mean_after}"
        );
        // and an empty centre is a real no-op, so the flag can be off.
        assert_eq!(centred(&raw[0], &[]), raw[0]);
    }

    /// the shape of the real finding, on data where the answer is known: the
    /// examples differ along one dimension the negatives vary in too, so a
    /// cosine to the nearest reference is dominated by the noisy ones and a
    /// fitted direction ignores them.
    #[test]
    fn a_linear_head_separates_what_a_cosine_vote_cannot() {
        let noise = |i: usize, k: usize| ((i * 37 + k * 101) % 23) as f32 / 23.0 - 0.5;
        let positive: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                unit_vec(
                    (0..8)
                        .map(|k| if k == 0 { 2.0 } else { noise(i, k) })
                        .collect(),
                )
            })
            .collect();
        let negative: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                unit_vec(
                    (0..8)
                        .map(|k| if k == 0 { -2.0 } else { noise(i + 7, k) })
                        .collect(),
                )
            })
            .collect();

        let (w, b) = fit_logistic(&positive, &negative);
        let p: Vec<f32> = positive.iter().map(|v| probe_score(v, &w, b)).collect();
        let n: Vec<f32> = negative.iter().map(|v| probe_score(v, &w, b)).collect();
        assert!(auc(&p, &n) > 0.95, "probe scored {}", auc(&p, &n));
    }

    /// the distinction the whole rule turns on. both of these are unlike the
    /// other examples; only one of them is a bad example.
    #[test]
    fn an_unusual_aspect_is_kept_and_a_street_like_crop_is_not() {
        //            typical  typical  rear view  mislabel
        let to_examples = [0.88, 0.87, 0.60, 0.60];
        let to_street = [0.70, 0.70, 0.50, 0.72];
        let odd = odd_ones_out(&to_examples, &to_street);
        let flagged: Vec<usize> = odd
            .iter()
            .filter(|(_, _, f)| *f)
            .map(|(i, _, _)| *i)
            .collect();
        assert_eq!(
            flagged,
            vec![3],
            "expected only the street-like crop: {odd:?}"
        );
    }

    /// a rear view sits as far from the other examples as a mislabel does, so a
    /// rule keyed on that alone throws away the coverage the references exist
    /// to get.
    #[test]
    fn distance_from_the_other_examples_alone_would_flag_both() {
        let to_examples = [0.88, 0.87, 0.60, 0.60];
        assert_eq!(to_examples[2], to_examples[3]);
    }

    #[test]
    fn belonging_measures_each_passage_against_the_others_and_the_street() {
        let a = unit_vec(vec![1.0, 0.0, 0.0]);
        let b = unit_vec(vec![0.98, 0.2, 0.0]);
        let street = vec![unit_vec(vec![0.0, 1.0, 0.0]), unit_vec(vec![0.0, 0.9, 0.1])];
        let (own, out) = belonging(&[vec![a], vec![b]], &street);
        assert!(own[0] > 0.9, "two near-identical passages scored {own:?}");
        assert!(out[0] < 0.3, "unrelated street crops scored {out:?}");
        assert!(
            odd_ones_out(&own, &out).is_empty(),
            "two passages is too few to judge"
        );
    }

    #[test]
    fn farthest_point_sampling_spreads_rather_than_clustering() {
        // three nearly identical passages and one far away. picking two must
        // take the outlier, not two of the clump.
        let centres = vec![
            unit_vec(vec![1.0, 0.0, 0.0]),
            unit_vec(vec![0.99, 0.01, 0.0]),
            unit_vec(vec![0.98, 0.02, 0.0]),
            unit_vec(vec![0.0, 0.0, 1.0]),
        ];
        let none = [false; 4];
        assert!(
            choose_references(&centres, &none, 2).contains(&3),
            "the outlier was not chosen"
        );
        assert_eq!(
            choose_references(&centres, &none, 9),
            vec![0, 1, 2, 3],
            "asking for more than exists"
        );

        // and when that outlier is the one flagged as street-like, spread must
        // not go and fetch it anyway -- which is what it did before.
        let skip = [false, false, false, true];
        assert!(
            !choose_references(&centres, &skip, 2).contains(&3),
            "a flagged passage was chosen for being unlike the rest"
        );
        // unless dropping it leaves too little to choose from at all.
        assert_eq!(choose_references(&centres, &[true; 4], 2).len(), 2);
    }

    #[test]
    fn a_summary_reports_the_spread_not_just_the_mean() {
        let mut values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let s = Summary::of(&mut values).unwrap();
        assert_eq!(s.n, 5);
        assert!((s.mean - 0.3).abs() < 1e-6);
        assert!((s.median - 0.3).abs() < 1e-6);
        assert!(s.lo < s.median && s.median < s.hi);
        assert!(Summary::of(&mut []).is_none());
    }
}
