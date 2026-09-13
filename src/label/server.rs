//! the labelling page, and the train button on it.
//!
//! labelling and measuring are one loop: what to label next is decided by what
//! the last measurement was weak at. having them as two commands meant
//! labelling sessions that never got measured, so they are one page.
//!
//! **a separate server from the preview, deliberately.** it loads a 351 MB
//! embedder and embeds the whole harvest on first use, which the live pipeline
//! must never be made to do just in case somebody opens a page. `--label` is a
//! mode: no camera, no gate, no detector.
//!
//! **the human is the ground truth.** the queue is *ordered* by similarity to
//! what has been labelled so far, which is only a way of putting the likely
//! ones on screen first; nothing the embedding says is ever written as a label.

use super::{Entry, Labels, Vectors, Via, eval};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// how many of each pool to offer in one sitting. a few hundred is about an
/// hour of honest attention; past that people start clicking.
const RANDOM_OFFERED: usize = 200;
const RANKED_OFFERED: usize = 200;

/// temporal neighbours of a known example, offered so a passage can be
/// completed. they are *not* auto-labelled: the crops either side of a real
/// sighting routinely include a different vehicle in the same frames, and one
/// of those in the reference set is exactly the poisoning this all guards
/// against. each one is still a human decision.
const NEIGHBOURS_OFFERED: usize = 60;

/// how many of the classifier's own verdicts to put up at once.
///
/// generous, because these are the crops most worth deciding: `split_negatives`
/// draws what fired ahead of every other tier. the subject is well under a
/// percent of the harvest, so the cap is rarely reached at all.
const VERDICTS_OFFERED: usize = 200;

/// how many labelled crops to put up for a second look, in each direction.
const SECOND_LOOK_OFFERED: usize = 50;

const PAGE: &str = include_str!("index.html");

pub struct Session {
    /// what `--harvest` named: one directory of crops, or a tree of sets.
    root: PathBuf,
    /// the crop directories under it, each with its own labels and vector
    /// cache beside it. apart from the state so a crop is served while a pass
    /// over the harvest holds that lock to embed.
    sets: Mutex<Vec<PathBuf>>,
    /// the embedder, for a crop that arrived after the page was opened.
    model: PathBuf,
    /// where `trained/` is written when a row of the report is saved.
    references: PathBuf,
    /// `[track] confirm_m` and `confirm_n`, which the report counts under.
    confirm: (u32, u32),
    state: Mutex<State>,
    subject: String,
}

struct State {
    labels: Labels,
    cache: Vectors,
    names: Vec<String>,
    queue: Vec<Offered>,
    /// what the last sweep wrote, so one click can take it back.
    last_sweep: Vec<String>,
    /// crops of the passages the last measurement held out as unlike the class,
    /// offered for a second look. already labelled, so nothing else would put
    /// them back on screen.
    review: Vec<(String, f32)>,
    /// what the classifier fired on at the last measurement, strongest first.
    alerts: Vec<(String, f32)>,
    /// the crops stage two named, read off their own filenames. unlike `alerts`
    /// this needs no measurement, so a harvest whose every verdict is wrong --
    /// which is how the first evening went -- can still be worked through.
    verdicts: Vec<String>,
    /// the margin those were taken at, for the page to show.
    alert_margin: f32,
    /// the last measurement, so a row of its curve is saved exactly as shown.
    report: Option<eval::Report>,
}

#[derive(Clone)]
struct Offered {
    name: String,
    via: Via,
    score: Option<f32>,
}

impl Session {
    pub fn open(
        harvest: &Path,
        model: &Path,
        subject: &str,
        references: &Path,
        confirm: (u32, u32),
    ) -> Result<Self> {
        let roots = [harvest.to_path_buf()];
        // the walk `--train` makes, so a tree of sets is labelled as the one
        // harvest it is trained as.
        let (labels, cache, names) = super::loaded(&roots, model)?;
        let sets = super::sets_under(&roots);
        // what is already known, so a session that is about to measure nothing
        // says so before the browser does. a subject named with a typo is the
        // common case and looks identical to one with no examples yet.
        anyhow::ensure!(
            !cache.is_empty(),
            "no crop could be embedded: check that {} is a readable embedder",
            model.display()
        );
        let known = labels.named(subject).len();
        tracing::info!(
            "{} crops, {} embedded, {} labelled {subject}; the file holds {}",
            names.len(),
            cache.len(),
            known,
            if labels.subjects().is_empty() {
                "no subjects yet".to_string()
            } else {
                labels.subjects().join(", ")
            }
        );
        let verdicts = verdicts(&sets, subject);
        if !verdicts.is_empty() {
            tracing::info!(
                "{} crops carry a {subject} verdict; the verdicts tab is where they are rejected",
                verdicts.len()
            );
        }
        let review = second_look(&labels, &cache, &names, subject);
        let doubted = review
            .iter()
            .filter(|(n, _)| labels.entries.get(n).is_some_and(|e| e.truth == subject))
            .count();
        tracing::info!(
            "second look: {doubted} labelled {subject} voting most with the negatives, \
             {} labelled {} voting most with {subject}",
            review.len() - doubted,
            super::NEGATIVE
        );
        let queue = build_queue(&names, &labels, &cache, subject, &review, &[], &verdicts);
        Ok(Self {
            root: harvest.to_path_buf(),
            sets: Mutex::new(sets),
            model: model.to_path_buf(),
            references: references.to_path_buf(),
            confirm,
            subject: subject.to_string(),
            state: Mutex::new(State {
                labels,
                cache,
                names,
                queue,
                last_sweep: Vec::new(),
                review,
                alerts: Vec::new(),
                verdicts,
                alert_margin: 0.0,
                report: None,
            }),
        })
    }

    /// serve the page, taking in new crops of the harvest every `every`.
    ///
    /// a pass is a directory walk plus whatever the cache has never seen, and the
    /// embedder is the slow part of noticing -- so the noticing happens on a clock
    /// rather than when somebody presses something. what it does *not* do is rebuild
    /// the pool, which is the one thing that would be rude to do under an open page.
    pub fn serve(self, addr: &str, every: std::time::Duration) -> Result<()> {
        let server = tiny_http::Server::http(addr)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("binding the labelling server to {addr}"))?;
        let shown = if addr.starts_with("0.0.0.0") {
            addr.replace("0.0.0.0", "localhost")
        } else {
            addr.to_string()
        };
        let often = if every.is_zero() {
            "when asked".to_string()
        } else {
            format!("every {}s", every.as_secs())
        };
        tracing::info!(
            "labelling {} at http://{shown}/, taking in new crops {often}",
            self.subject
        );
        std::thread::scope(|runs| {
            if !every.is_zero() {
                runs.spawn(|| {
                    loop {
                        std::thread::sleep(every);
                        match self.ingest() {
                            Err(e) => tracing::warn!("could not take in the harvest: {e:#}"),
                            // silence is the right answer to a quiet street: a line every
                            // minute is a log nobody reads, and a pass redraws nothing on
                            // the page, so there is nothing else it changed.
                            Ok(0) => {}
                            Ok(fresh) => tracing::info!("{fresh} new crops taken in"),
                        }
                    }
                });
            }
            for request in server.incoming_requests() {
                if let Err(e) = self.handle(request) {
                    tracing::warn!("request failed: {e:#}");
                }
            }
        });
        Ok(())
    }

    fn handle(&self, mut request: tiny_http::Request) -> Result<()> {
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or("");
        match (request.method(), path) {
            (tiny_http::Method::Get, "/") => {
                let page = PAGE.replace("__SUBJECT__", &self.subject);
                respond(request, "text/html; charset=utf-8", page.into_bytes())
            }
            (tiny_http::Method::Get, "/favicon.png") => {
                // the same mark the preview serves. these run side by side on
                // two ports and used to be two identical blank tabs.
                respond(request, "image/png", crate::preview::FAVICON.to_vec())
            }
            (tiny_http::Method::Get, "/queue") => {
                let state = self.state.lock().unwrap();
                respond(
                    request,
                    "application/json",
                    self.queue_json(&state).into_bytes(),
                )
            }
            (tiny_http::Method::Get, p) if p.starts_with("/crop/") => {
                self.serve_crop(request, &p[6..])
            }
            (tiny_http::Method::Post, "/label") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body)?;
                let ok = self.set_label(&body)?;
                respond(request, "application/json", ok.into_bytes())
            }
            (tiny_http::Method::Post, "/sweep") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body)?;
                let done = self.sweep(body.trim())?;
                respond(request, "application/json", done.into_bytes())
            }
            (tiny_http::Method::Post, "/undo") => {
                let done = self.undo()?;
                respond(request, "application/json", done.into_bytes())
            }
            (tiny_http::Method::Post, "/refresh") => {
                respond(request, "application/json", self.refresh()?.into_bytes())
            }
            (tiny_http::Method::Post, "/train") => {
                let body = self.train();
                respond(request, "application/json", body.into_bytes())
            }
            (tiny_http::Method::Post, "/save") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body)?;
                let done = self.save(body.trim());
                respond(request, "application/json", done.into_bytes())
            }
            _ => {
                let _ = request.respond(tiny_http::Response::empty(404));
                Ok(())
            }
        }
    }

    /// serve one crop by name.
    ///
    /// resolved against the crop directories by its file name only, so a
    /// crafted path cannot reach outside them. this serves a whole harvest to a
    /// browser, so that guard is the whole of the security story here.
    fn serve_crop(&self, request: tiny_http::Request, name: &str) -> Result<()> {
        let safe = Path::new(name)
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|n| crate::harvest::parse_name(n).is_some());
        let held = |n| self.holding(n).into_iter().next().map(|d| d.join(n));
        let Some(file) = safe.and_then(held) else {
            let _ = request.respond(tiny_http::Response::empty(404));
            return Ok(());
        };
        respond(request, "image/jpeg", std::fs::read(file)?)
    }

    /// the sets holding a crop of this name, which is usually one.
    fn holding(&self, name: &str) -> Vec<PathBuf> {
        let sets = self.sets.lock().unwrap();
        sets.iter()
            .filter(|d| d.join(name).is_file())
            .cloned()
            .collect()
    }

    /// write what was just decided about `touched` beside the crops it names.
    ///
    /// **each set keeps its own `labels.txt`**, which is what lets a set travel
    /// with its judgements, so a label made across a tree goes home rather than
    /// into one file for the lot. a crop two sets share is written in both: one
    /// answer in one and none in the other becomes two answers the day the
    /// other is labelled, and a tree whose sets disagree does not open. a crop
    /// no set holds any more -- evicted since it was offered -- is filed with
    /// the first, where a single harvest has always filed it.
    ///
    /// only the touched rows are written over what the file holds, so saying
    /// the same thing twice leaves it byte for byte as it was, and a label the
    /// live page made in between is kept and adopted (`save_merged`).
    fn save_labels(&self, state: &mut State, touched: &[String], dropped: &[String]) -> Result<()> {
        let first = self.sets.lock().unwrap().first().cloned();
        let mut homes: std::collections::BTreeMap<PathBuf, Labels> = Default::default();
        for name in touched {
            let mut held = self.holding(name);
            if held.is_empty() {
                held.extend(first.clone());
            }
            for dir in held {
                let part = homes.entry(dir).or_default();
                if let Some(entry) = state.labels.entries.get(name) {
                    part.entries.insert(name.clone(), entry.clone());
                }
            }
        }
        for (dir, mut part) in homes {
            part.save_merged(&super::paths(&dir).0, dropped)?;
            state.labels.entries.extend(part.entries);
        }
        Ok(())
    }

    /// `<name> <truth>` or `<name> <truth> <via>`; an unknown truth clears it.
    fn set_label(&self, body: &str) -> Result<String> {
        let mut parts = body.split_whitespace();
        let (Some(name), Some(truth)) = (parts.next(), parts.next()) else {
            anyhow::bail!("expected `<name> <truth>`");
        };
        let mut state = self.state.lock().unwrap();
        // an existing entry keeps the pool it was drawn from. re-deciding a
        // crop says nothing about how it was sampled, and letting the review
        // pool overwrite that would take a crop out of the random sample the
        // false positive rate is measured on.
        let via = state
            .labels
            .entries
            .get(name)
            .map(|e| e.via)
            .or_else(|| parts.next().and_then(Via::parse_public))
            .or_else(|| state.queue.iter().find(|o| o.name == name).map(|o| o.via))
            .unwrap_or(Via::Seed);
        let dropped = match truth {
            "none" => {
                state.labels.entries.remove(name);
                vec![name.to_string()]
            }
            _ => {
                state
                    .labels
                    .entries
                    .insert(name.to_string(), Entry::new(truth, via));
                Vec::new()
            }
        };
        self.save_labels(&mut state, &[name.to_string()], &dropped)?;
        Ok(format!(
            r#"{{"ok":true,"labelled":{}}}"#,
            state.labels.entries.len()
        ))
    }

    /// mark every *unlabelled* crop in one pool as not the subject.
    ///
    /// the subject is well under a percent of the harvest, so a random
    /// screenful is almost all negative and clicking each one adds nothing but
    /// wear. this is the same judgement, made once.
    ///
    /// two things it will not do. it never overwrites a decision already made,
    /// so the yesses picked out first survive it. and it takes one pool at a
    /// time rather than the page: the ranked pool is where the examples
    /// actually are, and sweeping it along with the random one would bury them.
    fn sweep(&self, pool: &str) -> Result<String> {
        let Some(via) = Via::parse_public(pool) else {
            anyhow::bail!("unknown pool {pool:?}");
        };
        let mut state = self.state.lock().unwrap();
        let fresh = sweepable(&state.queue, &state.labels, via);
        for name in &fresh {
            state
                .labels
                .entries
                .insert(name.clone(), Entry::new(super::NEGATIVE, via));
        }
        self.save_labels(&mut state, &fresh, &[])?;
        let n = fresh.len();
        // the names, not just the count: the page marks exactly these in place
        // rather than rebuilding the queue, which would throw away the scroll
        // position and drop whoever pressed it back at the top of the harvest.
        let body = format!(r#"{{"swept":{n},"names":{}}}"#, strings_json(&fresh));
        state.last_sweep = fresh;
        tracing::info!(
            "swept {n} unlabelled crops in the {pool} pool to {}",
            super::NEGATIVE
        );
        Ok(body)
    }

    /// put back exactly what the last sweep wrote.
    ///
    /// a sweep is one click that writes hundreds of labels, so it needs one
    /// click that takes them back. only the crops that sweep created are
    /// removed, so a yes picked out since is left alone.
    fn undo(&self) -> Result<String> {
        let mut state = self.state.lock().unwrap();
        let undone = std::mem::take(&mut state.last_sweep);
        for name in &undone {
            state.labels.entries.remove(name);
        }
        self.save_labels(&mut state, &undone, &undone)?;
        Ok(format!(
            r#"{{"undone":{},"names":{}}}"#,
            undone.len(),
            strings_json(&undone)
        ))
    }

    /// the queue, and what is left outside it.
    ///
    /// the counts are the point as much as the items. the random pool is a
    /// prefix of a fixed order and a crop stays in it until it is decided, so
    /// skipping one and pressing train brings it straight back -- which looks
    /// like the queue is stuck unless the totals are on screen saying otherwise.
    /// the queue as the page draws it, and the totals beside it.
    ///
    /// takes the state rather than locking it itself, because the handler that has
    /// just rebuilt the queue is holding the lock already, and this page's mutex is
    /// not one you can take twice.
    fn queue_json(&self, state: &State) -> String {
        let labelled = state.labels.entries.len();
        let subject = state.labels.named(&self.subject).len();
        let counts = format!(
            r#""total":{},"labelled":{labelled},"remaining":{},"subject":{subject},"alert_margin":{:.4}"#,
            state.names.len(),
            state.names.len().saturating_sub(labelled),
            state.alert_margin,
        );
        let rows: Vec<String> = state
            .queue
            .iter()
            .map(|o| {
                let truth = state
                    .labels
                    .entries
                    .get(&o.name)
                    .map(|e| format!(r#""{}""#, e.truth))
                    .unwrap_or_else(|| "null".into());
                let score = o
                    .score
                    .map(|s| format!("{s:.3}"))
                    .unwrap_or_else(|| "null".into());
                // what stage two called it, parsed here rather than in the
                // browser so the crop name has one reader rather than two.
                let verdict = crate::harvest::parse_name(&o.name)
                    .and_then(|s| s.subject)
                    .map(|s| format!(r#""{}""#, escape(&s)))
                    .unwrap_or_else(|| "null".into());
                format!(
                    r#"{{"n":"{}","via":"{}","score":{score},"truth":{truth},"verdict":{verdict}}}"#,
                    o.name,
                    o.via.slug()
                )
            })
            .collect();
        format!("{{{counts},\"items\":[{}]}}", rows.join(","))
    }

    /// take in whatever the harvest has gained, on the clock.
    fn ingest(&self) -> Result<usize> {
        let mut state = self.state.lock().unwrap();
        self.take_in(&mut state)
    }

    /// the walk, the embeds and the two files, with the lock already held.
    ///
    /// the count is what the cache had never seen rather than the difference in the
    /// harvest's size: the disk budget evicts the oldest crops, so an evening of
    /// harvesting can leave the total exactly where it was and the whole of it new.
    ///
    /// **the pool is not rebuilt here**, which is what lets this run unattended. which
    /// pool a crop was offered from is the provenance of its label, and answering a
    /// click against a pool that changed underneath the page would quietly misfile it.
    fn take_in(&self, state: &mut State) -> Result<usize> {
        // the listing is cached for thirty seconds, and the crops worth taking in are
        // the ones that arrived inside them: the pipeline writes them from another
        // process, so nothing would have bumped the generation this one reads.
        let known = self.sets.lock().unwrap().clone();
        for dir in known.iter().chain([&self.root]) {
            crate::harvest::forget(dir);
        }
        // walked again, because `--prepare` cuts a new set into the tree while the
        // page is open. a set new to this session brings the vectors it already has,
        // or its crops would be embedded a second time into the cache beside them.
        let sets = super::sets_under(std::slice::from_ref(&self.root));
        let fresh: Vec<PathBuf> = sets
            .iter()
            .filter(|d| !known.contains(d))
            .cloned()
            .collect();
        state.cache.adopt(super::merged(&fresh)?)?;
        let before = state.cache.len();
        state.names = super::scan(&sets, &self.model, &mut state.cache)?;
        // reloaded rather than kept, because the live page labels the same file over
        // the other port and an answer given there should not wait for a measurement
        // to be counted.
        state.labels = super::labels_of(&sets)?;
        state.verdicts = verdicts(&sets, &self.subject);
        *self.sets.lock().unwrap() = sets;
        Ok(state.cache.len() - before)
    }

    /// offer another screenful, **without measuring anything**.
    ///
    /// the queue is a bounded sample of a harvest that keeps growing while somebody
    /// labels it, and until now the only way back to the rest of it was the train
    /// button, which runs the embedder, measures the whole label file and reports a
    /// curve -- all of which is worth doing, and none of it asked for by "show me
    /// more".
    fn refresh(&self) -> Result<String> {
        let mut state = self.state.lock().unwrap();
        let fresh = self.take_in(&mut state)?;
        state.queue = build_queue(
            &state.names,
            &state.labels,
            &state.cache,
            &self.subject,
            &state.review,
            &state.alerts,
            &state.verdicts,
        );
        tracing::info!(
            "{fresh} new crops, {} in the harvest, {} unlabelled",
            state.names.len(),
            state.names.len() - state.labels.entries.len(),
        );
        Ok(self.queue_json(&state))
    }

    /// measure what the labels are worth so far.
    ///
    /// **the split is not offered as a choice.** it is by passage, automatic,
    /// and the reference side never overlaps the eval side -- choosing which
    /// examples to test on is exactly how a 100% recall gets reported, so the
    /// button does not expose it.
    fn train(&self) -> String {
        let mut state = self.state.lock().unwrap();
        let opts = eval::Options {
            subject: self.subject.clone(),
            confirm_m: self.confirm.0,
            confirm_n: self.confirm.1,
            ..Default::default()
        };
        let report = eval::measure(&state.labels, &state.cache, &state.names, &opts);
        // the queue is rebuilt from the fresh labels, so the next screenful is
        // ordered by what this measurement just learned rather than by what was
        // known when the page was opened.
        // recomputed rather than taken from the report: `odd_crops` names whole
        // passages, and what is worth a second look is one crop.
        state.review = second_look(&state.labels, &state.cache, &state.names, &self.subject);
        state.alerts = report.alerts.clone();
        state.alert_margin = report.alert_margin;
        // re-read rather than reused: the pipeline goes on harvesting while the
        // page is open, and a verdict that arrived since is exactly the one
        // worth deciding.
        state.verdicts = verdicts(&self.sets.lock().unwrap(), &self.subject);
        state.queue = build_queue(
            &state.names,
            &state.labels,
            &state.cache,
            &self.subject,
            &state.review,
            &state.alerts,
            &state.verdicts,
        );
        let json = report_json(&report);
        state.report = Some(report);
        json
    }

    /// write `trained/<subject>/` at a margin a person picked off the report.
    ///
    /// **only a row of the curve that was shown.** a margin means something only
    /// against the measurement it was read off, so there must have been one, and
    /// the numbers saved beside it are that row's rather than recomputed. the
    /// pipeline reads `trained/` when it starts, so it takes effect on a restart.
    fn save(&self, margin: &str) -> String {
        let error = |e: String| format!(r#"{{"error":"{}"}}"#, escape(&e));
        let Ok(margin) = margin.parse::<f32>() else {
            return error(format!("{margin:?} is not a margin"));
        };
        let state = self.state.lock().unwrap();
        let Some(report) = state.report.as_ref() else {
            return error("press train first: a margin is read off a measurement".into());
        };
        let Some(row) = report.row_at(margin) else {
            return error(format!("{margin} is not a row of the measured curve"));
        };
        match super::write_trained(report, &state.cache, &self.references, Some(row)) {
            Ok(()) => format!(
                r#"{{"ok":true,"margin":{:.3},"passages":{},"passage_recall":{:.4},"fpr":{:.5},"wrote":"{}"}}"#,
                row.margin,
                row.passages,
                row.passage_recall,
                row.fpr,
                escape(&self.references.join(&self.subject).display().to_string())
            ),
            Err(e) => error(format!("{e:#}")),
        }
    }
}

impl Via {
    fn parse_public(s: &str) -> Option<Self> {
        match s {
            "random" => Some(Via::Random),
            "ranked" => Some(Via::Ranked),
            "neighbour" | "seed" => Some(Via::Seed),
            "review" => Some(Via::Review),
            "alert" => Some(Via::Alert),
            _ => None,
        }
    }
}

/// labelled crops whose label the classifier's own vote disputes, both ways.
///
/// **computed from the labels and the vectors, not out of a measurement** --
/// the same argument the verdict pool rests on: both exist when the session
/// opens, so the pool is there before `train` has been pressed.
///
/// first the crops labelled this subject that vote most with the negatives,
/// then the crops labelled `other` that vote most with this subject, then the
/// ones marked `unclear` that vote with this subject at all -- up to
/// `SECOND_LOOK_OFFERED` of each and always, with no line to cross. a ranking
/// always has a top, and on a clean set that top is correct labels; that is the
/// price of never hiding a mislabel behind a cutoff.
///
/// **`unclear` is the third direction because nothing else ever offers it
/// back.** it means a person looked and could not tell, so the measurement
/// excludes it from both classes -- counting it either way would invent the
/// answer they could not give -- and it then sits in `labels.txt` forever. what
/// changed since is the classifier's vote, and the crops it now votes for are
/// exactly the ones whose open question it has an opinion about.
fn second_look(
    labels: &Labels,
    cache: &Vectors,
    names: &[String],
    subject: &str,
) -> Vec<(String, f32)> {
    let labelled = |keep: &dyn Fn(&str) -> bool| -> Vec<(String, Vec<f32>)> {
        names
            .iter()
            .filter(|n| labels.entries.get(*n).is_some_and(|e| keep(&e.truth)))
            .filter_map(|n| cache.get(n).map(|v| (n.clone(), v.clone())))
            .collect()
    };
    let vectors = |set: &[(String, Vec<f32>)]| -> Vec<Vec<f32>> {
        set.iter().map(|(_, v)| v.clone()).collect()
    };
    let mine = labelled(&|t| t == subject);
    // everything a person decided is not this subject, as `split_negatives`
    // counts it: `other` and every other subject. `unclear` is in neither half.
    let against = labelled(&|t| t != subject && t != super::UNCLEAR);
    // only `other` is offered back, though. a crop labelled another subject
    // belongs to that subject's session: a click here would overwrite `waymo`
    // with this subject or with `other`, recording something false.
    let others = labelled(&|t| t == super::NEGATIVE);

    let unclear = labelled(&|t| t == super::UNCLEAR);

    let mut look = eval::disputed(
        &mine,
        &mine,
        &vectors(&against),
        &format!("crops labelled {subject}"),
    );
    look.truncate(SECOND_LOOK_OFFERED);
    let mut back = eval::disputed(
        &others,
        &against,
        &vectors(&mine),
        &format!("crops labelled {}", super::NEGATIVE),
    );
    back.truncate(SECOND_LOOK_OFFERED);
    // scored against the street the same way `other` crops are, so a positive
    // score means the same thing on both: it votes with this subject.
    let mut parked = eval::disputed(
        &unclear,
        &against,
        &vectors(&mine),
        &format!("crops marked {}", super::UNCLEAR),
    );
    parked.truncate(SECOND_LOOK_OFFERED);
    look.extend(back);
    look.extend(parked);
    look
}

/// the crops stage two named as this subject, newest first.
///
/// taken off the filenames rather than out of a measurement. the name is the
/// only index the harvest has, and it records what actually fired on the
/// deployment rather than what the current references would fire on now -- so
/// these are on screen before anything has been labelled, which is the state a
/// harvest of false positives is in.
///
/// **filtered to this session's subject.** `other` means "none of the subjects
/// we know about", so marking a waymo verdict `other` in a go-4 session would
/// record something false about the crop rather than about the verdict.
fn verdicts(sets: &[PathBuf], subject: &str) -> Vec<String> {
    let mut named: Vec<crate::harvest::Saved> = sets
        .iter()
        .flat_map(|dir| {
            crate::harvest::page(dir, None, usize::MAX, crate::harvest::Want::Recognised)
        })
        .filter(|s| s.subject.as_deref() == Some(subject))
        .collect();
    // newest first across the tree, and a crop two sets share offered once.
    named.sort_by(|a, b| b.at_millis.cmp(&a.at_millis).then(a.name.cmp(&b.name)));
    named.dedup_by(|a, b| a.name == b.name);
    named
        .into_iter()
        .map(|s| s.name)
        .take(VERDICTS_OFFERED)
        .collect()
}

/// what to put on screen: the random sample, then temporal neighbours of known
/// examples, then whatever most resembles them.
fn build_queue(
    names: &[String],
    labels: &Labels,
    cache: &Vectors,
    subject: &str,
    review: &[(String, f32)],
    alerts: &[(String, f32)],
    verdicts: &[String],
) -> Vec<Offered> {
    let unlabelled: Vec<String> = names
        .iter()
        .filter(|n| !labels.entries.contains_key(*n))
        .cloned()
        .collect();

    let mut queue: Vec<Offered> = Vec::new();

    let mut by_hash = unlabelled.clone();
    by_hash.sort_by_key(|n| super::hash_rank(n, 0));
    let random: Vec<String> = by_hash.iter().take(RANDOM_OFFERED).cloned().collect();

    let mut taken: std::collections::BTreeSet<String> = random.iter().cloned().collect();
    queue.extend(random.into_iter().map(|name| Offered {
        name,
        via: Via::Random,
        score: None,
    }));

    let known = labels.named(subject);
    for name in neighbours(names, &known, &taken)
        .into_iter()
        .take(NEIGHBOURS_OFFERED)
    {
        if labels.entries.contains_key(&name) {
            continue;
        }
        taken.insert(name.clone());
        queue.push(Offered {
            name,
            via: Via::Seed,
            score: None,
        });
    }

    let references: Vec<Vec<f32>> = known.iter().filter_map(|n| cache.get(n).cloned()).collect();
    // the second look goes last, after everything still undecided. these crops
    // already have an answer, so they are not what the session is for -- they
    // are worth revisiting once the undecided ones in front of them are done,
    // not worth interrupting that to argue about.
    // the score rides along: how much more the crop votes with the opposite
    // class than with its own. it is the reason the crop is on screen, so it is
    // on the tile rather than only in the ordering.
    let second_look: Vec<Offered> = review
        .iter()
        .filter(|(n, _)| labels.entries.contains_key(n))
        .map(|(name, margin)| Offered {
            name: name.clone(),
            via: Via::Review,
            score: Some(*margin),
        })
        .collect();
    if references.len() >= crate::classify::VOTE_K {
        // **the other half of what opening a session costs.** every undecided
        // crop is scored against every known example, so this grows on both
        // axes as labelling goes on: the harvest fills up and the reference
        // side of the comparison gets longer with it.
        let waiting: Vec<&String> = unlabelled.iter().filter(|n| !taken.contains(*n)).collect();
        let mut progress =
            super::Progress::new("ranking", "ranked", "undecided crops", waiting.len());
        let mut ranked: Vec<(f32, String)> = waiting
            .into_iter()
            .filter_map(|n| {
                progress.tick();
                let v = cache.get(n)?;
                Some((
                    crate::classify::top_k_mean(v, &references, crate::classify::VOTE_K),
                    n.clone(),
                ))
            })
            .collect();
        progress.finished();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        queue.extend(
            ranked
                .into_iter()
                .take(RANKED_OFFERED)
                .map(|(s, name)| Offered {
                    name,
                    via: Via::Ranked,
                    score: Some(s),
                }),
        );
    }
    queue.extend(second_look);
    // what the classifier fired on. shown in its own view rather than mixed
    // into the labelling queue: these are the output, not the backlog, and most
    // of them already have a decision.
    //
    // the verdicts lead, because they are the stronger evidence of the two:
    // they really did fire, on the deployment, and they are on disk whether or
    // not anything has been measured. then whatever the references as they
    // stand would fire on now.
    //
    // a verdict already in another pool is left there as well as here. the
    // random pool is a prefix of a fixed order and a false positive rate is
    // measured on it, so quietly removing the crops that fired would measure
    // the rate at which everything *else* fires.
    queue.extend(verdicts.iter().map(|name| Offered {
        name: name.clone(),
        via: Via::Alert,
        score: None,
    }));
    queue.extend(
        alerts
            .iter()
            .filter(|(name, _)| !verdicts.contains(name))
            .map(|(name, margin)| Offered {
                name: name.clone(),
                via: Via::Alert,
                score: Some(*margin),
            }),
    );
    queue
}

/// what a sweep of one pool would write.
///
/// the two things it must never do, kept here where they can be tested rather
/// than only observed: it takes **one pool**, so the ranked pool -- which is
/// where the examples actually are -- is not buried along with the random one;
/// and it takes only crops with **no decision yet**, so the yesses picked out
/// before sweeping survive it.
fn sweepable(queue: &[Offered], labels: &Labels, via: Via) -> Vec<String> {
    queue
        .iter()
        .filter(|o| o.via == via && !labels.entries.contains_key(&o.name))
        .map(|o| o.name.clone())
        .collect()
}

/// unlabelled crops within a passage gap of a known example.
///
/// these are what completes a passage, and a passage is the unit of an example,
/// so they are worth more per click than anything the ranker finds.
fn neighbours(
    all: &[String],
    known: &[String],
    taken: &std::collections::BTreeSet<String>,
) -> Vec<String> {
    let stamp = |n: &String| crate::harvest::parse_name(n).map(|s| s.at_millis);
    let anchors: Vec<u128> = known.iter().filter_map(stamp).collect();
    let mut out: Vec<String> = all
        .iter()
        .filter(|n| !taken.contains(*n) && !known.contains(*n))
        .filter(|n| {
            stamp(n).is_some_and(|t| {
                anchors
                    .iter()
                    .any(|a| t.abs_diff(*a) <= super::PASSAGE_GAP_MS)
            })
        })
        .cloned()
        .collect();
    out.sort();
    out
}

fn summary_json(s: Option<eval::Summary>) -> String {
    match s {
        Some(s) => format!(
            r#"{{"mean":{:.4},"median":{:.4},"lo":{:.4},"hi":{:.4},"n":{}}}"#,
            s.mean, s.median, s.lo, s.hi, s.n
        ),
        None => "null".into(),
    }
}

/// the passages least like the rest, so the page can offer them for a second
/// look. a mislabel and a crop the subject is merely present in look the same
/// from here, and both are poor references.
fn odd_json(rows: &[(String, f32, bool)]) -> String {
    let items: Vec<String> = rows
        .iter()
        .map(|(name, mean, flagged)| {
            format!(
                r#"{{"n":"{}","mean":{mean:.4},"excluded":{flagged}}}"#,
                escape(name)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn strings_json(items: &[String]) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|s| format!(r#""{}""#, escape(s)))
        .collect();
    format!("[{}]", quoted.join(","))
}

/// the minimum needed for the strings this emits: label text and crop names,
/// which the harvester already sanitises to `[A-Za-z0-9._-]`. shared with the
/// preview, which emits configured directory names on the same terms.
pub(crate) fn escape(s: &str) -> String {
    s.replace('\\', r"\\").replace('"', "\\\"")
}

fn report_json(r: &eval::Report) -> String {
    if let Some(e) = &r.error {
        return format!(r#"{{"error":"{}"}}"#, escape(e));
    }
    let passages: Vec<String> = r
        .passages
        .iter()
        .map(|p| {
            format!(
                r#"{{"first":"{}","crops":{},"span_s":{:.1},"reference":{},"best":{:.4},"px":{}}}"#,
                p.first, p.crops, p.span_s, p.reference, p.best, p.px
            )
        })
        .collect();
    let rows = |curve: &[eval::Row]| -> String {
        curve
            .iter()
            .map(|s| {
                format!(
                    r#"{{"margin":{:.3},"passages":{},"passage_recall":{:.4},"crop_recall":{:.4},"fpr":{:.5},"per_hour":{:.1},"alerts_per_hour":{:.1}}}"#,
                    s.margin,
                    s.passages,
                    s.passage_recall,
                    s.crop_recall,
                    s.fpr,
                    s.per_hour,
                    s.alerts_per_hour
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    // the curve the operating point is chosen from and the margin chosen, from
    // the server, so the row the page highlights is the row `--train` writes.
    let curve = if r.sweep_near.is_empty() {
        "all"
    } else {
        "near"
    };
    let operating = r
        .operating_point()
        .map_or("null".to_string(), |row| format!("{:.3}", row.margin));
    format!(
        r#"{{"subject":"{}","n_crops":{},"n_labelled":{},"n_negatives":{},"per_hour":{:.1},"gap":{:.4},"self_self":{},"other_other":{},"self_other":{},"passages":[{}],"sweep":[{}],"curve":"{curve}","decision":[{decision}],"operating_margin":{operating},"confirm_m":{confirm_m},"confirm_n":{confirm_n},"fatal":{},"caveats":{},"screen":{},"leave_one_out":{},"odd":{odd},"alerts":{alerts},"alert_margin":{margin:.4},"go4":{{"self":{GO4_SELF},"other":{GO4_OTHER},"car":{CAR_SELF}}}}}"#,
        escape(&r.subject),
        r.n_crops,
        r.n_labelled,
        r.n_negatives,
        r.per_hour,
        r.gap(),
        summary_json(r.self_self),
        summary_json(r.other_other),
        summary_json(r.self_other),
        passages.join(","),
        rows(&r.sweep),
        strings_json(&r.fatal),
        strings_json(&r.caveats),
        strings_json(&r.screen),
        r.leave_one_out,
        decision = rows(r.decision_curve()),
        confirm_m = r.confirm_m,
        confirm_n = r.confirm_n,
        odd = odd_json(&r.odd),
        alerts = r.alerts.len(),
        margin = r.alert_margin,
        GO4_SELF = eval::GO4_SELF,
        GO4_OTHER = eval::GO4_OTHER,
        CAR_SELF = eval::CAR_SELF,
    )
}

fn respond(request: tiny_http::Request, kind: &str, body: Vec<u8>) -> Result<()> {
    let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], kind.as_bytes())
        .map_err(|_| anyhow::anyhow!("bad content type {kind}"))?;
    request
        .respond(tiny_http::Response::from_data(body).with_header(header))
        .context("writing response")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(millis: u128) -> String {
        format!("{millis}_car_085_200x200.jpg")
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("metermate-server-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// **a renamed id leaves a half-dead page.** the lookup returns null, every
    /// use of it is guarded, and a count or a button quietly stops working with
    /// nothing in the log and nothing on screen to say so.
    ///
    /// an id ending in a dash is built per pool, so it is checked against where
    /// it is written rather than against a literal in the markup.
    #[test]
    fn every_element_the_page_looks_up_exists() {
        let missing: Vec<&str> = PAGE
            .split("getElementById(\"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .filter(|id| {
                let literal = match id.ends_with('-') {
                    true => format!("id=\"{id}"),
                    false => format!("id=\"{id}\""),
                };
                !PAGE.contains(&literal) && !PAGE.contains(&format!(".id = \"{id}"))
            })
            .collect();
        assert!(missing.is_empty(), "the page looks up {missing:?}");
    }

    /// **the page highlights what `--train` writes.** the operating point is
    /// chosen over the near curve when there is one, and the page used to pick
    /// its own "best" over every crop -- so the highlighted row and the saved
    /// margin could differ with nothing saying so.
    #[test]
    fn the_report_carries_the_curve_and_margin_train_chooses() {
        let row = |margin: f32, passage_recall: f32, fpr: f32| eval::Row {
            margin,
            passage_recall,
            fpr,
            ..Default::default()
        };
        let report = eval::Report {
            sweep: vec![row(-0.01, 1.0, 0.02), row(0.0, 0.9, 0.01)],
            sweep_near: vec![row(-0.01, 0.9, 0.005), row(0.0, 0.5, 0.001)],
            ..Default::default()
        };
        assert_eq!(report.operating_point().map(|r| r.margin), Some(-0.01));
        let json = report_json(&report);
        assert!(json.contains(r#""curve":"near""#), "{json}");
        assert!(json.contains(r#""operating_margin":-0.010"#), "{json}");
        assert!(json.contains(r#""decision":[{"margin":-0.010"#), "{json}");
    }

    #[test]
    fn a_labelled_crop_is_never_offered_again() {
        let names: Vec<String> = (0..10)
            .map(|i| name(1_789_250_000_000 + i * 60_000))
            .collect();
        let mut labels = Labels::default();
        labels
            .entries
            .insert(names[0].clone(), Entry::new("waymo", Via::Random));
        let cache = Vectors::load(Path::new("/nonexistent")).unwrap();
        let queue = build_queue(&names, &labels, &cache, "waymo", &[], &[], &[]);
        assert!(!queue.iter().any(|o| o.name == names[0]));
        assert_eq!(queue.len(), names.len() - 1);
    }

    /// the crops either side of a known example are what completes a passage,
    /// and a passage is the unit of an example.
    #[test]
    fn the_neighbours_of_a_known_example_are_offered_for_review() {
        let base = 1_789_250_000_000u128;
        // three crops seconds apart, and one an hour later.
        let names: Vec<String> = [0u128, 2_000, 4_000, 3_600_000]
            .iter()
            .map(|d| name(base + d))
            .collect();
        let known = vec![names[0].clone()];
        let found = neighbours(&names, &known, &Default::default());
        assert_eq!(found, vec![names[1].clone(), names[2].clone()]);
    }

    /// they are offered, never applied: the frames around a real sighting
    /// routinely hold a different vehicle, and one of those in the reference
    /// set is the poisoning this all exists to prevent.
    #[test]
    fn neighbours_are_offered_rather_than_labelled() {
        let base = 1_789_250_000_000u128;
        let names: Vec<String> = [0u128, 2_000].iter().map(|d| name(base + d)).collect();
        let mut labels = Labels::default();
        labels
            .entries
            .insert(names[0].clone(), Entry::new("waymo", Via::Seed));
        let cache = Vectors::load(Path::new("/nonexistent")).unwrap();
        let queue = build_queue(&names, &labels, &cache, "waymo", &[], &[], &[]);

        assert!(
            queue.iter().any(|o| o.name == names[1]),
            "neighbour not offered"
        );
        assert_eq!(
            labels.entries.len(),
            1,
            "a neighbour was labelled by the queue"
        );
    }

    fn offered(names: &[(u128, Via)]) -> Vec<Offered> {
        names
            .iter()
            .map(|(m, via)| Offered {
                name: name(*m),
                via: *via,
                score: None,
            })
            .collect()
    }

    /// the yesses are picked out first and the rest marked in one go, so the
    /// sweep must not undo the picking.
    #[test]
    fn a_sweep_leaves_every_decision_already_made_alone() {
        let base = 1_789_250_000_000u128;
        let queue = offered(&[
            (base, Via::Random),
            (base + 60_000, Via::Random),
            (base + 120_000, Via::Random),
        ]);
        let mut labels = Labels::default();
        labels
            .entries
            .insert(name(base), Entry::new("waymo", Via::Random));

        let fresh = sweepable(&queue, &labels, Via::Random);
        assert_eq!(
            fresh.len(),
            2,
            "a labelled crop was about to be overwritten"
        );
        assert!(!fresh.contains(&name(base)));
    }

    /// the ranked pool is dense with examples. sweeping it because the random
    /// pool was swept would bury exactly what the session is looking for.
    #[test]
    fn a_sweep_takes_one_pool_and_not_the_page() {
        let base = 1_789_250_000_000u128;
        let queue = offered(&[
            (base, Via::Random),
            (base + 60_000, Via::Ranked),
            (base + 120_000, Via::Seed),
        ]);
        let labels = Labels::default();
        assert_eq!(sweepable(&queue, &labels, Via::Random), vec![name(base)]);
        assert_eq!(
            sweepable(&queue, &labels, Via::Ranked),
            vec![name(base + 60_000)]
        );
    }

    #[test]
    fn a_swept_pool_has_nothing_left_to_sweep() {
        let base = 1_789_250_000_000u128;
        let queue = offered(&[(base, Via::Random), (base + 60_000, Via::Random)]);
        let mut labels = Labels::default();
        for name in sweepable(&queue, &labels, Via::Random) {
            labels
                .entries
                .insert(name, Entry::new(crate::label::NEGATIVE, Via::Random));
        }
        assert!(sweepable(&queue, &labels, Via::Random).is_empty());
    }

    /// **a verdict is a listing, not a measurement.** the pool used to come out
    /// of `eval::measure`, so it was empty until train had been pressed -- and
    /// train needs labelled positives, which is exactly what a harvest full of
    /// false positives does not have.
    #[test]
    fn what_the_classifier_named_is_offered_before_anything_is_measured() {
        let base = 1_789_250_000_000u128;
        let names: Vec<String> = (0..3).map(|i| name(base + i * 60_000)).collect();
        let cache = Vectors::load(Path::new("/nonexistent")).unwrap();
        let queue = build_queue(
            &names,
            &Labels::default(),
            &cache,
            "waymo",
            &[],
            &[],
            &[names[1].clone()],
        );
        let fired: Vec<&String> = queue
            .iter()
            .filter(|o| o.via == Via::Alert)
            .map(|o| &o.name)
            .collect();
        assert_eq!(fired, vec![&names[1]], "no measurement, so no verdict");
    }

    /// one tile, one decision. the crop that fired on the deployment is also
    /// the one the current references fire on, which is the normal case rather
    /// than a coincidence.
    #[test]
    fn a_verdict_the_references_also_fire_on_is_offered_once() {
        let base = 1_789_250_000_000u128;
        let names: Vec<String> = (0..2).map(|i| name(base + i * 60_000)).collect();
        let cache = Vectors::load(Path::new("/nonexistent")).unwrap();
        let queue = build_queue(
            &names,
            &Labels::default(),
            &cache,
            "waymo",
            &[],
            &[(names[0].clone(), 0.02)],
            &[names[0].clone()],
        );
        assert_eq!(
            queue
                .iter()
                .filter(|o| o.via == Via::Alert && o.name == names[0])
                .count(),
            1
        );
    }

    /// the usual answer on this pool is "every one of these is wrong", so it is
    /// one action -- and it must leave the one that was right alone.
    #[test]
    fn sweeping_the_verdicts_takes_only_the_undecided_ones() {
        let base = 1_789_250_000_000u128;
        let queue = offered(&[(base, Via::Alert), (base + 60_000, Via::Alert)]);
        let mut labels = Labels::default();
        labels
            .entries
            .insert(name(base), Entry::new("waymo", Via::Alert));
        assert_eq!(
            sweepable(&queue, &labels, Via::Alert),
            vec![name(base + 60_000)]
        );
    }

    /// **`other` means "none of the subjects we know about".** so a waymo
    /// verdict rejected in a go-4 session would record something false about
    /// the crop rather than something true about the verdict.
    #[test]
    fn only_this_subjects_verdicts_are_offered_for_rejection() {
        let d = tmpdir("verdicts");
        let mine = "1789250000000_truck_go4_075_240x200.jpg";
        for n in [
            mine,
            "1789250060000_car_waymo_081_240x200.jpg",
            // stage two named nothing here, which is nearly every crop.
            "1789250120000_car_090_240x200.jpg",
        ] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        // the same directory twice is a crop two sets share: offered once.
        let sets = [d.clone(), d];
        assert_eq!(verdicts(&sets, "go4"), vec![mine.to_string()]);
        assert!(verdicts(&sets, "sweeper").is_empty());
    }

    #[test]
    fn a_crop_name_from_the_url_cannot_escape_the_harvest_directory() {
        for bad in [
            "../../Cargo.toml",
            "..%2fCargo.toml",
            "nope.jpg",
            "",
            "a/b.jpg",
        ] {
            let safe = Path::new(bad)
                .file_name()
                .and_then(|n| n.to_str())
                .filter(|n| crate::harvest::parse_name(n).is_some());
            assert!(safe.is_none(), "{bad} passed the guard");
        }
        let good = "1789259517126_car_085_673x396.jpg";
        assert!(crate::harvest::parse_name(good).is_some());
    }

    #[test]
    fn an_error_report_serialises_as_an_error_rather_than_as_empty_numbers() {
        let r = eval::Report {
            error: Some("nothing is labelled waymo".into()),
            ..Default::default()
        };
        let json = report_json(&r);
        assert!(json.contains(r#""error""#), "{json}");
        assert!(!json.contains("sweep"), "{json}");
    }

    #[test]
    fn quotes_in_a_label_cannot_break_out_of_the_json() {
        assert_eq!(escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }
}
