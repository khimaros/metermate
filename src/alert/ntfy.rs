//! notifying a phone directly, over ntfy.
//!
//! **mqtt is the integration surface; this is the doorbell.** a rule engine
//! wants the facts and a person crossing the room wants a sentence and a
//! picture, and the two are not the same message: `server` is usually somebody
//! else's machine, so what leaves the network here is deliberately not the wire
//! payload. no track ids, no boxes, no dwell times -- a headline, a line of
//! prose, and the crop that caused it.
//!
//! **nothing here runs on the pipeline thread.** r2.1 gives an alert 1.5s from
//! the roi to the broker and r2.2 says evidence capture may never hold that up;
//! an http post to a server on the other side of a domestic uplink is exactly
//! the kind of thing that stalls for thirty seconds. so a send is a message on
//! a bounded channel, and a worker does the talking.

use crate::config::{NTFY_QUEUE, NTFY_TIMEOUT_SECS, NtfyCfg};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::time::Duration;

/// one notification, as much of it as the pipeline thread is willing to build.
///
/// the crop is a path rather than bytes: reading it is io, io belongs on the
/// worker, and by the time the worker gets there the harvester has long since
/// written the file.
pub struct Note {
    pub title: String,
    pub body: String,
    pub priority: String,
    pub tags: String,
    pub click: String,
    pub crop: Option<PathBuf>,
}

pub struct Ntfy {
    tx: SyncSender<Note>,
    outcomes: Vec<String>,
    priority: String,
    click: String,
}

impl Ntfy {
    /// start the worker. the config is validated already, so a failure here is
    /// a thread that would not start rather than a setting nobody checked.
    pub fn start(cfg: &NtfyCfg) -> Self {
        let (tx, rx) = sync_channel(NTFY_QUEUE);
        let post = Post::new(cfg);
        std::thread::spawn(move || {
            for note in rx {
                if let Err(e) = post.send(&note) {
                    // logged and dropped rather than retried: r4.4 asks that
                    // network loss need no operator action, and a notification
                    // about a vehicle that has since left is worse than none.
                    tracing::warn!("ntfy: {e:#}");
                }
            }
        });
        Self {
            tx,
            outcomes: cfg.outcomes.clone(),
            priority: cfg.priority.clone(),
            click: cfg.click.clone(),
        }
    }

    pub fn wants(&self, outcome: super::Outcome) -> bool {
        self.outcomes.iter().any(|o| o == outcome.slug())
    }

    /// queue a notification, or drop it. **never blocks**: a full queue means
    /// the uplink is down, and waiting for it would hold up the detection loop
    /// that produced the alert.
    pub fn notify(
        &self,
        subject: &str,
        outcome: super::Outcome,
        note: &str,
        crop: Option<PathBuf>,
    ) {
        let note = Note {
            title: outcome.headline(subject),
            body: note.to_string(),
            priority: self.priority.clone(),
            tags: outcome.slug().to_string(),
            click: link(&self.click, crop.as_deref()),
            crop,
        };
        match self.tx.try_send(note) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "ntfy: {NTFY_QUEUE} notifications already waiting, dropping this one"
                )
            }
            Err(TrySendError::Disconnected(_)) => tracing::warn!("ntfy: the sender has stopped"),
        }
    }
}

/// **where a tap on the notification goes**, as a hash link into the preview.
///
/// every view of the page is addressable (r8), and r4.5 is what makes the
/// specific one safe to promise: a notification is only sent once its crop is on
/// the verdict page, so naming that crop in the link cannot dead-end. an outcome
/// with no evidence of its own -- a vehicle that stopped, or left -- names the
/// page it would have appeared on instead of a bare address.
///
/// empty in, empty out: a deployment with no preview the phone can reach keeps
/// its notifications and sends no `Click` header at all.
pub fn link(base: &str, crop: Option<&Path>) -> String {
    let base = base.trim_end_matches('/');
    if base.is_empty() {
        return String::new();
    }
    // nothing is escaped: the harvester writes names out of letters, digits,
    // dashes, underscores and dots, and ntfy passes the header through as it
    // arrived.
    match crop.and_then(|p| p.file_name()) {
        Some(name) => format!("{base}/#/verdict/{}", name.to_string_lossy()),
        None => format!("{base}/#/verdicts"),
    }
}

/// what the worker needs to talk to the server, and nothing about scheduling.
pub struct Post {
    url: String,
    token: String,
}

impl Post {
    pub fn new(cfg: &NtfyCfg) -> Self {
        Self {
            url: format!("{}/{}", cfg.server.trim_end_matches('/'), cfg.topic),
            token: cfg.token.clone(),
        }
    }

    /// **the picture is the body when there is one.** ntfy takes an attachment
    /// as a raw `PUT` with the words in headers, which is what puts the crop on
    /// the phone itself -- the alternative is a link the phone can only follow
    /// from the same lan, and the whole point is being told while out.
    pub fn send(&self, note: &Note) -> Result<()> {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(NTFY_TIMEOUT_SECS)))
            .build()
            .new_agent();
        let mut request = match &note.crop {
            Some(_) => agent.put(&self.url),
            None => agent.post(&self.url),
        };
        request = request
            .header("Title", &note.title)
            .header("Priority", &note.priority)
            .header("Tags", &note.tags);
        if !self.token.is_empty() {
            request = request.header("Authorization", &format!("Bearer {}", self.token));
        }
        if !note.click.is_empty() {
            request = request.header("Click", &note.click);
        }

        let reply = match &note.crop {
            Some(path) => {
                let name = path.file_name().map_or_else(
                    || "evidence.jpg".to_string(),
                    |n| n.to_string_lossy().into(),
                );
                let bytes = std::fs::read(path)
                    .with_context(|| format!("reading the crop {}", path.display()))?;
                request
                    .header("Filename", &name)
                    // the message would otherwise be the filename, which says
                    // nothing the title has not already said.
                    .header("Message", &note.body)
                    .send(&bytes[..])
            }
            None => request.send(note.body.as_bytes()),
        };
        let reply = reply.with_context(|| format!("posting to {}", self.url))?;
        let status = reply.status();
        anyhow::ensure!(status.is_success(), "{} answered {status}", self.url,);
        Ok(())
    }
}

/// a line of prose about what was seen, for a person rather than a rule.
///
/// the facts a rule needs -- track id, box, protected state -- are on the mqtt
/// topic. what belongs here is what changes whether somebody gets up.
///
/// the margin is named the way `trained.toml` names it, and carries its sign: it
/// is the same number the verdict page shows for the crop, and the number a person
/// moves when the alerting is wrong, so the two spellings have to agree.
/// `None` is a deployment with no classifier running, which has no margin to say.
pub fn describe(subject: &str, dwell_s: f32, margin: Option<f32>) -> String {
    let mut said = format!("{subject} recognised");
    if dwell_s >= 1.0 {
        said += &format!(", {dwell_s:.0}s in frame");
    }
    if let Some(margin) = margin {
        said += &format!(", margin {margin:+.3}");
    }
    said
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::Outcome;

    // where the phone can reach the preview, as an operator writes it.
    const PREVIEW: &str = "http://preview.example.com:8420";

    fn cfg() -> NtfyCfg {
        NtfyCfg {
            enabled: true,
            server: "https://ntfy.example/".into(),
            topic: "street".into(),
            token: "tk".into(),
            outcomes: vec!["sighting".into(), "stopped".into()],
            priority: "high".into(),
            click: String::new(),
        }
    }

    #[test]
    fn the_url_is_the_server_and_the_topic_however_the_server_was_written() {
        assert_eq!(Post::new(&cfg()).url, "https://ntfy.example/street");
        let mut bare = cfg();
        bare.server = "http://kairos:8080".into();
        assert_eq!(Post::new(&bare).url, "http://kairos:8080/street");
    }

    /// r4.5's other half: the crop a notification attaches is on the verdict page,
    /// so the tap can be aimed at it rather than at the front of a grid.
    #[test]
    fn a_notification_links_to_the_verdict_it_attaches() {
        let crop = Path::new("/var/crops/1789000000000_truck_go4_075_240x200.jpg");
        assert_eq!(
            link(PREVIEW, Some(crop)),
            format!("{PREVIEW}/#/verdict/1789000000000_truck_go4_075_240x200.jpg")
        );
        // the trailing slash is the operator's, and a hash does not want two.
        assert_eq!(
            link(&format!("{PREVIEW}/"), Some(crop)),
            link(PREVIEW, Some(crop))
        );
    }

    /// stopped and departed carry no crop of their own, and no address at all is
    /// no header rather than an empty one.
    #[test]
    fn an_outcome_with_no_crop_links_to_the_verdicts() {
        assert_eq!(link(PREVIEW, None), format!("{PREVIEW}/#/verdicts"));
        assert_eq!(link("", None), "");
        assert_eq!(link("", Some(Path::new("a.jpg"))), "");
    }

    /// **an outcome nobody asked for must not reach a phone.** the filter is
    /// the difference between a useful alert and a device that buzzes every
    /// time a tracked vehicle moves a metre.
    #[test]
    fn only_the_configured_outcomes_notify() {
        let n = Ntfy::start(&cfg());
        assert!(n.wants(Outcome::Sighting));
        assert!(n.wants(Outcome::Stopped));
        assert!(!n.wants(Outcome::Moving));
        assert!(!n.wants(Outcome::Departed));
    }

    /// r2.2: a stalled uplink may not hold up the loop that found the vehicle.
    /// with no worker draining it, the queue fills and every further send
    /// returns immediately.
    #[test]
    fn a_full_queue_drops_rather_than_waits() {
        let (tx, _rx) = sync_channel(1);
        let n = Ntfy {
            tx,
            outcomes: vec!["sighting".into()],
            priority: "high".into(),
            click: String::new(),
        };
        let started = std::time::Instant::now();
        for _ in 0..NTFY_QUEUE * 4 {
            n.notify("go4", Outcome::Sighting, "seen", None);
        }
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "notify blocked"
        );
    }

    /// the headline is what shows on a locked screen, so it says which subject
    /// and what it did rather than "metermate".
    #[test]
    fn the_headline_names_the_subject_and_the_outcome() {
        assert_eq!(Outcome::Stopped.headline("go4"), "GO4 stopped");
        assert_eq!(
            Outcome::Sighting.headline("sweeper"),
            "SWEEPER on the block"
        );
    }

    #[test]
    fn what_is_described_is_prose_and_never_the_payload() {
        let said = describe("go4", 12.0, Some(0.042));
        assert!(said.contains("12s in frame"), "{said}");
        // named the way `trained.toml` names it, so the number on a phone and the
        // bar a person tunes are obviously the same quantity.
        assert!(said.contains("margin +0.042"), "{said}");
        for wire in ["track", "box", "protected", "{"] {
            assert!(!said.contains(wire), "{said} leaks {wire}");
        }
    }

    /// the sign is what the number is for. a shipped margin may be negative, so a
    /// verdict can sit nearer the street than its own class, and that is the one
    /// case on a phone worth reading twice.
    #[test]
    fn a_margin_below_the_street_is_said_with_its_sign() {
        assert!(describe("go4", 12.0, Some(-0.007)).contains("margin -0.007"));
    }

    /// a short sighting has no dwell worth reporting, and an outcome with no
    /// classifier behind it has no margin: neither should read as "0s" or "+0.000".
    #[test]
    fn nothing_is_said_about_a_number_that_says_nothing() {
        assert_eq!(describe("go4", 0.4, None), "go4 recognised");
    }
}
