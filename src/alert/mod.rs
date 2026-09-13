//! mqtt publishing and home assistant discovery.
//!
//! publishing is fire-and-forget from the caller's point of view. the alert path
//! must never block on the network, because a late alert is a useless one (r2.2).

use crate::config::{MQTT_CHANNEL_CAPACITY, MQTT_KEEPALIVE_SECS, Mqtt};
use anyhow::{Context, Result};
use rumqttc::{Client, Event, Incoming, LastWill, MqttOptions, QoS};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// what happened, on its own topic, for one subject (r4.1, r10.2).
///
/// **not severity.** these were five ranked tiers -- sighting, approaching,
/// dwell, dismount -- which asserted that a dismount matters more than a
/// sighting. it might, and whether it does is a home assistant question about
/// this street on this day, downstream of the broker (r10). metermate reports
/// what it saw and attaches the facts a rule would need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// recognised as the subject, confirmed over several looks (r1.4).
    Sighting,
    /// and it is in motion.
    ///
    /// ROADMAP called this `approaching`, which asserts a direction metermate
    /// cannot currently measure: there is nothing yet for it to be approaching.
    /// once the protected vehicle exists (r9.1) the heading is computable from
    /// the track and the honest name comes back.
    Moving,
    /// and it has since stayed put. what a vehicle does before a ticket.
    Stopped,
    /// a person adjacent to a stopped subject (r1.3).
    Dismount,
    /// its track ended. the counterpart to `Sighting`, so a rule can tell
    /// "still here" from "was here", which a retained ON alone cannot.
    Departed,
}

impl Outcome {
    pub fn slug(self) -> &'static str {
        match self {
            Outcome::Sighting => "sighting",
            Outcome::Moving => "moving",
            Outcome::Stopped => "stopped",
            Outcome::Dismount => "dismount",
            Outcome::Departed => "departed",
        }
    }

    pub fn all() -> [Outcome; 5] {
        [
            Outcome::Sighting,
            Outcome::Moving,
            Outcome::Stopped,
            Outcome::Dismount,
            Outcome::Departed,
        ]
    }

    /// every slug, for a config that names outcomes and an error that has to
    /// say which ones exist.
    pub fn slugs() -> Vec<&'static str> {
        Self::all().iter().map(|o| o.slug()).collect()
    }

    /// how a notification opens: the words a person reads before deciding
    /// whether to get up. `Stopped` is the one that costs money.
    pub fn headline(self, subject: &str) -> String {
        let subject = subject.to_uppercase();
        match self {
            Outcome::Sighting => format!("{subject} on the block"),
            Outcome::Moving => format!("{subject} moving"),
            Outcome::Stopped => format!("{subject} stopped"),
            Outcome::Dismount => format!("{subject}: officer out of the vehicle"),
            Outcome::Departed => format!("{subject} gone"),
        }
    }
}

/// the raw gate signal. not subject-keyed, because nothing has been recognised
/// at this point -- it is "something moved", and it predates the cascade.
pub const MOTION_TOPIC: &str = "motion";

/// `<base>/<subject>/<outcome>`, or `<base>/motion` for the gate signal.
///
/// subjects are slugs from config rather than an enum, so adding one needs no
/// recompile (r10.1) and need not be a vehicle (r10.3).
pub fn topic_for(base: &str, subject: &str, outcome: Outcome) -> String {
    format!("{base}/{subject}/{}", outcome.slug())
}

pub mod ntfy;

/// the human half of an alert: a sentence, and the crop that caused it.
///
/// separate from `Facts` because the two audiences are different. a rule wants
/// the box and the track id; a person wants to know whether to get up, and the
/// crop answers that faster than any number.
#[derive(Default)]
pub struct Told {
    pub note: String,
    pub crop: Option<std::path::PathBuf>,
}

pub struct Alerter {
    /// **absent when there is no broker to talk to.** ntfy is a reason to
    /// alert on its own -- avoiding the broker-then-bridge-then-app chain is
    /// most of why it exists -- so a config with `[ntfy]` and no `[mqtt]` is a
    /// working deployment rather than a mistake.
    client: Option<Client>,
    base: String,
    /// **the same call publishes and notifies**, so the broker and the phone
    /// cannot come to disagree about what fired: there is one place that
    /// decides an outcome happened.
    ntfy: Option<ntfy::Ntfy>,
}

impl Alerter {
    /// `subjects` are the slugs discovery is generated for.
    pub fn connect(cfg: &Mqtt, subjects: &[String], ntfy: Option<ntfy::Ntfy>) -> Result<Self> {
        let mut opts = MqttOptions::new(&cfg.client_id, &cfg.host, cfg.port);
        opts.set_keep_alive(Duration::from_secs(MQTT_KEEPALIVE_SECS));
        if let (Some(u), Some(p)) = (&cfg.username, &cfg.password) {
            opts.set_credentials(u.clone(), p.clone());
        }
        let availability = format!("{}/availability", cfg.base_topic);
        // the broker announces our absence if we die, so home assistant shows the
        // entities as unavailable rather than silently stale (r4.4).
        opts.set_last_will(LastWill::new(
            &availability,
            "offline",
            QoS::AtLeastOnce,
            true,
        ));

        let (client, connection) = Client::new(opts, MQTT_CHANNEL_CAPACITY);
        // the event loop keeps its own handle, for republishing discovery on
        // every reconnect.
        let c = client.clone();
        let alerter = Self {
            client: Some(client),
            base: cfg.base_topic.clone(),
            ntfy,
        };

        let discovery = discovery_messages(cfg, subjects);
        std::thread::spawn(move || run_event_loop(connection, c, availability, discovery));
        Ok(alerter)
    }

    /// publish an outcome for a subject. failures are logged, never propagated:
    /// a broker outage must not take down detection.
    ///
    /// `facts` is the wire payload and `note` is the sentence a person reads;
    /// `crop` is the picture that caused it, when the harvester kept one.
    /// both channels leave from here so neither can be given an outcome the
    /// other never heard about.
    pub fn publish(&self, subject: &str, outcome: Outcome, facts: &str, told: Told) {
        self.send(&topic_for(&self.base, subject, outcome), facts);
        if let Some(n) = &self.ntfy
            && n.wants(outcome)
        {
            n.notify(subject, outcome, &told.note, told.crop);
        }
    }

    /// the raw gate signal, which belongs to no subject.
    pub fn publish_motion(&self, body: &str) {
        self.send(&format!("{}/{}", self.base, MOTION_TOPIC), body);
    }

    fn send(&self, topic: &str, body: &str) {
        let Some(client) = &self.client else {
            return;
        };
        if let Err(e) = client.try_publish(topic, QoS::AtLeastOnce, false, body) {
            tracing::warn!("publish to {topic} failed: {e}");
        }
    }
}

/// alerting with nothing but a phone on the other end.
///
/// the mqtt half of this is what a rule engine consumes, and plenty of
/// deployments have no rule engine: one camera, one street, one person who
/// wants to be told. `[ntfy]` alone is that deployment.
pub fn notifier_only(ntfy: ntfy::Ntfy) -> Alerter {
    Alerter {
        client: None,
        base: String::new(),
        ntfy: Some(ntfy),
    }
}

/// the event loop owns reconnection. on every fresh connection we republish
/// discovery and availability, because a broker that restarted has forgotten us.
fn run_event_loop(
    mut connection: rumqttc::Connection,
    client: Client,
    availability: String,
    discovery: Vec<(String, String)>,
) {
    for event in connection.iter() {
        match event {
            Ok(Event::Incoming(Incoming::ConnAck(_))) => {
                tracing::info!("mqtt connected");
                for (topic, payload) in &discovery {
                    let _ = client.try_publish(topic, QoS::AtLeastOnce, true, payload.as_bytes());
                }
                let _ = client.try_publish(&availability, QoS::AtLeastOnce, true, "online");
            }
            Ok(_) => {}
            Err(e) => {
                // rumqttc retries on its own; log and let it.
                tracing::warn!("mqtt connection error: {e}");
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
}

/// home assistant discovery: one binary sensor per subject and outcome, plus
/// the subject-less motion signal (r4.2).
///
/// generated from the configured subjects rather than an enum, so adding one
/// makes its entities appear with no yaml and no recompile (r10.1).
fn discovery_messages(cfg: &Mqtt, subjects: &[String]) -> Vec<(String, String)> {
    let mut out = vec![sensor(
        cfg,
        "motion",
        "metermate motion",
        MOTION_TOPIC.to_string(),
    )];
    for subject in subjects {
        for o in Outcome::all() {
            let slug = format!("{subject}_{}", o.slug());
            let name = format!("metermate {subject} {}", o.slug());
            out.push(sensor(cfg, &slug, &name, format!("{subject}/{}", o.slug())));
        }
    }
    out
}

/// one binary sensor. `suffix` is the topic below the base; `slug` has to be
/// unique per entity, since home assistant keys on `unique_id` forever.
fn sensor(cfg: &Mqtt, slug: &str, name: &str, suffix: String) -> (String, String) {
    let topic = format!(
        "{}/binary_sensor/metermate_{}/config",
        cfg.discovery_prefix, slug
    );
    let payload = format!(
        r#"{{"name":"{name}","unique_id":"metermate_{slug}","state_topic":"{base}/{suffix}","value_template":"{{{{ value_json.state }}}}","json_attributes_topic":"{base}/{suffix}","payload_on":"ON","payload_off":"OFF","device_class":"motion","availability_topic":"{base}/availability","device":{{"identifiers":["metermate"],"name":"metermate","manufacturer":"khimaros","model":"metermate"}}}}"#,
        base = cfg.base_topic,
    );
    (topic, payload)
}

/// milliseconds since the unix epoch, for event payloads.
pub fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// facts about one outcome, for a rule downstream to decide on.
///
/// no severity, no priority, no "urgent" -- r10 puts that in home assistant.
/// what goes in is what a rule cannot work out for itself: which subject, which
/// track, how long it has been stopped, how sure the classifier was, and
/// whether it is near the protected vehicle (r9.1, as a fact rather than a
/// tier).
pub struct Facts<'a> {
    pub subject: &'a str,
    pub on: bool,
    pub track: u64,
    pub dwell_s: f32,
    pub confidence: f32,
    pub protected: Protected,
    pub box_: crate::gate::Rect,
}

/// where the subject is relative to the protected vehicle (r9.1).
///
/// **four states, not a boolean.** a flag would collapse "the go-4 is nowhere
/// near the van" and "the van is not parked here at all" into one `false`, and
/// those want opposite rules downstream -- the second is r9.2, where there is
/// nothing to protect and the whole thing is uninteresting. severity is not
/// ours to assign, so the payload has to carry enough for a rule to implement
/// r9.2 itself.
// the producer is phase 2b, which learns where the protected vehicle parks.
// these are the wire format the payload already commits to, so a downstream
// rule can be written against them before the detector side lands -- and so
// that shape is decided once rather than retrofitted around a boolean.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protected {
    Adjacent,
    Near,
    Far,
    /// the protected vehicle is not in its space (r9.2).
    Away,
    /// no protected vehicle is configured, so the question does not apply.
    Unknown,
}

impl Protected {
    pub fn slug(self) -> &'static str {
        match self {
            Protected::Adjacent => "adjacent",
            Protected::Near => "near",
            Protected::Far => "far",
            Protected::Away => "away",
            Protected::Unknown => "unknown",
        }
    }
}

pub fn event_payload(e: &Facts) -> String {
    format!(
        r#"{{"state":"{state}","subject":"{subject}","track":{track},"dwell_s":{dwell:.1},"confidence":{conf:.3},"protected":"{protected}","box":{{"x":{x},"y":{y},"w":{w},"h":{h}}},"at":{at}}}"#,
        state = if e.on { "ON" } else { "OFF" },
        subject = e.subject,
        track = e.track,
        dwell = e.dwell_s,
        conf = e.confidence,
        protected = e.protected.slug(),
        x = e.box_.x,
        y = e.box_.y,
        w = e.box_.w,
        h = e.box_.h,
        at = now_millis(),
    )
}

/// build a motion event payload. kept as a pure function so the e2e suite can
/// assert on the exact shape without a broker.
pub fn motion_payload(on: bool, changed_frac: f32, regions: &[crate::gate::Rect]) -> String {
    let list = regions
        .iter()
        .map(|r| format!(r#"{{"x":{},"y":{},"w":{},"h":{}}}"#, r.x, r.y, r.w, r.h))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"state":"{}","changed_frac":{:.5},"ts":{},"regions":[{}]}}"#,
        if on { "ON" } else { "OFF" },
        changed_frac,
        now_millis(),
        list
    )
}

pub fn check_broker_reachable(cfg: &Mqtt) -> Result<()> {
    use std::net::TcpStream;
    TcpStream::connect_timeout(
        &format!("{}:{}", cfg.host, cfg.port)
            .parse()
            .with_context(|| format!("bad broker address {}:{}", cfg.host, cfg.port))?,
        Duration::from_secs(3),
    )
    .with_context(|| format!("mqtt broker {}:{} unreachable", cfg.host, cfg.port))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Mqtt {
        Mqtt {
            host: "127.0.0.1".into(),
            port: 1883,
            client_id: "metermate".into(),
            base_topic: "metermate".into(),
            discovery_prefix: "homeassistant".into(),
            username: None,
            password: None,
        }
    }

    fn subjects() -> Vec<String> {
        vec!["go4".to_string(), "sweeper".to_string()]
    }

    #[test]
    fn a_topic_names_the_subject_and_what_happened() {
        assert_eq!(
            topic_for("metermate", "go4", Outcome::Stopped),
            "metermate/go4/stopped"
        );
        assert_eq!(
            topic_for("metermate", "sweeper", Outcome::Sighting),
            "metermate/sweeper/sighting"
        );
    }

    /// adding a subject must make its entities appear with no yaml (r4.2) and
    /// no recompile (r10.1).
    #[test]
    fn discovery_covers_every_subject_and_outcome() {
        let msgs = discovery_messages(&cfg(), &subjects());
        assert_eq!(msgs.len(), subjects().len() * Outcome::all().len() + 1);
        for s in subjects() {
            for o in Outcome::all() {
                let want = format!(
                    "homeassistant/binary_sensor/metermate_{s}_{}/config",
                    o.slug()
                );
                assert!(
                    msgs.iter().any(|(topic, _)| topic == &want),
                    "missing {want}"
                );
            }
        }
    }

    /// the gate signal predates recognition, so it belongs to no subject.
    #[test]
    fn motion_keeps_its_own_subjectless_topic() {
        let msgs = discovery_messages(&cfg(), &subjects());
        let (_, payload) = msgs
            .iter()
            .find(|(t, _)| t.contains("metermate_motion"))
            .unwrap();
        assert!(
            payload.contains(r#""state_topic":"metermate/motion""#),
            "{payload}"
        );
    }

    #[test]
    fn discovery_payload_points_at_the_outcome_topic_and_availability() {
        let msgs = discovery_messages(&cfg(), &subjects());
        let (_, payload) = msgs
            .iter()
            .find(|(t, _)| t.contains("go4_stopped"))
            .unwrap();
        assert!(
            payload.contains(r#""state_topic":"metermate/go4/stopped""#),
            "{payload}"
        );
        assert!(payload.contains(r#""availability_topic":"metermate/availability""#));
        assert!(payload.contains(r#""unique_id":"metermate_go4_stopped""#));
    }

    /// **severity is not ours to assign** (r10). if one of these ever appears
    /// in a payload, the decision has been taken in the wrong place -- see the
    /// alerting section of DESIGN.md for why it sits downstream of the broker.
    #[test]
    fn a_payload_carries_facts_and_never_a_severity() {
        let body = event_payload(&Facts {
            subject: "go4",
            on: true,
            track: 12,
            dwell_s: 43.0,
            confidence: 0.87,
            protected: Protected::Adjacent,
            box_: crate::gate::Rect {
                x: 1,
                y: 2,
                w: 3,
                h: 4,
            },
        });
        assert!(body.contains(r#""subject":"go4""#), "{body}");
        assert!(body.contains(r#""track":12"#), "{body}");
        assert!(body.contains(r#""dwell_s":43.0"#), "{body}");
        assert!(body.contains(r#""protected":"adjacent""#), "{body}");
        for banned in ["severity", "priority", "urgent", "tier", "critical"] {
            assert!(!body.contains(banned), "{banned} in {body}");
        }
    }

    /// r9.2: "the van is not there" and "the go-4 is far from it" want opposite
    /// rules downstream, so they cannot share a value.
    #[test]
    fn protected_distinguishes_far_from_absent() {
        assert_ne!(Protected::Far.slug(), Protected::Away.slug());
        assert_ne!(Protected::Away.slug(), Protected::Unknown.slug());
    }

    #[test]
    fn motion_payload_shapes() {
        let off = motion_payload(false, 0.0, &[]);
        assert!(off.contains(r#""state":"OFF""#), "{off}");
        assert!(off.contains(r#""regions":[]"#), "{off}");

        let on = motion_payload(
            true,
            0.01234,
            &[
                crate::gate::Rect {
                    x: 1,
                    y: 2,
                    w: 3,
                    h: 4,
                },
                crate::gate::Rect {
                    x: 5,
                    y: 6,
                    w: 7,
                    h: 8,
                },
            ],
        );
        assert!(on.contains(r#""state":"ON""#), "{on}");
        assert!(
            on.contains(r#""regions":[{"x":1,"y":2,"w":3,"h":4},{"x":5,"y":6,"w":7,"h":8}]"#),
            "{on}"
        );
    }
}
