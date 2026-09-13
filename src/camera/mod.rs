//! talking to the camera's cgi, and watching where it is pointed (r5.2).
//!
//! the background model, the roi polygon, and every scale prior in the pipeline
//! are keyed to one pan/tilt position. move the camera and all three refer to
//! somewhere else, with no error anywhere -- it was panned once during
//! development and everything silently went stale. so position is polled, and a
//! move resets the model.
//!
//! only `getStatus` is ever requested. the streaming cgis are **stateful**:
//! fetching the mjpeg url reconfigures the substream encoder and leaves it that
//! way, which is a violation of r5.4 waiting to happen (see DESIGN.md).

pub mod md5;

use crate::config::Camera;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub pan: f32,
    pub tilt: f32,
    pub zoom: f32,
}

impl Position {
    /// has the camera been repointed by at least `degrees`?
    ///
    /// measured against this camera, a still one reports the same value every
    /// time -- 30 consecutive polls returned pan 25.4, tilt 42.0, zoom 1.0 with
    /// no variation at all -- so there is no dither to clear here. the threshold
    /// is for firmware that rounds, and it is configurable because guessing it
    /// wrong is expensive in both directions: too high and a real move goes
    /// unnoticed, too low and every reset costs `warmup_frames` of blindness and
    /// wipes scenery, so no parked vehicle can accumulate the time it needs.
    pub fn moved_from(&self, other: Position, degrees: f32) -> bool {
        (self.pan - other.pan).abs() >= degrees
            || (self.tilt - other.tilt).abs() >= degrees
            || (self.zoom - other.zoom).abs() >= degrees
    }
}

/// parse `ptz.cgi?action=getStatus`.
///
/// the key really is spelled `Postion`; that is the camera's typo, not ours,
/// and both spellings are accepted so a firmware that fixes it keeps working.
pub fn parse_status(body: &str) -> Option<Position> {
    let mut coords = [None; 3];
    for line in body.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if !key.contains("Postion") && !key.contains("Position") {
            continue;
        }
        let i = key
            .rsplit_once('[')
            .and_then(|(_, idx)| idx.trim_end_matches(']').parse::<usize>().ok());
        if let (Some(i), Ok(v)) = (i, value.trim().parse::<f32>())
            && i < 3
        {
            coords[i] = Some(v);
        }
    }
    Some(Position {
        pan: coords[0]?,
        tilt: coords[1]?,
        zoom: coords[2].unwrap_or(0.0),
    })
}

fn header<'a>(response: &'a str, name: &str) -> Option<&'a str> {
    response
        .lines()
        .find(|l| {
            l.to_ascii_lowercase()
                .starts_with(&name.to_ascii_lowercase())
        })
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim())
}

/// pull `name="value"` (or bare `name=value`) out of a digest challenge.
fn field(challenge: &str, name: &str) -> Option<String> {
    let at = challenge.find(&format!("{name}="))? + name.len() + 1;
    let rest = &challenge[at..];
    Some(match rest.strip_prefix('"') {
        Some(quoted) => quoted[..quoted.find('"')?].to_string(),
        None => rest.split(',').next()?.trim().to_string(),
    })
}

/// the digest response for one request, per rfc 2617 qop=auth.
fn authorization(
    challenge: &str,
    user: &str,
    pass: &str,
    path: &str,
    cnonce: &str,
) -> Option<String> {
    let realm = field(challenge, "realm")?;
    let nonce = field(challenge, "nonce")?;
    let qop = field(challenge, "qop");
    let ha1 = md5::hex(format!("{user}:{realm}:{pass}").as_bytes());
    let ha2 = md5::hex(format!("GET:{path}").as_bytes());
    let (response, extra) = match &qop {
        Some(q) => (
            md5::hex(format!("{ha1}:{nonce}:00000001:{cnonce}:auth:{ha2}").as_bytes()),
            format!(", qop={q}, nc=00000001, cnonce=\"{cnonce}\""),
        ),
        None => (
            md5::hex(format!("{ha1}:{nonce}:{ha2}").as_bytes()),
            String::new(),
        ),
    };
    Some(format!(
        "Digest username=\"{user}\", realm=\"{realm}\", nonce=\"{nonce}\", \
         uri=\"{path}\", response=\"{response}\"{extra}"
    ))
}

fn request(host: &str, path: &str, auth: Option<&str>, timeout: Duration) -> Result<String> {
    let mut stream = TcpStream::connect((host, 80)).context("connecting to the camera")?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let auth = auth
        .map(|a| format!("Authorization: {a}\r\n"))
        .unwrap_or_default();
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n{auth}\r\n"
    )?;
    let mut out = Vec::new();
    stream.read_to_end(&mut out)?;
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// one authenticated cgi GET: unauthenticated first, then answer the challenge.
///
/// the nonce is per-request, so there is no session to keep and nothing to
/// invalidate when the camera reboots.
pub fn get(cam: &Camera, path: &str) -> Result<String> {
    let first = request(
        &cam.host,
        path,
        None,
        Duration::from_secs(cam.cgi_timeout_secs),
    )?;
    if first.starts_with("HTTP/1.1 200") {
        return Ok(first);
    }
    let challenge = header(&first, "WWW-Authenticate")
        .context("camera refused the request without offering a challenge")?;
    // a counter would do: the nonce is the camera's, and this only has to
    // differ between requests sharing one.
    let cnonce = format!(
        "{:x}",
        std::time::Instant::now().elapsed().as_nanos() ^ 0x9e3779b9
    );
    let auth = authorization(challenge, &cam.username, &cam.password, path, &cnonce)
        .context("could not build a digest response")?;
    let second = request(
        &cam.host,
        path,
        Some(&auth),
        Duration::from_secs(cam.cgi_timeout_secs),
    )?;
    if !second.starts_with("HTTP/1.1 200") {
        bail!(
            "camera rejected the credentials: {}",
            second.lines().next().unwrap_or("")
        );
    }
    Ok(second)
}

pub fn position(cam: &Camera) -> Result<Position> {
    let path = format!("/cgi-bin/ptz.cgi?action=getStatus&channel={}", cam.channel);
    let body = get(cam, &path)?;
    parse_status(&body).context("no position in the camera's reply")
}

/// poll the camera's position, reporting each time it is repointed.
///
/// returns `None` when the camera has no ptz or cannot be reached, because a
/// watcher that cannot watch should not be mistaken for a camera that never
/// moves. detection runs either way -- this is a correctness aid, not a
/// dependency.
pub fn watch(cam: &Camera, every: Duration) -> Option<Receiver<Position>> {
    let start = match position(cam) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("pan/tilt watcher off: {e:#}");
            return None;
        }
    };
    tracing::info!(
        "pan/tilt watcher on: pan {:.1} tilt {:.1}, polling every {}s",
        start.pan,
        start.tilt,
        every.as_secs()
    );
    let (tx, rx) = channel();
    let cam = cam.clone();
    std::thread::spawn(move || {
        let mut last = start;
        loop {
            std::thread::sleep(every);
            match position(&cam) {
                Ok(now) if now.moved_from(last, cam.ptz_moved_degrees) => {
                    tracing::info!(
                        "camera repointed: pan {:.1}->{:.1}, tilt {:.1}->{:.1}",
                        last.pan,
                        now.pan,
                        last.tilt,
                        now.tilt
                    );
                    last = now;
                    if tx.send(now).is_err() {
                        return;
                    }
                }
                Ok(now) => last = now,
                Err(e) => {
                    tracing::warn!("pan/tilt poll failed, retrying: {e:#}");
                    std::thread::sleep(Duration::from_secs(cam.cgi_retry_secs));
                }
            }
        }
    });
    Some(rx)
}

/// did the camera move since this was last asked? drains the channel, so a
/// burst of movement is one reset rather than several.
pub fn repointed(rx: &Option<Receiver<Position>>) -> bool {
    let Some(rx) = rx else { return false };
    let mut moved = false;
    loop {
        match rx.try_recv() {
            Ok(_) => moved = true,
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return moved,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = "status.Postion[0]=29.5\r\nstatus.Postion[1]=13.6\r\n\
                        status.Postion[2]=0\r\nstatus.Zoom=0\r\nstatus.ZoomValue=0\r\n";

    #[test]
    fn parses_the_cameras_own_reply() {
        let p = parse_status(REAL).unwrap();
        assert_eq!(p.pan, 29.5);
        assert_eq!(p.tilt, 13.6);
        assert_eq!(p.zoom, 0.0);
    }

    /// a firmware that fixes the typo must not turn the watcher off silently.
    #[test]
    fn accepts_the_corrected_spelling() {
        let p = parse_status("status.Position[0]=10\nstatus.Position[1]=20\n").unwrap();
        assert_eq!((p.pan, p.tilt), (10.0, 20.0));
    }

    #[test]
    fn a_reply_without_a_position_is_not_a_position() {
        assert!(parse_status("Error\r\n").is_none());
        assert!(
            parse_status("status.Postion[0]=29.5\r\n").is_none(),
            "tilt missing"
        );
    }

    /// the reading dithers while the camera is still; only a real move counts,
    /// because a reset costs `warmup_frames` of not watching the street.
    #[test]
    fn dither_is_not_a_repoint() {
        let at = |pan, tilt| Position {
            pan,
            tilt,
            zoom: 0.0,
        };
        assert!(!at(29.5, 13.6).moved_from(at(29.4, 13.7), 1.0));
        assert!(at(29.5, 13.6).moved_from(at(31.0, 13.6), 1.0), "panned");
        assert!(at(29.5, 13.6).moved_from(at(29.5, 20.0), 1.0), "tilted");
        // and the threshold is honoured, since it is now a config value.
        assert!(at(29.5, 13.6).moved_from(at(29.9, 13.6), 0.2), "tighter");
        assert!(!at(29.5, 13.6).moved_from(at(31.0, 13.6), 5.0), "looser");
    }

    #[test]
    fn builds_a_digest_response_from_a_challenge() {
        let challenge = "Digest realm=\"Login to 1D01A\", qop=\"auth\", \
                         nonce=\"1234567890\", opaque=\"abc\"";
        let auth =
            authorization(challenge, "admin", "secret", "/cgi-bin/ptz.cgi", "deadbeef").unwrap();
        // ha1 = md5(admin:Login to 1D01A:secret), ha2 = md5(GET:/cgi-bin/ptz.cgi)
        let ha1 = md5::hex(b"admin:Login to 1D01A:secret");
        let ha2 = md5::hex(b"GET:/cgi-bin/ptz.cgi");
        let want = md5::hex(format!("{ha1}:1234567890:00000001:deadbeef:auth:{ha2}").as_bytes());
        assert!(auth.contains(&format!("response=\"{want}\"")), "{auth}");
        assert!(auth.contains("nc=00000001"), "{auth}");
        assert!(auth.contains("uri=\"/cgi-bin/ptz.cgi\""), "{auth}");
    }

    /// older firmware omits qop, and the response is computed differently.
    #[test]
    fn handles_a_challenge_without_qop() {
        let auth = authorization("Digest realm=\"r\", nonce=\"n\"", "u", "p", "/x", "c").unwrap();
        let ha1 = md5::hex(b"u:r:p");
        let ha2 = md5::hex(b"GET:/x");
        let want = md5::hex(format!("{ha1}:n:{ha2}").as_bytes());
        assert!(auth.contains(&format!("response=\"{want}\"")), "{auth}");
        assert!(!auth.contains("qop"), "{auth}");
    }

    #[test]
    fn pulls_quoted_and_bare_fields_out_of_a_challenge() {
        let c = "Digest realm=\"a b\", qop=auth, nonce=\"xyz\"";
        assert_eq!(field(c, "realm").as_deref(), Some("a b"));
        assert_eq!(field(c, "qop").as_deref(), Some("auth"));
        assert_eq!(field(c, "nonce").as_deref(), Some("xyz"));
        assert_eq!(field(c, "missing"), None);
    }
}
