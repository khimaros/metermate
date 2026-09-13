//! live preview: watch what metermate is seeing and deciding, in a browser.
//!
//! video and detections travel separately. the mjpeg stream carries pixels only,
//! and boxes are drawn as vector overlays in the browser from a json feed. that
//! is both cheaper (no drawing or font rendering server side) and better looking
//! (crisp labels at any zoom) than burning annotations into the jpeg.
//!
//! nothing is encoded unless somebody is watching (r8.3).

mod fmp4;

use anyhow::{Context, Result};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// frame boundary marker for multipart mjpeg.
const BOUNDARY: &str = "metermate";

/// where the harvest lives and how much disk it is allowed.
///
/// mirrors `Clips` deliberately: both grids answer the same questions about the
/// same kind of thing, and they had drifted apart because only one of them
/// carried its budget.
#[derive(Debug, Clone)]
pub struct Crops {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    /// whether stage two is running at all. the verdicts view would otherwise
    /// report "nothing recognised yet" when the truthful answer is that nothing
    /// can be -- the same distinction the events grid draws by saying "not
    /// recording" rather than "nothing recorded yet".
    pub classifying: bool,
}

/// where the event clips live and how much disk they are allowed.
///
/// `None` only when there is nothing to show and nothing being recorded, which
/// is when an empty tab would read as "the street was quiet" rather than as a
/// feature that is off. clips already on disk are reason enough on their own:
/// they outlive the setting that wrote them.
#[derive(Clone)]
pub struct Clips {
    pub dir: PathBuf,
    /// where the still frame for each card is cached. a directory of its own,
    /// so the events directory holds clips and nothing else.
    pub cache_dir: PathBuf,
    pub budget_bytes: u64,
    /// how far before the trigger a clip opens. the page seeks each card's
    /// poster past it, so the frame shown is the event rather than the seconds
    /// deliberately recorded ahead of it.
    pub preroll_ms: u128,
    /// the ceiling a clip cannot have run past. only a sanity bound on the
    /// length derived from the file's mtime, which a copy does not preserve --
    /// an events directory moved with `cp` reports every clip as however long
    /// ago it was recorded.
    pub max_clip_ms: u128,
    /// false when these are clips from a previous run. the page says so, or a
    /// list that never grows looks like a list that has stopped working.
    pub recording: bool,
}

/// the newest frame, shared by every viewer.
///
/// **behind `Arc`s, because every viewer copies this out on every frame.**
/// each one wakes on publish, takes the lock, and takes what it needs while
/// `publish` waits for the same lock to write the next frame -- so the cost of
/// that copy is paid in frame rate, once per viewer per frame.
///
/// measured on the deployment with a browser tab holding stale connections: the
/// loop ran at 11 fps against a 15 fps stream, and closing the tab returned it
/// to 15. an `Arc` clone is a pointer; a `Vec<u8>` clone was forty kilobytes.
#[derive(Default)]
struct Latest {
    jpeg: Arc<Vec<u8>>,
    detections: Arc<String>,
    seq: u64,
}

#[derive(Default)]
struct RelayState {
    /// `ftyp`+`moov`, kept so a viewer arriving mid-stream can be given the
    /// header it needs before any fragment makes sense.
    init: Vec<u8>,
    /// which connection is the live one. ffmpeg restarts, and for a moment two
    /// of them can be connected at once; only the newest may publish.
    generation: u64,
    /// one queue per viewer, rather than a single latest-fragment slot.
    ///
    /// the slot is what the mjpeg path uses and it is right there: every jpeg
    /// stands alone, so showing only the newest is exactly what is wanted. h.264
    /// is the opposite. a fragment holds a frame predicted from earlier ones, so
    /// skipping one leaves the decoder referencing data it never received and it
    /// stalls until the next keyframe -- which at this gop is most of a second,
    /// every time a viewer hesitates. so each viewer gets every fragment in
    /// order, or gets disconnected.
    subscribers: Vec<std::sync::mpsc::SyncSender<Vec<u8>>>,
}

/// fans one remuxed mp4 stream out to however many browsers are watching.
///
/// the stream comes from the ffmpeg that is *already* decoding the main stream
/// for crops, as a second output, rather than from a second rtsp session. that
/// distinction is the whole point of this type: opening another session pulled a
/// second 4 mbit/s off the camera, and over wifi it starved the gate badly enough
/// to drop detection from fifteen frames a second to one.
#[derive(Clone)]
pub struct Relay {
    inner: Arc<Mutex<RelayState>>,
    /// signalled when a header arrives, so a viewer that asked for the stream
    /// before ffmpeg produced one can be told the moment it exists.
    ready: Arc<Condvar>,
    /// fragments a viewer may fall behind before it is dropped. never defaulted:
    /// a zero here is a rendezvous channel, which would block the relay on the
    /// slowest browser connected to it.
    backlog: usize,
}

impl Relay {
    pub fn new(backlog: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RelayState::default())),
            ready: Arc::new(Condvar::new()),
            backlog,
        }
    }

    /// accept ffmpeg's connections and publish what it writes.
    ///
    /// loops rather than accepting once: the ingest supervisor restarts ffmpeg
    /// on a stalled camera, and the relay has to still be there afterwards.
    ///
    /// **a thread per connection, because one at a time is a restart loop.**
    /// this used to read each connection to completion inside the accept loop,
    /// which is correct exactly as long as the previous ffmpeg has really gone.
    /// when it has not -- a restart whose predecessor is still alive, or a
    /// socket the kernel has not torn down yet -- the new ffmpeg's connection
    /// waits unread in the backlog, its socket buffer fills, and its *other*
    /// output stalls with it, because one process writes both. metermate then
    /// sees no frames, the supervisor restarts it, and the replacement queues
    /// up behind the same stuck reader. that is a main-stream feed that
    /// restarts forever while the gate, which has no second output, runs
    /// perfectly.
    pub fn serve(&self, listener: std::net::TcpListener) {
        let relay = self.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                // **numbered here, in accept order, not inside the reader.**
                // taken inside, two connections are numbered in whatever order
                // their threads happen to be scheduled -- so an orphan can be
                // handed a *higher* generation than the replacement that
                // arrived after it, and the replacement then stands down in
                // favour of the connection it was meant to supersede. it shows
                // up as a stream that is read only sometimes.
                let Some(mine) = relay.next_generation() else {
                    return;
                };
                let relay = relay.clone();
                std::thread::spawn(move || relay.read_from(stream, mine));
            }
        });
    }

    /// claim the next connection number. `None` if the lock is poisoned, which
    /// means nothing can be published anyway.
    fn next_generation(&self) -> Option<u64> {
        self.inner
            .lock()
            .ok()
            .map(|mut s| {
                s.generation += 1;
                s.generation
            })
            .or_else(|| {
                tracing::warn!("the preview relay lock is poisoned; no video will be served");
                None
            })
    }

    /// how long a reader waits for bytes before checking it is still the
    /// connection in use. only a superseded reader is ever idle this long: a
    /// live camera delivers a fragment per frame.
    const STALE_CHECK: std::time::Duration = std::time::Duration::from_secs(5);

    /// read one ffmpeg connection, for as long as it is the newest one.
    ///
    /// newest wins because two connected ffmpegs is a transient, not a feature:
    /// interleaving both into one fragment stream would hand viewers a mix of
    /// two timelines. the older reader stops publishing the moment a newer one
    /// arrives and closes its socket, which is also what frees the orphan.
    fn read_from(&self, mut stream: std::net::TcpStream, mine: u64) {
        use std::io::{ErrorKind, Read};
        let _ = stream.set_read_timeout(Some(Self::STALE_CHECK));
        let mut split = fmp4::Split::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = match stream.read(&mut buf) {
                Ok(0) => return,
                Ok(n) => n,
                // nothing arrived in a while. that is either a dead camera,
                // which the ingest supervisor handles, or this reader having
                // been superseded, which it has to notice for itself.
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    match self.inner.lock() {
                        Ok(s) if s.generation == mine => continue,
                        _ => return,
                    }
                }
                Err(_) => return,
            };
            for chunk in split.push(&buf[..n]) {
                let Ok(mut s) = self.inner.lock() else {
                    return;
                };
                if s.generation != mine {
                    return;
                }
                match chunk {
                    // a restarted ffmpeg sends a fresh header. keep the
                    // newest, or viewers arriving later get one describing a
                    // stream that no longer exists.
                    fmp4::Chunk::Init(init) => {
                        s.init = init;
                        self.ready.notify_all();
                    }
                    // a viewer whose queue is full has stopped draining, so
                    // drop it rather than block the relay. it will reconnect
                    // and be given the header again, which recovers cleanly;
                    // feeding it a gap would not.
                    //
                    // the recorder is subscribed the same way and is dropped the
                    // same way, which is deliberate: it must never be able to
                    // push back on the socket, because ffmpeg writes the crop
                    // feed through the same process.
                    fmp4::Chunk::Fragment(f) => {
                        s.subscribers.retain(|tx| tx.try_send(f.clone()).is_ok());
                    }
                }
            }
        }
    }

    /// the init segment, if ffmpeg has produced one. never waits: the recorder
    /// asks for this from the frame loop, which must not block on the camera.
    pub fn init(&self) -> Option<Vec<u8>> {
        let s = self.inner.lock().ok()?;
        (!s.init.is_empty()).then(|| s.init.clone())
    }

    /// the init segment, waiting up to `timeout` for ffmpeg to produce one.
    ///
    /// **a viewer that arrives first has asked for a stream that is about to
    /// exist, not for one that is broken.** answering `503` in that moment was
    /// correct in isolation and wrong in combination: the page treats the
    /// failure as a dead stream and its reconnect throttle -- which exists so
    /// that a reconnect cannot trigger another and stutter forever -- then
    /// holds the retry off for five seconds. so every start of the h.264
    /// preview was blank for exactly that long, and it read as the recorder
    /// having broken the preview, because enabling the recorder is what put a
    /// remux there to be early for.
    ///
    /// waiting here rather than retrying sooner in the page is deliberate: the
    /// page would have to know that a `503` means "not yet" rather than "not
    /// there", which is this module's startup ordering written down in two
    /// places.
    pub fn init_within(&self, timeout: std::time::Duration) -> Option<Vec<u8>> {
        let s = self.inner.lock().ok()?;
        let (s, _) = self
            .ready
            .wait_timeout_while(s, timeout, |s| s.init.is_empty())
            .ok()?;
        (!s.init.is_empty()).then(|| s.init.clone())
    }

    /// every fragment from now on, in order.
    pub fn subscribe(&self) -> Option<std::sync::mpsc::Receiver<Vec<u8>>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(self.backlog);
        self.inner.lock().ok()?.subscribers.push(tx);
        Some(rx)
    }
}

/// a frame the loop has handed over, not yet encoded.
///
/// one slot, newest wins. the loop must never wait for the preview and must
/// never queue for it: a frame the encoder did not get to is worth nothing once
/// the next one exists.
struct Offered {
    rgb: Arc<Vec<u8>>,
    width: u32,
    height: u32,
    overlay: String,
}

pub struct Preview {
    latest: Arc<(Mutex<Latest>, Condvar)>,
    offered: Arc<(Mutex<Option<Offered>>, Condvar)>,
    viewers: Arc<AtomicUsize>,
}

impl Preview {
    /// `crops_dir` is the harvest directory, browsable at `/crops` (r8.5), and
    /// `clips` is the event-clip directory, browsable at `/clips` (r8.6) or
    /// `None` when nothing is being recorded. `video` is where the page should
    /// fetch pixels from, which may be the camera rather than us.
    // eight arguments is one over clippy's limit and it is right that it is a
    // smell -- these want to be a `Setup` struct. not done here only because
    // this signature is being edited in another session tonight and a struct
    // would collide with it; it is a five minute change once that settles.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        addr: &str,
        crops: Crops,
        clips: Option<Clips>,
        cfg: &crate::config::PreviewCfg,
        video: crate::config::PreviewVideo,
        relay: Option<Relay>,
        roi: &[[u32; 2]],
        perspective: &[[[u32; 2]; 2]],
        // the gate's own frame size. the overlay canvas is the *preview* size,
        // which is not the same thing, and the editor emits coordinates the
        // gate will read as its own -- so the page has to be able to convert.
        gate: (u32, u32),
        // the scene's shape, which is the main stream's: the encoded frame keeps
        // it, and the page has to lay the overlay over the picture rather than
        // over the box the picture was letterboxed inside.
        scene: (u32, u32),
        // the subject slugs the config watches for, which is what the
        // labelling picker opens on.
        subjects: &[String],
        stats: Arc<crate::stats::Stats>,
    ) -> Result<Self> {
        let server = tiny_http::Server::http(addr)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("binding preview server to {addr}"))?;

        let preview = Self {
            latest: Arc::new((Mutex::new(Latest::default()), Condvar::new())),
            offered: Arc::new((Mutex::new(None), Condvar::new())),
            viewers: Arc::new(AtomicUsize::new(0)),
        };
        // **the preview encodes on its own thread.**
        //
        // downscaling and jpeg-encoding every frame used to happen on the frame
        // loop, so the cost of showing a browser the picture was paid out of the
        // pipeline's frame rate -- and every viewer took the same lock the loop
        // needed to publish. measured on the deployment: 11 fps against a 15 fps
        // stream with a tab open, 15 fps the moment it was closed.
        //
        // a preview is a convenience and the pipeline is the product. no browser,
        // in any state, gets to slow it down.
        let (out_w, out_h) = encode_size(cfg.width, cfg.height, scene);
        encode_frames(
            preview.offered.clone(),
            preview.latest.clone(),
            video.needs_jpeg_encoding(),
            cfg.jpeg_quality,
            out_w,
            out_h,
        );
        let (src, kind) = video.page_src();
        let ctx = Serving {
            latest: preview.latest.clone(),
            viewers: preview.viewers.clone(),
            stats,
            crops,
            crops_listed: cfg.crops_listed,
            subjects: subjects.to_vec(),
            clips_listed: cfg.clips_listed,
            config: page_config(PageFacts {
                src,
                kind,
                events: clips.is_some(),
                cfg,
                gate,
                encode: (out_w, out_h),
                roi,
                perspective,
                subjects,
            }),
            clips,
            relay,
        };
        tracing::info!("preview at http://{addr}/ (video: {kind} from {src})");
        std::thread::spawn(move || serve(server, ctx));
        Ok(preview)
    }

    /// true when at least one browser is connected.
    ///
    /// this gates *publishing*, not encoding. the two were briefly conflated and
    /// it broke the page: in the modes where metermate does not supply the
    /// pixels this returned false forever, `publish` was never called, and the
    /// detections feed it also drives never emitted anything -- so the overlay
    /// stayed empty and the page sat on "waiting for frames" with a working
    /// video behind it. whether to encode jpeg is a separate question, asked
    /// inside `publish`.
    pub fn watched(&self) -> bool {
        self.viewers.load(Ordering::Relaxed) > 0
    }

    /// hand this frame to the preview and carry on.
    ///
    /// **never blocks and never queues.** the loop's job is the pipeline; the
    /// preview is a convenience. one slot, newest wins -- a frame the encoder did
    /// not get to is worthless the moment the next one exists, and waiting for it
    /// would make a browser able to slow detection down, which it must not be.
    pub fn offer(&self, rgb: Arc<Vec<u8>>, width: u32, height: u32, overlay: String) {
        if !self.watched() {
            return;
        }
        let (lock, cv) = &*self.offered;
        if let Ok(mut slot) = lock.lock() {
            *slot = Some(Offered {
                rgb,
                width,
                height,
                overlay,
            });
            cv.notify_one();
        }
    }
}

/// downscale and encode offered frames, off the frame loop.
///
/// this used to run inline, so the cost of showing a browser the picture came
/// out of the pipeline's frame rate: measured on the deployment, 11 fps against
/// a 15 fps stream with a tab open and 15 the moment it closed.
fn encode_frames(
    offered: Arc<(Mutex<Option<Offered>>, Condvar)>,
    latest: Arc<(Mutex<Latest>, Condvar)>,
    serves_video: bool,
    quality: u8,
    out_w: u32,
    out_h: u32,
) {
    std::thread::spawn(move || {
        loop {
            let work = {
                let (lock, cv) = &*offered;
                let Ok(mut slot) = lock.lock() else { return };
                loop {
                    if let Some(w) = slot.take() {
                        break w;
                    }
                    match cv.wait(slot) {
                        Ok(next) => slot = next,
                        Err(_) => return,
                    }
                }
            };

            let mut jpeg = Vec::new();
            if serves_video {
                let Some(px) =
                    crate::crop::downscale(&work.rgb, work.width, work.height, out_w, out_h)
                else {
                    continue;
                };
                let encoder = jpeg_encoder::Encoder::new(Cursor::new(&mut jpeg), quality);
                if let Err(e) = encoder.encode(
                    &px,
                    out_w as u16,
                    out_h as u16,
                    jpeg_encoder::ColorType::Rgb,
                ) {
                    tracing::warn!("preview encode failed: {e}");
                    continue;
                }
            }

            let (lock, cv) = &*latest;
            if let Ok(mut l) = lock.lock() {
                if serves_video {
                    l.jpeg = Arc::new(jpeg);
                }
                l.detections = Arc::new(work.overlay);
                l.seq += 1;
                cv.notify_all();
            }
        }
    });
}

/// everything a request handler needs that does not change between requests.
/// cloned per connection, which is cheap: the two shared states are behind
/// `Arc`s and the rest is a directory name and a few numbers.
#[derive(Clone)]
struct Serving {
    latest: Arc<(Mutex<Latest>, Condvar)>,
    stats: Arc<crate::stats::Stats>,
    viewers: Arc<AtomicUsize>,
    crops: Crops,
    crops_listed: usize,
    /// the subjects the config watches for, so the label picker can offer them
    /// before any of them has been used.
    subjects: Vec<String>,
    clips: Option<Clips>,
    clips_listed: usize,
    config: String,
    relay: Option<Relay>,
}

fn serve(server: tiny_http::Server, ctx: Serving) {
    for request in server.incoming_requests() {
        let full = request.url().to_string();
        let (url, query) = match full.split_once('?') {
            Some((path, q)) => (path.to_string(), q.to_string()),
            None => (full, String::new()),
        };
        let ctx = ctx.clone();
        // one thread per connection: the mjpeg and event streams never return,
        // and a preview is never going to have many viewers.
        std::thread::spawn(move || match url.as_str() {
            "/" => respond(
                request,
                "text/html; charset=utf-8",
                INDEX_HTML.as_bytes().to_vec(),
            ),
            "/favicon.png" => respond(request, "image/png", FAVICON.to_vec()),
            "/stream.mjpg" => stream_mjpeg(request, ctx.latest, ctx.viewers),
            // where the page should get video, and which tabs it can offer.
            // asked for on load, before the page commits to a source, so the
            // choice lives in one place -- the config -- rather than being
            // baked into the html.
            "/config" => respond(request, "application/json", ctx.config.into_bytes()),
            // **what the pipeline is doing, as numbers.** the only page that can
            // be read from somewhere else, which is where the questions come from
            // when something is wrong. every stall this program can have shows up
            // here as an age that has stopped moving.
            "/stats" => respond(
                request,
                "application/json",
                ctx.stats
                    .json(ctx.viewers.load(Ordering::Relaxed))
                    .into_bytes(),
            ),
            // the camera's own h.264, remuxed. only reachable when configured,
            // so a stray request cannot make us open an rtsp session.
            "/stream.mp4" => match &ctx.relay {
                Some(r) => stream_remux(request, r),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            // the detections feed counts as a viewer too. counting only mjpeg
            // clients meant a browser that opened this socket first would wait
            // forever: publishing is gated on the viewer count, and the count
            // was still zero.
            "/detections" => stream_detections(request, ctx.latest, ctx.viewers),
            // the crop and clip endpoints deliberately do *not* count as
            // viewers. browsing what was kept must not make the pipeline start
            // encoding preview frames for nobody (r8.3).
            "/crops" => respond(
                request,
                "application/json",
                crops_json(
                    &ctx.crops,
                    ctx.crops_listed,
                    &query,
                    &ctx.stats,
                    &ctx.subjects,
                )
                .into_bytes(),
            ),
            // one crop by name, which is what a hash link -- a tapped
            // notification, most of the time -- resolves against.
            "/crop" => match crop_json(&ctx.crops, &query) {
                Some(body) => respond(request, "application/json", body.into_bytes()),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            "/label" => label_crop(request, &ctx.crops.dir),
            "/labels" => respond(
                request,
                "application/json",
                labels_json(&ctx.crops.dir, &ctx.subjects),
            ),
            "/select" => match &ctx.clips {
                Some(c) => select_clip(request, &c.dir),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            "/unselect" => match &ctx.clips {
                Some(c) => unselect_clip(request, &c.dir),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            "/selections" => match &ctx.clips {
                Some(c) => respond(request, "application/json", selections_json(&c.dir)),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            path if path.starts_with("/crops/") => serve_crop(request, &ctx.crops.dir, &path[7..]),
            // 404 rather than an empty list when nothing is being recorded: the
            // page hides the tab, and a stray request should say the feature is
            // off rather than that the street was quiet.
            "/clips" => match &ctx.clips {
                Some(c) => respond(
                    request,
                    "application/json",
                    clips_json(c, ctx.clips_listed, &query).into_bytes(),
                ),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            // which recording was running when a crop was taken. asked only
            // when somebody clicks, rather than carried on every crop in every
            // page: it costs a listing, and a scroll asks for many pages.
            "/clip-at" => match &ctx.clips {
                Some(c) => match cursor_at(&query).and_then(|t| clip_at_json(c, t)) {
                    Some(body) => respond(request, "application/json", body.into_bytes()),
                    None => {
                        let _ = request.respond(tiny_http::Response::empty(404));
                    }
                },
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            path if path.starts_with("/clips/") => match &ctx.clips {
                Some(c) => serve_clip(request, &c.dir, &path[7..]),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            // a still from inside the clip, because the browser cannot seek a
            // fragmented mp4 to cut one for itself.
            path if path.starts_with("/poster/") => match &ctx.clips {
                Some(c) => serve_poster(request, c, &path[8..]),
                None => {
                    let _ = request.respond(tiny_http::Response::empty(404));
                }
            },
            _ => {
                let _ = request.respond(tiny_http::Response::empty(404));
            }
        });
    }
}

/// serve the remuxed stream to one browser.
///
/// the bytes come from the relay, which is fed by the ffmpeg already decoding
/// the main stream for crops. no process is spawned here and no rtsp session is
/// opened, however many viewers arrive.
fn stream_remux(request: tiny_http::Request, relay: &Relay) {
    use std::io::Write;

    // a viewer joining mid-stream needs the header before any fragment.
    //
    // subscribe *before* waiting for the init, so no fragment can slip through
    // the gap between the two and leave the viewer starting from a hole. the
    // wait then costs nothing: fragments queue behind the header they belong
    // to, which is the order they are wanted in anyway.
    let (Some(fragments), Some(init)) = (
        relay.subscribe(),
        relay.init_within(crate::config::PREVIEW_START_WAIT),
    ) else {
        let _ = request.respond(tiny_http::Response::empty(503));
        return;
    };

    let mut writer = request.into_writer();
    // no content-length: the stream has no end. no-store because a browser
    // caching a live feed would replay it forever.
    let head = "HTTP/1.1 200 OK\r\nContent-Type: video/mp4\r\n\
                Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    if writer.write_all(head.as_bytes()).is_err() || writer.write_all(&init).is_err() {
        return;
    }

    for fragment in fragments {
        if writer.write_all(&fragment).is_err() {
            break; // the viewer navigated away
        }
    }
}

/// a page of crops, as json, for the viewer to render.
///
/// `?before=<millis>` walks backwards through the harvest as the viewer
/// scrolls. without it the newest page is returned.
/// the `before=` cursor both grids scroll on: the stamp of the oldest card the
/// page already holds. shared so the two cannot disagree about the spelling.
fn cursor(query: &str) -> Option<u128> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == "before")
        .and_then(|(_, v)| v.parse().ok())
}

/// which crops a listing is asking for. `?recognised=1` is the verdicts tab.
fn recognised_only(query: &str) -> bool {
    query.split('&').any(|kv| kv == "recognised=1")
}

/// what the verdicts tab lists, and what a person has taken off it.
#[derive(Default)]
struct Standing {
    names: std::collections::BTreeSet<String>,
    rejected: usize,
    unclear: usize,
}

/// the crops that stand as sightings of a subject, read off names and labels.
///
/// the classifier's verdict rides in the filename and a person's judgement
/// lives in `labels.txt`, and where both exist the person wins:
///
/// - **an unjudged verdict stands**, since the claim is all there is.
/// - **a crop labelled a watched subject stands**, whatever its name says. a
///   subject stage two missed is the sighting most worth seeing here, and it
///   used to sit unmarked among the cars.
/// - **a verdict labelled anything else is settled and leaves.** `other` or
///   another name is a rejection. `unclear` is not one, but it is an answer,
///   and a tab of open questions is no place to keep the ones already asked;
///   it is counted apart so an empty tab does not read as a wrong classifier.
///
/// **counted over the crops still here.** a label outlives the crop it names --
/// the harvest evicts oldest first under its disk budget, and of 6213 labels in
/// one set 1598 already named crops that were gone. a count taken straight off
/// the file would explain an empty tab with rejections it cannot show.
fn standing(crops: &Path, labels: &crate::label::Labels, subjects: &[String]) -> Standing {
    let mut out = Standing::default();
    for crop in crate::harvest::all(crops) {
        let truth = labels.entries.get(&crop.name).map(|e| e.truth.as_str());
        let stands = match truth {
            None => crop.subject.is_some(),
            Some(t) => crop.subject.as_deref() == Some(t) || subjects.iter().any(|s| s == t),
        };
        match (stands, &crop.subject, truth) {
            (true, _, _) => {
                out.names.insert(crop.name);
            }
            (false, Some(_), Some(crate::label::UNCLEAR)) => out.unclear += 1,
            (false, Some(_), _) => out.rejected += 1,
            (false, None, _) => {}
        }
    }
    out
}

/// `n=` out of a query: the name of the single thing being asked for.
fn named(query: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == "n")
        .map(|(_, v)| v.to_string())
}

/// `t=` out of a query: the millisecond something happened.
fn cursor_at(query: &str) -> Option<u128> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == "t")
        .and_then(|(_, v)| v.parse().ok())
}

/// the clip that was recording at `at_millis`, if one was.
fn clip_at_json(clips: &Clips, at_millis: u128) -> Option<String> {
    let all = crate::record::list(&clips.dir);
    let (clip, at) =
        crate::record::containing(&all, at_millis, clips.preroll_ms, clips.max_clip_ms)?;
    Some(format!(
        r#"{{"n":"{}","w":"{}","at":{at:.1},"t":{}}}"#,
        clip.name,
        clip.worth.slug(),
        clip.stamp
    ))
}

/// one crop, as the page's grid and as a link to that one crop both need it.
///
/// every field comes back out of the name, because the name is the record: the
/// harvester writes what it decided into the filename and keeps nothing else per
/// crop.
///
/// `s` is absent unless stage two named a subject, which is well under a percent
/// of the harvest, `m` only when that verdict's margin was recorded, and `a` only
/// on a verdict whose vehicle went on to alert -- a null on every crop would be
/// most of what either field ever says. `truth` is absent until a person has said
/// something.
fn crop_entry(
    crop: &crate::harvest::Saved,
    alerted: &std::collections::BTreeSet<String>,
    truth: Option<&str>,
) -> String {
    let verdict = crop
        .subject
        .as_ref()
        .map_or(String::new(), |s| format!(r#","s":"{s}""#));
    // the margin a verdict was reached at, which the verdicts tab captions with.
    // absent rather than zero when the crop predates the record: a number that
    // looks measured is worse than a gap in the record.
    let margin = crop
        .margin
        .map_or(String::new(), |m| format!(r#","m":{m:.3}"#));
    // a verdict is a single frame; `a` is the green border's one claim, that the
    // vehicle behind it alerted.
    let fired = if !verdict.is_empty() && alerted.contains(&crop.name) {
        r#","a":1"#
    } else {
        ""
    };
    let truth = truth.map_or(String::new(), |t| format!(r#","truth":"{t}""#));
    format!(
        r#"{{"n":"{}","c":"{}","p":{:.2},"t":{}{verdict}{margin}{fired}{truth}}}"#,
        crop.name, crop.label, crop.confidence, crop.at_millis
    )
}

/// `/crop?n=`: the listing entry for one named crop.
///
/// **the hash links are the page's urls**, and a link names a crop the grid has
/// not paged to -- usually several pages down, since the page opens on the newest
/// arrivals. asking for the one entry costs less than paging to it, and less than
/// a second copy of the filename's grammar in javascript would.
///
/// the file has to be there: the harvest evicts oldest first, so a link a week
/// old can name a crop that is gone, and the page would rather show the tab it
/// belongs to than an overlay of a broken image.
fn crop_json(crops: &Crops, query: &str) -> Option<String> {
    let name = named(query)?;
    let found = crate::harvest::parse_name(&name)?;
    // the name is a path a browser chose, so it is checked the way
    // `/crops/<name>` checks it, and then for existing.
    if !crop_name_is_safe(&name) || !crops.dir.join(&name).is_file() {
        return None;
    }
    Some(crop_entry(
        &found,
        &crate::harvest::alerted(&crops.dir),
        None,
    ))
}

fn crops_json(
    crops: &Crops,
    listed: usize,
    query: &str,
    stats: &crate::stats::Stats,
    subjects: &[String],
) -> String {
    // the decisions, read fresh each listing rather than cached: a labelling
    // session in another process is writing to the same file, and a grid that
    // does not show its own labels gets the same crop labelled twice, the
    // second time without seeing the first.
    let (labels_path, _) = crate::label::paths(&crops.dir);
    let labels = crate::label::Labels::load(&labels_path).unwrap_or_default();
    // built before the page is cut, because the filter has to be: cutting first
    // would hand back a short page and stall the scroll.
    let stands = match recognised_only(query) {
        true => standing(&crops.dir, &labels, subjects),
        false => Standing::default(),
    };
    let want = match recognised_only(query) {
        true => crate::harvest::Want::Named(&stands.names),
        false => crate::harvest::Want::Everything,
    };
    // read per listing, like the labels: the pipeline appends to it while this
    // page is open, and the green has to follow.
    let alerted = crate::harvest::alerted(&crops.dir);
    let items: Vec<String> = crate::harvest::page(&crops.dir, cursor(query), listed, want)
        .into_iter()
        .map(|c| {
            crop_entry(
                &c,
                &alerted,
                labels.entries.get(&c.name).map(|e| e.truth.as_str()),
            )
        })
        .collect();
    // how many standing verdicts alerted, so the tab can say how much of what
    // it lists is green without paging to the end to count.
    let standing_alerted = stands.names.intersection(&alerted).count();
    let (kept, used) = stats.harvest_on_disk_now();
    format!(
        r#"{{"kept":{kept},"used":{used},"budget":{},"classifying":{},"alerted":{standing_alerted},"rejected":{},"unclear":{},"crops":[{}]}}"#,
        crops.budget_bytes,
        crops.classifying,
        // **an empty verdicts tab has two meanings and they are opposite.**
        // nothing has fired, or everything that fired was wrong and has been
        // thrown out. on the first evening stage two ran, every crop it named
        // was a false positive, so the second is the case that actually
        // happens -- and reporting it as "nothing recognised yet" would hide
        // exactly the result worth knowing.
        stands.rejected,
        stands.unclear,
        items.join(",")
    )
}

/// serve one harvested crop by name.
///
/// record what a person said about a crop, from the live page.
///
/// **the same `labels.txt` the labelling page writes** (r6.1): a decision made
/// while watching the street is a decision `--measure` and `--train` see, with
/// no import step and no second format. it is appended rather than merged,
/// because a labelling session is usually a second process against the same
/// file and a read-modify-write here would drop whatever it wrote in between.
///
/// the crop name decides which file is named, so it is checked the way
/// `/crops/<name>` checks it, and then checked again for existing: a label
/// naming a crop that is not in the harvest is a line nothing can ever use.
fn label_crop(mut request: tiny_http::Request, dir: &Path) {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        let _ = request.respond(tiny_http::Response::empty(400));
        return;
    }
    let (Some(name), Some(truth)) = (field(&body, "n"), field(&body, "truth")) else {
        let _ = request.respond(tiny_http::Response::empty(400));
        return;
    };
    if !crop_name_is_safe(&name) || crop_bytes(dir, &name).is_none() {
        tracing::warn!("rejected a label for {name:?}");
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    }
    // a subject invented in a text field must be one the config would accept,
    // or the labels are stranded under a name no `[[subject]]` can carry.
    let known = truth == crate::label::NEGATIVE || truth == crate::label::UNCLEAR;
    if !known && !crate::config::is_subject_name(&truth) {
        let _ = request.respond(
            tiny_http::Response::from_string(
                "a subject name holds lowercase letters, digits and single dashes",
            )
            .with_status_code(400),
        );
        return;
    }
    let (labels, _) = crate::label::paths(dir);
    match crate::label::Labels::append(&labels, &name, &truth, crate::label::Via::Preview) {
        Ok(()) => {
            tracing::info!("labelled {name} {truth}");
            let _ = request.respond(tiny_http::Response::from_string(r#"{"ok":true}"#));
        }
        Err(e) => {
            tracing::warn!("{e:#}");
            let _ = request.respond(tiny_http::Response::empty(500));
        }
    }
}

/// the labels in use, most worth offering first.
///
/// **frecency, not an alphabet.** on any given day two or three labels are
/// being applied and the rest are history: the one typed a minute ago is
/// almost certainly the next one, and a list sorted by name buries it under
/// whatever begins with an `a`. so a label's score is how often it has been
/// used divided by how long ago it last was, and the page offers the top of
/// that before anybody types anything.
///
/// "how long ago" is the timestamp in the crop's own name rather than when the
/// decision was written: `labels.txt` records no decision times, and a crop's
/// stamp is what it is about.
fn labels_json(dir: &Path, subjects: &[String]) -> Vec<u8> {
    let (labels_path, _) = crate::label::paths(dir);
    let labels = crate::label::Labels::load(&labels_path).unwrap_or_default();
    let mut counted: std::collections::BTreeMap<String, (usize, u128)> = Default::default();
    // the reserved answers and the configured subjects are always offerable,
    // with no uses yet: a fresh deployment must not open on an empty list.
    for name in subjects
        .iter()
        .map(String::as_str)
        .chain([crate::label::NEGATIVE, crate::label::UNCLEAR])
    {
        counted.entry(name.to_string()).or_insert((0, 0));
    }
    for (crop, entry) in &labels.entries {
        let at = crate::harvest::parse_name(crop).map_or(0, |s| s.at_millis);
        let seen = counted.entry(entry.truth.clone()).or_insert((0, 0));
        seen.0 += 1;
        seen.1 = seen.1.max(at);
    }
    let newest = counted.values().map(|(_, at)| *at).max().unwrap_or(0);
    let mut ranked: Vec<(f64, String, usize)> = counted
        .into_iter()
        .map(|(name, (uses, last))| {
            let days = (newest.saturating_sub(last)) as f64 / 86_400_000.0;
            (uses as f64 / (1.0 + days), name, uses)
        })
        .collect();
    // ties by name, so the order is stable between listings rather than
    // shuffling under the pointer.
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let rows: Vec<String> = ranked
        .iter()
        .map(|(_, name, uses)| format!(r#"{{"truth":"{name}","uses":{uses}}}"#))
        .collect();
    format!(r#"{{"labels":[{}]}}"#, rows.join(",")).into_bytes()
}

/// mark a passage of a clip to be cut into a set later.
///
/// **written down rather than acted on.** cutting crops reads every frame of
/// the clip through the detector, and this page is served by the process
/// watching the street (r2.2, r3.1). so the window is parsed here -- failing
/// now rather than in `--prepare` an hour later -- and then filed.
fn select_clip(mut request: tiny_http::Request, events: &Path) {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        let _ = request.respond(tiny_http::Response::empty(400));
        return;
    }
    let (Some(clip), Some(window)) = (field(&body, "n"), field(&body, "window")) else {
        let _ = request.respond(tiny_http::Response::empty(400));
        return;
    };
    if !clip_name_is_safe(&clip) || !events.join(&clip).is_file() {
        tracing::warn!("rejected a selection of {clip:?}");
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    }
    if let Err(e) = crate::harvest::dense::parse_window(&window) {
        let _ = request
            .respond(tiny_http::Response::from_string(format!("{e:#}")).with_status_code(400));
        return;
    }
    match crate::record::select::append(&crate::record::select::path(events), &clip, &window) {
        Ok(()) => {
            tracing::info!("selected {window} of {clip} for cropping");
            let _ = request.respond(tiny_http::Response::from_string(r#"{"ok":true}"#));
        }
        Err(e) => {
            tracing::warn!("{e:#}");
            let _ = request.respond(tiny_http::Response::empty(500));
        }
    }
}

/// take a mark back.
///
/// **a window is judged in a second, off a moving picture**, so marking the
/// wrong clip or three seconds of empty street is ordinary. without this the
/// way back is editing a file on the deployment, or letting `--prepare` spend
/// minutes cutting a passage nobody wants.
fn unselect_clip(mut request: tiny_http::Request, events: &Path) {
    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        let _ = request.respond(tiny_http::Response::empty(400));
        return;
    }
    let (Some(clip), Some(window)) = (field(&body, "n"), field(&body, "window")) else {
        let _ = request.respond(tiny_http::Response::empty(400));
        return;
    };
    if !clip_name_is_safe(&clip) {
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    }
    match crate::record::select::remove(&crate::record::select::path(events), &clip, &window) {
        Ok(gone) => {
            if gone > 0 {
                tracing::info!("unmarked {window} of {clip}");
            }
            let _ = request.respond(tiny_http::Response::from_string(format!(
                r#"{{"removed":{gone}}}"#
            )));
        }
        Err(e) => {
            tracing::warn!("{e:#}");
            let _ = request.respond(tiny_http::Response::empty(500));
        }
    }
}

/// what is waiting to be cut, so the tab can say so rather than leaving a
/// person to wonder whether the click landed.
fn selections_json(events: &Path) -> Vec<u8> {
    let waiting = crate::record::select::load(&crate::record::select::path(events))
        .unwrap_or_default()
        .iter()
        .map(|s| format!(r#"{{"n":"{}","window":"{}"}}"#, s.clip, s.window))
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"selections":[{waiting}]}}"#).into_bytes()
}

/// one string field out of a flat json object, without a json parser.
///
/// the two fields this takes are a crop name and a subject name, and both are
/// already constrained to `[a-z0-9._-]` by the checks above -- so anything an
/// escape sequence could smuggle in is refused a line later. a parser would be
/// a dependency for two `"key":"value"` lookups.
fn field(body: &str, key: &str) -> Option<String> {
    let at = body.find(&format!("\"{key}\""))?;
    let rest = body[at + key.len() + 2..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// the name arrives from the url, so it decides which file is read. it is
/// checked two ways rather than one: it must contain no path separator or
/// parent reference, and it must parse as a crop name. a percent-encoded
/// `..%2f` defeats a naive separator check but not the shape check, and a
/// filename that is merely odd defeats the shape check but not the separator
/// one.
fn serve_crop(request: tiny_http::Request, dir: &Path, name: &str) {
    if !crop_name_is_safe(name) {
        tracing::warn!("rejected crop request for {name:?}");
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    }
    match crop_bytes(dir, name) {
        // a crop never changes: its name carries the millisecond it was taken,
        // and the harvester only ever adds or deletes whole files. so tell the
        // browser to keep it forever and not even revalidate -- the viewer
        // rebuilds its grid every few seconds, and without this every visible
        // crop is re-requested each time.
        Some(bytes) => respond_with(
            request,
            "image/jpeg",
            bytes,
            &[("Cache-Control", "public, max-age=31536000, immutable")],
        ),
        None => {
            let _ = request.respond(tiny_http::Response::empty(404));
        }
    }
}

/// the event clips, newest first, with what the directory as a whole costs.
///
/// `kept` and `used` are sent alongside the page rather than left implicit: a
/// listing truncated to `listed` would otherwise read as the whole of what was
/// recorded, and a viewer deciding whether the budget is too small needs the
/// total rather than the visible part of it.
fn clips_json(clips: &Clips, listed: usize, query: &str) -> String {
    let all = crate::record::list(&clips.dir);
    // every file on disk, because this is measured against the disk budget.
    let used: u64 = all.iter().map(|c| c.bytes).sum();
    // **one card per event, not per file.** an event is recorded once per
    // configured stream -- main for watching, sub because the gate reads the
    // camera's own anamorphic encode and a replay needs it -- so a grid built
    // from the file list shows everything twice. main is what a person wants to
    // look at; sub is the fallback for a `streams = ["sub"]` deployment and for
    // every directory written before the pair existed.
    let mut events: Vec<&crate::record::Clip> = Vec::new();
    for c in &all {
        match events
            .iter_mut()
            .find(|e| e.stamp == c.stamp && e.worth == c.worth)
        {
            Some(chosen) => {
                if c.stream == crate::record::MAIN_STREAM {
                    *chosen = c;
                }
            }
            None => events.push(c),
        }
    }
    let kept = events.len();
    // **the cursor cuts the deduped list, never the file list.** an event is one
    // card but up to two files, so a boundary drawn across the directory can
    // fall between a main/sub pair -- and the sub half then arrives on the next
    // page as a card of its own. deduping per page instead loses the event
    // entirely when its halves straddle the cut.
    let items: Vec<String> = events
        .iter()
        .filter(|c| cursor(query).is_none_or(|before| c.stamp < before))
        .take(listed)
        .map(|c| {
            // how long it ran, from the file's mtime -- the only record of when
            // a clip stopped, since its name carries only when it started. the
            // card shows it because "this is 19 seconds of a 2 second event" is
            // the thing worth knowing about a clip before opening it.
            //
            // zero, meaning the card says nothing, when that works out longer
            // than a clip is allowed to run. an mtime does not survive a copy,
            // so a directory moved with `cp` claims every clip lasted as long
            // as it has been since it was recorded, and a wrong number on a
            // card is worse than no number.
            let ran = c.ended_ms.saturating_sub(c.stamp) + clips.preroll_ms;
            let secs = if ran <= clips.max_clip_ms + clips.preroll_ms {
                ran as f64 / 1000.0
            } else {
                0.0
            };
            format!(
                r#"{{"n":"{}","t":{},"w":"{}","s":"{}","b":{},"d":{secs:.1}}}"#,
                c.name,
                c.stamp,
                c.worth.slug(),
                c.stream,
                c.bytes
            )
        })
        .collect();
    format!(
        r#"{{"kept":{kept},"used":{used},"budget":{},"recording":{},"preroll":{},"dir":"{}","clips":[{}]}}"#,
        clips.budget_bytes,
        clips.recording,
        // seconds into a clip that its trigger sits at, so the page can seek
        // each card's poster past the pre-roll rather than opening on the empty
        // street the pre-roll exists to capture.
        clips.preroll_ms as f64 / 1000.0,
        // **exactly as configured, relative path and all.** it is there so a
        // card can offer the path `--dense` takes, and that argument is
        // resolved against the working directory metermate itself was started
        // in -- so `data/events` is the useful answer and its absolute
        // expansion is a path the person reading it did not write.
        crate::label::server::escape(&clips.dir.to_string_lossy()),
        items.join(",")
    )
}

/// what the page is told on load, before it commits to anything.
///
/// **the gate's frame size is in here because the overlay canvas is not it.**
/// the canvas is sized to the preview image -- `[preview] width`/`height`,
/// 640x360 by default -- while `[gate] roi` is read in `[stream]
/// gate_width`/`gate_height`, 640x480. the editor drew in one and wrote out the
/// other, so every polygon ever drawn had its y compressed by 0.75, and on the
/// deployment that put a region meant to cover the road onto the buildings
/// above it. the page needs both numbers to convert, so it is given both.
///
/// the roi and the lines go too, so the editor opens on what is configured
/// rather than on a blank canvas -- which is how an existing polygon gets
/// replaced by accident.
struct PageFacts<'a> {
    src: &'a str,
    kind: &'a str,
    events: bool,
    cfg: &'a crate::config::PreviewCfg,
    gate: (u32, u32),
    /// what the frame is actually encoded to, which is what the browser shows.
    encode: (u32, u32),
    roi: &'a [[u32; 2]],
    perspective: &'a [[[u32; 2]; 2]],
    subjects: &'a [String],
}

fn page_config(facts: PageFacts) -> String {
    let PageFacts {
        src,
        kind,
        events,
        cfg,
        gate,
        encode,
        roi,
        perspective,
        subjects,
    } = facts;
    let points: Vec<String> = roi.iter().map(|p| format!("[{},{}]", p[0], p[1])).collect();
    let lines: Vec<String> = perspective
        .iter()
        .map(|[a, b]| format!("[[{},{}],[{},{}]]", a[0], a[1], b[0], b[1]))
        .collect();
    format!(
        // positional rather than named: `concat!` expands before
        // `format_args!`, which then refuses to capture from scope.
        concat!(
            r#"{{"src":"{}","kind":"{}","events":{},"#,
            r#""buffer_ms":{},"max_buffer_ms":{},"reconnect_ms":{},"catch_up":"{}","#,
            r#""show_parked":{},"gate":[{},{}],"encode":[{},{}],"roi":[{}],"perspective":[{}],"subjects":[{}]}}"#
        ),
        src,
        kind,
        events,
        cfg.target_buffer_ms,
        cfg.max_buffer_ms,
        cfg.reconnect_min_ms,
        cfg.catch_up.as_str(),
        // whether the page may promise a dashed box at all: scenery is filtered
        // out of the feed rather than left for the browser to hide, so a legend
        // that described it would be describing nothing.
        cfg.show_parked,
        gate.0,
        gate.1,
        encode.0,
        encode.1,
        points.join(","),
        lines.join(","),
        // what the picker offers before anybody types a new one. the config is
        // the authority on which subjects are *watched for*; the page can name
        // one that is not here, and those labels train, but nothing fires on a
        // subject until a `[[subject]]` block says so.
        subjects
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// the size a gate frame is encoded to: `[preview] width`x`height` is a box, and
/// the picture keeps the scene's aspect inside it.
///
/// the scene's aspect is the main stream's, since the two streams share a field
/// of view, so this un-squashes an anamorphic substream for display -- 640x480 of
/// pixels showing a 16:9 scene becomes 640x360, which is what the two numbers
/// have always spelled out by hand -- leaves a square-pixel substream alone, and
/// stops squeezing a 64:29 street into 16:9.
///
/// the aspect matters beyond looks. a browser contains an image whose shape
/// differs from its box while the overlay canvas covers the box, so the two
/// disagreeing is boxes floating above the vehicles they belong to.
pub fn encode_size(box_w: u32, box_h: u32, scene: (u32, u32)) -> (u32, u32) {
    if scene.0 == 0 || scene.1 == 0 {
        return (box_w, box_h);
    }
    let w = (box_h * scene.0 / scene.1).clamp(2, box_w);
    let h = ((w as u64 * scene.1 as u64 / scene.0 as u64) as u32).clamp(2, box_h);
    (w, h)
}

/// a still from inside one clip, for its card in the events grid.
///
/// **cut from the middle rather than from the trigger.** the trigger is the
/// moment motion fired, which is before the vehicle has gone anywhere, and the
/// first frame is the pre-roll, which is the one part of a clip guaranteed to
/// hold nothing. a clip is the pre-roll, the passage, and a tail of about the
/// same length as the pre-roll, so its midpoint is the middle of the passage --
/// and never earlier than the trigger, for the short clips where it would be.
fn serve_poster(request: tiny_http::Request, clips: &Clips, name: &str) {
    if !clip_name_is_safe(name) {
        tracing::warn!("rejected poster request for {name:?}");
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    }
    // the name has already been proved to be a clip name, so the path follows
    // from it -- rather than walking the directory to find the entry it names,
    // which is a full listing per card, and twice as many entries now that both
    // streams are recorded.
    let path = clips.dir.join(name);
    let (Some((stamp, _, _)), Some(ended_ms)) = (
        crate::record::parse_name(name),
        std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis()),
    ) else {
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    };
    let preroll = clips.preroll_ms as f64 / 1000.0;
    let ran = (ended_ms.max(stamp).saturating_sub(stamp) + clips.preroll_ms) as f64 / 1000.0;
    let at = if ran > 0.0 && ran <= (clips.max_clip_ms + clips.preroll_ms) as f64 / 1000.0 {
        (ran / 2.0).max(preroll)
    } else {
        preroll
    };
    // **taken from the substream when there is one.** the same moment of the
    // same event at a twentieth of the pixels: measured, 55 MB to decode
    // against 256 for the main stream, which is what makes this affordable
    // inside a memory-capped service. main is the fallback, for a
    // `streams = ["main"]` deployment and for clips written before the pair
    // existed -- the rule the cards use, inverted.
    let from = crate::record::sibling(&path, crate::record::SUB_STREAM).unwrap_or(path.clone());
    let cache = crate::record::poster_path(&clips.cache_dir, &path);
    match crate::record::poster(&from, &cache, at) {
        Ok(bytes) => respond_with(
            request,
            "image/jpeg",
            bytes,
            // a clip never changes once listed, so neither does a frame of it.
            &[("Cache-Control", "public, max-age=31536000, immutable")],
        ),
        Err(e) => {
            tracing::warn!("no poster for {name}: {e:#}");
            let _ = request.respond(tiny_http::Response::empty(404));
        }
    }
}

/// serve one event clip by name.
///
/// streamed from the file rather than read into memory: a clip is up to a
/// minute of the full-resolution main stream, which is tens of megabytes, where
/// a crop is a few kilobytes.
///
/// cached hard, for the same reason crops are: a listed clip is finished, and
/// its name carries the worth it was finished with -- the recorder writes to a
/// name this refuses until the moment it renames it into place. so a name that
/// appears here will never describe different bytes.
fn serve_clip(request: tiny_http::Request, dir: &Path, name: &str) {
    if !clip_name_is_safe(name) {
        tracing::warn!("rejected clip request for {name:?}");
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    }
    let Ok(file) = std::fs::File::open(dir.join(name)) else {
        let _ = request.respond(tiny_http::Response::empty(404));
        return;
    };
    let mut response = tiny_http::Response::from_file(file);
    for (name, value) in [
        ("Content-Type", "video/mp4"),
        ("Cache-Control", "public, max-age=31536000, immutable"),
    ] {
        if let Ok(h) = tiny_http::Header::from_bytes(name, value) {
            response = response.with_header(h);
        }
    }
    let _ = request.respond(response);
}

/// a crop worth answering with, or `None` for one that is missing or empty.
///
/// the two cases are the same answer on purpose. the harvester renames crops
/// into place so an empty one should not exist, but if one ever does it is a
/// file with no bytes yet -- and an empty 200 carrying `immutable` is a
/// permanent wrong answer, cached under a name that will never be re-requested.
/// a 404 is temporary.
fn crop_bytes(dir: &Path, name: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(dir.join(name)).ok()?;
    (!bytes.is_empty()).then_some(bytes)
}

/// block until a frame newer than `sent` exists, then hand back a share of it.
/// `None` means the stream is over and the connection should close.
///
/// the two clones are `Arc` clones: the lock is released the instant this
/// returns, and what happens after -- writing to a socket that may be slow, or
/// gone -- happens without it.
fn wait_for_frame(
    latest: &(Mutex<Latest>, Condvar),
    sent: &mut u64,
) -> Option<(Arc<Vec<u8>>, Arc<String>)> {
    let (lock, cv) = latest;
    let mut l = lock.lock().ok()?;
    while l.seq == *sent {
        l = cv.wait(l).ok()?;
    }
    *sent = l.seq;
    Some((Arc::clone(&l.jpeg), Arc::clone(&l.detections)))
}

fn respond(request: tiny_http::Request, content_type: &str, body: Vec<u8>) {
    respond_with(request, content_type, body, &[]);
}

fn respond_with(
    request: tiny_http::Request,
    content_type: &str,
    body: Vec<u8>,
    extra: &[(&str, &str)],
) {
    let mut response = tiny_http::Response::from_data(body)
        .with_header(tiny_http::Header::from_bytes("Content-Type", content_type).unwrap());
    for (name, value) in extra {
        if let Ok(h) = tiny_http::Header::from_bytes(*name, *value) {
            response = response.with_header(h);
        }
    }
    let _ = request.respond(response);
}

fn stream_mjpeg(
    request: tiny_http::Request,
    latest: Arc<(Mutex<Latest>, Condvar)>,
    viewers: Arc<AtomicUsize>,
) {
    use std::io::Write;

    viewers.fetch_add(1, Ordering::Relaxed);
    let mut writer = request.into_writer();
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary={BOUNDARY}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n"
    );
    if writer.write_all(head.as_bytes()).is_err() {
        viewers.fetch_sub(1, Ordering::Relaxed);
        return;
    }

    let mut sent = 0u64;
    while let Some((jpeg, _)) = wait_for_frame(&latest, &mut sent) {
        let part = format!(
            "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpeg.len()
        );
        if writer.write_all(part.as_bytes()).is_err()
            || writer.write_all(&jpeg).is_err()
            || writer.write_all(b"\r\n").is_err()
        {
            break;
        }
    }
    viewers.fetch_sub(1, Ordering::Relaxed);
}

/// server-sent events carrying the detections for the most recent frame.
fn stream_detections(
    request: tiny_http::Request,
    latest: Arc<(Mutex<Latest>, Condvar)>,
    viewers: Arc<AtomicUsize>,
) {
    use std::io::Write;

    viewers.fetch_add(1, Ordering::Relaxed);
    let mut writer = request.into_writer();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    if writer.write_all(head.as_bytes()).is_ok() {
        let mut sent = 0u64;
        while let Some((_, payload)) = wait_for_frame(&latest, &mut sent) {
            if writer
                .write_all(format!("data: {payload}\n\n").as_bytes())
                .is_err()
            {
                break;
            }
            let _ = writer.flush();
        }
    }
    viewers.fetch_sub(1, Ordering::Relaxed);
}

const INDEX_HTML: &str = include_str!("index.html");

/// **both servers serve this, so it lives outside either of them.** the
/// preview and the labelling page are separate processes on separate ports and
/// a person runs them side by side; identical blank tabs is how you end up
/// reloading the wrong one.
pub const FAVICON: &[u8] = include_bytes!("../../assets/favicon.png");

/// the same check `serve_crop` applies, factored out so it can be tested
/// without standing up a server.
fn crop_name_is_safe(name: &str) -> bool {
    !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
        && !name.contains('%')
        && crate::harvest::parse_name(name).is_some()
}

/// the same two checks, against the clip naming scheme. the shape check is what
/// keeps `.part` out: a clip still being written must not be served, because it
/// is cached as immutable and its final name is not yet decided.
fn clip_name_is_safe(name: &str) -> bool {
    !name.contains('/')
        && !name.contains('\\')
        && !name.contains("..")
        && !name.contains('%')
        && crate::record::parse_name(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **the other half of the crop that showed as black.** a crop caught
    /// between creation and its bytes reads as a valid, empty file, and an empty
    /// 200 marked `immutable` is a permanent answer: the browser will keep
    /// serving nothing for that name for a year, which is why clicking the tile
    /// changed nothing and only a reload did. an empty crop is one that is not
    /// there yet, and must be answered like one.
    #[test]
    fn an_empty_crop_file_is_not_served() {
        let d = std::env::temp_dir().join(format!("metermate-empty-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let name = "1789260000000_car_090_20x10.jpg";
        std::fs::write(d.join(name), b"").unwrap();
        assert!(crop_bytes(&d, name).is_none(), "an empty crop was served");

        std::fs::write(d.join(name), b"jpeg").unwrap();
        assert_eq!(crop_bytes(&d, name).as_deref(), Some(&b"jpeg"[..]));
        assert!(crop_bytes(&d, "1789260000001_car_090_20x10.jpg").is_none());
        std::fs::remove_dir_all(&d).ok();
    }

    /// the crop name comes from the url and picks a file to read, so this is
    /// the one place in the preview where being wrong matters.
    #[test]
    fn crop_names_that_escape_the_harvest_directory_are_refused() {
        for attempt in [
            "../../../etc/passwd",
            "../metermate.local.toml",
            "..%2f..%2fetc%2fpasswd",
            "%2e%2e/passwd",
            "/etc/passwd",
            "subdir/1757700000123_car_086_64x64.jpg",
            "..\\windows\\system32",
            // right shape, but still a traversal.
            "../1757700000123_car_086_64x64.jpg",
        ] {
            assert!(!crop_name_is_safe(attempt), "{attempt} was allowed");
        }
    }

    #[test]
    fn things_that_are_not_crops_are_refused() {
        for attempt in ["metermate.local.toml", "detector.onnx", "", "index.html"] {
            assert!(!crop_name_is_safe(attempt), "{attempt} was allowed");
        }
    }

    #[test]
    fn a_real_crop_name_is_allowed() {
        assert!(crop_name_is_safe("1757700000123_car_086_640x640.jpg"));
        assert!(crop_name_is_safe(
            "1757700000123_traffic-light_042_64x64.jpg"
        ));
    }

    /// the viewer fetches `/crops/<name>`, so a name the harvester writes must
    /// survive url encoding unchanged. otherwise the browser sends `%20`, the
    /// guard refuses `%`, and the crop 404s in the viewer for no visible
    /// reason. this is why the harvester sanitises labels on write.
    #[test]
    fn written_names_need_no_url_encoding() {
        let name = "1757700000123_traffic-light_042_64x64.jpg";
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)),
            "{name} would be percent-encoded by a browser"
        );
        assert!(crop_name_is_safe(name));
    }

    #[test]
    fn an_empty_harvest_directory_lists_as_empty_json() {
        let json = crops_json(
            &crops_of(Path::new("/nonexistent/metermate")),
            120,
            "",
            &Default::default(),
            &[],
        );
        assert!(json.contains(r#""crops":[]"#), "{json}");
    }

    /// **the harvest reports what it costs, as the events grid always has.**
    /// the two tabs answer the same question about the same kind of thing, and
    /// a grid that cannot say how full the disk is lets it fill unnoticed --
    /// which for a harvest that evicts oldest-first means losing the training
    /// data silently.
    #[test]
    fn the_harvest_listing_carries_its_budget() {
        let stats = crate::stats::Stats::default();
        stats.harvest_on_disk(412_093, 19_847_000_000);
        let json = crops_json(
            &crops_of(Path::new("/nonexistent/metermate")),
            120,
            "",
            &stats,
            &[],
        );
        assert!(json.contains(r#""kept":412093"#), "{json}");
        assert!(json.contains(r#""used":19847000000"#), "{json}");
        assert!(
            json.contains(&format!(r#""budget":{}"#, 20u64 << 30)),
            "{json}"
        );
    }

    /// **the one crop in two hundred worth looking at has to look different.**
    ///
    /// the grid is the first place a sighting is visible: a crop is written
    /// whenever a vehicle moves, long before enough looks agree for a clip to
    /// be named `subject`. the classifier's verdict rides in the crop's name,
    /// so the listing carries it and the grid can border those cards.
    #[test]
    fn the_harvest_listing_says_which_crops_were_recognised() {
        let d =
            std::env::temp_dir().join(format!("metermate-preview-verdict-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("1789260000200_car_go4_083_640x480.jpg"), b"x").unwrap();
        std::fs::write(d.join("1789260000100_car_079_640x480.jpg"), b"x").unwrap();

        let json = crops_json(&crops_of(&d), 120, "", &Default::default(), &[]);
        assert!(json.contains(r#""c":"car","p":0.83"#), "{json}");
        assert!(
            json.contains(r#""s":"go4""#),
            "recognised crop unmarked: {json}"
        );
        assert_eq!(
            json.matches(r#""s":"#).count(),
            1,
            "unrecognised crop marked: {json}"
        );

        // and the verdicts tab is the same listing with the query the page
        // sends, which is the only part of the path the grid cannot test.
        let only = crops_json(&crops_of(&d), 120, "recognised=1", &Default::default(), &[]);
        assert!(only.contains(r#""s":"go4""#), "{only}");
        assert_eq!(only.matches(r#""n":"#).count(), 1, "cars came back: {only}");
        // paging keeps the filter, or a scroll would hand back the harvest.
        let older = crops_json(
            &crops_of(&d),
            120,
            "recognised=1&before=1789260000200",
            &Default::default(),
            &[],
        );
        assert!(older.contains(r#""crops":[]"#), "{older}");
        std::fs::remove_dir_all(&d).ok();
    }

    /// **the verdict page is where a margin gets tuned, so the margin has to be
    /// on it.** the notification quotes `margin +0.020` and the bar it cleared is
    /// in `trained.toml`; deciding whether to move the bar means seeing the gaps
    /// that cleared it, which nothing but the crop itself records.
    #[test]
    fn the_verdicts_listing_carries_the_margin() {
        let d =
            std::env::temp_dir().join(format!("metermate-preview-margin-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("1789260000200_car_go4_20_083_640x480.jpg"), b"x").unwrap();
        // the same verdict from before the margin was recorded: no number, and
        // not the detector's standing in for one.
        std::fs::write(d.join("1789260000100_car_go4_083_640x480.jpg"), b"x").unwrap();

        let only = crops_json(&crops_of(&d), 120, "recognised=1", &Default::default(), &[]);
        assert!(only.contains(r#""m":0.020"#), "{only}");
        assert_eq!(
            only.matches(r#""m":"#).count(),
            1,
            "a crop with no recorded margin was given one: {only}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    /// the clip name comes from the url and picks a file to read, exactly as a
    /// crop name does. a clip is the full-resolution main stream, so being
    /// wrong here hands over rather more than a thumbnail.
    #[test]
    fn clip_names_that_escape_the_events_directory_are_refused() {
        for attempt in [
            "../../../etc/passwd",
            "../metermate.local.toml",
            "..%2f..%2fetc%2fpasswd",
            "/etc/passwd",
            "subdir/1789-subject-main.mp4",
            "..\\windows\\system32",
            "../1789-subject-main.mp4",
            "metermate.toml",
            "",
        ] {
            assert!(!clip_name_is_safe(attempt), "{attempt} was allowed");
        }
        assert!(clip_name_is_safe("1789268224097-subject-main.mp4"));
    }

    /// **a clip still being written must not be served.** it is answered with
    /// `immutable`, so a browser that caches the half of it that exists now
    /// will keep replaying that half for a year -- under a name the recorder is
    /// about to rename away from anyway.
    #[test]
    fn a_clip_that_is_still_being_written_is_not_served() {
        assert!(!clip_name_is_safe("1789268224097-main.mp4.part"));
    }

    /// the listing carries the whole directory's cost, not the visible page's.
    /// a truncated list that did not say so would read as "this is everything
    /// that was recorded", which is how a full disk goes unnoticed.
    #[test]
    fn the_clip_listing_reports_what_it_held_back() {
        let d = std::env::temp_dir().join(format!("metermate-clips-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for stamp in 1..=3u128 {
            std::fs::write(d.join(format!("{stamp}-motion-main.mp4")), b"xx").unwrap();
        }
        let clips = Clips {
            budget_bytes: 100,
            ..clips_of(&d)
        };

        let json = clips_json(&clips, 2, "");
        assert!(json.contains(r#""kept":3"#), "{json}");
        assert!(json.contains(r#""used":6"#), "{json}");
        assert!(json.contains(r#""budget":100"#), "{json}");
        assert!(json.contains(r#""n":"3-motion-main.mp4""#), "{json}");
        assert!(
            !json.contains(r#""n":"1-motion-main.mp4""#),
            "the page was not truncated: {json}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    /// **the shape of a main-stream feed that restarts forever.** ffmpeg writes
    /// the crop feed and this remux from one process, so whatever blocks the
    /// remux socket stalls the crops with it. reading connections one at a time
    /// meant a predecessor that had not gone away yet left the replacement
    /// unread in the accept backlog, its socket buffer filling until it stalled,
    /// metermate seeing no frames, and the supervisor restarting it into the
    /// same queue. the gate, which has no second output, is untouched -- which
    /// is exactly how it presented.
    #[test]
    fn a_connection_that_lingers_does_not_wedge_the_one_after_it() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let relay = Relay::new(16);
        relay.serve(listener);

        // a predecessor that connects and then goes quiet without closing.
        let _orphan = std::net::TcpStream::connect(addr).unwrap();

        // the replacement. nothing about it may depend on the orphan.
        let mut live = std::net::TcpStream::connect(addr).unwrap();
        let rx = wait_for_subscriber(&relay);

        live.write_all(&[boxed(b"ftyp", b"isom"), boxed(b"moov", b"h")].concat())
            .unwrap();
        live.write_all(&[boxed(b"moof", b"f"), boxed(b"mdat", b"pixels")].concat())
            .unwrap();
        live.flush().unwrap();

        let got = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the second connection was never read");
        assert_eq!(
            got,
            [boxed(b"moof", b"f"), boxed(b"mdat", b"pixels")].concat()
        );
        assert!(relay.init().is_some(), "the header never arrived");
    }

    /// **the five second blank start.** a viewer that asks before ffmpeg has
    /// emitted its header was answered `503`, the page read that as a dead
    /// stream, and its reconnect throttle -- there so a reconnect cannot
    /// trigger another and stutter forever -- held the retry off for five
    /// seconds. both halves right on their own, and together a guaranteed five
    /// seconds of black at every start.
    #[test]
    fn a_viewer_that_arrives_before_the_header_waits_for_it() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let relay = Relay::new(16);
        relay.serve(listener);

        // the viewer is first, as it is at every start: the page opens the
        // video element while ffmpeg is still connecting.
        assert!(relay.init().is_none(), "the header cannot be there yet");
        let waiter = relay.clone();
        let asked =
            std::thread::spawn(move || waiter.init_within(std::time::Duration::from_secs(10)));

        std::thread::sleep(std::time::Duration::from_millis(150));
        let mut ffmpeg = std::net::TcpStream::connect(addr).unwrap();
        ffmpeg.write_all(&header_then_fragment(b"late")).unwrap();

        let got = asked.join().unwrap();
        assert!(
            got.is_some_and(|i| i.ends_with(b"late")),
            "the viewer was turned away instead of waiting"
        );
    }

    /// and it does not wait for a stream that is not coming: a relay with no
    /// ffmpeg behind it has to say so rather than hold the connection.
    #[test]
    fn a_viewer_is_not_held_forever_when_there_is_no_stream() {
        let relay = Relay::new(16);
        let began = std::time::Instant::now();
        assert!(
            relay
                .init_within(std::time::Duration::from_millis(120))
                .is_none()
        );
        assert!(began.elapsed() < std::time::Duration::from_secs(2));
    }

    /// two ffmpegs connected at once is a transient, not a feature: interleaved
    /// into one stream they would hand a viewer a mix of two timelines.
    #[test]
    fn only_the_newest_connection_publishes() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let relay = Relay::new(16);
        relay.serve(listener);

        // a header alone produces no init: the split emits one only when the
        // first `moof` proves the header is finished. so each connection sends
        // a fragment behind its header.
        let mut old = std::net::TcpStream::connect(addr).unwrap();
        // the first connection has to be read before the second arrives, or
        // "newest" is decided by whichever thread happened to start first.
        old.write_all(&header_then_fragment(b"old")).unwrap();
        wait_until(|| relay.init().is_some_and(|i| i.ends_with(b"old")));

        let mut new = std::net::TcpStream::connect(addr).unwrap();
        new.write_all(&header_then_fragment(b"new")).unwrap();
        wait_until(|| relay.init().is_some_and(|i| i.ends_with(b"new")));

        let rx = wait_for_subscriber(&relay);
        old.write_all(&[boxed(b"moof", b"stale"), boxed(b"mdat", b"stale")].concat())
            .unwrap();
        new.write_all(&[boxed(b"moof", b"live"), boxed(b"mdat", b"live!")].concat())
            .unwrap();

        let got = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert!(
            String::from_utf8_lossy(&got).contains("live"),
            "the superseded connection published: {}",
            String::from_utf8_lossy(&got)
        );
    }

    fn header_then_fragment(tag: &[u8]) -> Vec<u8> {
        [
            boxed(b"ftyp", tag),
            boxed(b"moov", tag),
            boxed(b"moof", tag),
            boxed(b"mdat", tag),
        ]
        .concat()
    }

    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    /// subscribing races the accept thread, so wait for it to be possible.
    fn wait_for_subscriber(relay: &Relay) -> std::sync::mpsc::Receiver<Vec<u8>> {
        relay.subscribe().expect("the relay lock was poisoned")
    }

    fn wait_until(mut done: impl FnMut() -> bool) {
        for _ in 0..500 {
            if done() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("condition never became true");
    }

    /// **the roi editor drew in one coordinate space and wrote out another.**
    /// the overlay canvas is the preview size, `[gate] roi` is read at the gate
    /// size, and nothing told the page they differ -- so every polygon drawn in
    /// that tab had its y scaled by the ratio between them. on the deployment,
    /// 640x360 against 640x480: a region meant to cover the road was handed to
    /// the gate covering the buildings above it, and every threshold tuned
    /// since was tuned against the wrong pixels.
    #[test]
    fn the_page_is_told_the_gates_frame_size_and_not_only_the_previews() {
        let cfg = crate::config::PreviewCfg::default();
        let json = page_config(PageFacts {
            src: "/stream.mjpg",
            kind: "mjpeg",
            events: false,
            cfg: &cfg,
            gate: (640, 480),
            encode: encode_size(cfg.width, cfg.height, (2560, 1440)),
            roi: &[[10, 20], [30, 40]],
            perspective: &[[[1, 2], [3, 4]]],
            subjects: &["go4".to_string()],
        });
        assert!(json.contains(r#""gate":[640,480]"#), "{json}");
        // what the browser is shown, so that it can lay the stage out under the
        // picture rather than around it.
        assert!(json.contains(r#""encode":[640,360]"#), "{json}");
        assert!(json.contains(r#""roi":[[10,20],[30,40]]"#), "{json}");
        assert!(json.contains(r#""perspective":[[[1,2],[3,4]]]"#), "{json}");
        // and the preview's own size is a different number, which is the whole
        // point: a page given only one of them cannot convert between them.
        assert_ne!(
            (cfg.width, cfg.height),
            (640, 480),
            "the default preview size has stopped differing from the gate's, \
             so this test no longer proves anything"
        );
    }

    /// the encoded frame keeps the scene's shape, because a picture that is
    /// contained inside a box of another shape leaves the overlay canvas covering
    /// the box -- which is a box drawn above a car rather than on it.
    #[test]
    fn the_encoded_frame_keeps_the_scenes_shape() {
        // the anamorphic pair this camera shipped with for a year: 640x480 of
        // pixels showing a 16:9 scene, which is where 640x360 came from.
        assert_eq!(encode_size(640, 360, (2560, 1440)), (640, 360));
        // the development camera's 64:29 street: as wide as the box allows.
        assert_eq!(encode_size(640, 360, (4096, 1856)), (640, 290));
        // a square-pixel 4:3 sub of a 4:3 scene: narrower, not squeezed.
        assert_eq!(encode_size(640, 360, (1024, 768)), (480, 360));
        // and with no scene to match, the box is the answer.
        assert_eq!(encode_size(640, 360, (0, 0)), (640, 360));
    }

    /// nothing configured is still valid json, and the editor opens empty
    /// rather than on a page that failed to parse its own config.
    #[test]
    fn an_unconfigured_roi_and_perspective_are_empty_arrays() {
        let json = page_config(PageFacts {
            src: "/stream.mp4",
            kind: "mp4",
            events: true,
            cfg: &crate::config::PreviewCfg::default(),
            gate: (640, 480),
            encode: encode_size(640, 360, (4096, 1856)),
            roi: &[],
            perspective: &[],
            subjects: &[],
        });
        assert!(json.contains(r#""roi":[]"#), "{json}");
        assert!(json.contains(r#""perspective":[]"#), "{json}");
        // and the picker opens on nothing rather than on a name nobody
        // configured: a subject invented by the page is deliberate, not a
        // default it fell into.
        assert!(json.contains(r#""subjects":[]"#), "{json}");
    }

    /// **the page is one file with no build step and no linter.** a renamed or
    /// mistyped element id is caught by nobody: `getElementById` hands back
    /// null, the script throws while loading, and the preview is a blank page
    /// with a perfectly healthy server behind it.
    #[test]
    fn every_element_the_page_looks_up_exists() {
        let missing: Vec<&str> = INDEX_HTML
            .split("getElementById(\"")
            .skip(1)
            .filter_map(|rest| rest.split('"').next())
            .filter(|id| !INDEX_HTML.contains(&format!("id=\"{id}\"")))
            .collect();
        assert!(missing.is_empty(), "the page looks up {missing:?}");
    }

    /// **the card shows a frame of the clip, so the listing has to say where
    /// the trigger is.** the page seeks every poster past the pre-roll, because
    /// a clip opens deliberately before anything happened and its first frame
    /// is therefore the emptiest one it contains.
    #[test]
    fn the_listing_says_where_the_trigger_sits_in_a_clip() {
        let json = clips_json(&clips_of(Path::new("/nonexistent/metermate")), 60, "");
        assert!(json.contains(r#""preroll":5"#), "{json}");
    }

    /// **an event is recorded once per stream and shown once.** main is what a
    /// person wants to look at; sub exists because the gate reads the camera's
    /// own anamorphic encode and a replay needs it. a grid built from the file
    /// list would show every event twice, and the page size would halve.
    #[test]
    fn an_event_recorded_on_two_streams_is_one_card() {
        let d = std::env::temp_dir().join(format!("metermate-pair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for name in [
            "100000-detection-main.mp4",
            "100000-detection-sub.mp4",
            // an older event, or a `streams = ["sub"]` deployment: sub alone.
            "90000-detection-sub.mp4",
        ] {
            std::fs::write(d.join(name), b"xx").unwrap();
        }

        let json = clips_json(&clips_of(&d), 60, "");
        assert!(json.contains(r#""kept":2"#), "events, not files: {json}");
        assert!(
            json.contains(r#""n":"100000-detection-main.mp4""#),
            "{json}"
        );
        assert!(
            !json.contains(r#""n":"100000-detection-sub.mp4""#),
            "both streams of one event were listed: {json}"
        );
        assert!(
            json.contains(r#""n":"90000-detection-sub.mp4""#),
            "an event with no main stream was dropped rather than fallen back on: {json}"
        );
        // the budget is about disk, so it still counts every file.
        assert!(json.contains(r#""used":6"#), "{json}");
        std::fs::remove_dir_all(&d).ok();
    }

    /// **nothing in a card is matched to a clip by timestamp any more.** the
    /// harvest holds a better picture of the vehicle, and for a while the card
    /// showed the clearest crop taken during the clip -- but which crops those
    /// are has to be inferred, and every way of inferring it was wrong
    /// somewhere. measured on the deployment, three consecutive cards showed
    /// one picture of a vehicle in none of them. a frame out of the clip's own
    /// bytes cannot be wrong about which clip it belongs to.
    #[test]
    fn a_card_is_not_illustrated_by_anything_outside_its_own_clip() {
        let d = std::env::temp_dir().join(format!("metermate-thumb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("100000-detection-main.mp4"), b"xx").unwrap();

        let json = clips_json(&clips_of(&d), 60, "");
        assert!(
            !json.contains("crop"),
            "the listing still carries a crop to match: {json}"
        );
        assert!(
            json.contains(r#""n":"100000-detection-main.mp4""#),
            "{json}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    /// **the scroll cursor cuts the deduped list, not the file list.** an event
    /// is one card but up to two files, so a boundary drawn across the
    /// directory can fall between a main/sub pair -- and the sub half then
    /// arrives on the next page as a card of its own, which is a duplicate
    /// event on any two-stream deployment. deduping per page instead drops the
    /// event altogether when its halves straddle the cut.
    #[test]
    fn a_page_boundary_never_splits_an_event_into_two_cards() {
        let d = std::env::temp_dir().join(format!("metermate-page-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        // four events, both streams each: eight files, four cards.
        for stamp in [100u128, 200, 300, 400] {
            for stream in ["main", "sub"] {
                std::fs::write(d.join(format!("{stamp}-detection-{stream}.mp4")), b"xx").unwrap();
            }
        }

        let first = clips_json(&clips_of(&d), 2, "");
        assert!(
            first.contains(r#""kept":4"#),
            "four events, not eight: {first}"
        );
        assert!(
            first.contains(r#""t":400"#) && first.contains(r#""t":300"#),
            "{first}"
        );
        assert!(
            !first.contains(r#""t":200"#),
            "page one ran past its size: {first}"
        );

        // the next page starts below the oldest card already held.
        let next = clips_json(&clips_of(&d), 2, "before=300");
        assert!(
            next.contains(r#""t":200"#) && next.contains(r#""t":100"#),
            "{next}"
        );
        assert!(
            !next.contains(r#""t":300"#),
            "the cursor is inclusive and repeats a card: {next}"
        );
        // and every card on it is a main-stream file, never a stranded sub half.
        assert!(
            !next.contains("-sub.mp4"),
            "a sub half surfaced as its own card: {next}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    fn crops_of(dir: &Path) -> Crops {
        Crops {
            dir: dir.to_path_buf(),
            budget_bytes: 20 << 30,
            classifying: true,
        }
    }

    /// **a verdict somebody has already settled is not a verdict, and a crop
    /// somebody called the subject is one.**
    ///
    /// the tab exists to show what stands as a sighting. a crop judged and
    /// labelled `other` is settled -- and on the first evening the classifier
    /// ran, every crop it named was wrong, so without this the tab is almost
    /// entirely a record of work already done.
    ///
    /// the labels live beside the harvest, written by the labelling page, and
    /// this is the only place the two halves meet.
    #[test]
    fn the_live_listing_is_what_stands_after_the_labels() {
        let d =
            std::env::temp_dir().join(format!("metermate-preview-rejected-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        let crops = d.join("crops");
        std::fs::create_dir_all(&crops).unwrap();
        let kept = "1789260000300_car_go4_090_640x480.jpg";
        let thrown = "1789260000200_truck_go4_083_640x480.jpg";
        let unclear = "1789260000100_car_go4_081_640x480.jpg";
        let by_hand = "1789260000050_car_077_640x480.jpg";
        let renamed = "1789260000040_car_waymo_077_640x480.jpg";
        let unwatched = "1789260000030_car_077_640x480.jpg";
        for name in [kept, thrown, unclear, by_hand, renamed, unwatched] {
            std::fs::write(crops.join(name), b"x").unwrap();
        }
        std::fs::write(
            d.join("labels.txt"),
            format!(
                "{kept} go4 alert\n{thrown} other alert\n{unclear} unclear alert\n\
                 {by_hand} go4 preview\n{renamed} go4 preview\n{unwatched} trailer preview\n"
            ),
        )
        .unwrap();

        let json = crops_json(
            &crops_of(&crops),
            120,
            "recognised=1",
            &Default::default(),
            &["go4".to_string()],
        );
        for (name, why) in [
            (kept, "a confirmed verdict was dropped"),
            (by_hand, "a subject the classifier missed is not listed"),
            (
                renamed,
                "a verdict a person renamed to a watched subject is not listed",
            ),
        ] {
            assert!(json.contains(name), "{why}: {json}");
        }
        for (name, why) in [
            (thrown, "a rejected verdict is still listed"),
            (unclear, "a verdict marked unclear is still listed"),
            (
                unwatched,
                "a label nothing watches for is listed as a sighting",
            ),
        ] {
            assert!(!json.contains(name), "{why}: {json}");
        }

        // the counts are what explain the tab, so they name only verdicts it
        // could have shown, and `unclear` is not counted as the classifier
        // being wrong.
        assert!(json.contains(r#""rejected":1,"unclear":1"#), "{json}");
        std::fs::remove_dir_all(&d).ok();
    }

    /// **a label outlives the crop it names.** the harvest evicts oldest first
    /// under its disk budget, and of 6213 labels in one set 1598 already named
    /// crops that were gone. a rejection counted off the file rather than
    /// against the directory would explain an empty tab with crops it cannot
    /// show -- and would keep reading plausibly if the labels beside this
    /// harvest turned out to describe a different harvest.
    #[test]
    fn rejections_are_counted_against_the_crops_that_are_still_here() {
        let d =
            std::env::temp_dir().join(format!("metermate-preview-evicted-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        let crops = d.join("crops");
        std::fs::create_dir_all(&crops).unwrap();
        let here = "1789260000300_car_go4_090_640x480.jpg";
        let evicted = "1789259000000_truck_go4_083_640x480.jpg";
        std::fs::write(crops.join(here), b"x").unwrap();
        std::fs::write(
            d.join("labels.txt"),
            format!("{here} other alert\n{evicted} other alert\n"),
        )
        .unwrap();

        let json = crops_json(
            &crops_of(&crops),
            120,
            "recognised=1",
            &Default::default(),
            &[],
        );
        assert!(
            json.contains(r#""rejected":1"#),
            "a crop that is gone was counted: {json}"
        );
        assert!(!json.contains(here), "a rejected crop is listed: {json}");
        std::fs::remove_dir_all(&d).ok();
    }

    /// **green is what alerted, not what was named.** a verdict is one frame;
    /// an alert needs the track to agree over several looks, and the pipeline
    /// writes the crops of a track that did into `alerted.txt` beside the
    /// harvest. a crop in that file with no verdict is not marked either: the
    /// mark is about a verdict, and there is none to colour.
    #[test]
    fn only_verdicts_of_a_vehicle_that_alerted_are_marked() {
        let d =
            std::env::temp_dir().join(format!("metermate-preview-alerted-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        let crops = d.join("crops");
        std::fs::create_dir_all(&crops).unwrap();
        let alerted = "1789260000300_car_go4_090_640x480.jpg";
        let named = "1789260000200_car_go4_083_640x480.jpg";
        let plain = "1789260000100_car_090_640x480.jpg";
        for name in [alerted, named, plain] {
            std::fs::write(crops.join(name), b"x").unwrap();
        }
        std::fs::write(
            d.join(crate::harvest::ALERTED),
            format!("{alerted}\n{plain}\n"),
        )
        .unwrap();

        let json = crops_json(&crops_of(&crops), 120, "", &Default::default(), &[]);
        let entry = |name: &str| {
            json.split('{')
                .find(|s| s.contains(name))
                .unwrap_or_else(|| panic!("{name} not listed: {json}"))
                .to_string()
        };
        assert!(entry(alerted).contains(r#""a":1"#), "{json}");
        assert!(!entry(named).contains(r#""a":1"#), "{json}");
        assert!(!entry(plain).contains(r#""a":1"#), "{json}");
        std::fs::remove_dir_all(&d).ok();
    }

    fn clips_of(dir: &Path) -> Clips {
        Clips {
            dir: dir.to_path_buf(),
            cache_dir: dir.join("cache"),
            budget_bytes: 1 << 30,
            preroll_ms: 5_000,
            max_clip_ms: 60_000,
            recording: true,
        }
    }

    #[test]
    fn an_empty_events_directory_lists_as_no_clips() {
        let json = clips_json(&clips_of(Path::new("/nonexistent/metermate")), 60, "");
        assert!(json.contains(r#""clips":[]"#), "{json}");
    }
}
