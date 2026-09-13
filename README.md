# metermate

<img src="assets/logo.svg" alt="" width="96" align="right">

spot san francisco parking enforcement from an ip camera and say so fast enough
to move the car.

sfmta runs three-wheeled go-4 interceptors. metermate watches a block, decides
whether what just moved is one of them, and publishes to mqtt and home assistant
within about a second and a half, while staying close to idle the rest of the
time.

## status

phase 0 is complete: ingest, motion gate, and mqtt work end to end against a real
camera. detection and classification are next. see [ROADMAP.md](ROADMAP.md).

what runs today: a supervised rtsp ingest, a background-model motion gate that
groups changed pixels into per-object regions, and mqtt publishing with home
assistant discovery.

## how it works

a cascade, where each stage gates the next and is far cheaper than it. the street
is full of permanently parked cars, so a detector running every frame would spend
all day rediscovering them. a background model folds them into the scene instead.

```
substream         -> motion gate -> detector -> classifier -> tracker -> mqtt
 6% of a core        0.2ms/frame    gated       gated         cheap
 any size
```

on a quiet street the whole pipeline costs about **4% of one core** and 155 mb,
because the expensive stage never runs.

full-resolution pixels come from a second decode of the main stream, keyframes
only, because only the main stream has enough detail to tell a go-4 from any
other white compact.

both streams are read at the size they actually send, which is asked of the
camera at startup rather than remembered from the last one. a resolution is a
camera setting, and getting it wrong is not an error: 4096x1856 pixels cut into
2560x1440 frames are half an image each, so the detector sees noise and the
harvest saves nothing while every log line says the pipeline is healthy. a
camera that cannot be asked -- rebooting, or off -- is read at
`[stream] main_width`/`main_height` and `gate_width`/`gate_height`, and the log
says so. whatever the two sizes are, they are mapped per axis, since one field of
view at two aspects is what anamorphic substreams are.

the detector is yolo26s rather than the nano. through a hazy window at a wide
angle a typical car is only ~77px in the detector's input, and the nano missed
the largest car in the frame outright. the gate is what makes the bigger model
affordable.

[DESIGN.md](DESIGN.md) has the measurements behind each of those choices,
including the ones that were rejected.

both servers serve the mark at `/favicon.png`, so the preview and the labelling
page are told apart in a tab strip rather than by reading their titles. it is a
go-4 in solid silhouette carrying sfmta's single amber beacon: the body, the cab
glass and two wheels. an earlier version was an outline with `MM` filling the
body, and its lines were a third of a pixel wide at 16px, which is the size it
is seen at most, so it rendered as gray haze. this one is drawn on a 16 unit
grid so its edges land on whole pixels there, and the beacon is the one piece of
colour, which keeps it findable on a dark tab bar. the backdrop is a squircle
with nothing behind it, since a square of near-black draws its own box on every
surface that is not exactly that black.

**`assets/logo.svg` is the mark.** `make logo` renders the two pngs from it,
including the favicon the binary compiles in, so a change to the vector reaches
both after a rebuild. this page shows the svg, since 96px is a size no raster
was exported at.

## requirements

- ffmpeg on `PATH`
- an mqtt broker, unless running with `--dry-run` or alerting over `[ntfy]` alone
- a dahua-family camera. developed against an amcrest ip4m-1041b.

## quick start

    mise install
    make                                  # the binary
    make models                           # the two onnx models, ~390 MB

    cp metermate.toml metermate.local.toml
    $EDITOR metermate.local.toml          # camera host, credentials, broker

    ./target/release/metermate

with no `--config`, metermate uses `metermate.local.toml` when it exists and
falls back to `metermate.toml`. the local file is gitignored, so real credentials
stay out of the repo and no flag is needed day to day. an explicit `--config`
always wins, including when the path is wrong, so a typo fails loudly instead of
quietly loading something else.

### choosing what the preview plays

`[preview] source` picks where the browser gets pixels. the detection overlay is
drawn from a separate json feed and rescales itself, so it works the same on all
of them.

`[preview] show_parked` picks which vehicles are in that overlay. the detector
looks at the whole frame, so every look reports the cars that have been on the
kerb for years: drawn dashed, they show what the pipeline decided to leave alone,
which is the point of r8.2. on a street where that is most of what is reported it
is a thicket over the box worth checking, and `show_parked = false` leaves only the
vehicles a person might still disagree with. scenery keeps deciding things behind
the hidden boxes -- it still vetoes a harvest and still counts -- and the boxes are
dropped from the feed rather than sent for the browser to hide, since that feed
goes out every frame to whoever is watching.

the **boxes** tickbox in the header is the other half of the same wish and does the
opposite: it hides the drawing, in either view, and leaves the pipeline, the feed and
the motion dashes alone. it exists because sometimes the question is what the street
looks like.

| `source` | what the browser plays | resolution | cost to metermate |
|---|---|---|---|
| `main-h264` *(default)* | the camera's h.264 main stream, remuxed to fragmented mp4 | **2560x1440** | a stream copy, no decode or encode |
| `server-sub` | jpeg metermate encodes from the gate frame | 640x360 | a jpeg encode per frame |
| `camera-sub` | mjpeg fetched straight from the camera | whatever the substream is | none |

the default is `main-h264` because it is both the best picture and about the
cheapest: `-c copy` passes the bitstream through untouched, measured at 0.16s to
remux ten minutes of 2560x1440. ffmpeg is spawned per viewer and killed when they
leave, so an unwatched preview costs nothing (r8.3).

`camera-sub` looks free and is not. the camera has **one** substream encoder
(`MaxExtraStream=1`) and the gate is using it, so a browser asking it for mjpeg
changes what the detector sees -- that is how the gate once silently dropped from
h.264 at 15fps to mjpeg at 10fps. it also needs the camera to allow anonymous
access, because browsers refuse credentials in a subresource url, and it needs
the browser to reach the camera directly, which a vpn breaks. `main-h264` has
none of these problems: it reads the main stream, and only metermate talks to the
camera.

to change it:

    [preview]
    source = "server-sub"

### reading it from a phone

**the live picture takes the window it is left with.** the stream is as large as
the screen allows at the shape it was encoded to, with the header above it and the
legend below both on screen at the same time -- measured against what the page
spends on those, since the header rewraps and the legend is prose.

**the bar the tabs live in stays on screen.** the crops grid and the events list
are each longer than any window, and a bar that scrolls away with the page means
scrolling to the top to change tab and losing where you were.

the page is otherwise laid out for the pointer it is given: on a touch device the
type and every button grow, the tabs take a row of their own, and nothing is a
fifteen-pixel target. a desktop stays as dense as it was, because that is what a
mouse is for.

**every view is a url, and so is every crop, verdict and clip.** the page is a
wall of tiles and the tile worth showing somebody is one of them, so the hash is
the navigation rather than a note of where the page happens to be:

| link | what it opens |
|---|---|
| `#/live`, `#/crops`, `#/verdicts`, `#/events`, `#/roi` | one view |
| `#/crop/<name>` | that crop, whole, on the crops tab |
| `#/verdict/<name>` | the same, on the verdicts tab |
| `#/event/<name>` | that clip, playing, on the events tab |

the tabs assign the hash and `hashchange` renders it, so the back button is the
browser's own, a link pasted into a message opens the thing it names, and a
crop's tile can be found again days later. opening a tile leaves its own link in
the location bar, so whatever is being looked at is whatever can be copied out. the crop in a link is looked up by
name rather than paged to, so a link works however far the grid has scrolled,
and brings the instant it was taken with it -- which is what lets the overlay
offer the recording it was taken during. a link to a crop the budget has since
evicted lands on its tab rather than on a broken image.

### running the harvester

the harvester is on by default, so the quick start above is already collecting.
it writes full-resolution vehicle crops to `data/crops`, capped at 10 GB with
the oldest deleted first -- except a crop somebody has labelled, which is never
deleted (`[harvest] keep_labelled`, on by default). labelled crops still count
towards the 10 GB. at startup it logs how many labelled crops it is keeping and
which `labels.txt` it read them from; a zero on a machine you have labelled on
means the labels are somewhere other than beside `data/crops`.

**the event clips get the larger budget of the two**, 20 GB against the
harvest's 10. both streams are recorded, so an event replays through the whole
pipeline and every crop of that passage can be cut again -- while no crop
reconstructs a clip. the harvest is derived data; when the disk is tight, the
derived half is what should go.

this is the part worth starting early. enforcement passes a few times a day, so
the training set grows in wall-clock time and no amount of later effort recovers
a day that was not recorded.

    ./target/release/metermate                    # foreground, watch the log
    ls data/crops | wc -l                         # how many crops so far

the **crops** tab of the preview browses them without leaving the browser: newest
first, with class, confidence and the time each was taken, how much of the disk
budget is spent, and click to see one at its own size. worth looking at early --
a harvest full of garage doors or letterboxed strips is invisible until somebody
looks at a crop.

**a crop of a vehicle that alerted carries a green border.** on the crops tab a
crop stage two recognised has a caption leading with what it was called:
`GO4 . car 0.83`, the verdict first and the detector class kept beside it, since
which of the two disagreed is the interesting case. green is stricter: the
tracker confirmed that vehicle, so an alert was published -- or would have been,
with no broker connected. the pipeline records every verdict crop of a track in
`alerted.txt` beside the harvest once the track confirms, including the crops it
saved on the looks before confirming. a page of the harvest is two hundred cars
and the subject is well under a percent of them, so the one worth looking at has
to survive being scrolled past.

the verdict is recorded in the crop's filename, as
`<epoch_ms>_<class>[_<subject>[_<margin>]]_<conf>_<w>x<h>.jpg` -- the margin in
thousandths, so `..._go4_20_...` cleared its negatives by `+0.020` -- because the
name is the only index the harvest has. it is a statement about that one frame: what the
classifier made of those exact pixels at the margin it was trained at, before the
tracker has had its say. an alert needs more -- the same subject on `confirm_m`
of the last `confirm_n` looks -- so a crop can carry a name that was never
published, and the difference between the two is the unconfirmed-look rate.

**it is evidence of what fired, never a label.** on the first evening it was
recorded, every crop it named was wrong. building a training set by globbing
`*_go4_*` would therefore train on the classifier's own false positives, which
is the worst shape of error this project has: a reference set that looks fully
populated while being mostly wrong. what a crop *is* lives in `labels.txt`,
written by a person, and nothing else is a source of truth -- not the filename,
and not which directory the crop sits in.

the **verdicts** tab is that filter as a view: every crop still claimed, newest
first, with the same paging and the same cards as the crops grid -- except that a
card there is captioned `GO4 +0.02`, the margin and not the detector's class. that
tab exists to answer "what is it firing on, and by how much", which is the
judgement a margin is set from, and `car 0.83` beside a margin invites reading one
number as the other. a crop claimed before the margin was recorded keeps its name
and shows no number.

**a verdict somebody has thrown out drops off it.** the labelling page is where
a crop named `go4` gets labelled `other`, and once that has happened the
classifier's claim has been answered -- leaving it here would make the tab a
record of work already done rather than of what the deployment is firing on. the
count says how many were rejected, so an empty tab still distinguishes "nothing
has fired" from "everything that fired was wrong", which are opposite results
and the second is the one that actually happened on the first evening. a verdict
marked `unclear` leaves as well and is counted apart (`1 marked unclear`): it is
not a rejection, but it is an answer, and the labelling page's second look is
where it comes back from.

**a crop labelled a subject by hand is on it too**, captioned `GO4 by hand`. a
go-4 stage two missed, or named as something else, is a sighting all the same,
and once somebody has labelled it from the crops tab it is listed here with the
verdicts. only a name in `[[subject]]` counts; any other label is a label.

three more things bound what can appear. only moving vehicles are classified at
all, so a subject parked at the kerb is judged on its approach and its departure
and not while it sits there; it starts from when the verdict began being
recorded, since crops harvested before that carry no name whatever they were;
and with `[classifier] enabled = false` the tab says so rather than saying the
street has been quiet.

**a crop can open the recording it was taken during.** click one and, if a clip
was being written at that moment, the overlay offers it along with how far in
the crop falls. that direction is answerable where the reverse is not: a clip
covers `[trigger - pre-roll, close]`, both ends known, so either the moment is
inside one or no clip was open -- which on a street with gaps between events is
often the honest answer.

it opens the clip rather than seeking into it. these files carry no index:
measured in chrome against a finished twenty second clip, `duration` reads 1.4,
`seekable` is `[0, 0]`, and assigning `currentTime` lands back at zero. so the
offset is told rather than jumped to.

for unattended running there are two units in `contrib/`. pick by what the
machine is:

| | |
|---|---|
| `metermate-system.service` | an always-on host. starts at boot, runs as its own user, no login needed |
| `metermate.service` | a machine you log into. runs as you, needs `enable-linger` to survive logout |

**on a server**, prefer the system unit:

    sudo useradd --system --home /var/lib/metermate --create-home metermate
    sudo cp -r target/release/metermate models metermate.local.toml /var/lib/metermate/
    sudo chown -R metermate: /var/lib/metermate
    sudo chmod 600 /var/lib/metermate/metermate.local.toml   # camera credentials

    sudo cp contrib/metermate-system.service /etc/systemd/system/metermate.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now metermate
    journalctl -u metermate -f

**on a laptop or desktop**, the user unit:

    mkdir -p ~/.config/systemd/user
    cp contrib/metermate.service ~/.config/systemd/user/
    $EDITOR ~/.config/systemd/user/metermate.service    # set WorkingDirectory
    systemctl --user daemon-reload
    systemctl --user enable --now metermate
    sudo loginctl enable-linger $USER                   # survive logout

    journalctl --user -u metermate -f                   # follow the log

both set `StartLimitIntervalSec=0`. the default is to give up after five
restarts in ten seconds, which is backwards for this: a camera rebooting or
wifi dropping is the normal case the ingest supervisor is built to survive
(r5.1), and giving up permanently turns a transient fault into silent downtime.

both also state the resource requirements as limits rather than hopes:
`MemoryMax=1G` is r3.3, and a leak shows up as memory pressure in the journal
instead of as a machine that swaps. **the cgroup holds the ffmpeg children too**:
measured on the deployment, 652 MB steady and 860 MB peak -- metermate 439 MB,
main-stream ffmpeg 191 MB, gate ffmpeg 108 MB. an older note here said 344 MB,
which was true before the crop feed decoded 2560x1440.
`CPUQuota=200%` is a runaway stop, deliberately well above the r3.1 idle budget,
because the detector is meant to burst while a vehicle is in frame.

to run it on another machine, copy the binary, `models/`, and your
`metermate.local.toml`. nothing else is needed at runtime: no python, no torch.

### without a camera or a broker

the same binary runs from a video file, which is how the tests work and how you
develop away from the camera:

    ./target/release/metermate --source clip.mp4 --dry-run

### measuring a change against real traffic

a live street is a bad test bench: the light moves, and a car either drove past
during the measurement or it did not. record a few minutes instead and replay it,
so two builds can be compared on the same input.

    ffmpeg -rtsp_transport tcp -t 600 -i "rtsp://USER:PASS@HOST:554/cam/realmonitor?channel=1&subtype=1" \
        -c copy data/eval/traffic-sub.mp4

    make eval CLIP=data/eval/traffic-sub.mp4 EVAL_ARGS="--out before.json"
    # change something, rebuild
    make eval CLIP=data/eval/traffic-sub.mp4 EVAL_ARGS="--compare before.json"

it reports vehicles found, how confidently, how many crops were harvested, the
detector's measured rate, and what fraction of the stream it managed to look at.

there are two ways to replay, and they bracket what a real machine does:

| mode | pacing | frames |
|---|---|---|
| default | wall-clock, as the camera runs | skipped when the loop falls behind |
| `--offline` | as fast as it can | every one, nothing skipped |

the default is what a deployment actually experiences, and a slow machine sees
less of the street. `--offline` removes that: it blocks the decoder rather than
dropping frames and ignores the detector's rate limit, so the same clip gives
the same answer on any machine. use the default to ask "how will this host
behave", and `--offline` to ask "did this change help".

**record both streams at once**, as `<stamp>-sub.mp4` and `<stamp>-main.mp4`.
the gate reads the substream and crops are cut from main-stream pixels, so a
replay given only one of them measures a pipeline nobody runs -- feed the
substream to both and every vehicle reaches the detector a quarter of its real
size.

`make eval` has no ground truth in it, so its output is for diffing between
builds. for a number that means something on its own, label a clip and ask the
whole pipeline:

    make transits STAMP=20260912-103138   # find the transits, then fill in `truth`
    make score LABELS=tests/e2e/fixtures/labels/20260912-103138.json
    make endtoend STAMP=20260912-103138

`score` answers "when a vehicle crossed, did the detector see it". `endtoend`
answers "did a crop of it reach the harvest", which is the question that
matters: everything between the two -- the movement threshold, the scenery
filter, the per-place rate limit -- can discard a vehicle the detector found.

`transits` writes one entry per **moving object**, which is usually several per
stretch of motion, and a `_clean.png` for each showing where to look. set every
`truth` to one of `vehicle`, `not-a-vehicle`, or `unclear`. re-running it later
picks up the decisions already made:

    make transits STAMP=20260912-103138 TRANSITS_ARGS="--from tests/e2e/fixtures/labels/20260912-103138.json"

`endtoend` reports two numbers. **recall** comes from those labels. **precision**
does not -- it asks an independent motion scan whether each crop's own box was
full of pixels that changed, so it cannot be flattered by an incomplete label
set. two flags are worth knowing:

    make endtoend STAMP=... E2E_ARGS="--keep-crops /tmp/crops"   # look at them
    make endtoend STAMP=... E2E_ARGS="--declined"                # see below the threshold

`--declined` logs the movement scores that `must_have_moved` turned away. without
it the distribution is truncated at the threshold, and tuning the threshold from
the crops that cleared it is circular.

### recording the moments something happened

`make record` writes a fixed ten-minute window when you ask for one. the thing a
stage-2 eval needs to contain arrives about twice an hour, so those two facts do
not meet: checked against the labels, **none of the recorded clips holds a single
labelled waymo passage**, and the nearest miss starts 85 seconds too late.

event clips fix that by recording when something happens:

    [record]
    enabled = true

    metermate --record-events            # or for one run, without editing config
    metermate --record-events=false      # and off for one run, keeping the setting

the flag wins over the config in both directions. it needs
`[preview] source = "main-h264"`, which is where the remuxed stream comes from.

it costs no inference and no second camera session. the main stream is already
remuxed as a second output of the ffmpeg feeding the crops -- a byte copy of what
the camera sent -- so recording it is a ring buffer and a file write.

two things about it are worth knowing, because both look like details and are
not:

- **a clip starts before its trigger.** motion fires once the vehicle is already
  in frame, so the last few seconds are held in a ring and written out ahead of
  it. without that a clip opens on a car halfway across and cannot answer when
  it first appeared.
- **what a clip is worth is decided when it closes**, not when it opens. at the
  trigger the detector and classifier have not run yet; by the time they have,
  the ring still holds the beginning. so clips are named
  `<stamp>-<motion|detection|subject>-main.mp4` for what turned up in them, and
  the disk budget evicts the least useful first rather than the oldest -- unlike
  the harvest, where the oldest crop is the cheapest to lose, a clip holding a
  recognised subject may be the only one there is.

the arithmetic that shapes the policy: the main stream is 30 MB a minute, a busy
ten minutes holds 27 motion events, so keeping every one is ~1.6 GB an hour
while keeping the ones a subject was recognised in is ~15 MB an hour.

by default only clips that turned out to hold something are kept:

    [record]
    keep = ["detection", "subject"]      # and not "motion"

a motion-only clip is one where the gate fired and nothing was ever established,
and on a street with parked cars permanently in frame those are the clips with
no vehicle in them at all. `keep = []` keeps every clip. the choice cannot be
made before recording, only before keeping -- at the trigger the detector has
not run yet -- so an unwanted clip is written and then deleted.

the **events** tab of the preview plays them without leaving the browser.
newest first, each card showing what the clip was named for, the time it
triggered and how long it ran, with a filter for the ones worth keeping. click a
card to play it full size.

each card also carries a small **copy**, which puts that clip's path on the
clipboard -- `data/events/1789660864671-subject-main.mp4`, under the directory
`[record] dir` names, which is the argument `--dense` below takes. clicking the
card plays the clip, so there is no way to select the path by hand without
opening it. the button says **copied** when it worked and **select it** when the
browser refused, since a clipboard cannot be read back to check.

**both grids are the same grid.** they answer the same questions about the same
kind of thing -- what is on disk, what it cost against the budget, what each
item is and when it happened -- so they say them in the same order, and both
scroll back through the whole of what is kept rather than stopping at a page.
they had drifted apart: only the events tab reported its budget, and only the
crops tab could scroll.

**what arrives while you are scrolled away waits for you.** both grids put new
cards at the top, so a crop that turned up four seconds ago used to move the row
under your pointer somewhere below it. the poll keeps running -- the bar above each
grid says what the harvest costs against its budget, and that has to stay current --
but what is new is only counted: the bar grows a button saying **3 waiting**, and
pressing it, or scrolling back to the top, brings them in. a tab that is empty is
never held like this, so opening one halfway down still shows it.

a card's time is a clock reading rather than "30s ago". looking a passage up
against a clip, a ticket or a memory of the street needs the wall-clock; the
date appears only when it was not today.

it appears when there are clips to show *or* something is recording, so turning
recording off does not also hide what it already recorded -- the tab says
`not recording` and goes on serving them. it stays out of the way only on a
deployment that has never recorded anything, where an empty tab would read as a
quiet street rather than as a feature that is off.

**a card shows the crop harvested during the clip, not a frame of the clip.**
the first frame is the pre-roll, which exists precisely so that the file opens
before anything happened -- so a grid of first frames is a grid of empty
streets, which reads as "nothing was captured". the harvest already holds a
picture of the vehicle itself, cached and a few kilobytes, so the tab borrows
it. clips with nothing harvested during them fall back to a frame from the
trigger, which is honest: an empty street is what they hold.

**an event writes both streams**, as `<stamp>-<worth>-main.mp4` and
`<stamp>-<worth>-sub.mp4`. `endtoend.py` wants the pair: the substream drives the
gate and the motion scan that judges precision, the main stream supplies the
crops, and the substream cannot be derived by downscaling main -- the gate reads
the camera's own encode, anamorphic and at its own frame rate, and metermate
deliberately never rescales on that path. a clip of main alone can be watched but
not replayed.

each rides along as a second output of the ffmpeg already pulling that stream, so
neither opens another camera session.

**recording both roughly doubles the disk budget**, because the substream is
currently mjpeg -- measured on one event, 1.81 mbit of sub against 1.85 mbit of
main. that is a camera setting rather than a fact of life: the substream was
h.264 at ~263 kbit until a browser asked it for mjpeg and permanently switched
its one encoder (see DESIGN, "fetching the mjpeg stream reconfigures the
camera"). switched back, the pair would cost about +14% instead of +100%.

    [record]
    streams = ["main"]    # for clips that are only ever watched

an unknown name there is refused at startup rather than ignored, for the reason
`keep` refuses one -- half a pair on disk looks exactly like a quiet street.

### asking what it is doing, from somewhere else

`/stats` answers json, so a deployment can be diagnosed without an ssh session
and a monitor can alert on it:

    curl -s http://metermate-host:8420/stats

it reports **ages, not just counts**. a counter says how much has happened;
`last_crop_frame_ms` says whether the crop feed is still arriving, and that is
the question worth asking when the pipeline looks fine and the harvest is empty.
`stream_seen_pct` is the share of offered frames the gate actually got.

**both inference stages run inline on the frame loop**, so what they cost is
frame rate rather than a curiosity, and `processing` reports both:

| field | what it settles |
|---|---|
| `inference_ms`, `classify_ms` | what the last single look cost in each stage |
| `inference_pct`, `classify_pct` | share of wall clock spent inside them since start |
| `classifications`, `recognised` | looks taken, and the few that named a subject |
| `last_recognised_ms` | how long since anything was recognised at all |

a stage that is slowing the pipeline down says so as a share: at 15fps a frame
is 66ms, so a stage sitting at 40% of wall clock is taking most of the budget
for every frame it touches. if the shares are low and the frame rate is still
down, the cause is below this program -- look at `limits` next, where memory
throttling presents as "slow while the machine looks idle".

`limits` is read from the process's own cgroup, so it is the same numbers the
kernel is acting on rather than an estimate:

| field | what it settles |
|---|---|
| `memory_anon_mb` | allocations that cannot be reclaimed. flat means no leak, whatever the total does |
| `memory_file_mb` | page cache. climbs while clips are browsed and comes back under pressure |
| `memory_throttled` | times the kernel forced reclaim at `MemoryHigh`, from `memory.events` |
| `memory_stall_pct` | share of the last ten seconds *everything* was stalled, from psi |
| `cpu_throttled` | quota periods cut short, from `cpu.stat` |

the anon/file split is there because the total cannot answer the only question
that matters at the limit. a cgroup sitting at `MemoryHigh` on page cache is the
kernel doing its job; sitting there on anonymous growth is a leak on its way to
an oom kill. both read as "memory is at 98%".

### the models

two, both fetched and exported by `make models` into `models/`. the export needs
python and torch; **the runtime needs neither**, so it runs in throwaway `uv`
environments and nothing it installs is kept.

| file | what | size | needed by |
|---|---|---|---|
| `detector.onnx` | yolo26s, stage one | 38 MB | the pipeline |
| `embedder.onnx` | clip vit-b/32 vision tower, stage two | 351 MB | `--label`, `--measure`, the classifier |

    make models          # both, skipping whatever is already there

each is a separate target, so `make models/embedder.onnx` builds just that one.

the detector export starts from an ultralytics checkpoint, kept in
`models/upstream/` so a second export reuses the download instead of dropping
another one in the repo root. **those `.pt` files are agpl-3.0**, which is viral
if metermate is published; the exported `detector.onnx` is what the runtime
loads, and nothing in `models/upstream/` is needed to run the binary.

**on a second machine, copy the files rather than re-exporting.** the embedder is
351 MB and the torch download needed to produce it is several times that, so
copying `models/` across is usually much faster than running the export again:

    rsync -av --exclude upstream/ models/ strix:/var/lib/metermate/models/

the binary needs nothing else at runtime -- no python, no torch, no network.

### looking at one crop

clicking a crop opens it over the page, and **it fills the window** -- a crop is a
photograph of a number plate or a light bar, and the grid draws it 150px wide only
because it draws two hundred of them at once. the shape stays the crop's own, so a
tall crop is bounded by the height and a wide one by the width.

the wheel zooms from there, toward wherever the cursor is, which is what makes every
part of a blown-up crop reachable without a second gesture to drag it about. zooming
out stops at the crop's own pixels, and in at twelve times the window -- past that a
crop is out of pixels rather than out of luck.

a clip fills the window the same way and is not zoomed: it is already the whole
street at the resolution it was encoded at, and its player keeps its own use for a
pointer over it. esc, or a click anywhere, closes either.

### labelling while you watch

the crops tab labels. under each crop is what it is labelled and an **edit**
button; edit opens a list of the labels in use, ten of them, ranked by how
often and how recently each was applied. type to filter, arrows to move, enter
to take the highlighted one, click to take any of them. a name nobody has used
is offered as a new label at the bottom of the list, so a subject can start as
a name and some crops long before anything can recognise one. clicking the
picture still opens it whole.

ranked that way because on any given day two or three labels are in play and
the rest are history -- an alphabet buries the one used a minute ago under
whatever begins with an `a`.

**the labels go in the same `labels.txt`** the labelling page writes, beside
the harvest, so a decision made here is one `--measure` and `--train` see with
nothing to import. they are recorded as `via preview`: the crops tab is
newest-first rather than a hash sample, so a rate measured over crops somebody
picked out while watching would be measuring the watcher, and the random pool
stays the only honest source of a false positive rate.

naming a subject here labels crops and nothing else. making one fire alerts is
still a `[[subject]]` block and a restart, because the config is what the
running pipeline reads and a server that rewrites the file it is running from
is a larger idea than this needs.

the pipeline and a labelling session are two processes writing one file, which
is how it is actually used. the live page appends a line rather than rewriting,
and a session re-reads before it saves, so neither can drop the other's work.

### marking a passage, and the two commands after it

the events tab marks. open a clip, and while it plays press **in** and **out**
to take the player's own clock, then **select**. what is marked shows on the
card it was taken from, each window with an **x** that takes it back -- a
window is judged in a second off a moving picture, so marking the wrong clip
or three seconds of empty street is ordinary. the passage is written to
`selections.txt` beside the events directory and nothing else happens yet:
cutting crops is a detector pass over every frame of the window, and the
process serving that page is the one watching the street (r2.2, r3.1). marking
only the start is allowed and means "to the end of the clip".

then, whenever it suits:

    make prepare     # cut every marked passage into sets/<subject>/<clip>-<window>/
    make label       # label what was cut: the labelling page over all of sets/
    make retrain     # embed and train every configured subject over the lot

or `metermate --prepare`, `metermate --label go4 --harvest sets/` and
`metermate --retrain` directly, which is what the targets run; `make label`
takes `SUBJECT=` and `SETS=` for another subject or a single set. **the middle
step is not optional**: a cut passage arrives unlabelled and an unlabelled crop
is counted as street, so a retrain straight after a prepare scores the subject
that was just cut as a false positive. **`--prepare` drains the queue**: a passage it cuts stops being
marked, so the events tab shows what is still waiting rather than everything
ever marked. marking the same passage again is caught by the set already being
on disk -- skipped and said so, because the detector pass costs minutes and a
second copy of one passage would be counted twice by everything downstream. `--retrain`
trains each `[[subject]]` over `[harvest] dir` *and* `[train] sets`, which is
the pairing that matters -- the harvest holds the deployment's own rejected
verdicts and the sets hold the dense passages, and dropping either is how a
retrain quietly loses the negatives that stopped the last false positive. a
subject that cannot be trained yet is reported and the others still are.

so a round is: label crops on the crops tab, mark passages on the events tab,
`make prepare`, `make label` for the new crops, `make retrain`, restart the
pipeline.

### labelling the harvest, and asking whether stage 2 works

the binary does this itself, so it runs wherever metermate runs, on the harvest
that is already there, with no second toolchain and nothing to copy:

    metermate --label waymo      # browser at localhost:8421, press t to train
    metermate --measure waymo    # the same numbers, on the command line
    metermate --train waymo      # write the reference file the classifier loads
    metermate --list-classes     # what `[detector] classes` accepts

`--label` puts crops on screen a pool at a time, one tab each, and records which
pool every label came from. which pool a crop came from is what decides what may
be measured on it, so it is a tab rather than a heading you scroll past:

| tab | what is in it | what it is for |
|---|---|---|
| **random** | drawn by a hash of the filename, so it is the same set on every run and does not reshuffle as the harvest grows | the only pool a false positive rate can honestly be measured on |
| **nearby** | the crops seconds either side of something already labelled | completes a passage, which is the unit of an example |
| **most likely** | whatever most resembles the known ones | finds the next ones fast; useless for measuring anything |
| **second look** | labelled crops the classifier's own vote disputes: labelled the subject but voting with the negatives, labelled `other` but voting with the subject, or marked `unclear` and now voting with the subject | possible mislabels and parked questions, fifty each way |
| **verdicts** | every crop stage two named as this subject | precision, and the negatives worth the most |

`y`, `n`, `u` and `x` label the crop under the cursor, `j` and `k` move it,
`1`-`5` switch tabs, `t` trains. the report `t` produces opens over the grid and
is dismissed with `escape`, the close button or a click outside it; the
labelling keys are held while it is open, so reading a page of numbers about
negatives cannot mark the crop the cursor was left on. the **already decided** switch in the bar hides
crops that had an answer when the pool was drawn, which collapses a pool to the
work left in it after one button has marked two hundred; the count on each tab
follows it, so the bar reads as how much is left where. a decision made *since*
the pool was drawn stays on screen until the pool is next drawn, so a misclick
can be clicked back rather than vanishing under the pointer.

**more** asks the directory again and offers another pool, without measuring
anything. the page shows a pool at a time out of a harvest that keeps growing while
somebody is labelling it, and the only way back to the rest of it used to be `t`, which
also runs the embedder over every crop in the harvest and reports a curve nobody asked
for. answers given on the crops tab in the meantime are picked up as well, because both
ports read the same `labels.txt`.

the page is rarely the thing that notices: `--label` walks the harvest and embeds what
is new every 60 seconds by itself, `--label-ingest 0` to stop it, because embedding is
the slow half and by the time anybody presses the button the crops should already be
there. what no pass ever does is redraw the pool you are working through: which pool a
crop came from is the provenance of its label.

every tile also carries a small **copy** button for that crop's filename, which
is what `--gather`, a line of `labels.txt` and a grep of the harvest all take.
selecting it by hand means clicking the tile, and clicking the tile labels the
crop. the button says **copied** when it worked and **select it** when the
browser refused, since a clipboard cannot be read back to check.

on the **second look** tab the switch is greyed out and held on: every crop in
that pool already has an answer -- that is what the pool is -- so there is
nothing for it to hide, and letting it apply would empty the tab. it lists the
fifty crops labelled the subject that vote most with the negatives, then the
fifty labelled `other` that vote most with the subject, then the fifty marked
`unclear` that vote most with the subject, each under its own heading. the
third is the one that is not about a mistake: `unclear` means a person looked
and could not tell, so every measurement leaves those crops out of both classes
and no other pool ever offers them again. the classifier's vote is what changed
since, and the ones it now votes for are the parked questions worth reopening. every tile carries its score, how much more the crop votes with the
other class than with its own; above zero, the classifier would call it the
other way. a score is a reason to look, not a verdict: change a label only if
the crop is actually wrong.

the number beside the switch is how much it is hiding on the tab you are on.
three of the pools only ever offer crops with no answer yet -- random and most
likely are drawn from the unlabelled harvest, and nearby skips whatever is
already decided -- so on a fresh page that number is zero and flipping the
switch changes nothing there. the zero is the point: it says the pool holds
nothing decided, rather than leaving you wondering whether the switch works.

the **verdicts** tab is the other half of that loop: every crop stage two named
as this subject, read off the filenames rather than computed, so it is there the
moment the page opens rather than after a train. the name is evidence of what
fired and never a label -- on the first evening the verdict was recorded, every
crop it named was wrong -- so `n` rejects the one under the cursor, a click does
the same, and one button rejects the rest of the screenful. a rejection is
written as `other` from the alert pool, which is the tier `--train` draws its
negatives from **first**: a vehicle the classifier actually fired on teaches it
more than a sedan that never confused it.

on first use it embeds the whole harvest, which takes a few minutes and happens
once: the vectors are cached beside the crops and a vector does not depend on
what is being looked for, so adding a subject later re-uses all of them.

the subject is a parameter, not a special case. `--label sweeper` labels street
sweepers into the same file, and nothing in the path assumes the thing is a
vehicle -- though `[detector] classes` has to be widened first, since anything
stage 1 drops can never be recognised later.

`--train` writes `[classifier] references` from those labels, printing the same
measurement first. **what it writes is the selection that was scored**, taken
off the report rather than chosen a second time: two selections agree until the
day they do not, and nothing would say which of them the deployment was running.
it refuses when the eval calls its own numbers void -- a reference set below the
vote size, or one that overlaps the eval set -- because a file built from those
ships the mistake instead of printing it.

    metermate --train go4 --harvest sets/ --train-out trained

`--harvest` points labelling, measuring, training or gathering at a directory of
crops that is not `[harvest] dir`, so an eval set staged from recorded clips is
trainable without editing the config. measuring and training take it more than
once, so a deployment's own harvest -- where its rejected verdicts live -- and
the sets cut from video train together:

    metermate --train go4 --harvest data/crops --harvest sets/

labelling walks a tree the same way, so every set under it is one session:

    metermate --label go4 --harvest sets/

each label is written to the `labels.txt` beside the crop it names, and each
vector to the `embeddings.bin` beside it, so a set still travels on its own. a
crop two sets share is offered once and labelled in both. running it again
changes nothing: no crop is embedded twice, and an answer repeated leaves every
file as it was.

it writes `trained/<subject>/references.txt` and `trained/<subject>/negatives.txt`.
**the negatives belong to the subject, not to the street**, so training one
subject leaves the others alone and a second subject's crops become negatives
for this one.

**every verdict you reject is a negative.** a verdict rejected on the labelling
page, and every crop of another subject, always goes into `negatives.txt`,
however many there are; only ordinary traffic is a sample, of thirty crops.
those rejections are what teach the classifier what it confuses. when they
shared one set of thirty with the street, 658 rejections left no cargo bike
among the negatives, and cargo bikes kept alerting. retrain after rejecting a
batch, then restart: the running pipeline knows only the negatives it started
with.

**the margin it measured is part of the artifact**, written to
`trained/<subject>/trained.toml` beside the vectors rather than copied into a
config by hand. there is no `[classifier] margin`: a margin is an output of a
measurement, and one set by hand on another machine describes a rule nobody ran.
so the whole directory travels to the deployment host, margin included:

    rsync -av trained/ kairos:/var/lib/metermate/trained/

the curve the margin was chosen from is in the same file, one `[[ladder]]` row
per margin of the sweep with its `passages`, `passage_recall` and `fpr`. the
margin says where the bar is and the ladder says what moving it would cost,
measured in the same run against the same references. the shape, with made-up
numbers:

    margin = -0.005
    fpr = 0.0
    ...
    [[ladder]]
    margin = -0.01
    passages = 3
    passage_recall = 1.0
    fpr = 0.004

`--train` writes the margin it picks: the lowest false positive rate that keeps
80% of passages. `<--` marks that row in the printed curve, and two flags move
the choice without reading a number off the table and typing it back in:

    metermate --train go4 --harvest sets/ --max-fpr 1%        # most recall under that rate
    metermate --train go4 --harvest sets/ --min-recall 90%    # cleanest margin keeping that many

`--max-fpr` is the rate you will tolerate and buys whatever recall fits under
it; `--min-recall` moves the default bracket and still takes the quietest margin
in it. both read `1%` or `0.01`, both apply to `--measure` too, and they combine.
**an operating point nothing on the curve meets is refused rather than replaced**
-- a rate you ruled out is not a rate to ship, and the artifact would carry no
record of what was asked for -- so the run says what each axis actually offered
and writes nothing.

to pick a row by hand instead, press **train** on the labelling page and press
**use** on any row of the table. it writes the same `trained/<subject>/` through
the same code, with that row's recall and false positive rate saved beside the
margin, to `[classifier] references`. the highlighted row is the one `--train`
would write with no flags. a running pipeline reads `trained/` at startup, so
restart it afterwards.

**both slow loops say how far along they are.** embedding a harvest runs at
roughly 18 crops a second on the deployment -- 1392 crops took 1m15s, measured
-- and the cross-validation behind the curve is a pass over every negative for
every labelled passage. each reports at every tenth of the way with the time
gone and an estimate of what is left, so a run that has minutes to go is
distinguishable from one that has stopped:

    embedding crops: 695/1392 (49%), 37s gone, ~37s left
    scoring passages: 12/104 (11%), 9s gone, ~1m6s left

the **passages** and **alerts/hr** columns count the way the deployment confirms:
a passage is caught, and a negative makes a false alert, only when `[track]
confirm_m` of `confirm_n` consecutive crops score above the margin. the values
come from the config the report runs with, so it should match the deployment's.

**crops** is the other recall: the share of the held-out passages' *individual*
crops that clear the margin, where **passages** is the share of the passages
themselves that fire. it is always the lower of the two and that is not a
problem -- a vehicle is cropped every few tenths of a second for several
seconds, and an alert needs a few of those frames rather than all of them. 100%
of passages at 39% of crops means every passage was caught on its best frames.
`fpr` is per crop as well, because that is what the margin moves, and crops are
fewer than the tracker's looks, so it is close to the deployment rather than
identical to it.

**a set has to own its crops, or it quietly stops being reproducible.** the
harvest keeps every crop named in the `labels.txt` beside it, but labels kept
anywhere else -- a set in `sets/`, or labels copied off another machine -- have
no such protection: the budget deletes oldest first, a label outlives the crop it
describes, and nothing fails when it does. measured on this camera before
labelled crops were kept, 1598 of 6213 labels in one set already named crops
that were gone.

    metermate --gather sets/waymo/learn --harvest data/crops

copies what that set's labels name into `sets/waymo/learn/crops/`, and reports
how many name crops beyond recovering rather than proceeding with the remainder.
the labels are the manifest, not the harvest listing, so it never pulls in crops
nobody judged. it is idempotent, and meant to be run again whenever labelling
continues.

**a recorded passage can be cut into a set whole.** the harvest keeps a few crops
a passage by design, so a go-4 caught on an event clip leaves a dozen crops where
there were hundreds of looks. watch the event in the preview, note where the
vehicle is in the clip, take the clip's path off the card's **copy** button, and
crop every detection in that window:

    metermate --dense data/events/1789660864671-subject-main.mp4 \
        --window 0:03-0:08 --into sets/go4/1789660864671
    metermate --label go4 --harvest sets/go4/1789660864671/crops

the window is `START-END` into the clip, as `SS`, `MM:SS` or `HH:MM:SS`, the way
the player shows it. either end may be left off: `0:03-` runs to the end of the
clip, `-0:08` from the start. every detection of the configured classes in every frame is
kept, parked ones included, at `[stream] crop_fps` -- the rate the live harvest
decodes the main stream at. crops are named by when their frame was taken, so
passages group as they would live, and the clip is copied into the set's
`clips/` so the set outlives the recording. expect most crops to be the parked
cars in frame, repeated every frame: label the passage, and sweep the rest.

**waymo is a rehearsal, not a feature.** it stands in for the go-4, which this
camera sees a few times a day and which has no same-camera references at all.
nothing about it ships. see DESIGN.md for what the rehearsal measured.

## configuration

everything tunable lives in one toml file; nothing needs a recompile. the two
settings worth understanding:

- `stream.gate_subtype` picks which camera stream feeds the motion gate. the
  substream (`1`) is the default and is five times cheaper to decode than the
  main stream. `gate_width` and `gate_height` must match that stream's **native**
  size, because metermate deliberately never rescales on this path.
- `gate.min_changed_frac` is the fraction of pixels that must change to count as
  motion. measured against this camera, a quiet scene sits at about `0.00045` at
  the 99th percentile, so the default of `0.002` has roughly 4.5x of headroom.
- `gate.roi` is the polygon the gate is allowed to look at, in gate pixels.
  worth setting: this camera looks past overhead wires, and they move in wind.

### drawing the roi

open the preview, click **roi**, and click on the street to place points. drag a
point to move it, `undo point` removes the last, and `copy toml` gives you a
block to paste into your config:

    [gate]
    roi = [[0, 210], [640, 160], [640, 480], [0, 480]]

it is drawn over the live video rather than a still, so you can see traffic
crossing the boundary you are placing. metermate reports what share of the frame
the polygon covers when it starts:

    roi: 4 vertices, 64% of the frame watched

worth reading. an roi that covers nothing produces a camera that detects nothing,
which looks exactly like a quiet street.

### when the camera moves

the background model, the roi, and every scale prior refer to one pan/tilt
position, and the camera reports nothing when it is moved. metermate polls its
position every `camera.ptz_poll_secs` seconds and resets the model when it
changes by more than a degree. set it to `0` to turn that off; a camera with no
ptz, or one that will not answer, logs a warning and carries on detecting.

### keeping office hours

enforcement is a daytime phenomenon, and a night of empty road spends the same
disk budget as a day of traffic. `[harvest] active_hours` and
`[record] active_hours` each take windows of local time:

    [harvest]
    active_hours = "07:00-19:00"

empty -- the default -- means always. a window whose start is past its end
wraps midnight (`"22:00-06:00"`).

days go in front of the times, and more than one window a day goes after them:

    [harvest]
    active_hours = "mon-fri 07:00-09:00,16:00-19:00"

so a window can be a weekday, or a morning, or both -- `"sat,sun 09:00-17:00"`,
or a week of groups separated by `;`. the day is the half a window of clock time
cannot say: it is what makes a stream shut on friday evening and open again on
monday morning, and the live page and `/stats` name the day as well as the hour
when that is the wait. a value that is not a window is refused at
startup, and when a window is set metermate says so on its startup line, so an
empty harvest directory says which of the street or the clock said no:

    harvesting crops to data/crops (budget 10240 MB, one per place per 5s, moved >= 0.08), only 07:00-19:00 local

the windows close *collections*, not decisions. detection, classification and
alerting run whatever the clock says -- a confirmed subject outside the window
still publishes, with its evidence crop -- and a clip open when its window
shuts finishes its own pre-roll and hangover rather than being cut.

`[stream] active_hours` is the third window, and the only one that reaches in
front of the pipeline: outside it the rtsp channels are closed, both ffmpegs
stopped, so the camera serves nothing and the machine decodes nothing. the
harvest and the recorder go quiet along with it, being driven by frames.

    stream shut until 07:00 local: rtsp channels closed, harvest and recording paused

`[stream] unload_models` goes one stage further while the window is shut and
hands the memory back too: the detector and the classifier are dropped rather
than held overnight for a street nobody is watching. it is off by default, it
can only happen while the window is shut -- a shut stream delivers no frames, so
nothing is mid-look when the weights go -- and the models are built again when
the window opens, before the first frame is looked at, and `/stats` reports what
the hand-back gave as `models_released_mb`.

what does *not* stop is the process. the preview keeps serving, and the crops
and event clips already on disk stay browsable through the night, because they
outlive the window that wrote them. when the window reopens the same process
starts pulling again and resets the gate's background model, since the street at
07:00 is not the street of the previous evening.

a closed stream says so where it was last seen. the live tab covers the picture
with **stream closed -- until 07:00 local** rather than holding the last frame it
was sent, because a still street reads exactly like a quiet one, and `/stats`
reports `"stream_shut": true` with the minute it reopens, so a stream closed on
purpose is never mistaken for one that has died.

## mqtt

one topic per subject and outcome, so a rule can be written against exactly the
thing it cares about.

| topic | meaning |
|---|---|
| `metermate/motion` | something moved. no subject: nothing is recognised yet |
| `metermate/<subject>/sighting` | recognised, and confirmed over several looks |
| `metermate/<subject>/moving` | and in motion |
| `metermate/<subject>/stopped` | and stationary |
| `metermate/<subject>/dismount` | a person stepped out of a stopped one |
| `metermate/<subject>/departed` | it has gone |

**metermate does not rank these.** whether a dismount matters more than a
sighting depends on your street and your day, so it is a home assistant
automation rather than something baked into a topic name. the payload carries
what a rule needs to decide:

```json
{"state":"ON","subject":"go4","track":12,"dwell_s":43.0,
 "confidence":0.87,"protected":"adjacent","box":{"x":..},"at":1789..}
```

`protected` is `adjacent`, `near`, `far`, `away` or `unknown` — `away` meaning
your vehicle is not in its space, so there is nothing to protect and the rule
can stay quiet.

messages are sent on transitions, not per frame, so a vehicle crossing produces a
handful rather than fifteen a second.

home assistant entities appear automatically via mqtt discovery, one per subject
and outcome — add a subject to the config and its entities show up with no yaml.
availability is published with a last will, so entities show as unavailable
rather than stale if metermate dies.

## notifying a phone

mqtt assumes something downstream is listening: a broker, then home assistant,
then an automation, then a companion app. `[ntfy]` is the short path — metermate
posts to [ntfy](https://ntfy.sh) itself, and the notification carries the crop
that caused it:

    [ntfy]
    enabled = true
    server = "https://ntfy.sh"     # or your own
    topic = "something-nobody-will-guess"
    outcomes = ["sighting"]        # sighting, moving, stopped, dismount, departed
    priority = "high"
    click = "http://10.1.100.250:8420/"   # the preview, at an address the phone reaches

    metermate --notify-test        # sends one now and says what the server answered

**a tap opens the verdict the notification carried**, not the front page of the
harvest: `click` is the preview's address and the link is aimed past it at
`#/verdict/<crop>`, which is safe to promise because **r4.5** refuses to notify
about anything the verdict page cannot show. an outcome with no crop of its own --
a vehicle that stopped, or left -- opens the verdicts instead. with `click` empty
no link is sent, so a deployment whose preview the phone cannot reach keeps its
notifications.

**the topic is a password** — anyone who knows it reads every alert — so there
is no default and metermate refuses to start with an empty one. a `token` goes
in `metermate.local.toml` beside the camera password, since that file is
gitignored. it works with or without `[mqtt]`: one camera, one street and one
person who wants to be told is a complete deployment.

**what leaves the network is a sentence and a picture, not the payload.** the
mqtt topic carries track ids, boxes and dwell times because a rule needs them;
ntfy.sh is somebody else's server, so the notification says `GO4 on the block`,
`go4 recognised, 12s in frame, margin +0.042`, and attaches the
crop — which is what makes moving the car a decision rather than a walk to a
laptop.

a notification is queued and sent by a worker, never inline: **r2.1** gives an
alert 1.5s end to end and **r2.2** forbids evidence capture from delaying one,
so a stalled uplink costs the pipeline nothing. when the queue backs up the
newest notification is dropped and logged, because a phone buzzing about a
vehicle that left ten minutes ago is worse than silence. `--notify-test` is the
exception and fails loudly: it exists to answer "does this work".

## development

    make          # release build
    make test     # rust unit tests
    make test-e2e # python end to end tests, no camera needed
    make precommit

see [CONTRIBUTING.md](CONTRIBUTING.md).

## a note on the camera

metermate treats the camera as read-only by default. it never writes camera
configuration without an explicit opt-in.
