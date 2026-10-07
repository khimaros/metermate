# design

## the problem

a fixed-ish camera watches a block. parking enforcement passes a few times a day.
we need an alert within about a second and a half, and we need the machine to be
essentially idle the other 99.99% of the time.

running a detector on every frame is the obvious approach and the wrong one: it
burns cpu continuously to find nothing, and it re-detects the same dozen parked
cars forever.

## the camera

metermate is written against a specific camera. the facts below are measured, not
assumed, and several of them constrain the design directly.

**amcrest ip4m-1041b**, sigmastar ssc327de, firmware `2.800.1ENN001.0.R`
(2024-10-29). dahua-family cgi api over http digest auth.

| property | value |
|---|---|
| main stream | 2560x1440 h.264 30fps, 4 mbit cbr |
| sub stream | 640x480 h.264 30fps, 512 kbit cbr |
| pan | 1 to 354 degrees |
| tilt | -4 to 79 degrees |
| zoom | none |
| absolute positioning | supported, and position is readable |
| presets | up to 25 |
| on-camera smart detection | not supported |

### the substream is squeezed, and both streams carry one field of view

both streams carry the same field of view, and the substream is whatever aspect
the encoder offers rather than the scene's. so mapping between the two is a pure
per-axis scale with no offset, and any crop taken from the substream is stretched.

two consequences, both easy to get silently wrong, so both are unit tested:

- gate coordinates map to main-stream pixels by one factor per axis, both derived
  from the sizes the two streams reported. measured pairs: `x * 4.0, y * 3.0` for
  a 640x480 sub of a 2560x1440 main, and `x * 3.20, y * 2.58` for a 1280x720 sub
  of a 4096x1856 main -- the pair the deployment runs.
- a substream frame has to be corrected to the scene's aspect before a detector
  trained on natural-aspect images sees it, and the scene's aspect is the main
  stream's: 640x360 for a 16:9 street, 640x290 for a 64:29 one. the same
  correction is what keeps the preview's overlay on the picture rather than
  floating above it.

### the streams' sizes are asked, not remembered

the main stream's size was a `const`. a second camera -- ip8m-dlb2998w-ai, dahua
family again -- sends 4096x1856, which cut into 2560x1440 frames is not an error
and not a stall. it is half an image per frame, so the gate ran, the crop feed
reported frames, the detector reported nothing, and the harvest wrote nothing.
the only symptom was `0 crops` next to a `--debug-detector` run, which reads the
*substream*, finding cars all day.

so both feeds are probed at startup, the pair is logged with the factors it
implies, and the config's numbers are the fallback for a camera that cannot be
answered -- with a warning naming the guess, because a silent fallback is the
same silence that hid this. measured: 1.6s per live rtsp probe, so about 3.2s of
startup for both; an unreachable stream fails on its own in about 3s, so no
timeout flag is needed.

the aspect difference is kept rather than normalised away. the substream is
cheaper to decode the smaller it is, and a 4:3 or 64:29 buffer is on most cameras'
menu; the mapping is per axis, so it costs nothing but a second factor. what does
not survive a size change is `[gate] roi` and `[gate] perspective`, both drawn by
hand in gate pixels, so a probe that disagrees with the config complains about
them by name rather than quietly rescaling someone's drawing.

### resolution is the binding constraint

a vehicle on the far side of the street is about 90px wide in the substream. a
go-4 is smaller than a sedan, so roughly 50px. that is not enough to tell an
sfmta go-4 from any white compact. the same vehicle is about 360px in the main
stream, which is ample.

**classification must read main-stream pixels.** this single fact is why the
pipeline decodes the main stream rather than the cheaper substream.

### it shoots through a window

the camera is indoors behind glass. daylight contrast is already reduced by
veiling glare. at night the ir illuminator reflects off the glass and washes out
the frame entirely, which is why night is deferred (r7.1) rather than pretended.

**the glass costs more than any other single factor measured so far.** the same
detector on the same substream, with the window shut and then opened:

| | vehicles found | best |
|---|---|---|
| through glass | 5 | 0.77 |
| window open | **9** | **0.91** |

frame contrast rose from 49.5 to 63.1. for comparison, going from the substream
to the full main stream through glass bought 5 -> 7 vehicles, so **opening the
window is worth more than quadrupling the resolution**.

this reorders the fixes. an open window is not something the system can rely on,
so the glare stays a design constraint, but it means the cheap wins are physical
-- a hood, a lens hood, a darker room, moving the camera to the opening -- rather
than architectural. two glare corrections in software, global autocontrast and
tiled clahe, were measured and both made detection *worse*.

the lens has strong barrel distortion. metermate does not correct it. the
distortion is fixed, so the stage-2 classifier simply learns it, which costs
nothing at runtime and avoids a calibration step.

### fetching the mjpeg stream reconfigures the camera

`GET /cgi-bin/mjpg/video.cgi?channel=1&subtype=1` looks like a read. it is not.
the substream has one encoder, and asking it for mjpeg switches
`Encode[0].ExtraFormat[0].Video.Compression` to `MJPG` and leaves it there. it
was observed by accident while probing what a browser could play directly:
the substream went from h.264 640x480 at 15fps and ~263 kbit/s to mjpeg 640x480
at 10fps and 1024 kbit/s cbr, and stayed that way.

this matters twice. it is a violation of r5.4 waiting to happen, since nothing
about the url suggests a write. and it silently changes the gate's frame rate,
which changes how much of the street is seen.

restoring it is one request:

    configManager.cgi?action=setConfig&Encode[0].ExtraFormat[0].Video.Compression=H.264

**treat every dahua cgi as potentially stateful.** `configManager.cgi` with
`action=getConfig` and `snapshot.cgi` are genuinely read-only; the streaming
endpoints are not.

the trade is not one-sided, which is why it is currently left as mjpeg: mjpeg is
all-intra, so there is no inter-frame smearing on the moving vehicle that is the
whole point, and per-frame quality is visibly better. against that it costs
several times the bitrate.

**the frame rate half of that trade has since gone away.** it switched at 10fps,
but measured today the substream is mjpeg at 15fps -- `ffprobe` on a `-c copy`
event clip reads 15.0, and the gate reports 14.9-15.0 at 100% of frames seen. so
the cost is now bitrate alone: 1.81 mbit measured, against ~263 kbit for h.264.

that bitrate is worth something real -- it doubles the `[record]` disk budget and
it is continuous wifi -- but switching back is **not** a free win, and the
measurement usually quoted for it does not support one. the detector reads
main-stream pixels for every inference, so a table of detector confidence cannot
show what the substream's codec does. the number that can is precision on
motion, which the gate does drive: 93% on the h.264 substream against 96% on
mjpeg. two clips, confounded with other changes, and pointing the wrong way for
anyone hoping to switch. anybody making this change should measure precision
before and after rather than assume the codec is invisible.

**nothing broke when it switched, and that is worth understanding.** a separate
deployment was running against the same camera throughout and kept working:

- the ingest never names a codec. the gate's filter is `format=rgb24` and ffmpeg
  detects the input, so mjpeg decodes into the same buffer h.264 did and no stage
  downstream can tell the difference.
- the cutover broke the live rtsp session, the supervisor saw no frames for
  `FRAME_STALL_TIMEOUT_MS`, killed ffmpeg and respawned it, and the new session
  negotiated mjpeg. this is r5.1 working rather than luck.

the part that *did* change silently is the frame rate, 15 to 10. nothing asserts
the input rate, so the gate simply sees a third fewer frames with no error
anywhere. that is why the loop now reports what fraction of the stream it
actually examined rather than assuming it keeps up.

### camera-side event push was evaluated and rejected

`eventManager.cgi` can long-poll and push motion events, which would let
metermate idle with no decode at all. it only pays off if we are not decoding
continuously. we are, because starting ffmpeg per event costs 300 to 800ms of
rtsp handshake plus keyframe wait, which does not fit the latency budget (r2.1).
the hook remains available as a redundancy input.

## pipeline

### the stream lies about its frame rate

this camera's rtsp stream declares `100 tbr` while actually sending 15 fps.
ffmpeg's default frame-rate handling pads output towards the nominal rate by
duplicating frames, and it was measured doing exactly that: **84 fps out of a 15
fps camera**, so five of every six frames were copies. every stage downstream
paid to decode, difference, detect and encode them.

`-fps_mode passthrough` emits only the frames that arrive. it is asserted in a
test, because nothing about the symptom points at the cause: the system simply
runs hot and drops frames.

### camera settings this assumes

both streams are configured at **15 fps** with an i-frame interval of **15**.
the camera refuses an interval below the frame rate, so **one keyframe per second
is a hard floor** on this hardware regardless of configuration.

15 fps is not a compromise here. nothing in the pipeline needs 30: the gate wants
to notice a vehicle, not measure its velocity precisely, and halving the frame
rate halves the decode bill.

### measured cost of every option

percent of one core on an i7-1360p, measured against this camera rather than
estimated. the middle column is why the design changed twice.

| approach | at 30 fps | at 15 fps | verdict |
|---|---|---|---|
| main 1440p, decode + rescale to gate size | 30% | -- | rejected: breaks r3.1 |
| main 1440p, full decode, no rescale | 13% | **7.5%** | **adopted** |
| main 1440p, keyframes only | 2% | ~2% | unnecessary, see below |
| substream 640x480 native gray | 6% | **4.5%** | **adopted** |
| `snapshot.cgi` full-res jpeg | ~1.0s latency | -- | rejected: eats the r2.1 budget |

two findings shaped this. first, software rescaling 3.7M pixels cost *more than
decoding them*, so the gate never rescales. second, once the camera dropped to
15 fps, full decode of the main stream became cheap enough that the keyframe
trick stopped being worth its latency.

### the shape that follows

two ffmpeg subprocesses, each doing the one thing it is cheap at:

| process | stream | output | cost | consumer |
|---|---|---|---|---|
| gate | sub, 640x480 | gray8, native, 15 fps | 4.5% | motion gate, tracker |
| crop | main, 2560x1440 | yuv420p, throttled to ~5 fps | 7.5% | classification, evidence |

**about 12% of one core**, against a 30% budget, leaving room for the detector
and classifier.

the gate runs on the substream at its native 640x480 with no scaling at all. the
substream is anamorphic, but the gate does not care: frame differencing is
indifferent to aspect ratio. only the detector needs correct geometry.

the crop process decodes every frame but only *emits* about five per second.
throttling the output does not reduce decode cost, it bounds pipe bandwidth:
full-res at 15 fps would be 83 MB/s of yuv420p for frames nothing would read.

### why keyframe-only decoding was dropped

it was the plan until the frame rate changed. decoding keyframes only costs ~2%
against 7.5%, but the camera's one-keyframe-per-second floor would have capped
full-res latency at a full second, inside a 1.5s end-to-end budget. spending an
extra ~5% of a core to get full-res pixels every 66ms instead of every 1000ms is
an obvious trade, and it removes a whole tier of latency-driven complexity from
the alert path.

```
  rtsp main stream
        |
   [ffmpeg decode]  ~20-25% of one core, the dominant fixed cost
        |
        +--> gate frames 640x360 gray @30
        |          |
        |     [motion gate]        ~0.2ms/frame, roi polygon only
        |          | fires
        |     [stage-1 detector]   ~5-10ms, coco vehicle + person
        |          | vehicle boxes
        +--> full frames 2560x1440 @5
                   |
             [crop, mapped to main-stream coords]
                   |
             [stage-2 classifier]  is this an sfmta go-4
                   |
             [tracker]             association, velocity, dwell, n-of-m
                   |
             [alert]               mqtt + home assistant discovery
```

each stage is cheap enough to gate the next. the gate is what makes the idle case
free: parked cars do not move, so background subtraction ignores them without a
single inference.

## why a cascade rather than one detector

the camera is fixed for a given pan/tilt position. that makes localization nearly
free, and it means the hard part is recognition, not detection.

- **finding things** is background subtraction. no neural net required.
- **recognizing** an sfmta go-4 is classification on a crop. classification needs
  roughly 100x less labeled data than detection, because labeling is sorting
  crops into folders rather than drawing boxes.
- training a detector to recognize "meter maid" directly would need thousands of
  boxed examples of a vehicle that passes a few times a day. that is a six month
  data collection problem, and it is avoidable.

so the stage-1 detector does a generic, pretrained job -- find vehicles and
people -- and the only project-specific model is the stage-2 classifier.

## does a go-4 register at all, and at what size

the whole stage-1 design rests on a stock coco model firing on a three-wheeled
vehicle that coco has no class for. rather than wait weeks for a sighting, this
was tested against seven reference photographs (`references/SOURCES.md`).

**it registers.** a go-4 reads as `truck` at 0.85 to 0.92 across cities, angles,
and day and night, including the san francisco vehicle at 0.85.

### fidelity is not the problem; size is

the reference photographs are sharp and this camera is not, so the obvious worry
was image quality. it was measured rather than assumed. on a daylight frame this
camera produces `mean 116.7, std 50.3, edge energy 3.78`, against `106-115,
62-73, 14.8-24.6` for the photographs: brightness already matches, contrast is a
little high, and **sharpness is four to six times too high**.

degrading the photographs to match closes that gap, landing them at edge energy
3.8-5.1. detection barely moves: 0.92 -> 0.92, 0.90 -> 0.88, 0.85 -> 0.84. so
this camera's softness, haze, compression and grain cost almost nothing.

what does cost something is apparent size. shrinking the vehicle to a given pixel
width, placed at native size so nothing is silently scaled back up:

| apparent width | detected, of 7 |
|---|---|
| 24px | 3 |
| 32px | 3 |
| 48px | 4 |
| **64px** | **7** |
| 96px and above | 7 |

**reliable at 64px and wider. unreliable below 48px.**

### what that means for this camera, and why the cascade already fixes it

in the gate stream a go-4 on the far side of the street is roughly 50px, which is
squarely in the unreliable band. near the protected vehicle it is 150-250px,
comfortably reliable. so the vehicle that actually matters -- one stopping beside
the van -- is well inside the detector's range, and far-side sightings are
best-effort.

the fix for the far side is already in the architecture. the detector does not
have to run on a whole downscaled frame. the gate produces a *region*, and that
region can be cropped from the **main** stream, where the same vehicle is four
times wider. a far-side go-4 at 50px in the gate stream is about 200px in a
main-stream crop, which moves it from the unreliable band to the reliable one.

this reframes the crop ingest from "needed for classification" to "needed for
detection at range too", and it is the strongest argument yet for running the
detector on gated main-stream crops rather than on whole frames.

### measured later: crops are worth much less than expected, except at range

the argument above was made from apparent size alone, before the detector had
been pointed at the street. measured on one frame with the window open:

| vehicle | whole-frame @640 | cropped |
|---|---|---|
| 1 (near) | 0.92 | 0.91 |
| 2 | 0.91 | 0.91 |
| 3 | 0.84 | 0.92 |
| 4 | 0.83 | 0.93 |

the whole-frame pass found seven vehicles at 0.66-0.92 unaided. cropping adds
nothing for near vehicles and about +0.09 for smaller ones. so the reasoning was
right about *range* and wrong about everything else: slicing earns its keep only
where the whole-frame pass is already failing.

hence the **hybrid**. every gated frame gets one whole-frame inference, and a
crop follows only where that pass could not settle the question:

- a motion region the whole-frame pass found nothing in -- the far-side case,
  where a go-4 at ~50px is under the floor whole-frame and ~200px cropped
- a detection that came back narrower than `SMALL_DETECTION_PX`, near enough to
  the floor that its box and class are not to be trusted

typical cost is one inference per gated frame rather than up to two, and the
coordinate mapping, the duplicate boxes, and most of the per-region budget
pressure go with it.

### replaying a clip needs its own clock

an offline replay examines every frame and takes longer than the clip it is
replaying -- about twenty-five minutes for ten minutes of video on the
development laptop. that breaks every judgement in the pipeline that is phrased
in elapsed time, and it breaks them silently:

- `scenery` asks whether a vehicle has stood in one place for `PARKED_AFTER`
  (90s). against wall-clock, twenty-five minutes of replay made *every* vehicle
  scenery, which zeroed its movement score and suppressed the harvest: 4 crops
  from a clip that yields 66 live.
- the harvest's own rate limit, the reinspect window, and the overlay linger are
  all the same shape of mistake.

so offline replay runs on **video time**: the clock is `origin + seq / fps`, with
fps probed from the clip. wall-clock is kept for exactly one thing, measuring how
long an inference took, which is a real duration either way.

the same error appeared one level up, in the eval's own parsing. it split tracks
on a three second gap between log timestamps, which are wall-clock, so in an
offline run nothing ever exceeded the gap and 57,000 sightings collapsed into 13
tracks. the detector's report now carries the frame number, and the eval keys off
that instead.

**the general point: a replay is not a slow live stream.** anything asking "how
long since" has to be told which clock it means.

### and the second decoder needs the same clock

the gate got a video clock. the crop feed did not, and the consequence was worse
than anything the clock itself caused.

there are two ffmpeg processes: the gate reads the substream, the crops read the
main stream, and they are joined by nothing but arrival order. live that is
sound -- both are fed by real time, so the newest main frame is at most one crop
interval (250ms at 4fps) behind the gate frame asking for it. offline there is
no real time to share. the crop decoder ran flat out, emptied a ten minute clip
in under half a minute, hit eof, and its slot held the final frame for the rest
of the run.

measured on `20260912-103138`: the inspected image stopped changing at gate
frame **200 of 6534**. the detector reported the same nine parked cars, at the
same confidences to two decimals, for the remaining 97% of the replay. every
transiting vehicle appeared in at most fourteen frames and most in one. the crop
feed was nine seconds *ahead* of the gate before it froze.

so the crop feed is now **pulled** rather than pushed: given the gate's position
in the clip, it advances to the newest main frame at or before that moment and
stops, which also backpressures its decoder into staying there. skew is bounded
by one crop interval, in one direction, by construction.

**what this invalidates.** every end-to-end number measured before it -- the
3/26 harvest, the `moved` distribution, the diagnosis that `scenery` had
condemned the whole road -- was scored against a still photograph. they are not
pessimistic estimates to be revised; they are measurements of a pipeline that
was not running. the detector-only figures survive, because `tools/label.py` and
`tools/score.py` never touch the crop path.

**what would have caught it.** nothing in the output said so. the detector's
report carried the gate frame number and not the main frame number, so a feed
that had stopped advancing looked identical to a street where nothing changed.
it now carries both, and `tests/e2e/test_sync.py` asserts the pairing directly:
that the two frame numbers stay within a crop interval of each other in video
time, and that no single main frame answers more inspections than the sampling
rate can explain. against a build of the old behaviour both fail -- at -9.2s of
skew and 284 consecutive inspections of one frame.

### what the two modes found, and what they cannot say

replaying the same clip both ways showed the harvest falling from 58 crops to 10.
chasing that turned up a real bug and a real limit of the eval, and they are
worth separating because only one of them was the cause.

**the bug.** `scenery` matched each sighting against a track's *most recent*
position and then drifted the track to it. consecutive sightings of a moving car
are always close together, so at high sampling rates the track walked across the
scene with the car, accumulated 90 seconds, and declared it parked. a test that
samples a crossing car at 15fps proves it: the car was called scenery having
travelled 4092 pixels. the anchor is now fixed at first sighting.

the shape of that bug is the interesting part: **it gets worse the faster the
machine**, so it could not be found on the machine it was written on, and would
have surfaced on the deployment host as vehicles quietly going missing.

**but it was not the cause.** fixing it moved the harvest from 10 crops to 11.
the real explanation is that most of these counts scale with sampling density:

| | realtime, 82% of frames | offline, 100% |
|---|---|---|
| sightings | 14,154 | 57,842 |
| over the 0.08 movement threshold | 826 | **2,003** |
| crops harvested | 58 | 11 |

offline clears the movement threshold *more* often, so nothing is being starved.
the harvest's per-place rate limit is simply working: densely sampled, a car's
successive positions overlap and are correctly refused as the same place, while
sparsely sampled the same car jumps far enough between looks to be cropped
several times in one passage. by the stated aim -- variety over volume -- the
offline number is the more honest one and the realtime 58 is inflated.

the same reasoning applies to track counts, for the same reason.

**so the eval compares builds, not modes.** a cross-mode comparison reads as a
large regression and means nothing; the tool now says so rather than printing a
column of "WORSE".

## measured detection quality

none of the above says whether a detection was *correct*. that needs labels, so
`tools/label.py` extracts the moments a vehicle actually transits the scene and
`tools/score.py` scores them against a hand label. the motion scan is its own
frame differencing rather than the project's gate or detector: an event list
drawn from the detector cannot contain a vehicle the detector never saw, which is
the quantity being measured.

over 40 minutes of recorded traffic, 107 transit events, 105 of them vehicles:

| clip | vehicles | found | median confidence |
|---|---|---|---|
| 10 min, sub h.264 15fps | 26 | 26 | 0.90 |
| 15 min, sub mjpeg 10fps | 39 | 39 | 0.90 |
| 15 min, main 30fps | 40 | 40 | 0.93 |
| **combined** | **105** | **105** | **0.91** |

false positives: **0** of 2 non-vehicle events.

the three clips were recorded either side of two camera changes -- the substream
switching to mjpeg, and the main stream going from 15 to 30fps -- and neither
moved the result.

### and what it was hiding

that table measures the **detector**. it says nothing about `must_have_moved`,
`scenery`, the per-place rate limit, or the skew between the gate frame and the
main frame a crop is cut from -- all of which run after the detector and decide
whether anything is actually kept.

asked end to end instead -- *when a vehicle crossed, did a crop of it reach the
harvest* -- `tools/endtoend.py` gives a second number, and the detector-only
figure should never be quoted without it.

with the two decoders in step:

| | 103138 | 133655 |
|---|---|---|
| transits (vehicles) | 26 | 40 |
| detected by yolo | 26/26 | 40/40 |
| **recall, end to end** | **26/26** | **40/40** |
| **precision, on motion** | **93%** | **96%** |
| crops per transit | median 6 | median 6 |
| `moved` of harvested | median 0.529 | median 0.552 |

the first run of this eval reported **3/26**, and every conclusion drawn from it
was wrong. the crop feed had raced to the end of the clip and frozen, so the
detector was inspecting a still photograph from gate frame 200 onward. the
`moved` values of the three survivors -- 0.095, 0.111, 0.111 against a 0.08
threshold -- were read at the time as evidence of a 250ms skew. they were
evidence of a nine second one. the same figures measured in step sit at 0.53,
five times the threshold.

**the detector is not the constraint and never was.** it was at 26/26 in the
broken run too.

**what the detector figure does not say**, which matters more than what it does:

- zero misses in 105 is not a 0% miss rate. the 95% upper bound is 3/105, so the
  honest claim is **recall above 97%**, not perfect.
- every one of these is a large near vehicle, 180px at the smallest. a go-4 on
  the far side of the street is smaller than anything in this sample.
- it scores the frame of *peak* motion, when the vehicle is most visible. r2.1
  asks how soon a vehicle is seen after it appears, which is not this.
- two non-vehicle events is not a false positive rate. neither the sky nor a
  pedestrian produced a vehicle box, which is reassuring and not a measurement.
- it scores the whole-frame pass. the hybrid's crops can only add detections, so
  the pipeline's recall is at least this.
- it is detection, not recognition. whether a vehicle is a go-4 (r1.4) is the
  classifier's question and the classifier is off.

incidentally the two clips were recorded either side of the substream switching
to mjpeg, and the codec made no difference: median confidence 0.89 on h.264 at
15fps against 0.88 on mjpeg at 10fps.

### precision does not come from the labels

recall needs labels: only a person can say a passing shape was a vehicle. asking
the same labels for precision -- counting a crop as a false positive if it fell
outside any labelled transit -- turned out to measure the labeller instead.

`label.py` emitted one entry per *event*, where an event is a stretch of time
the frame was changing, and boxed the largest blob in its peak frame. a city
intersection puts several vehicles in one such stretch. on a ten minute clip
that scored **171 crops as false positives**; pulled out and looked at, they
were a dark SUV, a yellow taxi, a white sedan, and 168 more like them, every one
cropped correctly. two manifests also sat at exactly 40 entries, which was an
undocumented `--max-events` cap silently discarding the rest.

so precision asks the motion scan directly: **was this crop's own box full of
pixels that changed.** no labels, nothing to truncate, and the same independence
argument as the transit list -- a median background differenced at 160x120 has
no way to inherit the blind spots of an 8.8 fixed-point exponential model at
native size.

three things had to be fixed before the answer meant anything, and the order
they were found in is the lesson:

1. **`binary_closing` erodes with `border_value=0`.** anything touching the edge
   of the array is treated as bordered by background and eaten -- a solid blob
   flush against column zero lost 40% of itself. so crops of vehicles entering
   or leaving the frame were scored as crops of a still street, and they
   clustered hard: 73-89% of everything called a false positive touched a frame
   edge, against 3% of what was accepted. fixing it moved precision from 89-91%
   to 93-96%.
2. **every blob, not the largest**, so the transit list stops being a list of
   busy moments and becomes a list of moving things: 27 events on one clip held
   **133 objects**.
3. **a fill floor**, because that then admits the utility wires strung across
   this block, which move in wind. size cannot separate them -- a wire spanning
   the frame covers more scan pixels than a distant car -- and neither can
   thickness, because a diagonal wire has a large square bounding box. what
   separates them is how much of that box is filled: vehicles at 0.35 on the 5th
   percentile against 0.16 on the 75th for everything else.

the residue is real and specific. what is left after all three is parked cars in
the **top-right corner**, where the detection box is clipped by the frame edge
and its size swings by a factor of three between looks -- so `scenery`, whose
anchor is deliberately fixed at first sighting, spawns a fresh track each time
and never accumulates the ninety seconds it needs to condemn the place. that is
a real defect with a known cause, which is what a working eval is for.

### the truncated distribution

`must_have_moved` is picked from the `moved` values the harvest reports, and
every harvested crop is by definition above it. the smallest number the eval
could ever see was the threshold itself -- measured at exactly 0.080 on all
three clips, which says only what was configured. choosing a threshold from that
is circular.

the pipeline now also logs what it *declined* and why, separating a scenery veto
from a vehicle that genuinely did not move: they both score zero and they need
opposite fixes. `Candidate` keeps `changed` and `parked` apart for this reason
rather than folding them into one number at the point of measurement.

### the detector's rate is derived, not chosen

the budget is a **duty cycle**, not a count: the detector may occupy
`DETECTOR_DUTY_CYCLE` of wall-clock while motion is present, and the rate follows
from a rolling average of measured inference time. a constant does not survive
contact with two machines -- yolo26s is 148ms on the development laptop and less
on the deployment host, and r5.5 says the same binary runs on both. a count tuned
for the laptop starves the server, which showed up as vehicles being "caught
late" there.

### but the rate was charged against the one look that matters

the budget was spent by a queue whose first entry is the whole-frame pass and
whose rest are magnified follow-ups, so a frame that ran out bought neither. the
whole-frame pass is the only look that covers the street: a frame that skips it
cannot find a vehicle at all.

that is bad in a way a per-second average hides, because **motion is bursty**. a
car crossing fires the gate on thirty consecutive frames, so the allowance runs
out exactly while a vehicle is in front of the camera. measured over 120 seconds
of this camera:

| | |
|---|---|
| frames where the gate fired and a main frame was paired | 292 |
| of those, frames the detector looked at | 121 |
| **frames where the detector never looked** | **171 (59%)** |
| regions skipped for budget | 717, of which 171 were the whole-frame pass |
| measured inference cost | ~130 ms, giving 4 looks/sec against a 15fps gate |

the symptom is a preview drawing a motion box with no detection inside it.

so the rate limit now governs follow-ups only, and the whole-frame pass always
runs when the gate fires (`inspections_allowed`). the worst case is one inference
per gate-firing frame, which is what not missing traffic costs. follow-ups are
magnification, and giving one up costs confidence on a distant vehicle rather
than the vehicle itself.

thread count is the other half of this and is a per-host setting, not a default.
measured on the development laptop, serially, on a recorded clip:

| threads | ms/inference | looks/sec | coverage of a 15fps gate |
|---|---|---|---|
| 1 | 213 | 2.8 | 19% |
| 2 | 126 | 4.8 | 32% |
| 4 | 101 | 5.9 | 40% |
| 8 | 76 | 7.9 | 53% |
| 12 | 69 | 8.7 | 58% |
| 16 | 78 | 7.7 | 51% |

past twelve it is oversubscription, not headroom.

## stage-1 detector

stock coco detector, filtered to car, truck, bus, motorcycle, and person. the
`person` class is not incidental; it is what the dismount trigger (r1.3) needs.

it also serves as the data harvester: from phase 1 onward it crops every vehicle
that passes, with no labeling effort, which is what makes the stage-2 training
set accumulate.

sits behind a `Detector` trait. it is agpl-3.0 and therefore viral if metermate
is published; d-fine and rt-detrv2 are apache-2.0 and close enough in accuracy
to be drop-in. the trait exists so that license decision stays a config change.

### why yolo26s and not the nano

the nano was the original default, on the reasoning that the gate makes the
detector rare so the cheapest credible model wins. measured against this camera,
that was wrong -- nano is not credible here.

on one frame from this street, at the 640x640 input the runtime actually uses:

| model | vehicles found | above 0.5 | above 0.7 | best | onnx ms |
|---|---|---|---|---|---|
| yolo26n | 4 | 2 | 0 | 0.70 | 47 |
| yolo26s | 5 | 5 | 5 | 0.88 | 148 |
| yolo26m | 8 | 6 | 4 | 0.91 | ~400 |

the counts understate it. nano missed the silver sedan in the foreground
entirely -- the largest, closest, least ambiguous car in the shot -- and drew the
van's box around a stretch of road. over a minute of live substream, nano's
median confidence was 0.39 against small's 0.75.

the scene explains why. the camera shoots through glass into the sun, so the
frame is veiled and low-contrast, and the wide lens puts a typical car at only
~77px in a 640 input. that is exactly the regime where a 2.4m-parameter model
degrades and a 9.5m one does not. a glare correction was tried first and
measured worse, so it was dropped; opening the window, measured later, was worth
more than either (see "it shoots through a window").

small over medium is the cost call: medium buys ~0.03 confidence for 2.7x the
inference. small is 3.2x nano's cost, which the motion gate absorbs -- idle
measures 4% of a core because the detector does not run at all without motion.
`MAX_DETECTIONS_PER_SEC` drops from 8 to 4 to match the new per-inference cost.

`tests/e2e/test_detector.py` pins this: it runs the real binary against a frame
from this camera and fails on nano's numbers.

### nms, and why it lives in the graph

yolo26 is marketed as nms-free, emitting one box per object with no suppression
step. that is not available to us: the released weights report `end2end=False`
on both the model and its `Detect` head, and the export duly produced a dense
`[1, 84, 8400]` tensor. forcing the flag on would run an untrained branch, so
the advertised property is simply absent from these weights.

so nms is baked into the exported graph instead, **class-agnostic**, giving a
final `[1, 300, 6]` of `x1, y1, x2, y2, confidence, class`. rust does nothing but
threshold.

class-agnostic matters more than it sounds. per-class nms cannot suppress a `car`
box and a `suitcase` box lying on the same van, because they are different
classes. measured on this camera before the change: 40 such pairs per 131 frames,
at iou up to 0.90. the phantom "suitcase" that kept appearing on the street was
never an object, it was a real vehicle wearing a second label. it is right for us
because the coco label answers no question metermate asks -- stage 2 decides what
a vehicle is -- and keeping both boxes would double-count vehicles, split one
track in two, and run the classifier twice on one crop.

the iou threshold is **0.7**, the ultralytics default, and it should not be
lowered. an earlier attempt to deduplicate in rust at 0.5 visibly deleted real
detections: vehicles parked along a street and viewed from this camera's oblique
angle overlap a great deal in image space, and genuine duplicates sit far higher,
at 0.81 to 0.90. after the change, cross-class duplicates measure zero while the
overlapping-but-distinct neighbours survive.

### is a natively nms-free model worth switching to

no. checked, and the answer is measured rather than assumed.

- **no yolo26 variant is nms-free.** `yolo26.yaml` ends in a plain `Detect`
  head, and the n/s/m/l/x scales differ only in depth and width multipliers, so
  every one of them behaves identically here.
- the only shipped configs with the natively nms-free `v10Detect` head are the
  **yolov10** family.
- the nms op costs **1.9 ms of 40.7 ms, about 4.6%**, confirmed by exporting
  with and without it: 412 graph nodes with one `NonMaxSuppression`, against 384
  nodes and a pair of `TopK` without.

so switching to yolov10n would trade a newer and more accurate model for at most
1.9ms on frames that are already gated, to remove a problem that is already
solved. revisit only if inference time becomes the binding constraint, which at
present it is not.

## subjects

a **subject** is a named thing worth publishing about. the go-4 is the first
one; r1.2 already asks for a second, a street sweeper is worth knowing about for
the same reason a go-4 is, and nothing about the machinery cares whether the
thing is a vehicle (r10).

metermate's job ends at the broker. it publishes what it recognised, where, how
confidently, and what was near it; **whether that is worth a notification is a
home assistant question**, and putting severity in here would mean recompiling
to change someone's mind about how urgent a street sweeper is.

so a subject is data:

    [[subject]]
    name = "go4"                          # also names trained/go4/
    detector_classes = ["car", "truck"]   # what stage 1 must have called it

the margin is deliberately absent. it is an output of a measurement rather than
a setting, so `--train` writes it to `trained/<subject>/trained.toml` beside the
vectors it was measured against; a margin typed into a config on another machine
is a rule nobody ran. a person may still choose a different row of the curve --
the labelling page's report can write any row of the curve the operating point
is chosen from, through the same `label::write_trained` as `--train`, with that
row's own recall and false positive rate saved beside it. the report sends that
curve and the chosen margin from the server, because the page used to pick its
own "best" over every crop while `--train` picked over the near curve, and the
highlighted row could be one `--train` would never write.

`trained.toml` carries that whole curve as `[[ladder]]`, one row per margin of
the sweep. the bar and its alternatives are then one measurement: "what would
-0.01 let in" is answered from the artifact on the deployment host rather than
from a terminal on the machine that trained, and never against a different set
of references. the reader ignores it -- `Trained` has no `deny_unknown_fields`,
so an older binary loads a newer file.

**which row is a choice, and the two axes are not interchangeable.** the default
is a recall bracket -- the cleanest margin keeping `OPERATING_RECALL` of the
passages -- because recall is counted on passages a person actually looked at,
while the false positive rate is estimated off the random pool and moves with
how much of the street has been sampled. an operator who knows what nuisance
rate they will tolerate wants to say *that* instead, so `--max-fpr` names the
ceiling and takes the most recall under it, and `--min-recall` moves the
bracket. under a ceiling the selection maximises recall rather than minimising
the rate again: an allowance that is set and then not spent buys nothing. ties
on both axes go to the more selective margin, spelled out rather than left to
which end of the sweep `max_by` happens to keep.

a constraint the curve cannot meet is an error that writes nothing, which is the
one place this differs from the default going unmet. the default failing is a
measurement reporting that stage 2 is not ready, and it still writes the vectors
with no margin; a rate an operator explicitly ruled out is a different thing,
and shipping the nearest row to it would look identical afterwards -- the
artifact records the margin, never the constraint it was chosen under.

the curve is counted under the deployment's confirmation. a held-out passage
used to count as caught, and a negative as a false alert, if any one crop
scored above the margin -- a one-of-one rule the pipeline never runs. both now
need `[track] confirm_m` of the last `confirm_n` crops in time order, the
tracker's own window, cleared after a passage gap as the tracker would lose the
vehicle. that is only possible because the negative pool is the whole harvest
rather than a sample: consecutive crops of one vehicle are in it. `fpr` and crop
recall stay per crop. measured on the go-4 set, 1 of 1 reports 0.6 false alerts
an hour at margin -0.05 and 3 of 5 reports none, while `fpr` is 2.58% under
both. every config table also parses with `deny_unknown_fields`,
so a key that has been removed cannot linger in a deployment pretending to have
an effect, and a misspelled one fails at startup instead of reading as set.

what that changes, in order of how expensive it is to retrofit:

- **each subject owns a directory.** `trained/<subject>/references.txt` and
  `trained/<subject>/negatives.txt`, still `<label> <floats>` per line, with
  `other` labelling the negative half. **the negatives belong to the subject,
  not to the street.** one shared pool reads as the cheaper design -- the street
  supplies a single set of "not any of these" -- but it cannot hold a second
  subject: a go-4 is a negative for a waymo, and a shared pool holding it would
  drag the go-4's own score down by exactly as much. so each subject is scored
  against its own negatives, and crops labelled for another subject are folded
  into them when that subject is trained. a subject with only one half built is
  skipped rather than judged, since "is this like a go-4" has no answer without
  "compared to what".
- **every confuser is a negative, and only the street is sampled.** the verdict
  is a comparison and nothing more: a crop is the subject when it sits nearer
  the references than the negatives, with no floor on how much it must resemble
  either. so a crop unlike every negative -- a cargo bike, when the negatives
  are cars -- is decided by which set it is *least* unlike. the negatives that
  teach the classifier what it confuses are the verdicts a person rejected and
  the crops of other subjects, and those all go in, however many there are;
  only ordinary traffic is a sample of thirty. they used to share one set of
  thirty, rejected verdicts first, and measured on the deployment that set
  drew 23 cars, 5 trucks and 2 buses out of 658 rejections of which 69 were
  cargo bikes. two cargo bikes then alerted at a +0.03 margin; with every
  rejected verdict among the negatives they scored -0.09 and -0.10. each crop's
  score against the negatives is taken once rather than once per fold, since
  the negatives do not change between folds and they are now most of the cost.
- **the verdict names a subject rather than being a boolean.** scoring is
  one-vs-rest: a top-k mean per subject against its own negatives, and the
  winner is whichever most exceeds its own margin. subjects that look alike are
  then a visible problem rather than a silent one.
- **stage 1 must stop being vehicle-only.** this is the cheap-now,
  expensive-later one (r10.3). the detector filter and the harvest's idea of
  what is worth keeping are currently compiled-in vehicle classes, so a subject
  that is a person or an object is discarded before recognition ever sees it.
  the filter should be the union of the subjects' `detector_classes`.
- **labels and cached embeddings are subject-independent.** a crop's embedding
  does not depend on what is being looked for, so adding a subject re-uses every
  vector already computed (r10.4). a label says which subject a crop *is*, and
  `other` means "none of the ones known when it was labelled" -- which is why
  the contamination screen matters more as the list grows, not less.
- **topics gain a subject segment**: `metermate/<subject>/<event>`, with
  `metermate/motion` staying subject-free because motion has no subject. the
  event names already exist -- sighting, approaching, dwell, dismount -- and are
  descriptions of what happened rather than of how much it matters.

none of this needs building before the go-4 works. what it needs is that the
pieces written now -- the label file, the embedding cache, the eval -- take a
subject name as a parameter rather than assuming one, which costs nothing today
and is most of the work if left until later.

## stage-2 classifier

no public dataset of sfmta go-4s exists and this camera sees a few per day, so
the design has to be useful at zero examples and improve as examples arrive.
**nothing here is trained from scratch.**

### stage 2 only ever sees what moved

`classify` runs inside `for i in moving`, and the harvest write is in the same
loop over the same list. a detection that fails `[harvest] must_have_moved` is
never classified, never harvested, and therefore never appears in a verdict, a
crop name, or an alert. **so `must_have_moved` is a stage-2 recall knob**, which
is invisible from either module alone: it reads as a harvesting threshold in one
and as silence in the other.

measured on the 2026-09-15 incident, replaying the two clips where a go-4 was
parked at the kerb writing a ticket: neither produced a single verdict, across
roughly 70 seconds of it sitting there. counted in detail on one of them
(`1789488218617`), 198 of 200 kerb-region detections were declined `below
must_have_moved`, `changed` between 0.015 and 0.30; the other was only observed
at info level, where 145 kerb-region detections likewise yielded nothing. the same vehicle fired
2.5s after becoming visible on arrival and again as it pulled away. **a go-4
doing the thing worth alerting about -- stopping -- is the case stage 2 cannot
see**, and no reference set or margin can fix it, because the pixels never reach
the classifier.

### frozen embedding, swappable head

one pretrained embedding model runs over the full-res vehicle crop and never
changes. only the head on top of it changes:

| stage | head | examples needed | training |
|---|---|---|---|
| day one | cosine match against reference crops | ~5-10 total | none |
| week one onward | logistic regression or small mlp | 20-50 per class | seconds, on cpu |
| only if that saturates | fine-tune the last backbone block | several hundred per class | minutes |

this replaces an earlier two-model design that would have swapped a few-shot
matcher out for a separately trained cnn. keeping the backbone fixed is better in
every dimension that matters here:

- the runtime cost is identical at every stage, so the resource budget never moves.
- harvested crops are embedded **once** and cached, so retraining a head is
  seconds rather than a training run, which makes the phase 5 feedback loop cheap.
- a linear probe on a strong frozen embedding is far more data-efficient than
  fine-tuning, which is the binding constraint when positives arrive a few a day.
- there is no model swap to validate, and no day where behaviour changes
  discontinuously.

### the file that ships is written by the thing that measured it

the head is a reference file -- `<label> <floats>` per line -- and for a while
the binary could label the harvest and measure it but not produce that file. it
came out of a python tool instead, so **what shipped was built by a different
toolchain from what was measured**, which is invisible in the worst way: the
pipeline keeps classifying and the eval keeps printing confident numbers about a
rule nobody runs.

`--train <subject>` closes it, and the shape matters more than the feature:

- **the selection is taken off the report, never made twice.** `measure` already
  chooses reference passages for spread and the clearest crops within them;
  training writes those, by name. two selections would agree until the day they
  did not, and nothing would say which of them the deployment held.
- **it refuses when the eval voided its own numbers.** the preconditions exist
  to stop confident claims about a rule nobody ran; writing a file from them
  would ship that mistake rather than print it.
- **the vectors written are uncentred**, whatever the measurement ran with.
  centring helps a trained head and does nothing for the shipped rule, whose
  two-sided subtraction already centres implicitly, so a centred file would
  leave the runtime comparing against a space it never sees.
- **the margin is part of the artifact.** a reference file and a `[classifier]
  margin` that were not measured together describe a rule nobody ran, so the
  training run prints the margin its own sweep chose.

### does a few-shot embedding actually work here: measured

tested before building, on crops cut from real images by the detector, so it
reflects what the pipeline would really feed the classifier. a stock clip
vit-b/32, 15 go-4 crops against 51 street-car crops:

| | sharp references | references degraded to video fidelity |
|---|---|---|
| go-4 <-> go-4 | 0.704 | 0.719 |
| car <-> car | 0.834 | 0.834 |
| go-4 <-> car | 0.666 | 0.682 |
| held-out go-4 nearest a go-4 | 14/15 | **13/15** |

the second column matters. the go-4 crops came from sharp photographs and the
car crops from video, so the model might have been separating *source* rather
than *vehicle*. degrading the references to this camera's fidelity removes that
confound, and separation survives: 87% rather than 93%. the signal is real.

**but the margin is thin**: 0.719 against 0.682 is 0.037 apart, while cars
resemble each other at 0.834. two consequences for the design:

- classify by **nearest neighbour among references**, never by an absolute
  cosine threshold. the ranking is reliable where the absolute value is not.
- reference count is the lever. fifteen crops already give 87%; the harvester
  exists to turn that into hundreds drawn from this camera's own viewpoint.

### the waymo rehearsal, and what it found

the 0.02 margin above had two candidate causes, needing opposite fixes: the
frozen embedding cannot do this task, or the seven go-4 references are sharp web
photographs of nypd and seattle vehicles shot from the pavement while our crops
are soft, shot downward through glass. **waymo separates the two**, because
waymos pass this block and appear in our own harvest, so their references come
from the same camera, glass, angle and light as the crops they are matched
against. nothing about waymo ships.

measured on 5 hand-labelled waymo passages (18 crops) against 770 harvested
crops of ordinary traffic, all from this camera. cross-passage only: two crops
0.4s apart score near 1.0 and report nothing but that a picture resembles
itself.

| | go-4, web references | waymo, same-camera references |
|---|---|---|
| positive <-> positive | 0.716 | 0.878 |
| car <-> car | 0.834 | 0.841 |
| positive <-> car | 0.694 | 0.843 |
| gap | +0.022 | **+0.034** |

the absolute numbers are not comparable across the columns and the gap is. every
crop in the right-hand column comes from one camera pointed at one street, so
everything resembles everything: even two unrelated cars sit at 0.841. the web
references sit far from all of it, which is what made the left column's numbers
low rather than discriminating.

**same-camera references widen the gap by half, and it is still nowhere near
enough.** leave-one-passage-out against the whole harvest as negatives, swept
across `margin`:

| margin | passages | fpr | false alerts/hour |
|---|---|---|---|
| -0.020 | 5/5 | 15.8% | 37.3 |
| -0.015 | 4/5 | 9.2% | 25.6 |
| -0.010 | 3/5 | 5.3% | 16.7 |
| +0.000 | 2/5 | 1.5% | 5.8 |

there is no operating point on that curve worth alerting from. and the
comparison that decides anything is *alerts* per hour, not crops: a vehicle is
cropped several times as it crosses, so one civilian car the classifier dislikes
produces a handful of firing crops within a couple of seconds and downstream
they are one alert. the same passages-not-crops argument that applies to the
positives.

that is not contamination flattering the negatives. every crop that fired at
`DEFAULT_MARGIN` was laid out as a contact sheet and looked at: white sedans,
dark suvs, a red hatchback, pickups. not one waymo among them.

#### three things that changed the answer, all of them setup rather than model

worth recording, because the first version of this measurement got all three
wrong and the numbers it produced looked perfectly reasonable:

- **the reference negatives were drawn off the front of a name-sorted sample**,
  which is the oldest crops, so all thirty came from one hour of one afternoon
  in one light. the negative half of the decision was being made by a single
  lighting condition. fixing it moved recall at margin 0 from 5/5 to 2/5.
- **the negative pool was a 770-crop sample rather than the harvest.**
  `other_score` is a top-k mean, so a small pool puts its nearest negatives
  further away and everything fires more easily. the pool size is not a
  presentational choice; it moves the operating point.
- **the margin sweep started at zero.** nothing in `go4_score > other_score +
  margin` requires the margin to be positive, and the useful region turns out to
  be negative. a sweep starting at zero reported "no margin keeps 80% of
  passages" when it had simply not looked -- the same truncation as reading
  `must_have_moved` off the crops that cleared it.

#### the head, not the backbone -- suggestive, not settled

the same leave-one-passage-out comparison, with a logistic regression fitted on
the same frozen vectors instead of a cosine vote:

| head | auc per crop | auc per passage |
|---|---|---|
| nearest neighbour against references (ships today) | 91.4% | 95.4% |
| linear probe on the same embeddings | **98.6%** | **97.7%** |

**per passage is the honest column**, and it narrows the gap a long way. an
alert fires on one crop of a passage rather than all of them, so a passage's
score is its best crop's; scoring per crop lets a passage that happened to be
cropped seven times outvote four that were cropped once, and two of the five
passages here have a single crop.

what the auc hides is a qualitative difference the per-passage scores show
plainly. the probe puts four of five passages at +0.31 to +1.81 against a
negative distribution centred near zero; the nearest-neighbour head puts three
of five *below* zero and separates the other two by 0.02. those are different
kinds of working.

**but this is five passages.** the per-passage auc is computed over five points,
so which single passage happens to be hardest moves it further than the choice
of head does -- and one passage does dominate it, see below. the direction is
consistent across every setup tried, and the size of the gap is not to be
trusted until there are twenty.

#### the hard case is occlusion, not aspect

one passage fails under both heads and both preprocessings:
`1789257341824_car_079_238x197.jpg`, the single rear-quarter crop, scores below
the civilian references every way it is measured. looking at it: roughly 40% of
the crop is deep building shadow, with a dark red car intruding at the bottom
corner. the waymo itself is lit and unambiguous to a person.

so the hard case here is **crop composition** rather than viewing angle, which
was the expected failure mode. that points at the harvest side -- crop margin
and how much context is kept around a box -- as much as at the classifier.

#### centring, and where it helps

clip's embedding space is anisotropic: everything shares a large common
component, and on crops that all come from one camera pointed at one street that
component is most of what a cosine measures. subtracting the harvest's mean
direction and renormalising needs no labels at all, and it collapses the shared
part almost entirely:

| | positive <-> positive | car <-> car | positive <-> car | gap |
|---|---|---|---|---|
| raw | 0.878 | 0.839 | 0.842 | +0.035 |
| centred | 0.186 | 0.001 | 0.002 | **+0.184** |

two unrelated cars go from 0.839 to 0.001. so 0.84 was never a statement about
cars resembling each other.

**it does not help the shipped rule.** measured on the two-sided decision,
centring moves nearest-neighbour auc 95.4% -> 94.9% per passage: no better, and
slightly worse. the reason is that `go4_score - other_score` is *already* an
implicit centring -- subtracting the negative-reference score removes most of
the common direction, so doing it explicitly first adds nothing.

it does help the linear probe, which has no such subtraction: 97.7% -> 98.7% per
passage, and at a threshold keeping 89% of crops the share of cars firing falls
from 2.07% to 1.27%. so centring is worth having with a trained head and is not
worth changing the day-one head for.

so clip vit-b/32 can see the difference clearly. what cannot use it is the
day-one head: a top-k cosine mean asks "which reference is nearest", and on
crops that all share a street, a camera and a lens, nearest is dominated by the
dimensions that encode the scene rather than the vehicle. a fitted direction
ignores those; a cosine cannot.

three consequences, in descending order of how much the evidence supports them:

- **"get same-camera references" does not rescue the day-one head.** that was
  the assumed fix and it is measurably not sufficient. this one does not depend
  on the sample size: it holds at every margin, under both preprocessings, and
  in two independently built setups. r6.2 -- useful before any project-specific
  model is trained -- is met by the nearest-neighbour head only in the sense
  that it runs, not in the sense that anyone could act on it.
- **a go-4 will be harder than this.** a waymo is a white i-pace with a roof
  dome and pillar lidar, and it is the most distinctive vehicle on this street.
  an sfmta go-4 is plain white and boxy. a head that cannot separate the easy
  case will not separate the hard one.
- **a trained head looks like the answer, on evidence that is not yet
  conclusive.** it needs no new backbone, no new export and no change to the
  resource budget -- the embeddings are already cached and fitting a linear head
  over them is seconds of cpu -- so it is cheap to try. but see the sample size
  below before reordering the roadmap around it.

the caveat that bounds all of it: 5 passages. the separation figures rest on 110
cross-passage crop pairs, which is enough to see 0.878 against 0.842. everything
else rests on 5 points. recall over 5 passages carries a 95% interval of 57-100%;
the per-passage auc that separates the two heads is an average over five
comparisons, one of which fails under every configuration tried. the collection
rate is about 1.7 passages an hour, so a reference set plus 25 held-out passages
is roughly a day of harvest, and that is the point at which the head comparison
is worth acting on rather than noting.

### choosing the backbone

wanted: clip-family semantics at mobile cost. mobileclip-family models are built
for exactly this tradeoff and land in low single-digit milliseconds, which the
3.51ms mobilenetv2 measurement suggests is realistic on cpu.

worth evaluating against it: a vehicle re-identification embedding, trained on
veri-776 or compcars. those are tuned for "same kind of vehicle", which is nearer
our actual question than generic image-text similarity. the harvest makes this a
cheap experiment: embed the same cached crops with each candidate and compare
head accuracy offline, with no change to the running system.

### bootstrapping labels

open-vocabulary detectors (owlv2, yolo-world) prompted with something like
"three-wheeled parking enforcement vehicle" are far too slow for the hot path at
hundreds of milliseconds, but they are excellent **offline** on the harvest. they
turn labeling from searching into verifying, which is the difference between an
afternoon and a month.

the classifier sits behind a `Classifier` trait so the backbone stays swappable.

### what an sfmta go-4 actually looks like

taken from news footage of san francisco enforcement vehicles, which is a better
guide than the wikimedia photographs: those are mostly nypd, and nypd livery is
misleading here.

- **sfmta go-4s are plain white.** not the blue and white of the nypd vehicles.
  colour alone therefore cannot separate a go-4 from the white sedans parked
  along this street, so stage 2 has to key on shape rather than paint.
- the silhouette *looks* distinctive to a human: a tall, narrow, boxy cab with a
  large upright windscreen. **but aspect ratio does not separate them**, and it
  was measured rather than assumed. go-4 reference photos have a median height
  over width of 0.64; mixed sf street traffic has 1.14, with the distributions
  broadly overlapping. the ratio conflates viewpoint with vehicle type -- the
  references are side-on close-ups and the street footage is mostly rear views --
  so it measures which way a vehicle is facing, not what it is. a cheap
  geometric prior is therefore not available, and the embedding has to do the
  work.
- the **amber roof beacon is real and lit while patrolling**. this was recorded
  as an unvalidated hypothesis and is now confirmed. a temporal flicker score
  over the roof region is cheap and highly specific, and worth building.

the footage also reproduces the class ambiguity seen on this street: one vehicle
drew `truck 0.85` and `car 0.89` in the same frame.

### a note on that footage

it is broadcast news material, used locally to understand the target and to
exercise the pipeline against real go-4s in motion. it is **not committed and
not redistributed**; only the observations above live in the repository. the
openly licensed wikimedia references are what `tools/fetch_references.py`
fetches, and they are what the shipped few-shot classifier will use.

## pan and tilt

the camera can be panned and tilted, which invalidates the background model, the
roi polygon, and the scale prior all at once -- and reports nothing. it was
panned once during development and every position-derived assumption silently
went stale.

absolute position is readable, so `src/camera` polls `ptz.cgi?action=getStatus`
every `ptz_poll_secs` and resets the gate, scenery, and the inspect window when
the position moves by more than a degree. the threshold is there because the
reading dithers by a tenth while the camera is still, and a reset costs
`warmup_frames` of not watching the street.

two deliberate choices:

- **it degrades to a warning.** a camera with no ptz, or one that cannot be
  reached, turns the watcher off and leaves detection running. a correctness aid
  that can take the pipeline down is a worse trade than the staleness it guards.
- **only `getStatus` is ever requested.** the streaming cgis are stateful --
  fetching the mjpeg url reconfigures the substream encoder and leaves it that
  way -- so the watcher is restricted to the one endpoint known to be read-only
  (r5.4).

digest auth is ~60 lines of md5 in `src/camera/md5.rs` rather than a dependency,
tested against the rfc 1321 vectors. it is used for nothing security-bearing;
the camera simply demands it.

metermate does not slew the camera to track a detection. moving the camera
destroys the background model and the scale prior at exactly the moment a clean
crop matters most.

### a bounding box is not a vehicle

`changed_fraction` asks what share of a detection's own box changed, and the
note above claims that answers "did *this object* move". it does not, and the
gap is where the harvest's remaining false positives came from.

a box is a rectangle. it also contains road, sky, and any other vehicle that
overlaps it. a car driving past a parked one intrudes on a corner of the parked
car's box, those pixels genuinely change, and the parked car is credited with
the movement and cropped -- again on the next car, and the next. it is the same
failure the iou approach had, arriving by a different route.

measured over the 283 crops from one ten minute clip, splitting them by whether
an independent scan agreed anything was moving:

| | crops of a moving vehicle | crops of a parked one |
|---|---|---|
| n | 263 | 20 |
| share of the motion in the box's middle half | **0.42** (p10 0.24) | **0.00** (p90 0.33) |
| centroid offset from the box centre | 0.37 | 0.85 |

more than half the bad crops had no changed pixel in the middle of the box at
all. so `motion_must_be_central` requires a share of the movement to sit in the
middle half of the vehicle's own box; motion clipping a corner belongs to
whatever is passing.

blob containment -- does the motion blob mostly lie *inside* the box -- was
measured too and rejected: 0.89 against 0.72, heavily overlapping. where the
motion sits separates; how much of it is enclosed does not.

### the roi

`[gate] roi` is a polygon in gate pixels. pixels outside it are skipped in
`mark_changed`, before anything reaches the background model, which is cheaper
than filtering regions afterwards.

the block is strung with utility wires and they move in wind: on one ten minute
clip, 45 of the 69 moving objects an independent scan found were wire against
sky. none reached the harvest, but each was a region the gate grouped and a
person later had to label.

empty means the whole frame, since an roi describes where one camera points.
the share of the frame it covers is logged at startup and a very small one
warns, because the failure mode is a camera that detects nothing and looks
exactly like a quiet street.

## alerting

one topic per subject and outcome. **metermate reports what it saw; it does not
decide what deserves attention** (r10).

| topic | condition |
|---|---|
| `metermate/motion` | the gate fired. no subject: nothing is recognised yet |
| `metermate/<subject>/sighting` | recognised, confirmed over several looks |
| `metermate/<subject>/moving` | and in motion |
| `metermate/<subject>/stopped` | and stationary for `stopped_after_secs` |
| `metermate/<subject>/dismount` | a person adjacent to a stopped subject (r1.3) |
| `metermate/<subject>/departed` | its track ended |

this replaced five severity-ranked tiers -- sighting, approaching, dwell,
dismount -- which asserted that a dismount matters more than a sighting. it
might, and whether it does is a question about this street on this day, which
home assistant is the right place to answer and metermate is not. the payload
therefore carries facts and no ranking:

```json
{"state":"ON","subject":"go4","track":12,"dwell_s":43.0,
 "confidence":0.87,"protected":"adjacent","box":{…},"at":1789…}
```

two of those repay explanation.

**`protected` is four states, not a flag.** `near_protected: false` would mean
both "the go-4 is nowhere near the van" and "the van is not parked here at all",
and those want opposite rules -- the second is r9.2, where there is nothing to
protect. so `adjacent | near | far | away | unknown`, with `away` being r9.2.
since severity moved downstream, the payload has to carry enough for a rule to
*implement* r9.2 rather than being told the answer.

**`approaching` is deliberately absent.** it asserts a direction nothing
measures: there is no protected vehicle yet for a go-4 to be approaching.
`moving` is the half that is computable from the tracker's velocity today, and
the original name returns when a heading has something to point at.

publishing is edge-triggered on transitions. at fifteen frames a second the
alternative is fifteen messages per vehicle per second all saying the same
thing. a track reports nothing until it has either genuinely moved or held still
long enough to mean it, because dwell starts at zero and announcing a newly-seen
parked car as `moving` is worse than saying nothing.

alert publication is decoupled from evidence capture (r2.2): the message goes out
first, and snapshot writing, crop harvesting, and disk io happen after, off the
latency path.

**the exception is a confirming look with nothing saved** (r4.5). a track stays
confirmed for as long as it is in view, and the harvest keeps one crop per place
per `min_interval_secs`, so every agreeing look can be deduplicated and the phone
notified about a vehicle the verdict page never listed. that look is now written
before the message goes, and it is the only look of that vehicle that publishes at
all -- which is what edge-triggered meant here all along. the write is not what
r2.2 was guarding against: a 184x449 crop encodes in **0.8 ms** and a close-up
600x1200 one in **6.6 ms**, once per vehicle, against a budget of 1.5s. a
confirmation whose crop cannot be written is logged and not sent, because an alert
nobody can check is a claim, and the page is where checking happens.

### the doorbell is not the integration surface

mqtt assumes something downstream is listening, and the chain that ends at a
phone is a broker, then home assistant, then an automation, then a companion
app. every link is a place a deployment quietly stops working, and the product
is "move the car before the ticket" -- so `[ntfy]` posts to an ntfy server
directly, with or without a broker. a config with `[ntfy]` and no `[mqtt]` is a
complete deployment: one camera, one street, one person who wants to be told.

**both channels leave from `Alerter::publish`.** one call decides an outcome
happened, and it publishes and notifies; the alternative is two code paths that
agree until the day they do not, with no way to tell which one the phone was
listening to.

what goes out is deliberately not a second copy of the payload. the topic
carries track ids, boxes, dwell and `protected` because a rule needs them to
implement r9.2 itself; `ntfy.sh` is somebody else's machine, and the facts a
rule needs are not facts a stranger needs. the notification is a headline, a
sentence, and the crop that caused it -- which is r4.3 as far as a notification
can take it, and the difference between a buzz that is a claim and one that is
evidence. the crop travels as the request body rather than as a link, because a
link to the preview is only followable from the same lan, and being told while
out is the whole point.

the send is a message on a bounded queue and a worker does the talking. an http
post to a server across a domestic uplink is exactly the kind of thing that
stalls for thirty seconds, and r2.1 gives the entire path 1.5s. a full queue
drops the newest and says so: these are real-time alerts, and a backlog of them
is a phone buzzing about vehicles that left ten minutes ago. failures are logged
rather than retried, per r4.4 -- network loss should need no operator action,
and a notification about a departed vehicle is worse than none.

`--notify-test` is the one synchronous path, and the one that fails loudly. a
token is the single setting the config cannot validate, since only the server
can judge it, and the alternative to a command is finding out it was wrong on
the evening a go-4 parks outside. it prints the link a tap would follow, for the
same reason: an address the phone cannot reach is a notification that goes
nowhere once it is on the screen.

**the tap is aimed at the crop, not at the page.** `click` holds the preview's
address and the notification is sent with `<preview>/#/verdict/<crop>` in it,
which is safe to promise only because r4.5 refuses to notify about anything the
verdict page cannot show -- the guarantee that makes the specific link cheaper
than the generic one, since a link to the front of a two hundred crop harvest is
a task, and a link to the crop is a glance. an outcome with no crop of its own
names the verdicts page, and an empty `click` sends no header at all.

## inference backend

the runtime is onnx runtime via the `ort` crate (2.0.0-rc.13, wrapping onnx
runtime 1.28). **cpu is the baseline and the only supported configuration.**

### measured, on an i7-1360p development laptop

| workload | cost |
|---|---|
| main stream h.264 decode, 2560x1440 | 1.27s cpu per 10s of video, **~13% of one core** |
| mobilenetv2 int8 224x224, single thread | **3.51 ms** per inference |
| yolo26n 640x640, two threads | **47 ms** per inference |
| yolo26s 640x640, two threads | **148 ms** per inference |
| whole pipeline, quiet street | **4% of one core**, 155 mb rss |

mobilenetv2 is a deliberately pessimistic stand-in: mobilenetv3-small at 128px is
three to five times cheaper, so the real stage-2 classifier lands under 1ms.

the detector is the one expensive piece, and the gate is what makes it
affordable: on a quiet street it does not run at all, which is why the whole
pipeline idles at 4% of a core against the 30% budget (r3.1). under sustained
motion `MAX_DETECTIONS_PER_SEC` bounds it to four inferences, ~590ms of work per
second of wall clock.

**decode costs more than inference does on an idle street.** that comparison
decides the backend question. a gpu would accelerate the cheap part of
the pipeline and leave the expensive part untouched.

### accelerators: evaluated, not adopted

vulkan compute was investigated in some depth. the conclusion is that it is not
worth the engineering for this workload:

- it does not reduce idle cost, which is decode plus the motion gate.
- it accelerates a few milliseconds of gated inference, which nothing is waiting on.
- it would make the binary harder to run on a development laptop, which is a
  requirement in its own right (r5.5).

the same reasoning rules out rocm on gfx1151 (preview quality, needs
`HSA_OVERRIDE_GFX_VERSION`) and the xdna2 npu (immature linux stack, windows-first
toolchain). neither is justified by a workload this small.

the execution provider remains a config value rather than a compile-time choice,
so this is revisited cheaply if a future model is heavy enough to change the
arithmetic. if that day comes, the ranked candidates are the onnx runtime plugin
ep api (ort >= 1.23, and we are on 1.28), `onnx-vulkan-rs` as a pure-rust native
vulkan ep, and burn with the cubecl vulkan backend. until then, no accelerator
code ships.

### if optimisation is ever needed, optimise decode

the target for future work is the 13%, not the 3ms:

- drop the gate to the substream and pull full-res crops only on demand, which is
  the `low_power` profile already in the design.
- vulkan **video decode** via radv is worth testing on the strix halo box, since
  va-api through radeonsi does not support that silicon and radv's video path is a
  different driver entirely. `-hwaccel vulkan` failed on the intel laptop and fell
  back to software, which proves nothing about radv. untested, and not assumed by
  any requirement.

## failure handling

the camera is on wifi and will drop. the design assumes every external thing is
unreliable:

- the ffmpeg subprocess is supervised and restarted with backoff on stall or exit.
- a stall is detected by frame arrival timeout, not by process liveness, because
  ffmpeg will happily sit on a dead rtsp socket.
- mqtt reconnects with backoff, and discovery is republished on reconnect (r4.4).
- harvest and evidence writers enforce a disk budget and delete oldest first
  (r6.4). a full disk degrades the harvest, never the alert path. labelled
  harvest crops count towards the budget but are never deleted.

### the browser cannot be trusted to fetch its own video

the preview's h.264 path pointed a `<video>` element at an endless fragmented
mp4 and let it fetch the stream itself. pictures appeared, so it looked finished.

it was unfixable in every way that mattered, because the element owned the
buffer and the resource had no duration and no byte ranges:

| asked for | got |
|---|---|
| `seekable` | `[0, 0]` -- nothing, ever |
| `buffered` | `[13.67, 17.27]` |
| `currentTime = 16.77` | clamped to `0`, then discarded |

so every correction written against it was inert. the drift seek had been in the
page since before any of this and had **never once fired successfully**, which
is why latency accumulated without bound: playing 10% fast was the only
mechanism that did anything, and it takes twenty seconds to close a two second
gap. a window left in the background came back and replayed the time it had
missed, because nothing could skip it. a `catch_up = "skip"` policy, added on
request, silently did nothing at all.

the page now feeds the element through a `MediaSource`: one `fetch`, the box
splitting ported from `preview::fmp4::Split` so client and server agree on what a
fragment is, appended to a `SourceBuffer` we evict behind the playhead. the
codec comes from `avcC` in the init segment rather than a guess.

| | progressive | media source |
|---|---|---|
| seekable | nothing | tracks live |
| buffer | unbounded | 6s, evicted |
| connections per open | 2 | 1 |
| latency | 2.6s, closed at 1.1x | **0.3-0.6s, steady** |

**the cushion is measured rather than chosen.** it exists so a fragment arriving
late has something to play through, so its right size is the worst lateness the
stream actually produces -- and that is visible in the spacing of the appends.
measured on this camera: fragments nominally 61ms apart, p99 204ms, worst 228ms,
so the lateness to cover is 167ms at its worst. the page keeps the last eight
seconds of gaps, takes the 95th percentile of lateness beyond nominal, doubles
it, and clamps between `[preview] target_buffer_ms` and `max_buffer_ms`. the
95th rather than the maximum so one hiccup does not raise latency for the next
eight seconds; the drift correction covers the rare outlier.

it settles at the 250ms floor on this stream.

**what this does not cover.** every measurement above is chrome. the deployment
runs firefox, which is where the chop that prompted the work appears, and
firefox's handling of progressive fragmented mp4 is the weaker path -- so the
thing this was built to fix is the thing least verified. a viewer joining
mid-GOP is also still unhandled: the relay hands out fragments from the next
boundary, which with per-frame fragmentation is almost always a P-frame.

### a crop is named before it has bytes

crops are cached `immutable` for a year -- a name carries the millisecond it was
taken, so it can never mean a different image -- and that is what keeps the
viewer from re-downloading the visible grid every four seconds.

that caching decision reaches back into the write path. `fs::write` creates the
file and then fills it, so for as long as the encode takes, a complete-looking
name sits in the harvest directory over zero bytes. the viewer lists that
directory on a timer and offers the name; the server finds the file readable and
answers `200` with an empty body and a year of `immutable`. that is a valid,
permanent, wrong answer, and it showed as a black tile clicking could not fix --
the zoom requests the same url and gets the same cached nothing. only a reload,
which revalidates past `immutable`, recovered it.

the write now goes to a temporary and is renamed, which is atomic within a
filesystem: a crop is either absent or whole. the listing skips empty files and
the server answers `404` rather than an empty `200` -- both unreachable now, and
both a line each, because a `404` is temporary and an empty `200` is forever.

### the name is the only index, so the verdict goes in it

a crop has no sidecar and no database: the filename is the whole record, which is
why it carries the millisecond, the detector class, the confidence and the size.
stage two's verdict is known at exactly the moment the crop is written -- it is
reached on those very pixels, one statement earlier in the same loop -- and was
being dropped, so the one crop in two hundred that is a go-4 sat in the grid
captioned `car 0.83` like the cars around it.

so the name gained an optional field: `<ms>_<class>[_<subject>]_<conf>_<w>x<h>.jpg`.
three decisions in that shape, each the cheap end of a trade:

- **beside the class, not instead of it.** "the detector called this a truck and
  stage two called it a go-4" is the case worth being able to find again, and
  either half alone hides it.
- **an underscore is the separator**, because `url_safe` reduces a class name and
  a subject name to letters, digits and dashes, so one cannot occur inside
  either. every name written before this parses unchanged and reads as having no
  verdict, which is what it means.
- **the frame's verdict, not the track's.** a crop is one look at one moment;
  confirmation is a property of a track over `confirm_m` of `confirm_n` looks.
  gating the name on it would mark the first crops of a passage differently from
  the last for a reason that is not about their pixels. the filenames are
  therefore a strict superset of what is ever published, and the difference
  between the two sets is the unconfirmed-look rate -- measurable, where before
  an unconfirmed look left no trace anywhere.

a verdict then gained one more field of its own, its margin: `<ms>_<class>[_<subject>
[_<margin>]]_<conf>_<w>x<h>.jpg`. thousandths, signed -- `20` is `+0.020`, `-07` is
`-0.007` -- and only ever beside a subject, since it is the gap that made the claim
and a crop stage two said nothing about has no claim to measure. three fields after
the millisecond can therefore only mean a verdict was measured, which is what keeps
every name written so far parsing exactly as it did.

it is there because the verdict page and the notification both quote the number, and
the name is the only place this project keeps anything per crop. a negative margin is
not a mistake and is not rounded away: this deployment ships at -0.005, so a crop can
be named while sitting nearer the street than its own class, which is the one verdict
worth reading twice. the same sign reaches the phone (`margin -0.007`) and the
caption (`GO4 -0.01`).

one parser reads it: `harvest::parse_name`, which the preview and the harvester both
go through. the browser is handed the fields as json and never parses a name, so a
change to the grammar is `Harvester::save`, that function, and the name builders in
`tests/e2e/labelling.py` -- which is how a field was added to it without touching the
page at all.

**the subject in a crop name is evidence of what fired, never a label.** it is
what the classifier made of those pixels at the configured margin, which is a
claim to be checked and not a judgement to train on: on the first evening the
field was recorded, every crop carrying it was a false positive. so
`*_go4_*` is exactly the right query for "what did it fire on" and exactly the
wrong one for "which crops are go-4s", and building a reference set from it
trains on the classifier's own mistakes -- with the set looking fully populated
the whole time, which is the shape of failure this project keeps meeting.

the same caution applies one level up. a set lives at `sets/<subject>/<id>/`, and
that subject is what the set was *collected for*, not what its crops are: the
go-4's own set is 586 positives against 806 negatives, because crops that share
the camera, the glass and the hour are exactly what a negative should be. truth
comes from `labels.txt`, written by a person. not the filename, and not the
directory holding it.

### a set carries its own labels, crops and vectors

    sets/<subject>/learn/        what the labelling ui accumulates
    sets/<subject>/<timestamp>/  a snapshot assembled by hand
      crops/  labels.txt  embeddings.bin  [clips/]
    trained/<subject>/
      references.txt  negatives.txt

`learn` is a reserved id beside `other` and `unclear`. the split is about who
wrote the set rather than when: the ui labels continuously and wants one growing
directory per subject, while a snapshot -- an incident cut from recorded video --
is a fixed thing worth naming for its day.

**the labels sit beside the crops they describe**, which `label::paths` already
gets right by deriving both from the crop directory's parent. that is what lets a
set be trained, re-measured, or handed to another machine on its own, and what
makes `--harvest sets/` able to merge several: each set's labels and vector cache
are read from its own directory, names are sorted across all of them by timestamp
so passage grouping does not invent a boundary at each seam, and a crop two sets
disagree about is refused rather than resolved by ordering.

`--label` reads its roots through the same walk (`label::loaded`) and writes back
per set. it takes any number of `--harvest` roots and, given none, the pair
`--retrain` reads (`with_sets`: `[harvest] dir` and `[train] sets`), so what is
trained on is what was offered for labelling. `--embed` is that walk with
nothing after it (`label::embed`): the embedder is the slow half of opening the
page and needs nobody watching, so `make embed` runs it ahead and the page
opens on vectors already written. the page still embeds what it finds missing,
which keeps the step an optimisation rather than a precondition. `Vectors` is one table with no path of its own: `ensure` appends to the
cache beside the directory a crop was found in, so the merged view is writable
without collapsing every set's vectors into one file. a label goes to every set
holding its crop (`Session::save_labels`), since a crop labelled in one of two
sets that share it is a disagreement waiting for the second label. only the
touched rows are written over what a file holds. the walk is idempotent: a crop
is embedded once, named once, and a repeated answer rewrites a file to the same
bytes. a set cut into the tree while the page is open is found on the next pass
and brings its own cache with it.

**a set is only reproducible if it owns its crops.** the harvest deletes oldest
first under a disk budget, so labels outlive what they describe -- measured, 26%
of one labelled set already named crops that were gone. so the harvest never
deletes a crop named in the `labels.txt` beside it (`[harvest] keep_labelled`, on
by default): keeping the first copy is simpler than making a second, and the
labels are what hours of attention produced. the file is re-read whenever it
changes, because labelling runs in another process. labelled crops still count
towards the budget, so r6.4 holds: the unlabelled rest rotates faster around
them, and if labels alone ever exceed the budget every other crop is deleted as
soon as it is written, the harvester says so once, and it stops walking the
directory looking for something it may delete. `--gather` remains for a set
that lives anywhere else.

**a passage on video is cut into a set whole.** the harvest keeps a few crops a
passage on purpose -- it rate-limits by place and drops what did not move -- so
an event clip of a go-4 leaves a dozen crops where there were hundreds of looks.
`harvest::dense` crops every detection in a window a person chose after watching
the clip, through the harvest's own writer and context margins. three choices in
it are worth recording:

- **named by the frame's time.** the recorder names a clip for the moment it was
  kept and opens it `preroll_secs` earlier, so a frame's time is that opening
  plus its offset. a crop named by the clock of the extraction would squeeze an
  event into a few milliseconds, and passages are grouped by time. crops of one
  frame are a millisecond apart, so a re-run writes the same names.
- **at `[stream] crop_fps`, never the container's rate.** a recorded clip claims
  100 fps for a camera sending 15; read literally, every real frame was cropped
  several times over at several times the cost.
- **parked vehicles kept.** a go-4 stopped at the kerb writing a ticket is a
  positive, so nothing is dropped for not moving, and a person labels the
  passage. measured on a real event, 18 seconds became 1070 crops, almost all
  of them the same four parked cars in every frame.

it runs the detector over the whole frame only, not the magnified follow-ups the
live pipeline adds, so a vehicle small enough to need magnifying is missed here.
the passages worth cutting are the near ones.

**so the page that checks the claim reads the same names.** the labelling
server's verdict pool is a directory listing filtered to this subject rather
than a measurement. `eval::measure` also reports what would fire, but only
against a reference set, which needs labelled positives to exist -- exactly what
a harvest of false positives does not have. the crops most worth deciding were
therefore the ones the page could not offer until the work was already done.
taken off the filenames they are on screen the moment the session opens, and
rejecting one writes `other` with `Via::Alert`: the tier `split_negatives` draws
ahead of every other, and the reason it is worth the priority is measured --
five crops the classifier called a go-4 were labelled `other`, none were drawn,
and all five still scored as the subject to four decimals.

two things it deliberately does not do. it does not take a rejected crop out of
whatever pool it was already in, because the random pool is a fixed prefix of a
hash order and a false positive rate measured after quietly removing the crops
that fired is the rate at which everything *else* fires. and it shows the
verdict only in that view, since priming the random pool with the classifier's
own opinion is the contamination the pool exists to avoid.

### asking the directory again

the harvest listing is cached for thirty seconds per process, because the crops tab
walks it every few seconds and a cold walk of a few hundred thousand crops measured 2.9
seconds. anything metermate itself writes bumps a generation and invalidates the cache;
a crop written by another process waits out the ttl. for a page redrawing itself that
is the right trade, and for the `more` button it is not -- "is there more?" is a
question about the directory, and a half-minute-old answer says no while the pipeline is
harvesting. `--label` usually runs as a second process beside that pipeline, so every
crop the button exists to find is one that arrived behind its back. `harvest::forget` is
the one call that says so, and a refresh makes it before walking.

what a refresh costs is the other half, and it is only paid when there is nothing
else to offer: while any embedded crop is undecided (`undecided`) a refresh re-reads
the labels and rebuilds the pool from what is in memory, without walking or embedding.
embedding an evening's arrivals on every press was minutes of waiting for crops that
sat behind the ones already on hand. when it does walk, `cache.ensure` embeds only what the cache has
never seen and the cache file lives beside the harvest, so the button can be pressed
twice without paying twice -- and the first press after an evening of harvesting is not
instant, which is why the page reads `reading...` while it works rather than looking
dead and getting pressed again.

the same call can be made on a clock, `--label-ingest <seconds>`, so an evening of
harvesting is not waiting on a click to be worth looking at. it is off by default: a
pass holds the state lock while it embeds, so a click made during one waits on crops
nobody asked for, and `--embed` now does that work ahead of the session. what the clock
does not do is rebuild the pool, and that is the whole split: which
pool a crop was offered from is the provenance of its label and decides what may be
measured on it, so a pool rebuilt underneath an open page would misfile the next click.
the button rebuilds, because pressing it is somebody saying the pool on screen is spent.
a pass over a harvest nobody added to costs a directory walk, says nothing at all, and
can be switched off entirely.

### the pool is the navigation

`Via` is the provenance of a crop and it decides what may be measured on it: a
rate is honest on `Random` and meaningless on `Ranked`, which is ordered by
resemblance to the answers. the page originally stacked all five pools down one
scroll under coloured headings, which made that distinction a boundary a person
scrolls past. it is one tab per `Via` instead -- random, nearby, most likely,
second look, verdicts -- so the pool is the thing you navigated to and the rule
down the side of the screen is about what is actually on it.

this is entirely a client-side regrouping. `/queue` already tags every item with
its pool and its truth, so the tabs and the **already decided** switch are both
filters over one payload, and the server keeps building one queue spanning every
pool. a crop offered in two pools at once is drawn in both tabs, which is the
same deliberate double-listing described above, and labelling it in one updates
the other because the decision is keyed by crop name rather than by tile.

the switch hides what a pool was drawn as, not what it is now. filtering on the
live truth would make a crop disappear the instant it was decided -- under the
pointer that just decided it, with no way back to a misclick -- so the page
snapshots each crop's decidedness when it draws the pool. the screen collapses
to the work remaining at the next draw, which is a tab switch, a train, or the
sweep that decided two hundred crops at once. the `review` pool is the case that
proves the rule: every crop in it is already decided by definition, so hiding
the decided ones empties it, and it says so rather than rendering blank.

the opposite case is `Random`, `Ranked` and `Seed`, which `build_queue` draws
from the unlabelled harvest alone -- so on a fresh page the switch changes
nothing at all on three of five tabs. the count beside it is what keeps that
from reading as a broken control: zero says the pool holds nothing decided.
worth stating because it is the one place the tab split exposes a server-side
rule the page cannot change. making the switch reach labels from *earlier*
sessions would mean offering already-decided crops in the random pool, and that
pool is a fixed prefix of a hash order whose whole value is being the one set a
false positive rate can be taken from.

### a long loop that says nothing is indistinguishable from a stuck one

embedding a harvest and cross-validating it are the two places this waits
minutes: 1392 crops took 1m15s measured, and the fold loop is a pass over every
negative for every labelled passage -- which is now every rejected verdict
rather than a sample of thirty, so it grew by the same change that fixed the
cargo bikes. announcing the total up front, which is all it used to do, says a
wait is coming and never how much of it is left; those are different questions,
one answered by waiting and the other by checking whether anything is still
running.

both report at deciles rather than on a timer, because the two loops differ by
orders of magnitude in what a step costs and one interval would be either
silence or a flood. the estimate is linear off the rate so far, which is sound
here for a reason particular to these loops: every step is the same jpeg decode,
or the same dot products, as the last one. the timing is left off entirely until
there is a second to report, so a loop that turns out to be fast says where it
got to without also predicting a wait that is already over.

### the live page labels, and says so in the provenance

labelling used to mean a second server on another port: see the thing go past,
then find its crop again in a pool and decide there. the crop is already on
screen on the page being watched, so the decision belongs there -- but *which*
crops are on that page is the whole question. the crops tab is newest-first,
so what it offers is whatever just happened, chosen by whoever was looking.

`Via::Preview` is what keeps that honest. the random pool is the only pool a
false positive rate may be measured on, and what makes it honest is that
nothing chooses which crops are in it; a label made off the live page is
chosen twice over, by the street and by the watcher. it is kept apart for the
same reason `Ranked` is, and the eval treats it the same way.

**two processes write `labels.txt`.** the pipeline serves the preview and a
labelling session is a second `metermate` against the same harvest, which is
the ordinary case rather than a corner. a session holds its labels in memory
and writes the whole file back, so a label appended by the live page while it
had the file open would vanish at its next save -- and `labels.txt` is hours of
attention with no second copy anywhere. so the live page appends a single line
under `O_APPEND` (`load` is last-line-wins, so appending *is* deciding) and a
session re-reads and merges before it writes. un-labelling is passed to that
merge explicitly, because "absent from what I hold" is also true of every label
the other writer just added.

the control is one `edit` per tile rather than a button per answer. a row of
buttons can only offer the answers somebody thought of in advance, and the
thing that actually happens on this street is seeing something the classifier
has no name for -- a street sweeper, a tow truck, a second enforcement livery.
so the tile shows its label and opens a picker: the labels in use, ranked by
frecency (uses over how long since the last one), filtered as you type, with
what you typed offered as a new label when it matches none of them. ten,
because that is what fits without scrolling and the answer is almost always
one of the last few used. the ranking is computed server-side from
`labels.txt`, so a session labelling in another process shows up here.

the picker is a box of its own rather than part of the tile. the card is what it
hangs from and not what clips it, because the list is taller than the tile; its
own height is bounded by the window rather than by a constant, so the tenth label
is never the one that scrolls; and on the last column it turns around to hang from
the card's other edge rather than run off the side of the screen. three rules that
only exist once boxes are measured, which is what `tests/e2e/test_picker.py` does.

a subject named on the page labels crops and nothing else. the config is what
the running pipeline reads, and having the server rewrite the file it is
running from is the same idea the roi tab already declined -- it emits toml to
paste instead. so a new subject trains as soon as it has crops, and fires once
a `[[subject]]` block exists and the pipeline has restarted.

### a selection is a note, not an extraction

marking a passage and cutting it are separated on purpose. cutting loads the
detector and reads every frame of the window, which is minutes of cpu on a
machine r3.1 budgets at 30% of a core for the whole pipeline -- and the page
offering the button is served by the process watching the street. a click that
started that work would spend the deployment to save somebody typing, so the
page writes `selections.txt` beside the events directory and `--prepare` does
the work later, when nothing is being missed.

the window is parsed at the click rather than at the cut: failing now, with
the clip on screen, beats failing an hour later in a batch of twelve. the
seconds come from the player's `currentTime`, which is also why there is no
frame-accurate scrub -- a fragmented mp4 has no index, `seekable` is `[0, 0]`,
and the clock that advances with playback is the clock there is. a second
either side costs a few crops of parked street, which one sweep removes.

`--prepare` is idempotent by directory, and the set is named for the clip
*and* the window: two passages of one clip are two examples, so naming a set
after the clip alone would either overwrite the first or merge both into one
burst that everything grouping by time counts as a single passage.

`--retrain` exists because the two-harvest invocation is the step that gets
forgotten. training over the sets alone drops the deployment's own rejected
verdicts, which are the negatives that stopped the last false positive; over
the harvest alone it drops the dense passages. the config already names the
subjects, so the command is the config plus both roots, and a subject that
cannot be trained yet -- the ordinary state of a new one -- is reported
without stopping the others.

### a name on screen is an argument somewhere else

both pages put names on cards and both make clicking a card mean something --
labelling the crop, playing the clip -- so there is no way to select what a card
names without also acting on it. hence a small button, with `stopPropagation`
ahead of the card's own handler. what it copies is what the next command wants:
the labelling page copies a crop's bare filename, which is how `labels.txt` and
`--gather` identify one, and the events tab copies a clip's path under the
directory `[record] dir` set, which is what `--dense` takes. the path is
reported exactly as configured rather than absolute: `--dense` resolves it
against the working directory metermate itself runs in, so `data/events/...` is
the argument a person will actually paste, and its absolute expansion is a path
nobody wrote.

**`navigator.clipboard` is undefined on every deployment this has.** it exists
only in a secure context, and both pages are served over plain http on the
camera's own lan address -- so the modern call is exactly the one that silently
does nothing where it is used. the roi tab's `copy toml` shipped using it and
had therefore never worked outside a developer's `localhost`, which is a secure
context by definition and hid it for as long as it existed. a hidden textarea
and `execCommand` work in both places. the outcome is reported on the button
because the clipboard is write-only to script: a failed copy and a successful
one are otherwise indistinguishable.

### a second look is a label the classifier would vote the other way

the tab is for catching mislabels, so it offers the labelled crops whose label
the classifier's own vote disputes, in both directions: crops labelled the
subject that vote most with the negatives, and crops labelled `other` that vote
most with the subject. `eval::disputed` scores a crop the way the classifier
votes -- the mean of its `VOTE_K` nearest in the opposite class, less the same in
its own -- through `classify::mean_of_top`, the one definition both share. above
zero, the classifier would call the crop the other way if its label were taken
back. both halves come from the labels and the vectors, which exist when the
session opens, so the pool is there before `train` has been pressed.

**its own passage is left out of its own class.** the frames of a passage are
near-copies, so leaving out only the crop itself lets its siblings vote for it
at close to 1.0, and every crop's own score becomes the same constant. the
ranking then collapses into raw similarity to the other class, which puts a
generic crop of the subject -- small, distant, near everything -- ahead of a
passage labelled wrong as a whole, and labels are made a passage at a time. a
class seen in only one passage has nothing outside it, so there only the crop
itself is left out. `tests/e2e/test_second_look.py` builds both cases.

the class a crop is voted against is what a person decided is not the subject,
as `split_negatives` counts it: `other` and every other subject. unlabelled
crops are in neither class, and `unclear` is in neither half, as everywhere.
only crops labelled `other` are offered back, though: one labelled another
subject belongs to that subject's session, and a click here would overwrite it
with this subject or with `other`.

**the top fifty of each direction are always offered, with no threshold.** the
version before this offered only crops sitting closer to "the street" -- the
mean of everything not the subject -- than to the rest of their class. it came
up empty on the deployment, and a crop's average resemblance to all traffic is
not a question anybody labelling needs answered. a ranking always has a top, so
on a clean set the tab holds correct labels, and the heading says a score is a
reason to look rather than a verdict. a flip made here moves `random_negatives`
like any other decision, which is why it should follow looking at the crop
rather than reading the score.

measured on the go-4 set, 586 labelled go-4 against 806 negatives, both
directions fill to fifty. the most disputed go-4 labels include a crop 43px
wide, and the most disputed `other` labels include verdicts stage two fired on
and a person rejected -- which is right: those are the crops that really do
look like the subject, and exactly the ones worth confirming.

two smaller consequences of always drawing the tab you are on. an empty pool now
renders a section where before it rendered nothing, so the sweep button has to
be withheld explicitly -- otherwise a pool holding no crops offers to mark them,
with a label written from a count that was never filled in. and the training
report moved from a block inserted above the grid to a modal over it: the report
is a read rather than a step in the loop, and pushing several hundred tiles down
the page to show it loses the place it was measured from. the labelling keys are
held while it is open, because `n` means "not one of these" to the grid and is
an ordinary keystroke to somebody reading a page about negatives.

the page is javascript and the rules above are only true if the page says so, so
`tests/e2e/test_learn.py` runs the real page's own script against the real
binary under node, and asks it which crops belong on which tab. the selection
rule is written as a pure function of the queue for exactly that reason. node is
a test-time toolchain entry in `mise.toml`; the binary contains no javascript
engine.

### green is what alerted

a verdict is one frame; an alert needs the same subject on `confirm_m` of the
last `confirm_n` looks. the live page coloured every verdict green, so the colour
said nothing about what reached mqtt. it now means the vehicle alerted.

that fact cannot ride in the crop's name, which is written on the look the crop
is taken, while confirmation arrives looks later -- and a crop gated on it would
mark a passage's first crops differently from its last for a reason that has
nothing to do with their pixels. so `harvest::Alerting` holds each track's
verdict crops until the track confirms that subject, then names them all at
once in `alerted.txt` beside the harvest, and names any later crop of that track
as it is saved. tracks that leave without confirming are forgotten. the preview
reads the file per listing, like `labels.txt`, and marks a crop only when it
carries a verdict and is named there. it is written in dry runs too, since
confirmation does not depend on a broker.

### a rejected verdict is no longer a verdict

the live page's verdicts tab and the labelling page's read the same filenames
and must not show the same thing. the live tab answers "what is stage two firing
on", so a crop already judged and labelled `other` has been answered and drops
off it. the labelling tab answers "what should I judge next", and it is where
that rejection gets made -- filtering it there would make a crop vanish under
the hand that just decided it, with no way back to a misclick. so `Want::Named`
carries the names that stand rather than the listing deciding for everyone: same
function, opposite needs, and the difference is stated at the call site.

**what stands is a fact about labels, so `harvest` does not decide it.** that
module knows filenames; the preview reads `labels.txt` beside the harvest and
passes names (`preview::standing`). where a person has spoken, the person wins:

- an unjudged verdict stands, since the claim is all there is.
- a crop labelled a configured subject stands whatever its name says, captioned
  as by hand. a subject stage two missed is the sighting most worth seeing.
- a verdict labelled anything else leaves. `other` or another name is a
  rejection. `unclear` is not one, but it is an answer, and it used to keep the
  crops nobody could settle on the tab for good; it is counted apart.

the counts travel with the listing, because without them an empty
tab has two opposite meanings -- nothing fired, or everything that fired was
wrong and has been thrown out. the second is the case that actually happens: on
the first evening stage two ran, every crop it named was a false positive, and a
tab reading "nothing recognised yet" would have hidden exactly that result.

### and an event clip is named after its worth is known

the same lesson, inverted. a clip's name carries what was found in it, and that
is not decided when the file is created -- the whole argument for the ring
buffer is that the detector and classifier have not run when the trigger fires.
so unlike a crop, a clip cannot be written to its final name and cannot be
renamed into place at the last moment either: it is appended to for up to a
minute first.

it is written under `<stamp>-<stream>.mp4.part`, a name the clip parser refuses,
and renamed when it closes. three things fall out of that one decision:

- the events tab never shows a growing file labelled `motion` that is about to
  become `subject`. a listing is a list of finished clips.
- the disk budget counts bytes that have landed rather than bytes still
  arriving, and cannot evict the clip being written.
- `/clips/<name>` can answer with a year of `immutable`, exactly as crops do,
  because a name that appears in a listing will never describe different bytes.

what that leaves is the clip open when the process stops. a stream that ends
normally closes it -- found by the events tab, which showed nothing after a
replay that stopped mid-event -- and a `.part` from a process that was killed is
adopted at the next start as `motion`, the floor. taking the floor is the
honest choice twice over: it is the one thing known to have happened, and it is
also what the budget evicts first, so being wrong about it costs the least.

### two outputs, one process: what the remux may never do

the remux that feeds the preview and the recorder is a second output of the
ffmpeg already decoding main for crops. that is the right shape -- a second rtsp
session costs a second 4 mbit/s off the camera and measurably starved the gate --
but it buys cheapness with a coupling that has to be respected rather than
forgotten: **one process writes both outputs, so whatever blocks the remux stalls
the crop feed with it.** the crop feed is the critical path (r6.1), and the stall
presents as no frames arriving, which is indistinguishable from a dead camera. the
supervisor restarts ffmpeg, and the replacement meets whatever stopped the first.

so the relay's single obligation is to drain the socket, always, whatever anyone
downstream is doing. three things follow from that, and each is load-bearing:

- **no subscriber may push back.** every subscription is a bounded channel and
  anyone who stops draining is dropped, recorder included. a recorder that could
  block would be a recorder that can stop the detector.
- **the recorder's disk write is not on this path.** it happens in the frame
  loop, so a slow disk costs detection latency rather than the socket. that is the
  lesser evil and it is deliberate, but it is a real cost and it belongs in any
  accounting of what the loop is doing per frame.
- **a connection is read on its own thread.** reading them one at a time inside
  the accept loop is correct only while each predecessor has really gone. when one
  has not, the replacement waits unread in the backlog, its socket buffer fills,
  its *other* output stalls with it, metermate sees no frames, the supervisor
  restarts it, and the replacement queues behind the same stuck reader. that is a
  main-stream feed restarting every fifteen to forty-five seconds forever while
  the gate, which has no second output, runs perfectly -- and it is the same
  signature as a camera fault, which is what it was first read as.

**recording the substream puts the gate under this same rule.** it was the
control that made the diagnosis above possible -- the one main-stream-shaped
process with nothing bolted to it, running perfectly while main restarted in a
loop -- and `[record] streams` takes that away by giving it a remux output of its
own. the mitigation is that it is the same `Relay`, with the same bounded
channels and the same obligation to drain, rather than a second mechanism that
would have to learn all of this again. what is genuinely gone is the control:
a future stall of this shape now has two candidates instead of one, and
`streams = ["main"]` is the way to put the gate back to a single output while
that is being read.

two connected ffmpegs is a transient rather than a feature, so the newest wins:
an older reader stops publishing the moment a newer arrives, rather than
interleaving two timelines into one fragment stream.

"newest" is decided **in the accept loop, not in the reader thread**. numbered
inside the reader, two connections are ordered by whichever thread the scheduler
ran first, so an orphan can outrank the replacement that arrived after it -- and
the replacement then stands down in favour of the connection it exists to
supersede. it presents as a stream that is read only sometimes: measured by
running the relay tests a hundred times, 14 failures with the number taken in the
reader and none with it taken at accept.

### one vehicle is worth one vehicle, wherever it is

`min_changed_frac` is a fraction of the frame, so the gate asks for the same
absolute number of changed pixels everywhere in it. the same vehicle does not
supply the same number: measured over 532 harvested boxes, apparent height runs
50-60px bottom-left against 153px mid-right, about nine times the area. one
threshold therefore either loses a distant vehicle or fires on noise near the
camera, and no single value avoids both.

`[gate] perspective` is a list of lines drawn along vehicles in the roi tab.
each one's midpoint says where and its length says how tall a vehicle is there.
from those the gate fits a height `h(x, y)` and weights every changed pixel by
`1 / h^2` -- area, because the pixels are what is being counted.

**the fit is a plane in both axes, not a horizon keyed on image row.** against
the same 532 boxes, apparent height explains R^2 0.36 on row alone, 0.50 on
column, and 0.76 on a plane in both: the depth axis of this street runs
diagonally across the frame, so a row-keyed model would be confidently wrong in
the corners.

**those 532 boxes are a biased sample and the numbers above are provisional.**
they come from the harvest, which only holds what the gate let through, and the
roi in force when they were taken was the compressed one -- the upper 71% of the
frame, with the near lane outside it entirely. a restricted sample cannot invent
a diagonal axis, so the shape of the finding holds; the specific heights
describe that band rather than the whole street. the arithmetic does not depend
on them either way, since it fits the lines that are drawn. one line gives a constant, two fix a gradient along the axis
joining them and nothing across it, three or more are least-squares. collinear
lines leave the plane undetermined and fall back rather than solving an
arbitrary system.

**the fit is held to the range the lines demonstrate.** a plane says something
everywhere, including where nobody drew, and a pixel weighted by `1 / h^2` turns
a small extrapolated height into enormous sensitivity. measured against the
first six lines drawn on the deployment -- which agree with each other to an rms
residual of 8.4px on a mean height of 70.9px -- the plane still runs to **-77px**
in the top-left, where no line was drawn. floored at a quarter of the shortest
line, that corner came out 2900 times more sensitive than the opposite one, none
of it from evidence. clamped to the drawn range instead, the widest ratio across
the frame is exactly the area ratio the lines themselves show. outside their
support the nearest measured value is the honest answer, and drawing another
line is how to say otherwise.

**the weights are normalised so their mean over watched pixels is one**, which
leaves `min_changed_frac` meaning what it did. the correction redistributes
sensitivity with depth; it does not raise it everywhere. unnormalised, drawing
two lines would quietly retune every threshold in the config at once.

what it buys is narrower than it first appears. tested against five real events,
a scale-aware threshold at matched sensitivity removed 3.7% of firing frames on
a clip holding no vehicle -- it does **not** fix over-triggering, which comes
from regions that are already vehicle-sized. it is for the opposite case:
holding on a small distant subject. empty means no correction, and the score is
the plain changed fraction every deployment runs today.

### a clip may only begin on a keyframe

the ring is bounded by time, so before this it began wherever the window
happened to fall -- which on an inter-coded stream is usually mid-gop. those
leading fragments are coded against a picture the ring no longer holds, so
nothing can decode them, and the container goes on declaring `start_time=0`
regardless. measured on the deployment: a main-stream keyframe interval of
2.00s, and an 8.73s clip whose first decodable frame was at 1.33s. a player
holds one still through it, so a third of a 3s pre-roll was not video.

the substream never showed it. mjpeg is all-intra, every fragment is a sync
sample, and its half of the same event decoded from 0.00s -- which is also why
this surfaced as "the pre-roll is frozen" only after the events tab began
preferring the main half.

so eviction drops **whole gops**: the ring runs from a keyframe and holds
between `preroll` and `preroll` plus a gop. the alternative -- keeping the hard
time ceiling and trimming forward to the next keyframe -- bounds the ring more
tightly but can leave as little as `preroll` minus a gop of actual pre-roll,
which at a 2s gop and a 3s pre-roll is one second. an extra second or two of
buffered fragments is worth less than the guarantee.

**`tfhd` is written before `trun`, and its default says non-sync on every
fragment.** dumping 900 fragments off the deployment: all of them carry both
boxes, every `tfhd` default_sample_flags sets `sample_is_non_sync_sample`, and
`trun` carries `first_sample_flags` only on the keyframes. a reader that takes
whichever box answers first therefore calls every frame in the stream a
non-keyframe. that is not a hypothetical: it emptied the ring on every push and
removed the pre-roll altogether, and the first fixture for it carried only
`trun`, so the tests passed while the deployment got worse. `trun` describes
this fragment's own first sample and must win; `tfhd` is only the default for
samples that say nothing.

### nothing spans an ffmpeg restart

the ingest supervisor restarts ffmpeg when the camera stalls, which on a wifi
camera is routine. the remux then begins a fresh timeline at zero, and a clip
open across that boundary is rebased against an origin an hour ahead of it: every
fragment goes negative, `saturating_sub` floors it, and the file claims all of
its frames happen at once. one clip on the deployed box held 103 fragments that
way -- 1.4 MB of real video that decoded to a single frame, in a file that
listed and played like any other.

so a decode time that goes *backwards* is treated as a new stream rather than as
a bad timestamp. with a fragment per frame the clock is otherwise monotonic, so
the signal is unambiguous. the open clip is finished with what it has, the ring
is dropped because it describes a stream that no longer exists, and the header is
dropped so the next one is asked for -- a restarted ffmpeg may have renegotiated
the stream, and fragments behind a stale `moov` are undecodable whatever their
timestamps say.

### closing two of three feeds saves nothing

r8.3 says the preview costs nothing when nobody is watching, and the page is
careful about it: leaving the live tab tears the mp4 stream down rather than
pausing it, because a paused element is a viewer the server drops anyway.

that care covered one of the three feeds. the mjpeg `<img>` had no equivalent --
`display: none` on its container does not close a connection, so the browser
went on pulling frames -- and the `/detections` socket was never closed at all.
that last one decides the whole question, because the server counts it as a
viewer *deliberately*: a browser that opened it before the video would otherwise
wait forever for frames gated on the viewer count. so a single open socket held
`watched()` true, and on the configurations where metermate supplies the pixels
that means encoding a 640x360 jpeg per gate frame, on the thread the detector
runs on, for a picture behind a hidden div.

the lesson generalises past this page: when several things each keep a resource
alive, closing all but one of them is worth nothing, and reads in review exactly
like closing all of them. the three feeds are now opened and closed together.

measured alongside it, for the tab that prompted the question: `/clips` against a
20,000-crop harvest costs 44 ms and is polled every six seconds, where `/crops`
costs 41 ms and is polled every four. the directory listings are not what makes
the preview expensive.

### a clip's first frame is its worst thumbnail

the pre-roll exists so that a clip opens before anything happened. that makes its
first frame the emptiest one it contains, and a grid of first frames a grid of
empty streets: the events tab shipped looking like it had captured nothing while
holding twenty-three real passages, and it was reported as exactly that.

the fix reuses the harvest rather than generating anything. a clip's stamp and
the crops from the passage that triggered it agree to the second, so the listing
carries the clearest crop taken inside the clip's window and the card shows that.
it is a picture of the vehicle rather than of the street, a few kilobytes rather
than the megabyte of video a poster frame costs to decode, and already cached
from the crops tab. a clip with nothing harvested during it has no crop to
borrow and falls back to a frame seeked past the pre-roll.

## keeping office hours

enforcement is a daytime phenomenon, so the harvest and the recorder can be
told to sleep. `active_hours` takes windows of local time and, by default, is
empty -- always. three decisions behind where the check lives and what it
covers:

**it sits on the two decisions that start collecting**, in the frame loop,
not inside `harvest` or `record`. the gate goes before the recorder's
`trigger` and before the harvest branch that calls `Harvester::wants`, so a
clip already open finishes on its own terms and -- crucially -- the evidence a
notification carries is untouched. `keep_evidence` runs on the alert path, and
r4.5 does not keep office hours: a subject confirmed at 02:00 outside the
window still writes its crop and still publishes (r11.2).

**the decision is minutes since local midnight, kept pure.** `Hours::covers_at`
takes two integers and returns a bool, and every case -- inclusive start,
exclusive end, a window that wraps midnight, a zero-length window read as the
whole day, a window on a day that is not today -- is a unit test. the only side
effect is reading the clock, `libc::localtime_r`, which is the one call that
knows the deployment's own timezone and its daylight saving; `std` hands out
epoch seconds and no local time. the common case -- no window -- returns before
touching the clock.

### the week is laid out at load, not tested per window

`"07:00-19:00"` was always enough for a street that sleeps, and never quite
enough for the ones that do not: enforcement on a school street is a morning and
an evening, and enforcement at a weekend market is six days. days go in front of
the times and extra windows go after them:

```text
"mon-fri 07:00-19:00"                  weekdays only
"mon-fri 07:00-09:00,16:00-19:00"      a morning and an evening
"mon-fri 07:00-19:00; sat 09:00-17:00" a week's worth of them
```

**a window that runs past midnight is written onto both of the days it is open
on, at parse time.** "22:00-06:00" every day becomes an evening on each day and
a morning on the next, which is what makes covering a minute cost nothing and,
more importantly, what makes "mon 22:00-06:00" mean tuesday at three in the
morning and *not* wednesday at three. testing the wrap against the window
instead would have meant carrying the window's own day through every question,
and the one question here is asked about a minute that has already left the day
it started in.

**days come before the times because a colon decides.** a token with a `:` in it
is a window and one without is a day, so the day list is the part that can be
told apart, and after the first window a comma is only ever between two windows.
that is the whole grammar: no new punctuation beyond `;` between groups of the
week, no modes, and an ordinary `"07:00-19:00"` still means and still prints
exactly what it did before days existed.

**a wait that runs past midnight names the day, because the day is the half that
matters.** a weekday window shut on friday evening does not reopen at 07:00; it
reopens at monday 07:00, and `/stats` and the live page say so. that is why the
reopening is carried as a string rather than as a minute of the day: a minute of
the day cannot hold a weekday, and the alternative was every reader working the
day out from a clock of its own.

**a window that is not a window is refused at load**, like `keep` and
`streams`. a typo silently keeps nothing, and an empty harvest directory is
indistinguishable from a quiet street. the value is parsed in `Deserialize`,
so the failure names the bad field and never starts a process.

### the window that closes the camera

`[stream] active_hours` is a fourth window and a different kind of thing. the
others decide what to keep from a stream that is already being pulled; this one
decides whether to pull it. at night that is the difference between an idle
machine and two ffmpegs decoding a street at r3.1's expense for twelve hours,
for a street that is not being watched.

Three things follow from where the switch sits:

**the feed holds itself shut rather than being torn down.** the window closes a
`ingest::Channels` handle both feeds share; a feed that is not wanted kills its
ffmpeg, waits, and spawns a new one when it is wanted again. the alternative --
stopping the ingest and rebuilding it -- would have to rebuild everything built
*at* it: the preview's video, the recorder's rings, the relay the remux writes
into. it would also mean a feed that ends, and an ingest that ends is a finished
source, which for a file replay is the end of the run.

**closing is reported as closing, everywhere it could be mistaken.** an ingest
that stops on purpose exits its read loop the same way a stalled stream does, so
the supervisor is told which it is: a closed window is neither warned about nor
counted as a restart, and `/stats` carries `stream_shut` so the ages that do
start growing have an explanation. the frame loop polls a second rather than
blocking in `recv`, because a closed window is quiet by definition and a loop
parked on a channel notices neither edge of it.

**what a closed channel leaves behind is finished, not left.** a clip open when
the channels close is closed, because it writes under the name the listing
refuses until it closes -- one nothing lists, evicts or plays. the newest
main-stream frame is discarded rather than kept, or the first vehicle through a
window that reopened at seven would be cropped from last evening's picture and
presented as current. and the background model is reset, the same way a
repointed camera resets it: the street at 07:00 is not the street of the
previous evening, and a model carried across the gap fires on the light.

the process itself keeps running, and that is deliberate. r8.5 and r8.6 are the
argument: crops and clips outlive the setting that wrote them, so a deployment
that sleeps from seven until seven is a deployment whose harvest and clips are
still browsable at midnight. what stops is the asking, not the answering.

**and the live page stops showing a street it has stopped watching.** a closed
window leaves the browser holding the last frame it was ever sent, and a still
street is indistinguishable from a quiet one -- which is the one wrong claim a
preview can make, since a quiet street is a fact about the world and that frame
is a fact about ten hours ago. so `/stats` carries the minute the channels reopen
and the live tab covers the stage with `stream closed, until 07:00`. it covers
the picture rather than taking the media elements down, which is the difference
between a notice and a second teardown path: the page polls, the frame reappears
by itself when the window opens, and nothing has to be reopened in the right
order. `[hidden]` is what hides it again, and a poll that cannot reach the server
says nothing rather than inventing a closure.

**the models can go with the window, and only with the window.**
`[stream] unload_models` drops the detector and the classifier while a
deployment's window is shut, which for a box with a 351 MB embedder is the same
r3.1 argument the channels are, measured in megabytes instead of bitrates. what
makes it safe is that it can only happen while the channels are down: a shut
stream delivers no frames, so there is no look in progress, no crop mid-cut and
no verdict half-written when the weights go, and nothing downstream has to be
told to be careful or to check whether anything is loaded. the ordering is the
feature -- close the channels, then let go; build the models, then reopen -- and
the inverse, an unload that could land between a frame and the look at it, is a
pipeline that is silently blind rather than one that is properly asleep. so the
loop looks the detector up with an `expect` rather than a branch: `None` while a
frame is in hand is a broken invariant, and a crash that says so is worth more
than a run that watches a street it cannot read.

it is opt-in and off by default, for the ordinary reason: the memory is doing
something useful in every deployment that has not asked otherwise, and a model
load at 07:00 is a quarter second paid by nobody.

`/stats` reports `models_released_mb`: what the last hand-back gave, measured on
either side of the drop by the code making it. The first version of this feature
was tested by watching the process's resident set from a test across the window
edge, and it failed for a reason worth keeping -- a process that has been
harvesting all evening grows by more than a detector is worth, so the models'
17 MB were real and invisible at the same time. The size of a drop and the size
of a heap are different questions, and only the code standing at the drop can
answer the first one. The megabytes are reported only when they can be measured,
so a platform without `/proc` says "released" and no number rather than "0 MB
back".

## the page is a url per thing, not a url per page

the preview had one address and five views, which made every share a
description: "the crops tab, third row, the one with the jeep". the hash now
carries the navigation -- `#/verdicts` for a view, `#/verdict/<crop>`,
`#/crop/<crop>` and `#/event/<clip>` for one thing in the overlay -- and the tab
buttons assign it rather than doing the work themselves, with `hashchange`
rendering whatever changed.

routing through the url rather than beside it costs one handler and buys three
things that were not otherwise available: the back button is the browser's own
history rather than a second stack to keep in step, a crop can be named in a
message to whoever is asking about that evening, and **a notification can arrive
at the evidence it attached** (see "the doorbell is not the integration
surface"). that last one is the feature; the other two are why the shape is worth
it even with the phone switched off.

a link naming one crop resolves through `/crop?n=`, which answers with the same
listing entry the grid draws its tiles from. the alternative was parsing the
filename in javascript, and the name is the harvester's grammar -- a second copy
in a second language is how the two drift. the endpoint reads the name rather
than walking the directory, and refuses a crop whose file is gone, which lets a
stale link land on the tab it belongs to instead of on a broken image; the
million-crop harvest a stale link is aimed at is exactly the one where paging to
the tile would be the expensive answer.

routing is also why the first hash is rendered after `/config` answers rather
than as the page parses. that call says where the video comes from, and the video
is opened the moment it is known -- so a link that arrives on the crops tab has to
be rendered after it, or the page opens the live stream for a picture nobody is
looking at and leaves it running. closing all of the feeds but one is worth
nothing (above), and this is the same rule at the front door.

opening a tile moves the url too, and by the other route: `history.replaceState`
rather than assigning the hash, because a tile click already knows what it is
looking at and routing through `hashchange` would fetch that crop's own entry
again in order to draw it a second time. assign for arriving, replace for
arriving-at-something-the-page-is-already-showing.

the layout followed the same argument a phone makes: at 13px inside a 16px button
the page is legible and unusable, and it had no viewport tag at all, so a phone
rendered it as a 980px desktop and asked for a pinch. the touch rules key on
`pointer: coarse` rather than on a width, because a tablet in landscape has a
desktop's pixels and a phone held sideways does not, and the desktop is untouched
-- the same page at the same density for whoever has a mouse.

### the picture takes the window it is left with

the live view used to size the stage at `min(100vw, 95vh)` of width, which on a
1440x900 window draws the street 719px wide with four hundred pixels of empty
screen under it and a legend below the fold. the stage now gets what the rest of
the page does not spend, and the shape it keeps is still the encoded frame's: a
browser contains an image whose shape differs from its box while the overlay
canvas covers the box, so a box of the wrong shape puts every vehicle under its
box rather than in it -- the same reason the stream is encoded to the street's
shape instead of to the `[preview]` box.

the arithmetic is left to css, since "as wide as there is, of a shape nobody knows
until a stream arrives" is a css question, and the two things only the page knows
are written into it. `--frame-w` and `--frame-h` are `/config`'s encode size, kept
as the pair the encoder sent rather than as a divided float. `--room` is what the
page spends around the picture -- the header, the roi editor, the legend -- and is
**measured** rather than assumed, because all three rewrap: the header becomes two
rows and the legend six lines on a phone, and the editor's toml gains a line with
every region. one `ResizeObserver` on the live view catches every way that changes
size inside it; a resize listener covers the one way that changes nothing inside
it, which is the window itself. a hidden live view reports no heights at all and
says nothing, because subtracting zero from the window hands the picture the whole
screen and pushes the legend off it.

measured in headless chrome against the stylesheet as served, at a 1440x757 window
and a 16:9 street: 1125x633 where the old rule gave 719x404, header and legend both
on screen with nothing to scroll, and the canvas coincident with the picture to the
pixel. the same arithmetic holds at 2560x1440, at a phone's 500x757 (width-bound at
468), with the roi editor open (the room grows by 110px and the frame gives it), and
in raw-detector mode where the frame is square.

rejected: a full-height flex column with the frame fitted by container query
(`100cqh`), which is fewer lines and the same numbers. query units in the block axis
resolve to nothing for an element whose height was never definite, and this page's
height is content-driven because the crops grid is longer than any window -- so the
flex column means every grid scrolling inside its own pane rather than inside the
document, and a sentinel measured inside a scrolling pane reported no intersection
at all, which is the crops grid quietly stopping halfway. what is kept instead is a
floor, expressed as a `min-width` so the shape survives it: a phone held sideways or
a toml on a short laptop scrolls rather than showing nobody a stream.

**the document is the scroller, so the bar that switches views is pinned to it.**
the same content-driven height that rules out a flex column means the header scrolls
off with the top of a long grid, and the tabs are the only way between views: sticky,
with a `z-index` above the label picker, since the tabs are reachable from every tab
and the picker is not, so only one of them may hang over the cards.

**and the scroller is what the grids read before they move.** both prepend, so a card
that arrives on a poll moves everything below it -- a crop four seconds into reading
the harvest used to shift the row under the pointer, and an afternoon of recording
moved the clip halfway down the screen every six seconds. the poll keeps running
because the bar above each grid reports what the harvest costs against its budget,
which must not go stale while somebody reads it; what arrives while the page is away
from the top is counted out of the listing and never merged, and releasing it is one
more poll rather than a remembered array, so a card the budget evicted in the
meantime is never drawn. a grid with nothing on it yet is not held, because the
failure that guard prevents is a tab opened halfway down showing a button beside an
empty grid.

### looking closer stops at the pixels

the overlay used to size a crop at its own pixels on the way in, which was right when
the crop was the thing being looked at and wrong in the way that mattered: a crop is a
photograph of a plate, and 200px of it on a 1440px screen answers a question nobody
asked. opening one now gives it the window at its own shape, and the wheel decides how
much of that window one part of it gets.

both ends of the wheel are the crop rather than the screen. **out stops at one screen
pixel per pixel of crop**, because below that the screen is inventing pixels rather
than showing the ones stage two saw, and the blur that made a verdict wrong is a
property of those pixels rather than of how big they are drawn; the number comes from
`naturalWidth` against the window, and until the bytes say otherwise there is no truth
to stop at, so the window is the floor. in stops at twelve times the window, where a
crop is out of pixels rather than out of luck. an unanchored zoom would strand whatever
was being read outside the screen, so the point under the cursor is the point that does
not move -- `t' = s - (s - t) * k'/k`, which is why the box the transform lives on is
the window itself with `transform-origin: 0 0`: the arithmetic then needs no layout at
all, and a test can check it against a cursor at 1000,400.

a clip fills the window too but keeps a box of its own rather than the window's, so its
controls sit under the picture instead of at the far edge of a letterbox and the dark
around it stays the backdrop that closes the overlay. it is not zoomed: it is the whole
street at the resolution it was encoded at, so filling the window is as close as close
goes, and its player has its own use for a pointer over it.

measured in headless chrome against the page as served, opening a 240x200 crop in a
1440x900 window paints it 1050x900 -- the height, and the crop's own 6:5.
