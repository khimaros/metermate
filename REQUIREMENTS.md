# requirements

product requirements for metermate. these are commitments. a release must not
regress any requirement that was previously met.

each requirement has a stable id so code, tests, and the roadmap can reference
it. status is one of: `planned`, `partial`, `met`.

## r1 -- detection

| id | requirement | status |
|---|---|---|
| r1.1 | detect sfmta go-4 interceptor parking enforcement vehicles in the camera view | planned |
| r1.2 | detect marked sfmta enforcement trucks and suvs | planned |
| r1.3 | detect a person adjacent to a stopped enforcement vehicle (dismount) | planned |
| r1.4 | distinguish enforcement vehicles from civilian vehicles with a false positive rate low enough that alerts stay trustworthy | planned |
| r1.5 | stretch: detect physical tire chalking behavior | planned |

note on r1.5: sfmta's go-4 fleet ships with lpr and digital chalking, so physical
chalking is largely obsolete. r1.5 is retained as a stretch goal but r1.3 is the
signal that a ticket is being written.

## r9 -- the protected vehicle

the point of the system is one specific parked vehicle: the van in the right
foreground of this camera's view. that it is the subject, and not just another
object in the scene, changes what the alerts should say.

| id | requirement | status |
|---|---|---|
| r9.1 | proximity to the protected vehicle is published as an attribute of a sighting; a sighting anywhere in frame is reported regardless (r10.2) | planned |
| r9.2 | when the protected vehicle is not parked in its space, the higher tiers are suppressed | planned |
| r9.3 | the protected vehicle's own detections never trigger alerts | planned |

r9.1 adds a dimension rather than replacing one. `sighting` stays useful on its
own: enforcement on the block is worth knowing about before it is adjacent.

r9.2 removes a class of useless alerts: enforcement passing while the vehicle is
out is not an event.

**r9.3 is about alerting, not detection.** an attempt to exclude the vehicle
from *detection* by a hard-coded rectangle was removed: it stopped working the
moment the camera was panned, and special-casing one object inside the detection
path is the wrong shape. keeping a static vehicle out of the harvest is a
general problem -- is the motion about this vehicle -- and is solved generically.

**presence is not decided by the detector.** measured over 900 frames, the
parked van's class flips between `car` and `truck` and its confidence averages
0.37, ranging 0.25 to 0.75. its *box* is stable to within about 25px on each
axis. so occupancy is judged from the background model and a reference image of
the space, which are stable, rather than from a class label that is not.

## r10 -- subjects

the go-4 is the motivating case, not the only one. a street sweeper is worth
knowing about for exactly the same reason a go-4 is, and r1.2 already asks for a
second vehicle type. what metermate recognises should therefore be a list, not a
compiled-in constant.

**metermate publishes outcomes; it does not decide what deserves attention.**
what reaches a phone is a home assistant question, downstream of the broker. so
the job here is to export each recognised thing as its own distinct outcome,
with enough attached to it that the downstream rule can be written.

| id | requirement | status |
|---|---|---|
| r10.1 | what metermate watches for is configuration, not code; adding a subject needs no recompile (r5.3) | planned |
| r10.2 | several subjects are recognised at once, each published as its own outcome | planned |
| r10.3 | a subject need not be a vehicle. nothing in the recognition or harvest path may assume it is | planned |
| r10.4 | adding a subject does not invalidate the harvest, the cached embeddings, or the labels already collected | planned |

r10.3 is the one that is cheap now and expensive later. the stage-1 detector is
currently filtered to coco vehicle classes and the harvest keeps what it finds,
so a subject that is a person, an animal, or an object on the step is excluded
before recognition ever sees it. the filter belongs in configuration, derived
from the subjects, rather than in the code.

r10.4 is what makes labelling worth doing before the subject list is settled. a
crop labelled as belonging to no known subject is only negative *for the
subjects known when it was labelled*; adding one later does not make those
labels wrong, but it does mean the new subject's examples may be sitting in
them. the contamination screen already exists to find exactly that.

this refines r4.1 rather than replacing it: distinct topics are still how
distinct outcomes are exported. what changes is that the topic carries which
subject as well as what happened, and that **severity is not metermate's to
assign**. proximity to the protected vehicle (r9.1) is published as a fact about
the sighting; whether it is urgent is decided downstream.

## r2 -- latency

| id | requirement | status |
|---|---|---|
| r2.1 | under 1.5s from an enforcement vehicle entering the roi to mqtt publish | planned |
| r2.2 | alert publication must never be blocked by evidence capture, recording, or crop harvesting | planned |

## r3 -- resource use

| id | requirement | status |
|---|---|---|
| r3.1 | idle cost (no motion in roi) stays under 30% of one cpu core | planned |
| r3.2 | no gpu, npu, or external accelerator required to meet r2 and r3 | planned |
| r3.5 | no model is trained from scratch; every model starts from pretrained weights | planned |
| r3.3 | steady-state memory under 1 gb | planned |
| r3.4 | resource use is verified by an automated test, not by inspection | planned |

r3.1 is expressed against one core rather than the whole machine so the number
stays meaningful if the deployment target changes.

## r4 -- alerting

| id | requirement | status |
|---|---|---|
| r4.1 | publish to mqtt, each outcome on its own topic, keyed by subject and by what happened (r10.2) | planned |
| r4.2 | emit home assistant mqtt discovery so entities appear without manual yaml | planned |
| r4.3 | attach an evidence snapshot to each alert | planned |
| r4.4 | survive broker restarts and network loss without operator action | planned |
| r4.5 | never notify about something the verdict page cannot show | planned |

r4.5 exists because a notification and its evidence are the same fact seen
twice: the crop the phone attaches is the file the verdict page lists. a track
stays confirmed for as long as it is in view, so a sighting used to notify on
every look of it, and a page entry only appeared where a look found the place
unphotographed. the first look to confirm now writes its crop whatever the
harvest's rate limit says, and it is the only look that notifies. a confirmation
whose crop cannot be written is logged and not sent.

## r5 -- operation

| id | requirement | status |
|---|---|---|
| r5.1 | run unattended for weeks; recover from camera reboot, stream stall, and wifi loss | planned |
| r5.2 | tolerate the camera being panned or tilted without emitting false alerts | planned |
| r5.3 | all tuning lives in one config file; no recompile to retune | planned |
| r5.4 | never mutate camera configuration without the operator opting in | planned |
| r5.5 | the same binary runs on a development laptop as on the deployment host, so the system can be exercised away from the target machine | planned |
| r5.6 | the streams' resolution and aspect ratio come from the streams; no camera model is compiled in | planned |

r5.6 is a camera swap. the main stream's size was a constant, so a camera
sending 4096x1856 was read as 2560x1440 chunks: half a frame of sheared pixels
per frame, no error anywhere, and a harvest that wrote nothing while every log
line said the pipeline was healthy.

## r6 -- data

| id | requirement | status |
|---|---|---|
| r6.1 | harvest vehicle crops continuously so a training set accumulates from day one | planned |
| r6.2 | be useful before any project-specific model is trained | planned |
| r6.3 | operator confirm/deny feedback flows back into the training set | planned |
| r6.4 | bound disk use for harvested crops and evidence; never fill the disk | planned |

## r11 -- schedule

| id | requirement | status |
|---|---|---|
| r11.1 | crop harvest and event recording can be restricted to configured windows of local time | met |
| r11.2 | restricting collection never restricts deciding: an alert outside the window still publishes with its evidence | met |
| r11.3 | the streams themselves can be restricted to a configured window; outside it nothing is asked of the camera, the live view says the stream is closed rather than showing the last frame it was sent, and what is already on disk stays browsable | met |
| r11.4 | a deployment can ask for the models to be released as well while its stream window is shut, and have them built again before the first frame after it opens | met |

enforcement is a daytime phenomenon and the disk budget is not infinite. a
window names when collection may run; it is configuration (r5.3), refused at
load when it is not a window, because an empty collection directory reads
exactly like a quiet street.

a window is as fine as the street needs it to be: clock time alone, days of the
week in front of it, or several windows in one day. that is the same requirement
and the same knob, not a new one -- the point of a window is that the street has
a shape, and a street with a school by it is not open all day.

r11.2 is what keeps r4.5 whole: the evidence a notification carries is written
by the alert path, which the windows do not touch.

r11.4 is the same argument one stage further in: the weights are hundreds of
megabytes held for a street that is not being watched. it is opt-in, and it can
only happen while the window is shut, which is what keeps it an unload rather
than a pipeline that cannot see.

r11.3 is the window in front of the pipeline rather than behind it, and it is
what r3.1 asks for: an idle machine should be idle, not decoding a street that
nobody is watching. It closes the channels, so the harvest and the recorder go
quiet with them, and it says so on the live view, because the alternative is a
page holding a frame hours old and looking exactly like a quiet street. It does
not close the process, because r8.5 and r8.6 say the crops and the clips are
there to be looked at, and nights are when anybody gets
round to looking.

## non-requirements

deliberately out of scope, recorded so they do not get rebuilt by accident.

- general purpose nvr. recording and playback belong to frigate or similar.
- reading license plates. the enforcement vehicle is the signal; plates are not
  needed and invite a privacy and legal surface we do not want.
- identifying individual people. the dismount trigger uses an anonymous person
  box associated with a vehicle, never identity.
- cloud or off-device inference. everything runs locally.
- multi-camera support in v1. the design should not preclude it.
- night operation in v1. sfmta enforcement is overwhelmingly daytime and the
  camera shoots through glass, so night is best-effort until r7 below is met.

## r8 -- observability

| id | requirement | status |
|---|---|---|
| r8.1 | a live view of the scene with detections drawn on it, in a browser, with nothing to install | planned |
| r8.2 | the view shows what the system is deciding, not just what the camera sees: roi, motion, boxes, classes, confidence | planned |
| r8.3 | the preview costs nothing when nobody is watching, and never delays an alert | planned |
| r8.4 | a raw detector mode showing every class the detector sees, ungated and unfiltered, for validating the model against this scene | planned |
| r8.5 | browse the harvested crops from the same view, so the training set can be judged without shell access | planned |
| r8.6 | watch the recorded event clips from the same view, so what was kept can be judged without shell access | planned |

r8.2 is the point of the feature. a preview that only shows video is a webcam;
the value is seeing why a frame did or did not fire, which is what makes the roi
and threshold tuning in phase 1 tractable.

r8.5 applies the same argument to the harvest. it is the critical path (r6.1) and
it was write-only: crops accumulated for days with nobody looking, and the two
times they were inspected they turned out to be garage doors and letterboxed
strips. it is also the groundwork for r6.3, which needs a place to click
confirm or deny.

r8.6 is that argument a third time, and the case is stronger: a clip is kept or
evicted on what the pipeline decided was in it, so a wrong decision costs the
recording of the event it was wrong about. that is only checkable by watching
the clips, and the clips live on whichever machine is watching the street.

## r7 -- deferred

| id | requirement | status |
|---|---|---|
| r7.1 | usable detection after dark through the window | planned |
| r7.2 | multi-camera support | planned |
