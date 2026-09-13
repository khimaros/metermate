"""helpers for driving the alerting path end to end.

**no test used to reach a confirmation.** the pipeline was exercised in pieces --
the detector against a street, the classifier against a trained set, ntfy against
a fake server -- and the seam where a confirmed subject becomes a notification was
only ever read. that is the seam where `r4.5` lives, so it is built here: a clip
with a vehicle in it, references trained from that clip, and a phone to receive
what comes out.

the clip is `fixtures/street.jpg` with one vehicle slid along the kerb, which is
what the detector is tested against elsewhere, and the references are that same
run's own crops. they are renamed into passages a gap apart because a reference is
one crop per passage: copied out of a single run they all belong to one vehicle
that was merely looked at twice.
"""

import http.server
import shutil
import socket
import subprocess
import threading
from pathlib import Path

from conftest import REPO
from test_resolution import street_clip

DET = REPO / "models" / "detector.onnx"
EMB = REPO / "models" / "embedder.onnx"
SUBJECT = "go4"

# an hour apart, so nothing groups into one passage. `label::PASSAGE_GAP_MS` is
# minutes, and the name is the only time a crop has.
GAP_MS = 3_600_000
# inside the gap, so crops written a few frames apart stay one passage.
FRAME_MS = 400
# the first name of the first passage. any 2020s millisecond will do; the crops
# this builds are never compared against the wall clock.
AT_MS = 1_789_000_000_000

TOKEN = "tk_secret"
TOPIC = "metermate-test"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeNtfy:
    """records what was posted, and answers however the test asks it to."""

    def __init__(self, status: int = 200):
        self.status = status
        self.seen: list[dict] = []
        recorder = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _take(self):
                length = int(self.headers.get("Content-Length", 0))
                recorder.seen.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": self.rfile.read(length),
                    }
                )
                self.send_response(recorder.status)
                self.end_headers()
                self.wfile.write(b'{"id":"fake"}')

            do_POST = _take
            do_PUT = _take

            def log_message(self, *args):
                pass

        self.port = free_port()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def base(url: str) -> str:
    """a config for a street nothing has been trained on yet.

    the gate is the whole frame: what matters here is what happens after a
    detection, and an roi would be one more thing between the test and it.
    """
    return f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"
url = "{url}"

[stream]
gate_subtype = 1

[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 1500

[detector]
model = "{DET}"

[track]
# one agreeing look is enough to confirm, so a run over a short clip is enough
# to produce an alert rather than a statistical expectation of one.
confirm_m = 1
confirm_n = 3
"""


def clips(root: Path, main=(1280, 720), sub=(640, 360), passes: int = 3):
    """one street at two sizes, with the vehicle crossing `passes` times."""
    root.mkdir(parents=True, exist_ok=True)
    for n, size in ((0, main), (1, sub)):
        clip = street_clip("ffmpeg", root / f"{n}.mp4", size[0], size[1], 8)
        again = root / f"{n}.x{passes}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-stream_loop",
                str(passes - 1),
                "-i",
                str(clip),
                "-c",
                "copy",
                str(again),
            ],
            check=True,
        )
        again.replace(clip)


def run(binary: Path, config: Path, *extra: str, timeout: int = 900):
    done = subprocess.run(
        [str(binary), "--config", str(config), "--preview", "off", *extra],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def harvested(binary: Path, root: Path) -> list[str]:
    """run the pipeline for its crops, which is how the references get made."""
    cfg = root / "harvest.toml"
    cfg.write_text(
        base(f"{root}/{{subtype}}.mp4") + f'\n[harvest]\nenabled = true\ndir = "{root / "raw"}"\n'
    )
    code, out = run(binary, cfg, "--dry-run")
    assert code == 0, out
    names = sorted(p.name for p in (root / "raw").glob("*.jpg"))
    assert names, f"the detector found nothing to harvest:\n{out}"
    return names


def set_of(root: Path, names: list[str], label: str, crops: Path):
    """copy the harvested crops into passages, labelled.

    the size and label fields of a name are what the crop is; only its
    millisecond changes, which is the whole of what makes it another vehicle.
    """
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(names):
        at = AT_MS + i * GAP_MS + (i % 3) * FRAME_MS
        copy = f"{at}_" + "_".join(name.split("_")[1:])
        shutil.copyfile(root / "raw" / name, crops / copy)
        rows.append(f"{copy} {label} seed")
    (crops.parent / "labels.txt").write_text("\n".join(rows) + "\n")
    return crops


def trained(binary: Path, work: Path, subject: str = SUBJECT) -> Path:
    """the work directory, holding the clips and a set trained from their crops.

    `<work>/trained` is the reference set, `<work>/big` the street an alerting run
    then drives; both are where they are because that is where this put them.
    """

    # one directory per set, because the labels belong to the set: `labels.txt`
    # lives beside a crop's directory, so two sets sharing a parent overwrite.
    def set_dir(name: str) -> Path:
        return work / "refs" / name / "crops"

    clips(work / "big")
    clips(work / "other", main=(1024, 576), sub=(512, 288))
    sets = [
        set_of(work / "big", harvested(binary, work / "big"), subject, set_dir(subject)),
        set_of(work / "other", harvested(binary, work / "other"), "other", set_dir("other")),
    ]
    out = work / "trained"
    cfg = work / "train.toml"
    cfg.write_text(base("http://nowhere/{subtype}.mp4") + f'\n[classifier]\nmodel = "{EMB}"\n')
    code, log = run(
        binary,
        cfg,
        "--dry-run",
        *[arg for s in sets for arg in ("--harvest", str(s))],
        "--train",
        subject,
        "--train-out",
        str(out),
    )
    assert code == 0, log
    return work


def alerting_config(
    clips: Path,
    server: str,
    crops: Path,
    trained: Path,
    harvest=True,
    click: str = "",
) -> Path:
    """the config of the run that alerts: the street, the references, the phone."""
    cfg = clips.parent / "alert.toml"
    cfg.write_text(
        base(f"{clips}/{{subtype}}.mp4")
        + f'\n[harvest]\nenabled = {str(harvest).lower()}\ndir = "{crops}"\n'
        + f'\n[classifier]\nmodel = "{EMB}"\nenabled = true\nreferences = "{trained}"\n'
        + f'\n[[subject]]\nname = "{SUBJECT}"\n'
        + f'\n[ntfy]\nenabled = true\nserver = "{server}"\n'
        + f'topic = "{TOPIC}"\ntoken = "{TOKEN}"\n'
        + (f'click = "{click}"\n' if click else "")
    )
    return cfg
