"""driving the labelling server, for the tests that need one.

the harness rather than the fixtures. the suite's convention is that a fixture
is defined in the module that uses it -- see `test_margin.py`, "the same shape
as `test_train`'s, defined here rather than imported" -- because a fixture is a
few lines and a shared one quietly couples two tests that meant to be
independent. an http client and a subprocess lifecycle are neither few lines
nor independent of each other, so those live here and each module wraps them in
its own fixture.
"""

import json
import shutil
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import REPO

READY_TIMEOUT_S = 120
POLL_S = 0.5

SUBJECT = "sweeper"
# what stage two named, and what it did not. the second group is the control:
# the harvest is almost entirely crops with no verdict at all, and a pool that
# swept those up would be marking the street rather than the classifier.
VERDICTS, PLAIN = 5, 6

AT = 1789000000000
# well past `label::PASSAGE_GAP_MS`, so nothing here groups into one passage and
# the neighbour pool stays out of the way.
SPACING_MS = 60_000

# `label::CACHE_MAGIC` and the record layout `Vectors::load` reads. the vectors
# are pre-seeded so a session has nothing left to embed: these tests are about
# which crops reach the page and what a key writes, and loading a 351 MB
# embedder to answer that would cost half a minute per case. a format change
# degrades rather than breaks -- an unreadable cache is discarded, and the
# session embeds for real.
CACHE_MAGIC = b"MMVEC001"
DIM = 4


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def verdict_name(millis: int, subject: str, conf: int = 75, margin: float | None = None) -> str:
    """`{millis}_{class}_{subject}[_{margin}]_{conf}_{w}x{h}.jpg`, what `parse_name` reads.

    the margin is stage two's own number -- how far the crop cleared its negatives,
    in thousandths -- and it rides only beside a subject, since a crop the
    classifier said nothing about has no margin worth recording.
    """
    gap = "" if margin is None else f"_{round(margin * 1000)}"
    return f"{millis}_truck_{subject}{gap}_{conf:03d}_240x200.jpg"


def plain_name(millis: int) -> str:
    """the same name with no verdict in it: stage two named nothing."""
    return f"{millis}_car_090_240x200.jpg"


def write_cache(path: Path, names: list[str], vectors: list[list[float]] | None = None) -> None:
    body = bytearray(CACHE_MAGIC + struct.pack("<I", DIM))
    for i, name in enumerate(names):
        body += struct.pack("<H", len(name)) + name.encode()
        # distinct per crop unless chosen, so nothing here can accidentally rank
        # by being a copy of everything else.
        vector = vectors[i] if vectors else (1.0, i * 0.01, -i * 0.01, 0.5)
        body += struct.pack(f"<{DIM}f", *vector)
    path.write_bytes(bytes(body))


def blank_crops(dir: Path, ffmpeg: str, names: list[str]) -> None:
    """the same gray jpeg under every name.

    what these tests exercise is names, labels and vectors, so the pixels only
    have to decode. called twice on one directory it adds to it, which is how a
    crop arrives while a server is already running.
    """
    dir.mkdir(exist_ok=True)
    template = dir.parent / "template.jpg"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=240x200",
            "-frames:v",
            "1",
            str(template),
        ],
        check=True,
    )
    for name in names:
        shutil.copy(template, dir / name)


def build_harvest(tmp_path: Path, ffmpeg: str) -> Path:
    """crops the classifier named, crops it did not, and a seeded vector cache."""
    names = [verdict_name(AT + i * SPACING_MS, SUBJECT) for i in range(VERDICTS)]
    names += [plain_name(AT + (VERDICTS + i) * SPACING_MS) for i in range(PLAIN)]
    dir = tmp_path / "crops"
    blank_crops(dir, ffmpeg, names)
    write_cache(tmp_path / "embeddings.bin", names)
    return dir


def build_config(tmp_path: Path, harvest: Path | None = None, sets: Path | None = None) -> Path:
    """`--label` reads crops off disk: no camera, no stream, no detector.

    `references` is always set, because the page can write `trained/` and the
    default is relative to wherever the test runs -- which is the repository.
    `harvest` and `sets` are for a run that names no `--harvest` and so reads
    the configured pair, whose defaults are relative the same way.
    """
    path = tmp_path / "metermate.toml"
    path.write_text(
        f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[classifier]
model = "{REPO / "models" / "embedder.onnx"}"
references = "{tmp_path / "trained"}"
"""
        + (f'\n[harvest]\ndir = "{harvest}"\n' if harvest else "")
        + (f'\n[train]\nsets = "{sets}"\n' if sets else "")
    )
    return path


def harvest_args(harvests: list[Path]) -> list[str]:
    """`--harvest` once per root, and not at all for the configured pair."""
    return [arg for h in harvests for arg in ("--harvest", str(h))]


def embed(binary: Path, config: Path, harvests: list[Path]) -> str:
    """run `--embed` to the end and hand back everything it said."""
    done = subprocess.run(
        [str(binary), "--config", str(config), *harvest_args(harvests), "--embed"],
        capture_output=True,
        text=True,
        timeout=READY_TIMEOUT_S,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout + done.stderr


class Labelling:
    """the labelling server, against the harvests on disk.

    `harvest` is one root, several, or none for the configured pair.
    """

    def __init__(
        self,
        binary: Path,
        config: Path,
        harvest: Path | list[Path] | None,
        subject: str = SUBJECT,
        ingest: int | None = None,
    ):
        harvests = [harvest] if isinstance(harvest, Path) else harvest or []
        self.port = free_port()
        self.log = config.parent / "label.log"
        self.handle = self.log.open("w")
        self.proc = subprocess.Popen(
            [
                str(binary),
                "--config",
                str(config),
                *harvest_args(harvests),
                "--label",
                subject,
                "--label-addr",
                f"127.0.0.1:{self.port}",
                *([] if ingest is None else ["--label-ingest", str(ingest)]),
                "--preview",
                "off",
            ],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
        )
        # the first root's, which is the only one when a test names one.
        self.labels = (harvests[0] if harvests else config).parent / "labels.txt"

    def output(self) -> str:
        self.handle.flush()
        return self.log.read_text()

    def get(self, path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, TimeoutError):
            return 0, b""

    def post(self, path: str, body: str = "") -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=body.encode(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def queue(self) -> dict:
        status, body = self.get("/queue")
        assert status == 200, self.output()
        return json.loads(body)

    def ready(self) -> "Labelling":
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                pytest.fail(f"the labelling server exited: {self.output()}")
            if self.get("/queue")[0] == 200:
                return self
            time.sleep(POLL_S)
        # stopped here, because `with Labelling(...).ready()` has not entered the
        # block yet and so would never reach `__exit__`.
        said = self.output()
        self.stop()
        pytest.fail(f"no queue within {READY_TIMEOUT_S}s: {said}")

    def entries(self) -> dict[str, tuple[str, str]]:
        """`labels.txt` as `{crop: (truth, via)}`, which is the whole artifact."""
        if not self.labels.exists():
            return {}
        rows = {}
        for line in self.labels.read_text().splitlines():
            line = line.split("#")[0].split()
            if len(line) >= 3:
                rows[line[0]] = (line[1], line[2])
        return rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.handle.close()
