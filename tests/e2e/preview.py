"""driving the live page: the pipeline with its preview reachable.

shared because three test files now ask the same thing of it -- labelling a
crop, marking a passage, and running the page's own javascript against it.
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

READY_TIMEOUT_S = 60


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(port: int, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError):
        return 0, b""


def post(port: int, path: str, body: dict) -> tuple[int, bytes]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError):
        return 0, b""


class Preview:
    """metermate against a file source, with its preview reachable."""

    def __init__(self, binary: Path, config: Path, source: Path | None = None, debug=False):
        self.port = free_port()
        self.log = config.parent / "preview.log"
        self.handle = self.log.open("w")
        cmd = [str(binary), "--config", str(config)]
        # without a source the config's own [camera] url is what runs, which is
        # how a stream of some other size reaches the preview.
        if source:
            cmd += ["--source", str(source)]
        # what the detector saw on a gated look, and how many parked vehicles were
        # standing there, is debug: a line per gated frame is too loud for a
        # deployment. this is where asking it is cheap.
        env = {**os.environ, "RUST_LOG": "metermate=debug"} if debug else None
        self.proc = subprocess.Popen(
            cmd
            + [
                "--dry-run",
                "--preview",
                f"127.0.0.1:{self.port}",
            ],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def ready(self, on: str = "/crops") -> "Preview":
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if get(self.port, on)[0] == 200:
                return self
            time.sleep(0.5)
        raise AssertionError(f"the preview never answered:\n{self.output()}")

    def crops(self) -> dict:
        return json.loads(get(self.port, "/crops")[1])

    def clips(self) -> dict:
        return json.loads(get(self.port, "/clips")[1])

    def selections(self) -> list:
        return json.loads(get(self.port, "/selections")[1])["selections"]

    def overlay(
        self, until=lambda frames: True, within: float = 30.0, read_for: float = 3.0
    ) -> list[dict]:
        """frames of the detections feed, as the browser would have drawn them.

        the feed is one json object per sse event and exists only while frames are
        flowing, so this connects, drains it for `read_for` at a time, and stops
        once `until` is happy with everything seen or `within` runs out. a test
        that wants to know whether something is *never* drawn has to keep reading
        after the point it would stop for something that is.
        """
        deadline = time.monotonic() + within
        seen: list[dict] = []
        while time.monotonic() < deadline:
            seen += self._events(read_for)
            if until(seen):
                return seen
        if not seen and self.proc.poll() is not None:
            raise AssertionError(f"the run ended before drawing anything:\n{self.output()}")
        return seen

    def _events(self, read_for: float = 3.0) -> list[dict]:
        """the frames sent in one connection, for as long as it is held open."""
        buf = b""
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=read_for) as s:
                s.sendall(b"GET /detections HTTP/1.1\r\nHost: metermate\r\n\r\n")
                while True:
                    chunk = s.recv(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
        except OSError:
            # nobody watching yet, or the source ended mid-read: either way this
            # connection saw nothing, and the caller decides whether that matters.
            return []
        frames = []
        for event in buf.split(b"\r\n\r\n", 1)[-1].split(b"\n\n"):
            line = event.strip()
            if not line.startswith(b"data: "):
                continue
            try:
                # a frame caught mid-write is not a frame.
                frames.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
        return frames

    def output(self) -> str:
        self.handle.flush()
        return self.log.read_text()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.handle.close()
