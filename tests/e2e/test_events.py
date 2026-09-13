"""the event clips, and reaching them from the browser (r8.6).

driven through the real binary rather than against the recorder directly,
because everything interesting here is in the wiring: the clip has to travel
from the gate's motion decision, through the ffmpeg remux, a ring buffer, a
file whose name is chosen when it closes, a directory listing, and an http
handler, before a browser sees anything. a recorder that is right in
`record/mod.rs` and wired up wrong is a preview tab that is always empty.
"""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO, in_page

# generous: the binary replays the clip through ffmpeg twice over -- once
# decoded for the gate and once stream-copied for the recorder -- and the clip
# only closes once the ceiling below is reached.
READY_TIMEOUT_S = 60
POLL_S = 0.5
# short enough that clips close while the six second fixture is still playing,
# which is what lets the listing be read from a running process.
MAX_CLIP_S = 2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_config(path: Path, events: Path, **record) -> Path:
    # `keep = []` because the fixture is a white block rather than a vehicle, so
    # nothing is ever detected in it and the shipped default would discard every
    # clip. what `keep` does is tested on its own, below.
    settings = {
        "enabled": "true",
        "max_clip_secs": MAX_CLIP_S,
        "hangover_secs": 1,
        "keep": "[]",
        **record,
    }
    lines = "\n".join(f"{k} = {v}" for k, v in settings.items())
    path.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"

[stream]
gate_subtype = 1
gate_width = {CLIP_W}
gate_height = {CLIP_H}

[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 500

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
enabled = false

[record]
dir = "{events}"
{lines}
""")
    return path


def get(port: int, path: str) -> tuple[int, bytes, str]:
    """fetch, returning the status even for the errors this asserts on."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")
    except (urllib.error.URLError, TimeoutError):
        return 0, b"", ""


def wait_for(port: int, done, timeout: float = READY_TIMEOUT_S):
    """poll `/clips` until `done` accepts it, returning the listing."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, body, _ = get(port, "/clips")
        if status == 200:
            last = json.loads(body)
            if done(last):
                return last
        time.sleep(POLL_S)
    pytest.fail(f"nothing satisfied the wait within {timeout}s; last listing was {last}")


class Running:
    """metermate against a file source, with its preview reachable."""

    def __init__(self, binary: Path, config: Path, source: Path, port: int, *args: str):
        self.port = port
        self.log = config.parent / "metermate.log"
        self.handle = self.log.open("w")
        self.proc = subprocess.Popen(
            [
                str(binary),
                "--config",
                str(config),
                "--source",
                str(source),
                "--dry-run",
                "--preview",
                f"127.0.0.1:{port}",
                *args,
            ],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
        )

    def output(self) -> str:
        self.handle.flush()
        return self.log.read_text()

    def wait(self, timeout: int = READY_TIMEOUT_S) -> str:
        """let the stream run out, which is how a replay ends."""
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        return self.output()

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


@pytest.fixture
def events_dir(tmp_path: Path) -> Path:
    return tmp_path / "events"


def test_a_recorded_clip_is_listed_and_playable(binary, tmp_path, moving_clip, events_dir):
    """the whole chain, end to end: motion becomes a file becomes a listing
    becomes bytes a browser can play."""
    port = free_port()
    cfg = write_config(tmp_path / "metermate.toml", events_dir)
    with Running(binary, cfg, moving_clip, port) as run:
        listing = wait_for(port, lambda d: d["clips"])

        assert listing["kept"] >= 1
        assert listing["used"] > 0, "a clip was listed with no bytes in it"
        assert listing["budget"] > 0

        clip = listing["clips"][0]
        assert clip["n"].endswith("-main.mp4"), clip
        assert clip["w"] in ("motion", "detection", "subject"), clip
        assert clip["b"] > 0 and clip["t"] > 0, clip

        status, body, kind = get(port, "/clips/" + clip["n"])
        assert status == 200, run.output()[-2000:]
        assert kind == "video/mp4", kind
        assert len(body) == clip["b"], "the listing and the file disagree on size"
        # a clip begins with its init header, or nothing can decode it.
        assert body[4:8] == b"ftyp", body[:16]
        assert b"moof" in body, "no fragments followed the header"


def test_a_clip_open_when_the_stream_ends_is_still_finished(
    binary, tmp_path, moving_clip, events_dir
):
    """**a clip is not lost because the stream stopped mid-event.**

    the ceiling and the hangover are both set past the length of the fixture
    here, so motion never goes quiet and the clip is still open when the replay
    runs out. left unclosed it stays under the name the recorder writes to while
    it is deciding -- which nothing lists, nothing evicts and nothing plays,
    since the whole point of that name is that it is not a finished clip.
    """
    cfg = write_config(
        tmp_path / "metermate.toml", events_dir, max_clip_secs=600, hangover_secs=600
    )
    with Running(binary, cfg, moving_clip, free_port()) as run:
        out = run.wait()

    assert "stream ended" in out, out[-2000:]
    unfinished = list(events_dir.glob("*.part"))
    finished = list(events_dir.glob("*.mp4"))
    assert not unfinished, f"a clip was abandoned mid-write: {unfinished}"
    assert finished, f"the open clip was never finished: {sorted(events_dir.iterdir())}"
    assert finished[0].stat().st_size > 0


def wait_for_config(port: int, run) -> dict:
    deadline = time.monotonic() + READY_TIMEOUT_S
    status, body = 0, b""
    while time.monotonic() < deadline:
        status, body, _ = get(port, "/config")
        if status == 200:
            return json.loads(body)
        time.sleep(POLL_S)
    raise AssertionError(f"the preview never answered: {run.output()[-2000:]}")


def test_the_events_tab_is_not_offered_with_nothing_to_show(
    binary, tmp_path, moving_clip, events_dir
):
    """an empty events tab would say the street was quiet, when the truth is
    that the feature is off. with nothing recorded and nothing recording, the
    tab is not offered and the endpoints behind it are not there."""
    port = free_port()
    cfg = write_config(tmp_path / "metermate.toml", events_dir, enabled="false")
    with Running(binary, cfg, moving_clip, port, "--record-events=false") as run:
        assert wait_for_config(port, run)["events"] is False
        assert get(port, "/clips")[0] == 404
        assert get(port, "/clips/1789-subject-main.mp4")[0] == 404


def test_clips_stay_browsable_after_recording_is_turned_off(
    binary, tmp_path, moving_clip, events_dir
):
    """**turning recording off must not hide what it already recorded.**

    disabling it is the first thing anyone does when it misbehaves, and the
    clips on disk are exactly what they then want to look at. they outlive the
    setting that wrote them.
    """
    events_dir.mkdir(parents=True)
    kept = events_dir / "1789413727055-detection-main.mp4"
    kept.write_bytes(b"not a real clip, but a real listing")

    port = free_port()
    cfg = write_config(tmp_path / "metermate.toml", events_dir, enabled="false")
    with Running(binary, cfg, moving_clip, port, "--record-events=false") as run:
        assert wait_for_config(port, run)["events"] is True, run.output()[-2000:]

        status, body, _ = get(port, "/clips")
        assert status == 200
        listing = json.loads(body)
        assert listing["kept"] == 1, listing
        assert listing["clips"][0]["n"] == kept.name, listing
        # and the page is told, or a list that never grows reads as broken.
        assert listing["recording"] is False, listing

        assert get(port, "/clips/" + kept.name)[0] == 200


def test_a_motion_only_clip_is_not_kept_by_default(binary, tmp_path, moving_clip, events_dir):
    """the shipped `keep` does not include motion, and the fixture is a white
    block that no detector calls a vehicle -- so this run triggers repeatedly,
    writes clips, and keeps none of them.

    the pairing matters: the same fixture with `keep = []` fills the directory,
    which is what every other test here relies on. so this proves the policy is
    doing the work rather than the fixture simply never triggering."""
    cfg = write_config(tmp_path / "metermate.toml", events_dir, keep='["detection", "subject"]')
    with Running(binary, cfg, moving_clip, free_port()) as run:
        out = run.wait()

    assert "event clip" in out, f"nothing ever triggered, so nothing was tested: {out[-2000:]}"
    assert "discarded" in out, out[-2000:]
    assert not list(events_dir.glob("*")), (
        f"a motion-only clip was kept: {list(events_dir.iterdir())}"
    )


def test_an_unknown_outcome_in_keep_is_refused(binary, tmp_path, moving_clip, events_dir):
    """a typo keeps nothing, and an events directory that stays empty looks
    exactly like a quiet street. so it is refused at startup rather than
    ignored."""
    cfg = write_config(tmp_path / "metermate.toml", events_dir, keep='["detections"]')
    with Running(binary, cfg, moving_clip, free_port()) as run:
        out = run.wait()
    assert "record.keep" in out and "detections" in out, out[-2000:]


def listed_clip(tmp_path: Path, events_dir: Path, name: str) -> Path:
    """a clip on disk with recording off, which is the cheapest real listing.

    the bytes are never decoded by anything these tests ask about -- the page's
    posters are loaded below the fold, which nothing here scrolls to.
    """
    events_dir.mkdir(parents=True, exist_ok=True)
    clip = events_dir / name
    clip.write_bytes(b"not a real clip, but a real listing")
    write_config(tmp_path / "metermate.toml", events_dir, enabled="false")
    return clip


def test_an_event_card_offers_the_clips_filename_to_copy(
    binary, tmp_path, moving_clip, events_dir, node
):
    """**the path is the argument `--dense` takes**, not the bare filename:
    pulling every crop out of a passage means naming the clip it is in, under
    the directory `[record] dir` set -- and the card plays the clip when it is
    clicked, so nothing on it can be selected by hand.
    """
    clip = listed_clip(tmp_path, events_dir, "1789413727055-subject-main.mp4")
    port = free_port()
    with Running(binary, tmp_path / "metermate.toml", moving_clip, port, "--record-events=false"):
        wait_for(port, lambda d: d["clips"])
        state = in_page(
            node,
            port,
            # the filter is a `<select>`, and its value comes from the option
            # the markup marks as selected -- which a stub document has not
            # got. everything kept, which is what the page opens on.
            "(async () => { worthFilter.value = 'motion'; showTab('events');"
            " await new Promise(r => setTimeout(r, 500));"
            " const find = e => e.className === 'copy' ? e :"
            "   (e.children || []).map(find).find(Boolean);"
            " const button = find(document.getElementById('clips'));"
            " button.onclick({stopPropagation() {}});"
            " return {copied: globalThis.__copied, said: button.textContent,"
            "   playing: zoomVid.src}; })()",
        )
        # as configured, which is how `--dense` will resolve it: the suite
        # configures an absolute directory, kairos configures `data/events`.
        assert state["copied"] == str(events_dir / clip.name), state
        # a copy that failed and one that worked look identical otherwise: the
        # clipboard cannot be read back.
        assert state["said"] == "copied", state
        # the card under the button plays the clip when clicked, and copying its
        # name is not asking to watch it.
        assert state["playing"] == "", state


def test_the_roi_toml_is_copied_without_the_clipboard_api(
    binary, tmp_path, moving_clip, events_dir, node
):
    """`navigator.clipboard` is only defined in a secure context, and the
    preview is served over plain http on the camera's own lan address -- so on
    every real deployment that api is undefined and the button that used it did
    nothing at all, silently, which is the worst way for a copy to fail.
    """
    listed_clip(tmp_path, events_dir, "1789413727055-subject-main.mp4")
    port = free_port()
    with Running(binary, tmp_path / "metermate.toml", moving_clip, port, "--record-events=false"):
        wait_for(port, lambda d: d["clips"])
        state = in_page(
            node,
            port,
            "(async () => { roi = [[10, 20], [30, 40], [50, 60]]; showRoi();"
            " document.getElementById('roi-copy').onclick({stopPropagation() {}});"
            " return {copied: globalThis.__copied, want: roiToml.textContent}; })()",
        )
        assert "roi = [[10, 20], [30, 40], [50, 60]]" in state["want"], state
        assert state["copied"] == state["want"], state


def test_a_clip_name_cannot_escape_the_events_directory(binary, tmp_path, moving_clip, events_dir):
    """the name comes from the url and picks a file to read. a clip is the full
    resolution main stream, so being wrong here hands over rather more than a
    thumbnail."""
    port = free_port()
    cfg = write_config(tmp_path / "metermate.toml", events_dir)
    with Running(binary, cfg, moving_clip, port):
        wait_for(port, lambda d: d["clips"])
        for attempt in (
            "/clips/../metermate.toml",
            "/clips/..%2fmetermate.toml",
            "/clips/%2e%2e/metermate.toml",
            "/clips/metermate.toml",
            # the name a clip is written under while it is still open: serving
            # it would cache half a file under `immutable`.
            "/clips/1789-main.mp4.part",
        ):
            assert get(port, attempt)[0] == 404, f"{attempt} was served"
