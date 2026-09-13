"""notifying a phone directly, over ntfy.

mqtt is the integration surface (r4.1) and it assumes something downstream is
listening: a broker, then home assistant, then a companion app. the thing this
project is for -- "move the car before the ticket" -- wants the shortest path
there is, so metermate posts to ntfy itself.

driven against a fake ntfy server rather than ntfy.sh: what is worth asserting
is the request metermate builds -- the topic it posts to, the token it carries,
the evidence it attaches -- and a real server would tell us nothing about that
while making the suite depend on someone else's uptime.
"""

import http.server
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from conftest import CLIP_H, CLIP_W, REPO

TOKEN = "tk_secret"
TOPIC = "metermate-test"
# the preview's address as an operator writes it in `[ntfy] click`: the phone has
# to reach it, so it is a lan address rather than localhost.
PREVIEW = "http://preview.example.com:8420"
AT = 1789000000000


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


def write_config(path: Path, server: str, **ntfy) -> Path:
    settings = {
        "enabled": "true",
        "server": f'"{server}"',
        "topic": f'"{TOPIC}"',
        "token": f'"{TOKEN}"',
        **ntfy,
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

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[ntfy]
{lines}
""")
    return path


def notify_test(binary, cfg, timeout: int = 60):
    return subprocess.run(
        [str(binary), "--config", str(cfg), "--notify-test"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_a_test_notification_reaches_the_server(binary, tmp_path):
    """**a deployment proves the phone works before it needs it to.**

    the alternative is finding out that the token was wrong on the evening a
    go-4 parks outside, which is the one moment nobody is reading logs.
    """
    with FakeNtfy() as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url)
        run = notify_test(binary, cfg)
        output = run.stdout + run.stderr

    assert run.returncode == 0, output
    assert len(ntfy.seen) == 1, ntfy.seen
    posted = ntfy.seen[0]
    assert posted["path"] == f"/{TOPIC}", posted["path"]
    assert posted["headers"]["authorization"] == f"Bearer {TOKEN}", posted["headers"]
    assert b"metermate" in posted["body"].lower(), posted["body"]
    # and it says so on the terminal, since that is what the command is for.
    assert TOPIC in output, output


def test_a_refused_test_notification_fails_loudly(binary, tmp_path):
    """the command exists to answer "does this work", so a server that says no
    must not exit zero -- an operator would read that as the phone being set
    up."""
    with FakeNtfy(status=403) as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url)
        run = notify_test(binary, cfg)
        output = run.stdout + run.stderr

    assert run.returncode != 0, output
    assert "403" in output, output


def test_an_unreachable_server_fails_loudly(binary, tmp_path):
    """the common setup mistake is a hostname nobody can resolve."""
    cfg = write_config(tmp_path / "metermate.toml", f"http://127.0.0.1:{free_port()}")
    run = notify_test(binary, cfg)
    output = run.stdout + run.stderr
    assert run.returncode != 0, output
    assert TOPIC in output or "ntfy" in output.lower(), output


def test_notifying_is_off_until_it_is_configured(binary, tmp_path):
    """**an empty topic is not a default anybody wants.** posting to a guessable
    ntfy.sh topic is publishing the street's comings and goings to whoever
    subscribes to it, so the feature refuses to start rather than pick one."""
    cfg = write_config(tmp_path / "metermate.toml", "https://ntfy.sh", topic='""')
    run = notify_test(binary, cfg)
    output = run.stdout + run.stderr
    assert run.returncode != 0, output
    assert "topic" in output.lower(), output


@pytest.mark.parametrize("bad", ['priority = "loudest"', 'outcomes = ["parked"]'])
def test_a_misspelled_setting_is_refused_at_startup(binary, tmp_path, bad):
    """the config is validated rather than shrugged at: a priority the server
    would ignore, or an outcome that does not exist, is a notification that
    silently never arrives."""
    key, value = bad.split(" = ")
    cfg = write_config(tmp_path / "metermate.toml", "https://ntfy.sh", **{key: value})
    run = notify_test(binary, cfg)
    output = run.stdout + run.stderr
    assert run.returncode != 0, output
    assert key in output, output


def test_the_crop_that_fired_is_the_notification_body(binary, tmp_path):
    """**r4.3, as far as a notification can take it.** a phone buzzing "go4"
    is a claim; the crop is the evidence for it, and deciding whether to move
    the car is a glance rather than a walk to a laptop.

    put rather than posted, with a filename, which is how ntfy takes an
    attachment -- so the image travels with the message instead of being a link
    the phone can only follow on the same lan.
    """
    crop = tmp_path / "evidence.jpg"
    crop.write_bytes(b"\xff\xd8\xff" + b"not really a jpeg" * 8)
    with FakeNtfy() as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url)
        run = subprocess.run(
            [str(binary), "--config", str(cfg), "--notify-test", "--notify-crop", str(crop)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = run.stdout + run.stderr

    assert run.returncode == 0, output
    posted = ntfy.seen[0]
    assert posted["method"] == "PUT", posted
    assert posted["body"] == crop.read_bytes(), posted["headers"]
    assert posted["headers"]["filename"].endswith(".jpg"), posted["headers"]
    # the words go in the headers when the body is the picture.
    assert "title" in posted["headers"], posted["headers"]


def test_a_notification_opens_the_verdict_it_was_about(binary, tmp_path):
    """**the tap is the second half of the evidence.**

    r4.5 guarantees the crop a notification attaches is on the verdict page, so
    the link can name that crop and cannot dead-end. a generic address made every
    notification land on the front page of a grid of two hundred crops.
    """
    crop = tmp_path / f"{AT}_truck_go4_075_240x200.jpg"
    crop.write_bytes(b"\xff\xd8\xff" + b"not really a jpeg" * 8)
    with FakeNtfy() as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url, click=f'"{PREVIEW}/"')
        run = subprocess.run(
            [str(binary), "--config", str(cfg), "--notify-test", "--notify-crop", str(crop)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert run.returncode == 0, run.stdout + run.stderr
    assert ntfy.seen[0]["headers"]["click"] == f"{PREVIEW}/#/verdict/{crop.name}", ntfy.seen[0]


def test_an_outcome_with_no_crop_of_its_own_reaches_the_verdicts(binary, tmp_path):
    """stopped and departed close a story the sighting opened and carry no crop of
    their own, so the nearest there is to point at is the page of verdicts."""
    with FakeNtfy() as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url, click=f'"{PREVIEW}"')
        run = notify_test(binary, cfg)
    assert run.returncode == 0, run.stdout + run.stderr
    assert ntfy.seen[0]["headers"]["click"] == f"{PREVIEW}/#/verdicts", ntfy.seen[0]


def test_a_notification_still_works_with_no_preview_to_link_to(binary, tmp_path):
    """`[ntfy] click` is empty by default, and a deployment with no preview the
    phone can reach keeps its notification and loses the link -- the header is
    left off rather than sent as an empty one the server would route to itself.
    """
    crop = tmp_path / f"{AT}_truck_go4_075_240x200.jpg"
    crop.write_bytes(b"\xff\xd8\xff" + b"not really a jpeg" * 8)
    with FakeNtfy() as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url)
        run = subprocess.run(
            [str(binary), "--config", str(cfg), "--notify-test", "--notify-crop", str(crop)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "click" not in ntfy.seen[0]["headers"], ntfy.seen[0]["headers"]


def test_the_config_records_what_would_be_sent(binary, tmp_path):
    """outcomes are a filter, and a filter nobody can see is a feature that
    looks broken: `--notify-test` prints the ones that will notify."""
    with FakeNtfy() as ntfy:
        cfg = write_config(
            tmp_path / "metermate.toml", ntfy.url, outcomes='["sighting", "stopped"]'
        )
        run = notify_test(binary, cfg)
        output = run.stdout + run.stderr

    assert run.returncode == 0, output
    assert "sighting" in output and "stopped" in output, output
    assert "departed" not in output, output


def test_json_payloads_are_never_sent_to_a_third_party(binary, tmp_path):
    """**what leaves the network is a sentence and a picture, not the wire
    format.** the mqtt payload carries track ids, boxes and dwell times for a
    rule to act on; ntfy.sh is somebody else's server, and the facts a rule
    needs are not facts a stranger needs.
    """
    with FakeNtfy() as ntfy:
        cfg = write_config(tmp_path / "metermate.toml", ntfy.url)
        run = notify_test(binary, cfg)
        assert run.returncode == 0, run.stdout + run.stderr

    body = ntfy.seen[0]["body"]
    assert not body.strip().startswith(b"{"), body
    for leak in (b"box", b"track", b"protected"):
        assert leak not in body.lower(), body
