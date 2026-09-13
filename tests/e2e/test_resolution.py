"""streams at whatever size the camera sends them (r5.6).

the main stream's size used to be a constant. a camera swapped in for the
development one sends 4096x1856, which read as 2560x1440 is not an error and not
a stall: it is half a frame of sheared pixels per frame, so the detector finds
nothing and the harvest writes no crops while every log line says the pipeline is
healthy. the only symptom anyone got was `0 crops`, next to a `--debug-detector`
run -- which reads the *substream* -- finding cars all day.

these run the binary at a pair of urls whose sizes the config does not know and
whose aspects differ, which is the case nothing else covered. a file source
probed itself, so the rtsp path had no test at all until `[camera] url` could
name one.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from conftest import REPO
from preview import Preview, get

FIXTURE = REPO / "tests" / "e2e" / "fixtures" / "street.jpg"
CROP = re.compile(r"^\d+_[a-z-]+(_[a-z0-9-]+)?_\d{3}_\d+x\d+\.jpg$")

# the sizes under test: a 4:3 sub and a 16:9 main of one field of view, which is
# the anamorphic pair real cameras emit and the reason the mapping is per-axis.
MAIN = (1280, 720)
SUB = (1024, 768)

# the jeep across the street, as a fraction of the 640x480 fixture it was cut
# from, so the same vehicle can be moved in a clip of any size.
JEEP = (0.375, 0.085, 0.15, 0.145)
# how far the slide runs, and when it starts: after `[gate] warmup_frames`.
SLIDE = 0.06
SLIDE_AFTER = 1.2
SECONDS = 5


def street_clip(ffmpeg: str, out: Path, width: int, height: int, seconds: int = SECONDS) -> Path:
    """the same street at one size, with the jeep sliding along the kerb.

    scaled from an anamorphic frame, which is what the camera emits: a 4:3
    substream and a 16:9 main over one field of view, so the two differ by a
    different factor per axis -- the mapping the harvest lives on.
    """
    jx, jy = int(JEEP[0] * width), int(JEEP[1] * height)
    jw, jh = int(JEEP[2] * width), int(JEEP[3] * height)
    rate = int(3 * SLIDE * width)
    out.parent.mkdir(parents=True, exist_ok=True)
    # the moving thing is a crop of the frame it is pasted back onto, so one
    # vehicle changing its mind is the only difference between two frames.
    filters = (
        f"[0:v]scale={width}:{height},setsar=1[base];"
        "[base]split[a][b];"
        f"[b]crop={jw}:{jh}:{jx}:{jy}[patch];"
        "[a][patch]overlay="
        f"x='if(lt(t,{SLIDE_AFTER}),{jx},{jx}+{rate}*(t-{SLIDE_AFTER}))':y={jy}[out]"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(FIXTURE),
            "-t",
            str(seconds),
            "-r",
            "15",
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "15",
            str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture
def streams(ffmpeg: str, tmp_path: Path):
    """a directory whose {subtype} is a stream number, for [camera] url."""

    def make(main=MAIN, sub=SUB, seconds=SECONDS, name="streams") -> Path:
        dir = tmp_path / name
        street_clip(ffmpeg, dir / "0.mp4", main[0], main[1], seconds)
        street_clip(ffmpeg, dir / "1.mp4", sub[0], sub[1], seconds)
        return dir

    return make


def write_config(path: Path, url: str, crops: Path, harvest: bool = True) -> Path:
    path.write_text(f"""
[camera]
host = "127.0.0.1"
username = "u"
password = "p"
url = "{url}"

[stream]
gate_subtype = 1
# what the streams really send is what the run has to use, so these stay at
# their defaults -- 2560x1440 and 640x480, both wrong -- and must lose.
main_width = 2560
main_height = 1440
gate_width = 640
gate_height = 480

[gate]
min_changed_frac = 0.002
warmup_frames = 15
latch_ms = 1500

[detector]
model = "{REPO / "models" / "detector.onnx"}"

[harvest]
enabled = {str(harvest).lower()}
dir = "{crops}"
""")
    return path


def run(binary: Path, config: Path, timeout: int = 300) -> str:
    proc = subprocess.run(
        [str(binary), "--config", str(config), "--dry-run", "--preview", "off"],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "RUST_LOG": "metermate=info"},
        check=False,
    )
    return proc.stdout + proc.stderr


def crops_in(dir: Path) -> list[str]:
    if not dir.is_dir():
        return []
    return sorted(p.name for p in dir.glob("*.jpg") if CROP.match(p.name))


def test_a_size_the_config_does_not_know_is_still_harvested(binary, tmp_path, streams):
    crops = tmp_path / "crops"
    dir = streams()
    cfg = write_config(tmp_path / "metermate.toml", f"{dir}/{{subtype}}.mp4", crops)
    out = run(binary, cfg)

    # said out loud, because a fallback that is silent is a wrong size that is
    # silent, and this failure mode runs with no error anywhere to find.
    assert f"crop feed: {MAIN[0]}x{MAIN[1]}" in out, out
    assert f"gate feed: {SUB[0]}x{SUB[1]}" in out, out

    # a 4:3 sub of a 16:9 main is one field of view at a different factor per
    # axis, and the pair is reported so it can be checked against the camera
    # rather than against a constant.
    assert "scale 1.25x0.94" in out, out

    assert [ln for ln in out.splitlines() if "motion on" in ln], f"no motion:\n{out}"
    assert crops_in(crops), f"nothing was harvested:\n{out}"


def first_jpeg(stream: bytes) -> bytes:
    """the first frame of an mjpeg stream, which is what a browser is sent."""
    start = stream.find(b"\xff\xd8\xff")
    end = stream.find(b"\xff\xd9", start)
    assert start >= 0 and end > start, f"no jpeg in {len(stream)} bytes"
    return stream[start : end + 2]


def jpeg_size(jpeg: bytes) -> tuple[int, int]:
    """the frame's own size, from its start-of-frame marker."""
    at = 2
    while at + 9 < len(jpeg):
        assert jpeg[at] == 0xFF, "not a jpeg"
        marker = jpeg[at + 1]
        # 0xC0..0xCF are the frame headers, but DHT, JPG and DAC sit in that
        # range and carry no size.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return (
                int.from_bytes(jpeg[at + 7 : at + 9], "big"),
                int.from_bytes(jpeg[at + 5 : at + 7], "big"),
            )
        at += 2 + int.from_bytes(jpeg[at + 2 : at + 4], "big")
    raise AssertionError("no start-of-frame marker in the jpeg")


def test_the_preview_shows_the_street_rather_than_the_preview_box(binary, tmp_path, streams):
    """the encoded frame keeps the scene's shape, at a street that is not 16:9.

    a browser contains an image whose shape differs from its box while the overlay
    canvas covers the box, so a frame squeezed into `[preview] width`x`height`
    puts every box above the vehicle it belongs to. the shape to keep is the main
    stream's, since that is the street both streams are looking at -- 64:29 on the
    development camera, which fits no preview box.
    """
    wide = streams(main=(1024, 464), sub=(640, 480), seconds=20, name="wide")
    crops = tmp_path / "crops"
    cfg = write_config(tmp_path / "preview.toml", f"{wide}/{{subtype}}.mp4", crops, harvest=False)
    with Preview(binary, cfg) as p:
        p.ready()
        served = json.loads(get(p.port, "/config")[1])
        status, stream = get(p.port, "/stream.mjpg")
        assert status == 200, p.output()

    width, height = jpeg_size(first_jpeg(stream))
    assert (width, height) == tuple(served["encode"]), "the page was told another size"
    assert abs(width / height - 1024 / 464) < 0.02, f"{width}x{height} is not the street"
    # and it is not the frame's own shape either: the substream here is 4:3.
    assert served["gate"] == [640, 480]


def test_a_stream_that_cannot_be_asked_says_which_size_it_guessed(binary, tmp_path):
    """a camera that is off or rebooting cannot be probed.

    the declared size is then all there is, and the run has to say it is working
    from a guess: a fallback that prints nothing is the same silence that made
    the compiled-in size cost a harvest.
    """
    missing = tmp_path / "nowhere"
    missing.mkdir()
    crops = tmp_path / "crops"
    cfg = write_config(tmp_path / "guessed.toml", f"{missing}/{{subtype}}.mp4", crops)
    out = run(binary, cfg)

    assert out.count("did not answer") == 2, out
    assert "reading it as 2560x1440" in out, out
    assert "reading it as 640x480" in out, out
    assert "panicked" not in out, out
