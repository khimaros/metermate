"""embed a labelled set with a candidate model served over http, for `--measure`.

the question this answers is whether another embedder would separate a subject
better than the one that ships, asked of the labels already made. the set's
crops and `labels.txt` are copied to `out`, each crop is sent to an openai-style
`/v1/embeddings` server (llama.cpp's `llama-server --embeddings --mmproj ...`
is one), and the vectors are written as the `embeddings.bin` `label::Vectors`
reads. `--measure` then scores them with the code that scores the shipped
embedder, so the two reports differ in the vectors and nothing else:

    uv run tools/embed_remote.py http://host:7860/v1/embeddings \\
        sets/go4/20260915 /tmp/candidate --model embeddinggemma-2:Q8_0
    metermate --measure go4 --harvest /tmp/candidate
    metermate --measure go4 --harvest sets/go4/20260915    # the baseline

a copy rather than in place, because a set holds one cache and vectors from two
models must never share it. the cache is appended a record at a time, so a run
that is killed resumes where it stopped. one connection is kept open for the
whole run: a connection per crop costs three round trips before any work, which
on a distant server is most of the time.

measured 2026-10-06 on `sets/go4/20260915` (586 go4 in 3 passages, 806 other),
clip vit-b/32 against embeddinggemma-2 at Q8_0: separation gap +0.060 against
+0.104, nearest neighbour per crop 98.7% against 99.6%, go4 crops kept at 0.09%
of the street firing 50% against 64%, and at none of it firing 39% against 41%.
both fire on 3 of 3 passages. a better embedder on this set, and three passages
from one stretch of one day cannot say whether it is a better detector.
"""

import argparse
import base64
import http.client
import json
import math
import shutil
import struct
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

# `label::CACHE_MAGIC`, and the record layout `Vectors::load` reads.
CACHE_MAGIC = b"MMVEC001"
REPORT_EVERY = 100
TIMEOUT_S = 300


def cached(path: Path) -> tuple[int, set[str]]:
    """the width of a cache and the names in it, or (0, nothing)."""
    if not path.exists():
        return 0, set()
    data = path.read_bytes()
    dim = struct.unpack("<I", data[8:12])[0]
    names, at = set(), 12
    while at + 2 <= len(data):
        n = struct.unpack("<H", data[at : at + 2])[0]
        end = at + 2 + n + dim * 4
        if end > len(data):
            break
        names.add(data[at + 2 : at + 2 + n].decode())
        at = end
    return dim, names


def connect(url: str) -> tuple[http.client.HTTPConnection, str]:
    parts = urlsplit(url)
    kind = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    return kind(parts.netloc, timeout=TIMEOUT_S), parts.path


def embed(conn: http.client.HTTPConnection, path: str, jpeg: bytes, model: str | None) -> list:
    """one crop's vector, l2 normalised so cosine is a dot product downstream."""
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    body = {"input": [{"content": [{"type": "image_url", "image_url": {"url": uri}}]}]}
    if model:
        body["model"] = model
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    reply = conn.getresponse()
    text = reply.read()
    if reply.status != 200:
        raise RuntimeError(f"{reply.status}: {text[:300]!r}")
    vector = json.loads(text)["data"][0]["embedding"]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def stage(source: Path, out: Path) -> None:
    """the set's crops and labels, copied where a second cache can sit beside them."""
    shutil.copytree(source / "crops", out / "crops", dirs_exist_ok=True)
    if (source / "labels.txt").exists():
        shutil.copy(source / "labels.txt", out / "labels.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url", help="the server's embeddings endpoint")
    ap.add_argument("set", type=Path, help="a set: a directory holding crops/ and labels.txt")
    ap.add_argument("out", type=Path, help="where the copy and its embeddings.bin go")
    ap.add_argument("--model", help="the model name, for a server that hosts several")
    args = ap.parse_args()

    stage(args.set, args.out)
    cache = args.out / "embeddings.bin"
    dim, done = cached(cache)
    crops = sorted(p for p in (args.out / "crops").glob("*.jpg") if p.name not in done)
    print(f"{len(done)} cached, {len(crops)} to embed", flush=True)
    conn, path = connect(args.url)
    started = time.monotonic()
    with cache.open("ab") as f:
        for i, crop in enumerate(crops, 1):
            vector = embed(conn, path, crop.read_bytes(), args.model)
            if dim == 0:
                dim = len(vector)
                f.write(CACHE_MAGIC + struct.pack("<I", dim))
            name = crop.name.encode()
            f.write(struct.pack("<H", len(name)) + name + struct.pack(f"<{dim}f", *vector))
            f.flush()
            if i % REPORT_EVERY == 0 or i == len(crops):
                each = (time.monotonic() - started) / i * 1000
                print(f"{i}/{len(crops)}, {dim} wide, {each:.0f} ms a crop", flush=True)
    print(f"measure it with `metermate --measure <subject> --harvest {args.out}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
