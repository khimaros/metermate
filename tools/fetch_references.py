"""fetch reference photographs of go-4 interceptors from wikimedia commons.

these seed the phase-2 few-shot classifier, which compares an embedding of each
vehicle crop against a handful of known go-4 images. a few reference photos is
all that stage needs, which is what makes metermate useful before any local
training data exists (r6.2).

the images are not committed. they carry attribution requirements and would bloat
the repository, so they are fetched on demand and listed in references/SOURCES.md.
"""

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCES = REPO / "references" / "go4"

COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"
# a modest user agent: wikimedia blocks unidentified bulk clients.
USER_AGENT = "metermate/0.1 (https://github.com/khimaros/metermate)"

# deliberately varied: angles, cities, day and night. the san francisco one is
# the closest match to what this camera will actually see.
REFERENCE_FILES = [
    ("SFPD parking enforcement vehicle side.JPG", "CC BY-SA 4.0", "BrokenSphere"),
    ("NYPD 3 wheeled car in Central Park.JPG", "CC BY-SA 3.0", "Kevin.B"),
    ("NYC NYPD Westward Go-4 Interceptor 3589.JPG", "CC0", "Benoit Prieur"),
    ("Nyc-interceptor-nypd.jpg", "CC0", "MartinThoma"),
    ("NYPD Westward Go-4 Interceptors at night, 2015.jpg", "CC BY-SA 4.0", "Andromeda2064"),
    ("NYPD vehicle on 8th Avenue.jpg", "CC BY-SA 4.0", "Kritzolina"),
    ("Westward Go-4, Seattle Police (Parking Enforcement).jpg", "CC BY 4.0", "Kyah117"),
]


# wide enough to crop a good vehicle region from, small enough to fetch quickly.
WIDTH = 1400


def fetch(name: str, out_dir: Path) -> Path | None:
    safe = name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    dest = out_dir / safe
    if dest.exists():
        return dest
    url = COMMONS + urllib.parse.quote(name) + f"?width={WIDTH}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
    except Exception as e:  # noqa: BLE001 - a missing reference is not fatal
        print(f"  failed {name}: {e}", file=sys.stderr)
        return None
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REFERENCES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    got = 0
    for name, lic, author in REFERENCE_FILES:
        path = fetch(name, args.out)
        if path:
            got += 1
            print(f"  {path.name}  [{lic}, {author}]")
    print(f"\n{got}/{len(REFERENCE_FILES)} go-4 references in {args.out}")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
