"""score a labelled manifest from label.py into recall and false positives.

the question this answers is the product's: when a vehicle transits the scene,
does the detector see it. not "how many boxes were drawn", which is what every
other number in the eval counts.

each event is one transit. `truth` says whether the moving thing was really a
vehicle; `detected_on_moved` says whether the detector put a vehicle box on it.
so:

    vehicle + detected      -> hit
    vehicle + nothing       -> MISS, a false negative
    not-a-vehicle + boxed   -> FALSE POSITIVE
    not-a-vehicle + nothing -> correct rejection

    uv run tools/score.py out/manifest.json
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()

    events = json.loads(args.manifest.read_text())
    unlabelled = [e["event"] for e in events if e.get("truth") is None]
    if unlabelled:
        print(f"{len(unlabelled)} events still unlabelled: {unlabelled}", file=sys.stderr)
        return 1

    hits, misses, false_pos, rejects, unclear = [], [], [], [], []
    for e in events:
        seen = bool(e["detected_on_moved"])
        match e["truth"]:
            case "vehicle":
                (hits if seen else misses).append(e)
            case "not-a-vehicle":
                (false_pos if seen else rejects).append(e)
            case _:
                unclear.append(e)

    vehicles = len(hits) + len(misses)
    print(f"{len(events)} transit events, {vehicles} of them vehicles"
          f"{f', {len(unclear)} unclear (excluded)' if unclear else ''}\n")

    if vehicles:
        print(f"  recall            {len(hits)}/{vehicles} = "
              f"{len(hits) / vehicles * 100:.0f}%")
        print(f"  false negatives   {len(misses)}"
              + (f"  {[e['event'] for e in misses]}" if misses else ""))
    non_veh = len(false_pos) + len(rejects)
    if non_veh:
        print(f"  false positives   {len(false_pos)}/{non_veh} non-vehicle events"
              + (f"  {[e['event'] for e in false_pos]}" if false_pos else ""))

    confs = [d["conf"] for e in hits for d in e["detected_on_moved"]]
    if confs:
        print(f"\n  confidence on the moving vehicle: "
              f"median {st.median(confs):.2f}  min {min(confs):.2f}  max {max(confs):.2f}")
        weak = [e["event"] for e in hits
                if max(d["conf"] for d in e["detected_on_moved"]) < 0.5]
        if weak:
            print(f"  found but under 0.50: {weak}")

    widths = [d["w"] for e in hits for d in e["detected_on_moved"]]
    if widths:
        print(f"  box width px: median {int(st.median(widths))}  "
              f"min {min(widths)}  max {max(widths)}")

    if misses:
        print("\nmissed:")
        for e in misses:
            print(f"  e{e['event']:02d} t={e['t']}s  motion={e['motion']}  "
                  f"box={e['moved_box']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
