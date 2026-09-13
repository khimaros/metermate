"""measuring the page's css in a real browser.

the node harness that covers the page's javascript has no layout: an element's width
and height there are whatever the test last wrote, which is exactly how a control can
be right in the source and clipped, overlapped or off screen on screen. nothing but a
browser knows what a rule did to a box.

the browser is handed the stylesheet out of src/preview/index.html and a document with
the ids and classes it keys on, so the rules are measured where they apply. chrome is
not everywhere, so callers skip rather than fail without one -- see the `pytestmark`
in the modules that use this.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

BROWSERS = ("google-chrome", "chromium", "chromium-browser")

ROOT = Path(__file__).resolve().parents[2]

# `box(el)` for a rect, and `hittable(el)` for whether a click at its centre lands on
# it: an element that is clipped away or covered reads as a normal box and is still
# unreachable, which is the whole failure this helper exists to catch.
PRELUDE = """
  const rect = (el) => {
    const r = el.getBoundingClientRect();
    return {
      x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom,
      scrollH: el.scrollHeight, clientH: el.clientHeight,
    };
  };
  // a clipped or covered element still reports a rect, so the question is what a
  // click there would actually land on: this element or one of its own descendants.
  const hittable = (el) => {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const at = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return !!at && el.contains(at);
  };
"""


def has_browser() -> bool:
    return any(shutil.which(b) for b in BROWSERS)


def style() -> str:
    """the preview page's stylesheet, which is where every layout rule lives."""
    html = (ROOT / "src" / "preview" / "index.html").read_text()
    blocks = re.findall(r"<style>(.*?)</style>", html, re.DOTALL)
    assert len(blocks) == 1, f"{len(blocks)} style blocks"
    return blocks[0]


def render(css: str, body: str, script: str, width: int = 1440, height: int = 900) -> dict:
    """lay out `body` under `css` and return what `script` assigns to `result`.

    the script runs on load with `rect` and `hittable` in scope; it has no other way to
    report back, being headless chrome printing a dom.
    """
    # the root font is left at the browser's own 16px rather than set here, because
    # every `rem` in the page's stylesheet -- the picker's height, a thumb's worth of
    # padding -- resolves against it, and a test document that set it differently
    # would measure a page nobody runs.
    doc = f"""<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>html {{ margin: 0 }} {css}</style>
{body}
<script>
let result = null;
addEventListener("load", () => {{
{PRELUDE}
{script}
  document.title = JSON.stringify(result);
}});
</script>
"""
    browser = next(b for b in BROWSERS if shutil.which(b))
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "doc.html"
        f.write_text(doc)
        out = subprocess.run(
            [
                browser,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--virtual-time-budget=2000",
                "--dump-dom",
                str(f),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    got = re.search(r"<title>(.*?)</title>", out.stdout, re.DOTALL)
    assert got, out.stdout[:400]
    return json.loads(got.group(1).replace("&quot;", '"'))
