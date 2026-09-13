"""the label picker's box, measured in a real browser.

the picker is the only control on a crop tile that is bigger than the tile: a field
and up to `LABELS_OFFERED` labels, which is several times the height of the row it is
opened from. so its whole layout problem is escaping the card it belongs to, and that
is a question about clipping, stacking and overflow -- which the node harness has no
layout to answer and no amount of reading the source settles.

the markup mirrors what the page builds: `makeFigure`'s figure, figcaption and
`labelStrip`'s row, and `openPicker`'s picker. the guard below keeps the mirror honest.
"""

import re

import pytest

from browser import ROOT, has_browser, render

pytestmark = pytest.mark.skipif(not has_browser(), reason="no browser to measure layout in")

PAGE = ROOT / "src" / "preview" / "index.html"

TILE = """
<figure>
  <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
  <figcaption><span><b>truck</b> 0.71</span><span><time>09:14</time></span></figcaption>
  <div class="labels"><span class="truth">unlabelled</span><button>edit</button></div>
</figure>
"""

# one screenful of crops, so the tile being labelled has tiles below it to fall over
# and behind it to be covered by.
GRID = """
<body class="grid">
  <div id="crops" class="on">
    <div class="grid-bar"><span>200 standing, 3 alerted</span></div>
    <div id="grid" class="crop-grid">%s</div>
  </div>
</body>
"""

# the picker, as `openPicker` appends it: a field and a scrolling list of labels,
# opened from the row the `edit` button lives in.
OPEN = """
  const open = (fig, labels) => {
    const row = fig.querySelector(".labels");
    const box = document.createElement("span");
    box.className = "picker";
    const field = document.createElement("input");
    const list = document.createElement("div");
    list.className = "options";
    for (const one of labels) {
      const o = document.createElement("div");
      o.className = "option";
      o.textContent = one;
      list.appendChild(o);
    }
    box.append(field, list);
    row.appendChild(box);
    // mirrored from `openPicker`: a picker hanging past the window turns around.
    if (box.getBoundingClientRect().right > innerWidth) box.style.right = "0";
    return box;
  };
"""


def test_the_picker_is_readable_and_clickable(browser_css):
    """**the whole picker is on screen and every label in it can be clicked.**

    the picker hangs off the bottom of the card, over the cards below it, and that has
    three ways to fail quietly: the card it belongs to clips it to its own border box,
    the cards after it in the dom paint over it, or the list is cut short and scrolled
    with no hint there is more. each leaves the field and the top few labels fine, so
    the picker looks like it works right up to the label somebody wanted.

    the ten labels are what `LABELS_OFFERED` promises to show without scrolling, and
    the tile is one with cards under it, since that is where covering would happen.
    """
    got = render(
        browser_css,
        GRID % (TILE * 30),
        OPEN
        + """
  const boxes = [...document.querySelectorAll("#grid figure")];
  const picker = open(boxes[12], ["sweeper", "go4", "other", "unclear", "bin", "trailer",
                                  "car", "van", "truck", "plant"]);
  const list = picker.querySelector(".options");

  const options = [...picker.querySelectorAll(".option")];
  result = {
    clipped: !hittable(picker.querySelector("input")),
    options: options.map((o) => ({ text: o.textContent, hit: hittable(o), box: rect(o) })),
    field: rect(picker.querySelector("input")),
    list: rect(list),
    inner: { w: innerWidth, h: innerHeight },
    sideways: document.documentElement.scrollWidth,
  };
""",
    )

    assert not got["clipped"], "the picker's field is not reachable"
    missed = [o["text"] for o in got["options"] if not o["hit"]]
    assert not missed, f"labels nobody can click: {missed}"

    list = got["list"]
    assert list["scrollH"] <= list["clientH"] + 1, "the list scrolls"
    assert list["bottom"] > got["field"]["bottom"], "the list did not hang below the card"
    assert got["sideways"] <= got["inner"]["w"] + 1, "sideways scroll"


def test_the_picker_stays_inside_the_window_it_hangs_off(browser_css):
    """**a picker opened on the last column still fits across.** the card is what the
    picker hangs from, so on the right of the grid it would hang off the side of the
    window: the labels past the edge are unreachable and the page gains a sideways
    scrollbar, which moves the tile being labelled out of sight. the picker turns
    around and hangs from the card's other edge instead.
    """
    # a window where the last column's picker is wider than the room beside it.
    got = render(
        browser_css,
        GRID % (TILE * 30),
        OPEN
        + """
  const boxes = [...document.querySelectorAll("#grid figure")];
  const rightmost = (el) => el.getBoundingClientRect().right;
  const picked = boxes.reduce((a, b) => (rightmost(b) > rightmost(a) ? b : a));
  const picker = open(picked, ["sweeper", "go4", "other", "unclear", "bin", "trailer",
                               "car", "van", "truck", "plant"]);
  const options = [...picker.querySelectorAll(".option")];
  result = {
    sideways: document.documentElement.scrollWidth,
    inner: { w: innerWidth, h: innerHeight },
    missed: options.filter((o) => !hittable(o)).map((o) => o.textContent),
    edge: Math.max(...options.map((o) => o.getBoundingClientRect().right)),
  };
""",
        width=900,
        height=700,
    )

    assert not got["missed"], f"labels nobody can click: {got['missed']}"
    assert got["sideways"] <= got["inner"]["w"] + 1, "sideways scroll"
    assert got["edge"] <= got["inner"]["w"] + 1, got["edge"]


def test_the_mirror_matches_the_page():
    """the page builds this markup in javascript, so the copy above is a copy. drift
    here means the layout test above measured a document nobody renders."""
    src = PAGE.read_text()
    assert 'box.className = "picker"' in src
    assert 'list.className = "options"' in src
    assert 'option.className = i === at ? "option on" : "option"' in src
    assert "row.appendChild(box)" in src
    assert 'if (box.getBoundingClientRect().right > innerWidth) box.style.right = "0"' in src
    assert re.search(r"figure \{ position: relative; \}", src)
