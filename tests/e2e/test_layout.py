"""the two layout rules of the page, measured in a real browser.

the node harness that covers the rest of the page has no layout: an element's width
and height there are whatever the test last wrote, which is exactly how the live stream
ended up filling the window on a phone and not on a desktop window of the same height,
and how a crop that opens "full screen" could be half width and five screens tall.
neither is visible without boxes, so nothing else catches them.
"""

import pytest

from browser import has_browser, render

pytestmark = pytest.mark.skipif(not has_browser(), reason="no browser to measure layout in")

# the live view: the header, the roi tools, the stage, the controls and the legend,
# which are the parts of the window a formula has to leave room for.
LIVE = """
<body class="grid">
  <div id="hdr"><div id="brand">cam one, and a second camera</div><div id="tabs"></div></div>
  <div class="view live active" id="live">
    <div id="roi-tools"><button></button></div>
    <div id="stage"><img><canvas></canvas></div>
    <div id="live-controls"></div>
    <div id="legend">a crop of a subject, quite long</div>
  </div>
</body>
"""

# the crops grid with a crop open over it.
GRID = """
<body class="grid">
  <div id="hdr"><div id="brand">cam one, and a second camera</div><div id="tabs"></div></div>
  <div class="view crops on" id="crops">
    <div class="bar"><input id="q"></div>
    <div id="grid">%s</div>
  </div>
  <div id="zoom" class="on"><img></div>
</body>
"""

# what the page writes into the stage once it knows the stream's encoded size.
SETUP = """
  const stage = document.getElementById("stage");
  if (stage) {
    stage.style.setProperty("--frame-w", 1600);
    stage.style.setProperty("--frame-h", 900);
  }
"""

MEASURE = """
  const box = (sel) => {
    const el = document.getElementById(sel) || document.querySelector(sel);
    return el ? rect(el) : null;
  };
  result = {
    stage: box("stage"), legend: box("legend"), crops: box("crops"), zoom: box("zoom"),
    img: box("#zoom img"),
    inner: { w: innerWidth, h: innerHeight },
    sideways: document.documentElement.scrollWidth,
    pageScrolls: document.documentElement.scrollHeight > innerHeight,
  };
"""


def measure(css: str, body: str, width: int, height: int) -> dict:
    return render(css, body, f"{SETUP}\n{MEASURE}", width, height)


def test_the_live_picture_fills_the_window_it_has(browser_css):
    """**the picture is as large as its window allows, in the shape the stream is.**
    on a desktop window it is width-limited; on a phone in portrait the width is spent
    and the height is what is left. a formula that subtracts chrome it only guesses at
    is wrong on exactly the viewport where it matters most, because that is where the
    header and the legend wrap to a second line.

    headless chrome refuses a window narrower than 500 css pixels, so the narrow cases
    below are narrow rather than a particular phone, and the rules that key on `pointer:
    coarse` are out of reach from here entirely -- what this does measure is a box that
    is tall for its width, which is the shape that used to get the picture wrong.
    """
    for w, h in ((1440, 900), (1280, 1024), (500, 900), (900, 420)):
        got = measure(browser_css, LIVE, w, h)
        stage = got["stage"]
        assert abs(stage["w"] / stage["h"] - 1600 / 900) < 0.02, (w, h, stage)
        assert stage["right"] <= got["inner"]["w"] + 1, (w, h, stage)
        room = stage["bottom"] + got["legend"]["h"]
        assert room <= got["inner"]["h"] + 1, (w, h, stage, got["legend"])
        assert got["sideways"] <= got["inner"]["w"] + 1, (w, h, "sideways")


def test_a_crop_opens_at_the_size_of_the_screen(browser_css):
    """**a crop fills the window, and the page -- not a pane -- is what scrolls.** the
    crop is contained in the window it is given, so its box is the window and the wheel
    has something to go closer into. the grid keeps the page as its scroller, because
    the lazy loading works by an element scrolling into view."""
    # tall enough that the page has to scroll if the pane is not going to.
    tiles = "".join("<div class='t' style='height:8rem'></div>" for _ in range(40))
    for w, h in ((1440, 900), (500, 900), (900, 420)):
        got = measure(browser_css, GRID % tiles, w, h)
        img = got["img"]
        # the crop's box *is* the window: `object-fit: contain` puts the picture inside
        # it and the zoom transform scales that picture, so at fill there is nothing
        # left for the wheel to go closer into.
        assert abs(img["w"] - got["inner"]["w"]) <= 1, (w, h, img)
        assert abs(img["h"] - got["inner"]["h"]) <= 1, (w, h, img)
        crops = got["crops"]
        assert crops["scrollH"] == crops["clientH"], (w, h, "the pane scrolls")
        assert got["pageScrolls"], (w, h, "the page should be the scroller")


# the live stage with the notice a closed stream is shown under. the picture is
# still in the dom behind it: the point of the notice is what is *seen*, not that
# the media was torn down -- taking the element down is a different code path, and
# the one the page already learned not to walk (see `closeVideo`).
SHUT = """
<body class="grid">
  <div id="hdr"><div id="brand">cam one</div><div id="tabs"></div></div>
  <div class="view live active" id="live">
    <div id="stage">
      <img><canvas></canvas>
      <div id="stream-shut"%s><b>stream closed</b><span id="stream-msg">until 07:00 local</span></div>
    </div>
    <div id="live-controls"></div>
    <div id="legend">a crop of a subject, quite long</div>
  </div>
</body>
"""

COVER = """
  const shut = document.getElementById("stream-shut");
  result = {
    stage: rect(document.getElementById("stage")),
    shut: rect(shut),
    seen: hittable(shut),
    sideways: document.documentElement.scrollWidth,
    inner: { w: innerWidth, h: innerHeight },
  };
"""

# where a click over the middle of the stage lands while the notice is hidden.
GONE = """
  const box = stage.getBoundingClientRect();
  const over = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
  const box2 = document.getElementById("stream-shut").getBoundingClientRect();
  result = {
    paints: box2.width > 0 || box2.height > 0,
    over: over ? over.tagName.toLowerCase() : null,
  };
"""


def test_a_closed_stream_covers_the_picture_it_replaces(browser_css):
    """**the frozen frame is the bug and the notice is the fix, so the notice has
    to be what the window shows.** a message in the corner of a still picture
    reads as a caption on that picture, and a caption on yesterday's frame is the
    same wrong claim in smaller type.

    the notice covers the stage rather than replacing it, which is also what keeps
    the media elements where the video plumbing expects them to be.
    """
    for w, h in ((1440, 900), (500, 900), (900, 420)):
        got = render(browser_css, SHUT % "", f"{SETUP}\n{COVER}", w, h)
        stage, shut = got["stage"], got["shut"]
        assert abs(shut["w"] - stage["w"]) <= 1, (w, h, stage, shut)
        assert abs(shut["h"] - stage["h"]) <= 1, (w, h, stage, shut)
        assert got["seen"], (w, h, "the notice is not what the window shows")
        assert abs(got["sideways"] - got["inner"]["w"]) <= 1, (w, h, "sideways")


def test_a_closed_notice_that_is_hidden_paints_nothing(browser_css):
    """**a rule about what the notice looks like must not outvote the rule about
    whether it is there.** the notice is a grid and the page hides things with
    `hidden`, which is `display: none` -- and a `display` on the element wins over
    that unless the hidden rule is stronger, which is exactly the mistake that
    leaves a permanent "stream closed" over a working stream.

    the stream being open is the common case, so this is the case that matters.
    """
    for w, h in ((1440, 900), (500, 900)):
        got = render(browser_css, SHUT % " hidden", f"{SETUP}\n{GONE}", w, h)
        assert not got["paints"], (w, h, got, "a hidden notice still has a box")
        assert got["over"] == "img", (w, h, got, "the notice is over the picture")


# the bar, over a grid long enough that the page has to scroll.
BAR = """
<body>
  <header>
    <h1>metermate</h1>
    <span class="tabs"><button class="on">live</button><button>crops</button></span>
    <span id="stats">12/s</span>
  </header>
  <div id="crops" class="on"><div id="grid" class="crop-grid">%s</div></div>
</body>
"""

# measured from the middle of a long grid, which is where the bar used to be left.
SCROLLED = """
  scrollTo(0, 600);
  const bar = document.querySelector("header");
  result = {
    bar: rect(bar),
    scrolled: scrollY,
    reachable: hittable(bar),
    inner: { w: innerWidth, h: innerHeight },
  };
"""


def test_the_top_bar_stays_in_sight(browser_css):
    """**the tabs are the only way between views, so they do not scroll away.** a
    crops grid and the events list are each longer than any window, and the bar that
    switches between them went off the top with the rest of the page -- so reaching
    the events tab from a crop halfway down meant scrolling to the top first, and
    where you had been was not recoverable except by scrolling back."""
    tiles = "".join("<figure style='height:10rem'></figure>" for _ in range(40))
    for w, h in ((1440, 900), (500, 900), (900, 420)):
        got = render(browser_css, BAR % tiles, SCROLLED, w, h)
        assert got["scrolled"] > 100, (w, h, got["scrolled"], "the page never scrolled")
        assert abs(got["bar"]["y"]) <= 1, (w, h, got["bar"])
        assert got["bar"]["h"] > 10, (w, h, got["bar"])
        # not merely at the top of a box of its own: a bar the cards paint over is a
        # bar whose buttons do not work.
        assert got["reachable"], (w, h, "something scrolls over the bar")
        assert abs(got["bar"]["w"] - got["inner"]["w"]) <= 1, (w, h, got["bar"])
