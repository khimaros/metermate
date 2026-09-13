// run a served page's own javascript against a running server.
//
// the two pages are the interface for the most valuable work in the project and
// nothing else exercises them, so a tab that draws the wrong pool is a defect
// only a person clicking around would ever find. node brings fetch, so a page
// can load its queue over http from the real binary; the only thing missing is
// a document, and the rules worth testing -- which crops belong on which tab --
// are written so they never touch one.
//
//   node page.mjs http://127.0.0.1:PORT/ '<expression>'
//
// the expression is evaluated in the page's own scope and printed as json, so
// the assertions stay in the test rather than in here.

const [url, expr] = process.argv.slice(2);

// out of whatever tree it was in. the grids move cards around rather than
// rebuilding them, so a child has to leave one parent as it joins the next or
// it is drawn in both.
const detach = (child) => {
  const kids = child.parent && child.parent.children;
  const at = kids ? kids.indexOf(child) : -1;
  if (at >= 0) kids.splice(at, 1);
  child.parent = null;
};

// enough of an element to absorb a render. what the page decided is mostly read
// from its own variables instead, which is the part a browser would not tell us
// anyway; the tree is kept so a control a render built can be found and clicked.
const element = () => {
  const node = {
    textContent: "",
    className: "",
    title: "",
    value: "",
    src: "",
    checked: false,
    disabled: false,
    hidden: false,
    width: 0,
    height: 0,
    // the geometry the page measures to size the picture: a stub element is at
    // home (a hidden one reports no offsetParent, and a test can say so), and the
    // heights are whatever the test needs them to be.
    offsetTop: 0,
    offsetHeight: 0,
    offsetParent: {},
    // a media element with nothing buffered: `canSeek` asks, and an element that
    // cannot answer is a page that throws rather than waits.
    seekable: { length: 0, start: () => 0, end: () => 0 },
    // custom properties are kept too: the page writes the shape of the picture
    // here, and nothing can read back what a stub dropped.
    style: {
      cssText: "",
      setProperty(name, value) {
        this[name] = String(value);
      },
      getPropertyValue(name) {
        return this[name] ?? "";
      },
    },
    dataset: {},
    children: [],
    parent: null,
    // tracked rather than swallowed: "is the overlay still open" is a class,
    // and a stub that always answers no cannot tell a test that it closed.
    classList: {
        held: new Set(),
        add(name) {
          this.held.add(name);
        },
        remove(name) {
          this.held.delete(name);
        },
        toggle(name, on) {
          const want = on === undefined ? !this.held.has(name) : on;
          if (want) this.held.add(name);
          else this.held.delete(name);
        },
        contains(name) {
          return this.held.has(name);
        },
      },
    querySelectorAll: () => [],
    // enough of a selector engine for what the pages ask: `.class` searched
    // through the children this stub keeps, and anything else -- a tag name --
    // answered with a fresh element, since the page only ever holds onto those.
    //
    // **`null` when a class is not there matters.** a page that asks "is the
    // picker already open" and is always told yes never opens one, and the
    // test then measures a stub rather than the page.
    querySelector(selector) {
      if (!selector.startsWith(".")) return element();
      const want = selector.slice(1);
      const find = (e) =>
        e.className === want ? e : (e.children || []).map(find).find(Boolean);
      return node.children.map(find).find(Boolean) || null;
    },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: node.width, height: node.height }),
    // drawn ops are recorded by name, because "was a box drawn at all" is a
    // question a test asks of a view that can be told to draw nothing.
    drawn: [],
    getContext: () => {
      const api = { measureText: () => ({ width: 10 }) };
      return new Proxy(api, {
        get: (t, k) =>
          k in t ? t[k] : (...args) => node.drawn.push(String(k)),
      });
    },
    appendChild(child) {
      detach(child);
      child.parent = node;
      node.children.push(child);
      return child;
    },
    append(...kids) {
      kids.forEach((kid) => node.appendChild(kid));
    },
    insertBefore(child, before) {
      detach(child);
      child.parent = node;
      const at = before ? node.children.indexOf(before) : -1;
      if (at < 0) node.children.push(child);
      else node.children.splice(at, 0, child);
      return child;
    },
    removeChild(child) {
      detach(child);
      return child;
    },
    remove() {
      detach(node);
    },
    // what a textarea offers the copy button.
    select() {
      globalThis.__selected = node.value;
    },
    play: () => Promise.resolve(),
    pause() {},
    load() {},
    focus() {},
    blur() {},
    attrs: {},
    getAttribute: (name) => (name in node.attrs ? node.attrs[name] : null),
    setAttribute(name, value) {
      node.attrs[name] = String(value);
    },
    removeAttribute(name) {
      delete node.attrs[name];
    },
    setPointerCapture() {},
    scrollIntoView() {},
    addEventListener() {},
  };
  // assigning innerHTML replaces an element's contents, so the stub drops the
  // children it held -- otherwise a second render finds the first one's tiles.
  let html = "";
  Object.defineProperty(node, "innerHTML", {
    get: () => html,
    set: (value) => {
      html = value;
      node.children = [];
    },
  });
  return node;
};

// elements are remembered by id rather than minted fresh per lookup, so state
// the page writes to one -- whether the train report is open, say -- is still
// there when it reads it back.
const byId = new Map();

// the preview page is a live view, so it reaches for the apis that make one:
// lazy loading below the fold, a frame clock, a detection stream and streamed
// video. none of them is what a test asks about, and all of them have to exist
// for the script to finish parsing.
class Nothing {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
  close() {}
  addEventListener() {}
  endOfStream() {}
}
globalThis.IntersectionObserver = Nothing;
globalThis.ResizeObserver = Nothing;
globalThis.EventSource = Nothing;
globalThis.MediaSource = Nothing;
globalThis.MediaSource.isTypeSupported = () => false;
globalThis.requestAnimationFrame = () => 0;
URL.createObjectURL = () => "blob:stub";
URL.revokeObjectURL = () => {};

globalThis.document = {
  onkeydown: null,
  hidden: false,
  getElementById: (id) => {
    if (!byId.has(id)) byId.set(id, element());
    return byId.get(id);
  },
  createElement: element,
  querySelectorAll: () => [],
  querySelector: () => null,
  addEventListener() {},
  body: element(),
  documentElement: element(),
  // the page copies through a hidden textarea rather than `navigator.clipboard`,
  // which is undefined over plain http. what landed on the clipboard is read
  // back from here.
  execCommand: (command) => {
    if (command !== "copy") {
      return false;
    }
    globalThis.__copied = globalThis.__selected;
    return true;
  },
};

// **a window with a location the page can navigate.** the preview's tabs and its
// hash links are the same navigation, and a stub whose hash never changes cannot
// tell a wired-up route from one nobody listens to: assigning a hash fires the
// event a browser would, so a test follows a link rather than calling the
// function a link calls.
let current = new URL(url).hash;
globalThis.location = {
  get hash() {
    return current;
  },
  set hash(next) {
    if (next === current) return;
    current = next;
    if (globalThis.window.onhashchange) globalThis.window.onhashchange();
  },
};

// and the other way the url moves: `replaceState` changes it without firing
// anything, which is how the page names whatever it just opened.
globalThis.history = {
  replaceState: (title, ignored, hash) => {
    current = String(hash);
  },
};

// `scrollY` and `scrollTo` are the labelling page's, and the resize listener the
// preview installs has nothing to be resized here. `innerWidth` is the width a zoom
// is measured against, and tests that say how big a crop is are written against it.
globalThis.innerWidth = 1440;
globalThis.window = { scrollY: 0, scrollTo: () => {}, addEventListener: () => {} };

// the page asks for "/queue"; give those the origin the server is actually on.
const net = globalThis.fetch;
const origin = new URL(url).origin;
globalThis.fetch = (path, opts) => net(String(path).startsWith("http") ? path : origin + path, opts);

const page = await (await net(url)).text();
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];

// the page calls load() as it parses, so give that round trip a moment to
// land before reading anything out of it.
const run = new Function(
  "EXPR",
  script + "\n;return (async () => { await new Promise(r => setTimeout(r, 200)); return eval(EXPR); })();",
);
console.log(JSON.stringify(await run(expr)));
// the preview refreshes its grids on a timer, and node will not exit while one
// is pending -- so the answer is printed and the process is done with.
process.exit(0);
