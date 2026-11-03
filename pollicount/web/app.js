"use strict";

/* Analysis stages, in order, with the share of the server progress (0 to 1) each one covers. */
const STAGES = [
  { key: "upload", label: "Uploading your video" },
  { key: "preparing", label: "Preparing the video", span: [0, 0.1] },
  { key: "tracking insects", label: "Looking for insects", span: [0.1, 0.4] },
  { key: "tracking flowers", label: "Finding the flowers", span: [0.4, 0.7] },
  { key: "finding visits", label: "Counting landings", span: [0.7, 0.75] },
  { key: "identifying insects", label: "Identifying each visitor", span: [0.75, 0.95] },
];

const GROUPS = {
  bumblebees: {
    name: "Bumblebees",
    one: "Bumblebee",
    about: "Large, round and furry bees, often banded with yellow and a white or buff tail.",
    traits: ["Very hairy, stocky body", "Two pairs of wings", "Elbowed antennae", "Black with yellow bands and a pale tail"],
  },
  honeybees: {
    name: "Honeybees",
    one: "Honeybee",
    about: "Slim, brown-banded bees from a hive. Workers carry pollen in baskets on their hind legs.",
    traits: ["Amber and brown banded abdomen", "Hairy thorax, less fluffy than a bumblebee", "Elbowed antennae", "Pollen baskets on the hind legs"],
  },
  "solitary bees": {
    name: "Solitary bees",
    one: "Solitary bee",
    about: "Bees that nest alone, in the ground or in hollow stems, from tiny and dark to bright ginger.",
    traits: ["Two pairs of wings", "Elbowed antennae", "Often smaller than a honeybee", "Colours from black to ginger"],
  },
  wasps: {
    name: "Wasps",
    one: "Wasp",
    about: "Hunters that also drink nectar. Smooth and shiny, with a narrow waist.",
    traits: ["Narrow waist", "Almost hairless, yellow and black", "Two pairs of wings, folded lengthwise at rest", "Elbowed antennae"],
  },
  hoverflies: {
    name: "Hoverflies",
    one: "Hoverfly",
    about: "Flies that dress up as bees and wasps. Some are slender and dark rather than striped.",
    traits: ["One pair of wings, where bees and wasps have two", "Very large eyes", "Short antennae, no narrow waist", "Often hovers still in the air before landing"],
  },
  "other flies": {
    name: "Other flies",
    one: "Fly",
    about: "Flies of all kinds, from house flies to metallic greenbottles.",
    traits: ["One pair of wings", "Large eyes", "Short antennae with a bristle", "Often bristly, sometimes metallic"],
  },
  "butterflies & moths": {
    name: "Butterflies and moths",
    one: "Butterfly or moth",
    about: "Scaled wings and a long tongue that uncoils to reach nectar.",
    traits: ["Two pairs of scaled wings", "Coiled tongue", "Clubbed antennae in butterflies, feathery or thread-like in moths"],
  },
  beetles: {
    name: "Beetles",
    one: "Beetle",
    about: "Hard wing cases that meet in a straight line down the back. Many come for the pollen.",
    traits: ["Hard wing cases over the body", "Wing cases meet in a straight line", "Chewing mouthparts"],
  },
  "small insects": {
    name: "Small insects",
    one: "Small insect",
    about: "Anything under 3 mm, such as pollen beetles and thrips, too small to sort further on a video.",
    traits: ["Shorter than 3 mm", "Often dark and shiny", "Easy to miss: look for moving dots"],
  },
  "other insects": {
    name: "Other insects",
    one: "Insect",
    about: "Every other visitor: bugs, ants, sawflies and more.",
    traits: ["Shield bugs carry a triangle on the back", "Ants have a narrow waist and often no wings", "Sawflies look like wasps without the narrow waist"],
  },
};


const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
let result = null;
let tracks = [];
let lastTime = -1;
let currentId = null;
let visitors = new Map();
let far = new Set();
let hoverId = null;
let spot = null;
/** Insect ids to stand out on the video, from the row selected in the list (null: all equal). */
/** Card shown when visits come from insects found by their movement, too small for SAM to outline. */
function farCard(all) {
  const card = el("div", { className: "far" });
  card.append(
    el("p", { className: "far-tag" }, svg('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/></svg>'), el("span", { textContent: "Seen from afar" })),
    el("p", { textContent: `${all ? "Insects" : "Some insects"} found by their movement: visits are counted, and sorted into groups like bees or hoverflies.` }),
    el("p", { className: "far-missing", textContent: "Not shown from this far: how many individual insects there were, and their time on flowers." }),
  );
  return card;
}
let rows = [];

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const svg = (markup) => {
  const span = document.createElement("span");
  span.innerHTML = markup;
  return span.firstElementChild;
};
const timecode = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const plural = (n, word) => `${n} ${n === 1 ? word : `${word}s`}`;
const seconds = (s) => `${s.toFixed(1)} s`;
const groupInfo = (group) => GROUPS[group] ?? { name: "Unidentified insects", one: "Insect", about: "", traits: [] };
const plateFor = (group) => `plates/${group.replace(" & ", "-").replace(/\s+/g, "-")}.webp`;
const videoDuration = () => (result ? result.video.frames / result.video.fps : video.duration || 0);
const CHEVRON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>';
const CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

function show(screen) {
  $("start").hidden = screen !== "start";
  $("footer").hidden = screen !== "start";
  $("index").hidden = screen !== "start";
  $("analysis").hidden = screen !== "analysis";
  $("side").hidden = screen !== "analysis";
  $("back").hidden = screen !== "analysis";
}

function fail(message) {
  $("progress").hidden = true;
  $("error-text").textContent = message;
  $("analysis-error").hidden = false;
}

/* Upload and progress */

const progress = { started: 0, stage: 0 };

function renderStages(index, fraction) {
  const list = $("stages");
  list.replaceChildren();
  STAGES.forEach((stage, i) => {
    const state = i < index ? "done" : i === index ? "now" : "";
    const item = el("li", { className: state }, el("span", { className: "dot" }), el("span", { textContent: stage.label }));
    if (state === "done") item.firstChild.append(svg(CHECK));
    if (state === "now") {
      const pct = Math.round(fraction * 100);
      const bar = el("span", { className: "bar" }, el("span"));
      bar.firstChild.style.width = `${pct}%`;
      item.append(el("span", { className: "pct", textContent: `${pct}%` }), bar);
    }
    list.append(item);
  });
}

function setStage(key, value) {
  const index = Math.max(1, STAGES.findIndex((s) => s.key === key));
  const [from, to] = STAGES[index].span;
  renderStages(index, Math.min(1, Math.max(0, (value - from) / (to - from))));
  if (key === "queued") $("stages").children[1].children[1].textContent = "Waiting for a free slot";
  if (!progress.started && value > 0) progress.started = performance.now() - 1;
  if (progress.started && value >= 0.15) {
    const elapsed = (performance.now() - progress.started) / 1000;
    const left = (elapsed * (1 - value)) / value;
    $("eta").textContent = left < 60 ? "< 1 min" : `≈ ${Math.ceil(left / 60)} min`;
  }
}

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "api/analyses");
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) renderStages(0, e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 202) resolve(xhr.response);
      else reject(new Error(xhr.response?.detail || "The upload was refused."));
    };
    xhr.onerror = () => reject(new Error("The upload stopped. Check your connection and try again."));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

async function follow(id) {
  for (;;) {
    const response = await fetch(`api/analyses/${id}`);
    if (response.status === 404) throw new Error("This analysis is no longer available: analyses are kept for 24 hours. Upload the video again.");
    const job = await response.json();
    if (job.status === "failed") throw new Error(job.error);
    if (job.status === "done") return job;
    setStage(job.stage, job.progress);
    await new Promise((r) => setTimeout(r, 700));
  }
}

function waitingScreen(title) {
  show("analysis");
  $("where").textContent = title;
  renderStages(0, 0);
  renderGhost();
}

/** Clears the previous result, when a new video is started from the results page. */
function reset() {
  result = null;
  tracks = [];
  rows = [];
  spot = null;
  hoverId = null;
  visitors = new Map();
  far = new Set();
  progress.started = 0;
  for (const id of ["results", "analysis-error", "download", "credit"]) $(id).hidden = true;
  $("progress").hidden = false;
  $("eta").textContent = "estimating";
}

async function start(file) {
  if (!file.type.startsWith("video/")) {
    if (!$("analysis").hidden) return fail("This file is not a video. Pick a video of a flower patch.");
    $("start-error").textContent = "This file is not a video. Pick a video of a flower patch.";
    $("start-error").hidden = false;
    return;
  }
  reset();
  history.replaceState(null, "", "./");
  waitingScreen("Your video");
  video.src = URL.createObjectURL(file);
  video.play().catch(() => {});
  try {
    const job = await upload(file);
    history.replaceState(null, "", `?a=${job.id}`);
    await follow(job.id);
    await showResult(job.id);
  } catch (error) {
    fail(error.message);
  }
}

async function resume(id) {
  waitingScreen("Your video");
  renderStages(1, 0);
  try {
    const job = await follow(id);
    if (job.example) showExample(job.example);
    await showResult(id);
  } catch (error) {
    fail(error.message);
  }
}

/* Examples */

async function listExamples() {
  const response = await fetch("api/examples").catch(() => null);
  const examples = response?.ok ? await response.json() : [];
  if (!examples.length) return;
  const list = $("examples-list");
  for (const example of examples) {
    const poster = el("img", { src: `api/examples/${example.id}/poster`, alt: "", loading: "lazy" });
    const credit = `${example.credit.replace(", Wikimedia Commons", "")}, ${example.license}`;
    const text = el("span", {}, el("strong", { textContent: example.title }), el("small", { textContent: credit }));
    const open = el("span", { className: "open" });
    open.dataset.tip = "Open this example";
    open.append(svg('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/></svg>'));
    list.append(el("li", {}, el("a", { className: "example", href: `?a=${example.id}` }, poster, text, open)));
  }
  $("examples").hidden = false;
}

/* New video menu, on the results page */

let panelFilled = false;
async function fillPanel() {
  if (panelFilled) return;
  panelFilled = true;
  const response = await fetch("api/examples").catch(() => null);
  const examples = response?.ok ? await response.json() : [];
  const list = $("panel-examples");
  for (const example of examples) {
    const poster = el("img", { src: `api/examples/${example.id}/poster`, alt: "", loading: "lazy" });
    list.append(el("li", {}, el("a", { href: `?a=${example.id}` }, poster, el("span", { textContent: example.title }))));
  }
  list.previousElementSibling.hidden = !examples.length;
}

function setPanel(open) {
  $("new-panel").hidden = !open;
  $("new").setAttribute("aria-expanded", String(open));
  if (open) fillPanel();
}

$("new").addEventListener("click", (e) => {
  e.stopPropagation();
  setPanel($("new-panel").hidden);
});
document.addEventListener("click", (e) => {
  if (!$("new-panel").hidden && !e.target.closest(".menu")) setPanel(false);
}, true);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("new-panel").hidden) {
    setPanel(false);
    $("new").focus();
    e.stopImmediatePropagation();
  }
}, true);

function showExample(example) {
  $("where").textContent = example.title;
  const credit = $("credit");
  credit.replaceChildren(`Video: ${example.credit.replace(", Wikimedia Commons", "")}, `);
  credit.append(el("a", { href: example.source, textContent: example.license }));
  credit.hidden = false;
}

/* Results */

async function showResult(id) {
  currentId = id;
  result = await (await fetch(`api/analyses/${id}/result`)).json();
  tracks = [
    ...result.flowers.map((t) => ({ ...t, kind: "flower", byFrame: index(t.boxes) })),
    ...result.insects.map((t) => ({ ...t, kind: "insect", byFrame: index(t.boxes) })),
  ];
  video.src = `api/analyses/${id}/video`;
  video.play().catch(() => {});

  const { record } = result;
  far = new Set(result.insects.filter((i) => i.from_motion).map((i) => i.id));
  visitors = visitorsOf(record);
  $("progress").hidden = true;
  $("results").hidden = false;
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
  $("download").href = URL.createObjectURL(blob);
  $("download").hidden = false;
  renderLedger(record);
  select(rows.find((r) => `#${r.id}` === location.hash) ?? rows[0], false);
}

function index(boxes) {
  const map = new Map();
  for (const [frame, ...box] of boxes) map.set(frame, box);
  return map;
}

/** Each insect seen close up that visited a flower, named "Hoverfly 1", "Hoverfly 2"... in order of arrival.
 * Insects found from their movement are left out: from afar, one insect leaving and coming back looks like two. */
function visitorsOf(record) {
  const found = new Map();
  const perGroup = {};
  for (const visit of [...record.visits].sort((a, b) => a.start_s - b.start_s)) {
    if (far.has(visit.insect)) continue;
    if (!found.has(visit.insect)) {
      perGroup[visit.group] = (perGroup[visit.group] ?? 0) + 1;
      const name = `${groupInfo(visit.group).one} ${perGroup[visit.group]}`;
      found.set(visit.insect, { id: visit.insect, name, group: visit.group, visits: [] });
    }
    found.get(visit.insect).visits.push(visit);
  }
  return found;
}

/* Ledger: a tree of rows, all visits > groups > individuals, each with its own timeline. */

function makeRow({ id, level, name, visits, parent, open, thumb, kind, data }) {
  const duration = videoDuration() || 1;
  const row = el("div", { className: `row${level === 1 ? " all" : ""}`, id, tabIndex: -1 });
  row.setAttribute("role", "treeitem");
  row.setAttribute("aria-level", String(level));
  row.setAttribute("aria-selected", "false");
  row.style.setProperty("--level", level);
  if (parent) row.dataset.parent = parent;
  Object.assign(row, { visits, kind, data });

  const label = el("span", { className: "label" });
  if (open !== undefined) {
    row.setAttribute("aria-expanded", String(open));
    const twisty = el("button", { className: "twisty", type: "button", tabIndex: -1 });
    twisty.setAttribute("aria-hidden", "true");
    twisty.append(svg(CHEVRON));
    label.append(twisty);
  } else {
    label.append(el("span", { className: "leaf" }));
  }
  if (thumb) {
    const image = el("span", { className: "thumb" });
    image.style.backgroundImage = `url(${thumb})`;
    label.append(image);
  }
  label.append(el("span", { className: "name", textContent: name }));

  const track = el("span", { className: "track" });
  for (const visit of visits) {
    const end = visit.start_s + visit.duration_s;
    const seg = el("button", { className: "seg", type: "button", tabIndex: -1 });
    seg.style.left = `${(visit.start_s / duration) * 100}%`;
    seg.style.width = `${(visit.duration_s / duration) * 100}%`;
    seg.dataset.span = `${timecode(visit.start_s)} - ${timecode(Math.round(end))}`;
    seg.dataset.stay = seconds(visit.duration_s);
    seg.dataset.start = visit.start_s;
    seg.setAttribute("aria-label", `Visit ${seg.dataset.span}, ${seg.dataset.stay}`);
    track.append(seg);
  }
  track.append(el("i", { className: "head" }));

  const count = el("span", { className: "count", textContent: String(visits.length) });
  count.dataset.tip = plural(visits.length, "visit");
  count.setAttribute("aria-label", count.dataset.tip);
  row.append(label, track, count);
  return row;
}

function renderLedger(record) {
  const ledger = $("ledger");
  ledger.classList.remove("ghost");
  ledger.replaceChildren();
  rows = [makeRow({ id: "row-all", level: 1, name: "All visits", visits: record.visits, open: true, kind: "all" })];
  const groups = Object.entries(record.counts).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  for (const [group] of groups) {
    const members = [...visitors.values()].filter((v) => v.group === group);
    const rowId = `row-${group.replace(/\W+/g, "-")}`;
    rows.push(makeRow({
      id: rowId, level: 2, name: groupInfo(group).name, parent: "row-all", open: members.length ? true : undefined, thumb: plateFor(group),
      visits: record.visits.filter((v) => v.group === group), kind: "group", data: { group, members },
    }));
    for (const visitor of members) {
      rows.push(makeRow({
        id: `row-insect-${visitor.id}`, level: 3, name: visitor.name, parent: rowId, visits: visitor.visits, kind: "insect", data: visitor,
      }));
    }
  }
  ledger.append(...rows);
  refresh();
}

function renderGhost() {
  const ledger = $("ledger");
  ledger.classList.add("ghost");
  ledger.replaceChildren(
    ...["All visits", "Insect groups"].map((name, i) => {
      const row = el("div", { className: "row" });
      row.style.setProperty("--level", i + 1);
      const skel = el("span", { className: "skel" });
      skel.style.animationDelay = `${i * 0.2}s`;
      row.append(
        el("span", { className: "label" }, el("span", { className: "leaf" }), el("span", { className: "name", textContent: name })),
        el("span", { className: "track" }, skel),
        el("span", { className: "count", textContent: "?" }),
      );
      return row;
    }),
  );
  ledger.setAttribute("aria-hidden", "true");
}

const parentOf = (row) => row.dataset.parent && $(row.dataset.parent);
const isOpen = (row) => row.getAttribute("aria-expanded") === "true";
const canOpen = (row) => row.hasAttribute("aria-expanded");
const selected = () => rows.find((r) => r.getAttribute("aria-selected") === "true");

function refresh() {
  for (const row of rows) {
    let visible = true;
    for (let p = parentOf(row); p; p = parentOf(p)) if (!isOpen(p)) visible = false;
    row.hidden = !visible;
  }
}

function setOpen(row, open) {
  if (!canOpen(row)) return;
  row.setAttribute("aria-expanded", String(open));
  refresh();
}

function select(row, focus = true) {
  if (!row) return;
  $("ledger").removeAttribute("aria-hidden");
  for (const r of rows) {
    r.setAttribute("aria-selected", String(r === row));
    r.tabIndex = r === row ? 0 : -1;
  }
  for (let p = parentOf(row); p; p = parentOf(p)) setOpen(p, true);
  const path = [];
  for (let r = row; r; r = parentOf(r)) path.unshift(r);
  $("crumbs").replaceChildren(
    ...path.flatMap((r, i) => {
      const name = el("span", { textContent: r.querySelector(".name").textContent });
      return i < path.length - 1 ? [name, svg(CHEVRON)] : [name];
    }),
  );
  spot = row.kind === "insect" ? new Set([row.data.id]) : row.kind === "group" ? new Set(row.visits.map((v) => v.insect)) : null;
  $("pane").replaceChildren(renderPane(row));
  draw(true);
  if (focus) row.focus();
}

$("ledger").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row || !rows.includes(row)) return;
  const seg = e.target.closest(".seg");
  if (seg) return seek(Number(seg.dataset.start));
  if (e.target.closest(".twisty")) return setOpen(row, !isOpen(row));
  const wasSelected = row.getAttribute("aria-selected") === "true";
  select(row);
  if (canOpen(row)) setOpen(row, wasSelected ? !isOpen(row) : true);
});

$("ledger").addEventListener("keydown", (e) => {
  const row = e.target.closest(".row");
  if (!row || !rows.includes(row)) return;
  const list = rows.filter((r) => !r.hidden);
  const i = list.indexOf(row);
  const move = {
    ArrowDown: () => list[i + 1] && select(list[i + 1]),
    ArrowUp: () => list[i - 1] && select(list[i - 1]),
    Home: () => select(list[0]),
    End: () => select(list[list.length - 1]),
    ArrowRight: () => canOpen(row) && (isOpen(row) ? list[i + 1] && select(list[i + 1]) : setOpen(row, true)),
    ArrowLeft: () => (canOpen(row) && isOpen(row) ? setOpen(row, false) : parentOf(row) && select(parentOf(row))),
    Enter: () => setOpen(row, !isOpen(row)),
  }[e.key];
  if (move) {
    e.preventDefault();
    move();
  }
});

/* Detail panes */

const onFlowers = (visits) => visits.reduce((sum, v) => sum + v.duration_s, 0);

/** Share of the video during which at least one insect is on a flower (overlapping visits merged). */
function busyShare(visits) {
  const spans = visits.map((v) => [v.start_s, v.start_s + v.duration_s]).sort((a, b) => a[0] - b[0]);
  let total = 0;
  let end = -Infinity;
  for (const [a, b] of spans) {
    if (b > end) {
      total += b - Math.max(a, end);
      end = b;
    }
  }
  return Math.min(1, total / (videoDuration() || 1));
}

function tally(items, className = "tally compact") {
  const list = el("dl", { className });
  for (const [term, value, note, lead, tip] of items) {
    const dt = el("dt", { textContent: term });
    if (tip) dt.dataset.tip = tip;
    const dd = el("dd", { textContent: value });
    if (note) dd.append(el("small", { textContent: note }));
    list.append(el("div", { className: lead ? "lead" : "" }, dt, dd));
  }
  return list;
}

function emptyMessage(record) {
  const insectsSeen = result.insects.length;
  if (insectsSeen > 0) return `${plural(insectsSeen, "insect")} passed by, but none stayed on a flower for at least 0.3 s.`;
  if (result.flowers.length > 0) return "No insect found in this video. Try a sunny, warm moment, when pollinators are active, and keep the phone still.";
  if (record.flowers_tracked === 0) return "No flowers or insects found. PolliCount needs a flower patch filling the frame: film it from 20 to 50 cm, phone held still.";
  return "No visit found in this video.";
}

function renderPane(row) {
  const pane = el("section", { className: "pane" });
  const { record } = result;
  if (row.kind === "all") {
    const time = onFlowers(record.visits);
    // Flower heads are not shown: measured on the Monash videos, their count runs about 50% high.
    const farVisits = record.visits.filter((v) => far.has(v.insect)).length;
    const items = [["Visits", String(record.total_visits), "", true]];
    if (!farVisits) {
      items.push(
        ["Pollinators", String(record.insects)],
        // The time is summed over insects; the share counts each moment once, so it never exceeds the video.
        ["Time on flowers", seconds(time), "all insects together"],
        ["Time with a visitor", `${Math.round(busyShare(record.visits) * 100)}%`, "of the video", false, "Share of the video with at least one insect on a flower"],
      );
    } else if (visitors.size) {
      // From afar, insects cannot be told apart and missed visits make the total time unreliable.
      items.push(["Pollinators seen close up", String(visitors.size)]);
    }
    pane.append(tally(items, "tally"));
    if (farVisits) {
      pane.append(farCard(farVisits === record.total_visits));
    }
    pane.append(el("p", { className: "hint", textContent: !record.total_visits ? emptyMessage(record) : visitors.size ? "Pick a group or an insect in the list to see its details." : "Pick a group in the list to see its details." }));
    return pane;
  }

  if (row.kind === "group") {
    const { group, members } = row.data;
    const info = groupInfo(group);
    const plate = el("div", { className: "lit" }, el("img", { src: plateFor(group), alt: `Engraved plate of ${info.name.toLowerCase()}` }));
    pane.append(
      el("div", { className: "specimen" }, plate, el("div", { className: "plaque" }, el("h2", { textContent: info.name }), el("p", { textContent: info.about }))),
      tally([
        ...(row.visits.some((v) => far.has(v.insect))
          ? members.length ? [["Seen close up", String(members.length)]] : []
          : [["In this video", String(members.length)]]),
        ["Visits", String(row.visits.length)],
        ["Average visit", seconds(onFlowers(row.visits) / row.visits.length)],
      ]),
    );
    if (row.visits.some((v) => far.has(v.insect))) pane.append(farCard(row.visits.every((v) => far.has(v.insect))));
    if (info.traits.length) {
      pane.append(el("h3", { textContent: "How to tell them apart" }), el("ul", { className: "traits" }, ...info.traits.map((t) => el("li", { textContent: t }))));
    }
    return pane;
  }

  const visitor = row.data;
  const insect = result.insects.find((i) => i.id === visitor.id);
  const heads = new Set(visitor.visits.map((v) => v.flower)).size;
  pane.append(
    el("h2", { className: "who", textContent: visitor.name }),
    tally([
      ["Flower heads visited", String(heads)],
      ["Time on flowers", seconds(onFlowers(visitor.visits))],
      ["Average visit", seconds(onFlowers(visitor.visits) / visitor.visits.length)],
    ]),
    el("h3", { textContent: "Identification" }),
  );
  const crops = el("div", { className: "crops" });
  for (const name of insect?.crops ?? []) {
    crops.append(el("img", { src: `api/analyses/${currentId}/crops/${name}`, alt: `${visitor.name}, close-up`, loading: "lazy" }));
  }
  if (crops.children.length) pane.append(crops);
  const confidence = insect?.confidence;
  const cues = (insect?.cues ?? []).filter((c) => c !== "no answer");
  const pct = el("span", { className: "verdict-pct", textContent: confidence == null ? "?" : `${Math.round(confidence * 100)}%` });
  pct.append(el("small", { textContent: "confidence" }));
  const verdict = el(
    "div",
    { className: "verdict" },
    el("div", { className: "verdict-head" }, el("span", { className: "verdict-name", textContent: confidence == null ? "Not identified" : groupInfo(visitor.group).one }), pct),
  );
  if (cues.length) verdict.append(el("ul", {}, ...cues.map((c) => el("li", { textContent: c }))));
  if (confidence == null || confidence < 0.5) verdict.append(el("p", { className: "check", textContent: "The model was unsure: check this one on the video." }));
  pane.append(verdict);
  return pane;
}

/* Overlay */

function contentRect() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const vw = result?.video.width || video.videoWidth || w;
  const vh = result?.video.height || video.videoHeight || h;
  const scale = Math.min(w / vw, h / vh);
  return { x: (w - vw * scale) / 2, y: (h - vh * scale) / 2, scale };
}

function boxAt(track, frame) {
  for (let d = 0; d <= 2; d++) {
    const box = track.byFrame.get(frame - d) || track.byFrame.get(frame + d);
    if (box) return box;
  }
  return null;
}

function activeVisit(insectId, time) {
  return result.record.visits.find((v) => v.insect === insectId && time >= v.start_s && time <= v.start_s + v.duration_s);
}

function draw() {
  const dpr = devicePixelRatio || 1;
  if (canvas.width !== canvas.clientWidth * dpr) {
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  const time = video.currentTime;
  if (time !== lastTime) {
    lastTime = time;
    const duration = video.duration || videoDuration() || 1;
    $("scrub").value = Math.round((time / duration) * 1000);
    $("scrub").style.setProperty("--played", `${(time / duration) * 100}%`);
    $("clock").textContent = `${timecode(time)} / ${timecode(video.duration || videoDuration() || 0)}`;
    const at = `${Math.min(100, (time / (videoDuration() || duration)) * 100)}%`;
    for (const head of document.querySelectorAll(".head")) head.style.left = at;
  }
  if (!result) return;

  const frame = Math.floor(time * result.video.fps);
  const { x, y, scale } = contentRect();
  const css = getComputedStyle(document.documentElement);
  const uv = css.getPropertyValue("--uv").trim();
  const pollen = css.getPropertyValue("--pollen").trim();

  const now = result.record.visits.filter((v) => time >= v.start_s && time <= v.start_s + v.duration_s);
  const hosts = new Set(now.map((v) => v.flower));
  const spotHosts = new Set(now.filter((v) => spot && spot.has(v.insect)).map((v) => v.flower));
  // The selected or hovered insect is drawn last, so its label sits on top of a neighbour's.
  const front = (t) => (t.id === hoverId || (spot && spot.has(t.id)) ? 1 : 0);
  if (spot) {
    const lit = tracks
      .filter((t) => t.kind === "insect" && spot.has(t.id))
      .map((t) => boxAt(t, frame))
      .filter(Boolean)
      .map((b) => b.map((v, i) => (i % 2 ? y + v * scale : x + v * scale)));
    spotlight(lit);
  }
  for (const track of [...tracks].sort((a, b) => front(a) - front(b))) {
    const box = boxAt(track, frame);
    if (!box) continue;
    const [x1, y1, x2, y2] = box.map((v, i) => (i % 2 ? y + v * scale : x + v * scale));
    if (track.kind === "flower") {
      // Only the flower an insect is visiting is outlined: it shows why a visit is counted, while outlines of
      // every detected flower (merged heads, false flowers) clutter the picture.
      if (!hosts.has(track.id)) continue;
      ctx.globalAlpha = spot && !spotHosts.has(track.id) ? 0.55 : 1;
      // A dark halo under the dashes keeps the outline readable on white petals and pale leaves.
      ctx.beginPath();
      ctx.ellipse((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2, 0, 0, Math.PI * 2);
      ctx.setLineDash([]);
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = "rgba(16, 32, 26, 0.75)";
      ctx.stroke();
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = uv;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      continue;
    }
    // Motion also picks up leaves in the wind: only the candidates confirmed as visitors are drawn.
    if (track.from_motion && !track.visiting) continue;
    const visiting = activeVisit(track.id, time);
    const picked = spot && spot.has(track.id);
    const hovered = hoverId === track.id;
    // The selected insect (or group) stands out; the others step back.
    ctx.globalAlpha = spot && !picked ? 0.55 : 1;
    drawTrail(track, frame, x, y, scale, pollen);
    ctx.globalAlpha = spot && !picked ? 0.55 : 1;
    if (picked || hovered) {
      ctx.fillStyle = "rgba(224, 185, 68, 0.14)";
      ctx.fillRect(x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6);
      ctx.shadowColor = "rgba(224, 185, 68, 0.8)";
      ctx.shadowBlur = 10;
    }
    ctx.strokeStyle = pollen;
    ctx.lineWidth = picked || hovered ? 1.5 : 1;
    ctx.strokeRect(x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6);
    ctx.lineWidth = (visiting ? 3.5 : 2.5) + (picked || hovered ? 1 : 0);
    const grow = hovered ? 3 : 0;
    corners(x1 - 3 - grow, y1 - 3 - grow, x2 + 3 + grow, y2 + 3 + grow);
    ctx.shadowBlur = 0;
    if (track.visiting) label(visitors.get(track.id)?.name ?? groupInfo(track.group).one, x1 - 3, y1 - 6, picked);
    ctx.globalAlpha = 1;
  }
}

function corners(x1, y1, x2, y2) {
  const k = Math.min(12, (x2 - x1) / 3, (y2 - y1) / 3);
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
    ctx.moveTo(cx + dx * k, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * k);
  }
  ctx.stroke();
}

function drawTrail(track, frame, x, y, scale, color) {
  ctx.beginPath();
  let started = false;
  for (let f = frame - 20; f <= frame; f++) {
    const box = track.byFrame.get(f);
    if (!box) continue;
    const px = x + ((box[0] + box[2]) / 2) * scale;
    const py = y + ((box[1] + box[3]) / 2) * scale;
    started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    started = true;
  }
  ctx.globalAlpha = 0.45;
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function label(text, x, y, lit = false) {
  ctx.font = "600 15px 'Cormorant Garamond', Georgia, serif";
  const w = ctx.measureText(text).width + 18;
  const top = Math.max(0, y - 24);
  ctx.fillStyle = lit ? "#c8a45a" : "rgba(16, 32, 26, 0.85)";
  ctx.fillRect(x, top, w, 24);
  ctx.strokeStyle = "#c8a45a";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, top + 0.5, w - 1, 23);
  ctx.fillStyle = lit ? "#10201a" : "#ece6d6";
  ctx.fillText(text, x + 9, top + 17);
}

/** Dims the picture outside the given boxes (with a margin), like a spotlight on the selected insects. */
const veil = document.createElement("canvas");
function spotlight(boxes) {
  if (!boxes.length) return;
  veil.width = canvas.width;
  veil.height = canvas.height;
  const v = veil.getContext("2d");
  v.setTransform(ctx.getTransform());
  v.fillStyle = "rgba(8, 16, 13, 0.3)";
  v.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  v.globalCompositeOperation = "destination-out";
  v.fillStyle = "#000";
  for (const [x1, y1, x2, y2] of boxes) {
    const m = Math.max(24, 0.4 * Math.max(x2 - x1, y2 - y1));
    v.beginPath();
    v.roundRect(x1 - m, y1 - m, x2 - x1 + 2 * m, y2 - y1 + 2 * m, m);
    v.fill();
  }
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(veil, 0, 0);
  ctx.restore();
}

function loop() {
  draw();
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

/* Player */

function seek(time) {
  const duration = video.duration || videoDuration();
  video.currentTime = Math.min(Math.max(0, time), Math.max(0, duration - 0.01));
  draw();
}

const togglePlay = () => (video.paused ? video.play().catch(() => {}) : video.pause());

function jump(direction) {
  const starts = (selected()?.visits ?? []).map((v) => v.start_s).sort((a, b) => a - b);
  if (!starts.length) return;
  const now = video.currentTime;
  const next = direction > 0 ? starts.find((t) => t > now + 0.05) ?? starts[0] : [...starts].reverse().find((t) => t < now - 0.5) ?? starts.at(-1);
  seek(next);
}

function toggleSpeed() {
  video.playbackRate = video.playbackRate === 1 ? 0.5 : 1;
  $("speed").textContent = video.playbackRate === 1 ? "1×" : "½×";
  $("speed").setAttribute("aria-pressed", String(video.playbackRate !== 1));
}

$("play").addEventListener("click", togglePlay);
$("prev").addEventListener("click", () => jump(-1));
$("next").addEventListener("click", () => jump(1));
$("speed").addEventListener("click", toggleSpeed);
const showPlaying = (playing) => {
  $("play").classList.toggle("playing", playing);
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
};
video.addEventListener("play", () => showPlaying(true));
video.addEventListener("pause", () => showPlaying(false));
$("scrub").addEventListener("input", (e) => seek((e.target.value / 1000) * (video.duration || 0)));

document.addEventListener("keydown", (e) => {
  if ($("analysis").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Escape" && spot && rows.length) {
    e.preventDefault();
    return select(rows[0], e.target.closest(".ledger") !== null);
  }
  if (e.target.closest(".ledger, input, textarea, select")) return;
  const frame = 1 / (result?.video.fps || 30);
  const action = {
    " ": togglePlay,
    ArrowLeft: () => seek(video.currentTime - 5),
    ArrowRight: () => seek(video.currentTime + 5),
    ",": () => { video.pause(); seek(video.currentTime - frame); },
    ".": () => { video.pause(); seek(video.currentTime + frame); },
    s: toggleSpeed,
    S: toggleSpeed,
  }[e.key];
  if (!action || (e.key === " " && e.target.closest("button, a"))) return;
  e.preventDefault();
  action();
});

video.addEventListener("loadedmetadata", () => {
  const ratio = video.videoWidth / video.videoHeight;
  $("frame").style.setProperty("--ratio", `${video.videoWidth} / ${video.videoHeight}`);
  $("frame").style.width = `min(100%, calc(70dvh * ${ratio}))`;
});

function insectAt(event) {
  if (!result) return null;
  const rect = canvas.getBoundingClientRect();
  const { x, y, scale } = contentRect();
  const px = (event.clientX - rect.left - x) / scale;
  const py = (event.clientY - rect.top - y) / scale;
  const frame = Math.floor(video.currentTime * result.video.fps);
  const pad = 20 / scale;
  return tracks.find((t) => {
    if (t.kind !== "insect" || !t.visiting) return false;
    const box = boxAt(t, frame);
    return box && px >= box[0] - pad && px <= box[2] + pad && py >= box[1] - pad && py <= box[3] + pad;
  }) ?? null;
}

canvas.addEventListener("pointermove", (event) => {
  const hit = insectAt(event);
  const id = hit ? hit.id : null;
  canvas.style.cursor = hit ? "pointer" : "default";
  if (id !== hoverId) {
    hoverId = id;
    draw(true);
  }
});
canvas.addEventListener("pointerleave", () => {
  hoverId = null;
  canvas.style.cursor = "default";
  draw(true);
});

// A click in the void turns the spotlight off: anywhere but a control, the detail panel being read, or the
// end of a text selection. On the video, only a click beside the insects does.
document.addEventListener("click", (event) => {
  if (!spot || $("analysis").hidden || !rows.length) return;
  if (event.target === canvas || String(getSelection()).trim()) return;
  if (event.target.closest("button, a, input, label, select, textarea, [role=treeitem], .detail, .player, .tip")) return;
  select(rows[0], false);
});

canvas.addEventListener("click", (event) => {
  const hit = insectAt(event);
  if (!hit && spot) return select(rows[0], false);
  if (hit && visitors.has(hit.id)) select($(`row-insect-${hit.id}`), false);
  else if (hit && far.has(hit.id) && hit.group) select($(`row-${hit.group.replace(/\W+/g, "-")}`), false);
});

/* Tooltips */

const tip = el("div", { className: "tip", hidden: true });
document.body.append(tip);

function showTip(target) {
  const box = target.getBoundingClientRect();
  if (target.dataset.span) {
    tip.replaceChildren(target.dataset.span, " ", el("span", { textContent: `(${target.dataset.stay})` }));
  } else {
    tip.textContent = target.dataset.tip;
  }
  const below = box.top < 90;
  tip.classList.toggle("below", below);
  tip.style.left = `${box.left + box.width / 2}px`;
  tip.style.top = `${below ? box.bottom : box.top}px`;
  tip.hidden = false;
}

const tipped = (e) => e.target.closest?.(".seg, [data-tip]");
document.addEventListener("pointerover", (e) => {
  const target = tipped(e);
  if (target && e.pointerType !== "touch") showTip(target);
  else tip.hidden = true;
});
document.addEventListener("focusin", (e) => {
  const target = tipped(e);
  if (target) showTip(target);
});
document.addEventListener("focusout", () => (tip.hidden = true));
addEventListener("scroll", () => (tip.hidden = true), { passive: true });

/* Start */

for (const input of document.querySelectorAll("[data-pick]")) {
  input.addEventListener("change", () => {
    if (!input.files[0]) return;
    setPanel(false);
    start(input.files[0]);
  });
}

const drop = $("drop");
for (const type of ["dragenter", "dragover"]) {
  drop.addEventListener(type, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
}
drop.addEventListener("dragleave", (e) => {
  if (!drop.contains(e.relatedTarget)) drop.classList.remove("over");
});
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const file = e.dataTransfer.files[0];
  if (file) start(file);
});

const resumeId = new URLSearchParams(location.search).get("a");
if (resumeId && /^[0-9a-z-]{1,40}$/.test(resumeId)) resume(resumeId);
else listExamples();
addEventListener("resize", draw);
loop();
