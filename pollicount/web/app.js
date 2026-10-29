"use strict";

const STAGES = {
  queued: "Waiting for a free slot",
  preparing: "Preparing the video",
  "tracking insects": "Looking for insects",
  "tracking flowers": "Finding the flowers",
  "finding visits": "Counting landings",
  "identifying insects": "Identifying each visitor",
  done: "Done",
};

const GROUP_NAMES = {
  bumblebees: "Bumblebees",
  honeybees: "Honeybees",
  "solitary bees": "Solitary bees",
  wasps: "Wasps",
  hoverflies: "Hoverflies",
  "other flies": "Other flies",
  "butterflies & moths": "Butterflies and moths",
  beetles: "Beetles",
  "small insects": "Small insects, under 3 mm",
  "other insects": "Other insects",
};

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let result = null;
let tracks = [];
let lastTime = -1;

function show(screen) {
  $("start").hidden = screen !== "start";
  $("analysis").hidden = screen !== "analysis";
}

function setStage(text, value) {
  $("stage-name").textContent = text;
  $("meter-fill").style.width = `${Math.round(value * 100)}%`;
}

function fail(id, message) {
  const el = $(id);
  el.textContent = message;
  el.hidden = false;
}

function groupName(group) {
  return group ? GROUP_NAMES[group] || group : "Unidentified insect";
}

/* Upload and progress */

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "api/analyses");
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setStage("Uploading your video", 0.1 * (e.loaded / e.total));
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
    if (response.status === 404) throw new Error("This analysis has expired. Upload the video again.");
    const job = await response.json();
    if (job.status === "failed") throw new Error(job.error);
    setStage(STAGES[job.stage] || job.stage, 0.1 + 0.9 * job.progress);
    if (job.status === "done") return;
    await new Promise((r) => setTimeout(r, 700));
  }
}

async function start(file) {
  show("analysis");
  video.src = URL.createObjectURL(file);
  video.play().catch(() => {});
  try {
    const job = await upload(file);
    history.replaceState(null, "", `?a=${job.id}`);
    await follow(job.id);
    await showResult(job.id);
  } catch (error) {
    $("progress").hidden = true;
    fail("analysis-error", error.message);
  }
}

async function resume(id) {
  show("analysis");
  try {
    await follow(id);
    await showResult(id);
  } catch (error) {
    $("progress").hidden = true;
    fail("analysis-error", error.message);
  }
}

/* Results */

async function showResult(id) {
  result = await (await fetch(`api/analyses/${id}/result`)).json();
  tracks = [
    ...result.flowers.map((t) => ({ ...t, kind: "flower", byFrame: index(t.boxes) })),
    ...result.insects.map((t) => ({ ...t, kind: "insect", byFrame: index(t.boxes) })),
  ];
  video.src = `api/analyses/${id}/video`;
  video.play().catch(() => {});

  const { record } = result;
  $("progress").hidden = true;
  $("results").hidden = false;
  document.querySelector(".playback").hidden = false;
  const plural = (n, word) => `${n} ${n === 1 ? word : `${word}s`}`;
  $("total").textContent = record.insects;
  $("total-label").textContent = record.insects === 1 ? "pollinator" : "pollinators";
  const minutes = record.duration_min < 1 ? `${Math.round(record.duration_min * 60)} s` : `${record.duration_min} min`;
  $("summary").textContent =
    record.insects === 0
      ? `No insect stayed on a flower long enough in ${minutes} of video. Try a sunnier moment or a patch in full bloom.`
      : `${plural(record.total_visits, "visit")} to ${plural(record.flowers_tracked, "tracked flower")} in ${minutes} of video. Each dot is one visit.`;
  renderLedger(record);
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
  $("download-json").href = URL.createObjectURL(blob);
}

function index(boxes) {
  const map = new Map();
  for (const [frame, ...box] of boxes) map.set(frame, box);
  return map;
}

function renderLedger(record) {
  const ledger = $("ledger");
  ledger.replaceChildren();
  const groups = Object.entries(record.counts).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  if (groups.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No visits to list yet.";
    ledger.append(li);
    return;
  }
  for (const [group, count] of groups) {
    const li = document.createElement("li");
    li.dataset.group = group;
    li.innerHTML = `<span class="name"></span><span class="count"></span><span class="grains"></span>`;
    li.querySelector(".name").textContent = groupName(group);
    li.querySelector(".count").textContent = count;
    for (const visit of record.visits.filter((v) => v.group === group)) {
      const grain = document.createElement("span");
      grain.className = "grain";
      grain.dataset.start = visit.start_s;
      grain.title = `Visit at ${visit.start_s.toFixed(1)} s, ${visit.duration_s.toFixed(1)} s long`;
      li.querySelector(".grains").append(grain);
    }
    ledger.append(li);
  }
}

function updateLedger(previous, time) {
  for (const grain of document.querySelectorAll(".grain")) {
    const start = Number(grain.dataset.start);
    if (!reducedMotion && previous < start && start <= time && time - previous < 1) {
      const row = grain.closest("li");
      row.classList.remove("flash");
      void row.offsetWidth;
      row.classList.add("flash");
    }
    grain.classList.toggle("seen", start <= time);
  }
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
  return result.record.visits.find(
    (v) => v.insect === insectId && time >= v.start_s && time <= v.start_s + v.duration_s,
  );
}

function draw() {
  const dpr = devicePixelRatio || 1;
  if (canvas.width !== canvas.clientWidth * dpr) {
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!result) return;

  const time = video.currentTime;
  const frame = Math.floor(time * result.video.fps);
  const { x, y, scale } = contentRect();
  const css = getComputedStyle(document.documentElement);
  const uv = css.getPropertyValue("--uv").trim();
  const pollen = css.getPropertyValue("--pollen").trim();

  for (const track of tracks) {
    const box = boxAt(track, frame);
    if (!box) continue;
    const [x1, y1, x2, y2] = box.map((v, i) => (i % 2 ? y + v * scale : x + v * scale));
    if (track.kind === "flower") {
      ctx.setLineDash([4, 6]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = uv;
      ctx.beginPath();
      ctx.ellipse((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      continue;
    }
    const visiting = activeVisit(track.id, time);
    drawTrail(track, frame, x, y, scale, pollen);
    ctx.lineWidth = visiting ? 3 : 2;
    ctx.strokeStyle = pollen;
    corners(x1 - 3, y1 - 3, x2 + 3, y2 + 3);
    if (track.visiting) label(groupName(track.group), x1, y1 - 8, pollen);
  }

  if (time !== lastTime) {
    updateLedger(lastTime, time);
    lastTime = time;
    $("scrub").value = Math.round((time / (video.duration || 1)) * 1000);
  }
}

function corners(x1, y1, x2, y2) {
  const k = Math.min(10, (x2 - x1) / 3, (y2 - y1) / 3);
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

function label(text, x, y, color) {
  ctx.font = "600 13px 'Atkinson Hyperlegible Next', sans-serif";
  const w = ctx.measureText(text).width + 14;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x, y - 18, w, 22, 11);
  ctx.fill();
  ctx.fillStyle = "#15302a";
  ctx.fillText(text, x + 7, y - 3);
}

function loop() {
  draw();
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

/* Interaction */

canvas.addEventListener("click", (event) => {
  if (!result) return;
  const rect = canvas.getBoundingClientRect();
  const { x, y, scale } = contentRect();
  const px = (event.clientX - rect.left - x) / scale;
  const py = (event.clientY - rect.top - y) / scale;
  const frame = Math.floor(video.currentTime * result.video.fps);
  const pad = 20 / scale;
  const hit = tracks.find((t) => {
    if (t.kind !== "insect") return false;
    const box = boxAt(t, frame);
    return box && px >= box[0] - pad && px <= box[2] + pad && py >= box[1] - pad && py <= box[3] + pad;
  });
  if (hit) openInsect(hit);
});

function openInsect(insect) {
  const visits = result.record.visits.filter((v) => v.insect === insect.id);
  $("insect-group").textContent = groupName(insect.group);
  $("insect-confidence").textContent =
    insect.confidence == null ? "Not identified: it never landed long enough." : `${Math.round(insect.confidence * 100)}% sure`;
  $("insect-cues").replaceChildren(
    ...insect.cues.map((cue) => Object.assign(document.createElement("li"), { textContent: cue })),
  );
  $("insect-visits").textContent = visits.length
    ? `${visits.length} ${visits.length === 1 ? "visit" : "visits"}: ${visits.map((v) => `${v.start_s.toFixed(1)} s`).join(", ")}`
    : "No visit counted.";
  $("insect").showModal();
}

video.addEventListener("loadedmetadata", () => {
  const stage = document.querySelector(".stage");
  stage.style.setProperty("--ratio", `${video.videoWidth} / ${video.videoHeight}`);
  stage.style.maxWidth = `calc(72dvh * ${video.videoWidth / video.videoHeight})`;
});

$("play").addEventListener("click", () => (video.paused ? video.play() : video.pause()));
video.addEventListener("play", () => ($("play").textContent = "Pause"));
video.addEventListener("pause", () => ($("play").textContent = "Play"));
$("scrub").addEventListener("input", (e) => {
  video.currentTime = (e.target.value / 1000) * (video.duration || 0);
});

for (const input of document.querySelectorAll("[data-pick]")) {
  input.addEventListener("change", () => input.files[0] && start(input.files[0]));
}

const resumeId = new URLSearchParams(location.search).get("a");
if (resumeId && /^[0-9a-f]{32}$/.test(resumeId)) resume(resumeId);
addEventListener("resize", draw);
loop();
