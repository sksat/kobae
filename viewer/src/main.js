import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const $ = (s) => document.querySelector(s);
const MAGIC = 0x4b4f4241;

// ---------------------------------------------------------------- data
const meta = await (await fetch("/api/meta")).json();
const pos = new Float32Array(await (await fetch("/api/positions")).arrayBuffer());
const scIdx = new Uint8Array(await (await fetch("/api/superclass")).arrayBuffer());
const N = meta.n;
$("#backend").textContent = `計算: ${meta.backend.toUpperCase()} · ${N.toLocaleString()} 細胞 · ${meta.edges.toLocaleString()} 結合`;
const SC_JA = { ENS: "腸管神経", ascending_neuron: "上行（体→脳）", cb_efferent: "中枢脳 遠心", cb_endocrine: "中枢脳 内分泌",
  cb_intrinsic: "中枢脳 内在", cb_motor: "中枢脳 運動", cb_sensory: "中枢脳 感覚", cb_sensory_tbc: "中枢脳 感覚(未確定)",
  descending_neuron: "下行（脳→体）", descending_neuron_tbc: "下行(未確定)", efferent_ascending: "遠心+上行", efferent_descending: "遠心+下行",
  ol_intrinsic: "視葉 内在", ol_sensory: "視葉 感覚（光受容体）", sensory_ascending: "感覚+上行", sensory_ascending_tbc: "感覚+上行(未確定)",
  sensory_descending: "感覚+下行", visual_centrifugal: "視覚 遠心", visual_projection: "視覚 投射", visual_projection_tbc: "視覚 投射(未確定)",
  vnc_efferent: "腹髄 遠心", vnc_endocrine: "腹髄 内分泌", vnc_intrinsic: "腹髄 内在", vnc_motor: "腹髄 運動", vnc_sensory: "腹髄 感覚",
  vnc_sensory_tbc: "腹髄 感覚(未確定)", vnc_tbc: "腹髄(未確定)" };
const scJa = (s) => SC_JA[s] || s;
const RO_JA = { DNa02: "旋回", DNp09: "前進", MDN: "後退", MN9: "吻を伸ばす（摂食）", DNp20: "視覚→下行 DNp20", DNpe017: "視覚→下行 DNpe017" };
const BTN_JA = { FOOD: "食べ物（味覚）", BITTER: "苦味", PAIN: "痛み", PHEROMONE: "フェロモン（接触）", MATE: "求愛回路", DOPAMINE: "ドーパミン（報酬）", WIND: "風・音", GIANT: "逃避（巨大線維）" };

// superclass palette (stable order = meta.superclasses)
const PALETTE = ["#7cc7ff","#ffb86b","#8be28b","#ff7b9c","#c9a2ff","#ffe36b","#6be8d8","#ff9b6b","#a7c4ff","#e6a4ff",
  "#9be0a0","#ffd0a0","#8ad0ff","#ffa0c0","#c0ffa0","#a0a8ff","#ffc4e0","#b0ffe0","#e0e080","#c0c0c0",
  "#ff8080","#80ffc0","#c080ff","#80c0ff","#ffc080","#c0ff80","#ff80c0"];
const scColor = meta.superclasses.map((_, i) => new THREE.Color(PALETTE[i % PALETTE.length]));
$("#legend").innerHTML = "<b>点の色 = 細胞のクラス</b><br>" + meta.superclasses.map((s, i) => `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${scJa(s)} ${meta.superclass_counts[i].toLocaleString()}</span>`).join(" ");

// ---------------------------------------------------------------- scene
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
const camera = new THREE.PerspectiveCamera(45, 1, 1, 20000);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;

const geom = new THREE.BufferGeometry();
geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
const colors = new Float32Array(N * 3);
for (let i = 0; i < N; i++) { const c = scColor[scIdx[i]]; colors[3*i] = c.r; colors[3*i+1] = c.g; colors[3*i+2] = c.b; }
geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
const lastSpike = new Float32Array(N).fill(-1e9);   // ms of simulated time
const lastAttr = new THREE.BufferAttribute(lastSpike, 1); lastAttr.setUsage(THREE.DynamicDrawUsage);
geom.setAttribute("lastSpike", lastAttr);
const bbox = new THREE.Box3().setFromBufferAttribute(geom.getAttribute("position"));
const center = bbox.getCenter(new THREE.Vector3()); controls.target.copy(center);
const radius = bbox.getBoundingSphere(new THREE.Sphere()).radius;
function view(kind) {
  const vfov = THREE.MathUtils.degToRad(camera.fov / 2);
  const hfov = Math.atan(Math.tan(vfov) * camera.aspect);
  const d = Math.max(radius / Math.sin(vfov), radius / Math.sin(hfov)) * 0.9;
  if (kind === "front") { camera.position.set(center.x, center.y, center.z + d); camera.up.set(0, 1, 0); }
  if (kind === "top")   { camera.position.set(center.x, center.y + d, center.z + 1); camera.up.set(0, 0, -1); }
  if (kind === "side")  { camera.position.set(center.x + d, center.y, center.z); camera.up.set(0, 1, 0); }
  controls.update();
}
document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => view(b.dataset.view));

const mat = new THREE.ShaderMaterial({
  uniforms: { uNow: { value: 0 }, uTau: { value: 80.0 }, uSize: { value: 2.6 }, uDim: { value: 0.55 } },
  vertexShader: `
    attribute float lastSpike; attribute vec3 color;
    uniform float uNow, uTau, uSize, uDim; varying vec3 vColor; varying float vGlow;
    void main() {
      float age = uNow - lastSpike;
      float g = age < 0.0 ? 0.0 : exp(-age / uTau);
      vGlow = g; vColor = color;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = (uSize + 7.0 * g) * (600.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vColor; varying float vGlow; uniform float uDim;
    void main() {
      vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard;
      vec3 c = mix(vColor * uDim, vec3(1.0), vGlow * 0.85) + vColor * vGlow * 0.6;
      gl_FragColor = vec4(c, 1.0);
    }`,
  transparent: false, depthWrite: true,
});
const points = new THREE.Points(geom, mat);
scene.add(points);
// landmark labels that follow the cloud when it is rotated
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.id = "labels";
canvas.parentElement.appendChild(labelRenderer.domElement);
for (const [name, p] of Object.entries(meta.landmarks || {})) {
  if (!p) continue;
  const el = document.createElement("div"); el.className = "landmark"; el.textContent = name;
  const o = new CSS2DObject(el); o.position.set(p[0], p[1], p[2]); scene.add(o);
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  labelRenderer.setSize(w, h);
  const r = canvas.getBoundingClientRect();
  Object.assign(labelRenderer.domElement.style, { position: "absolute", left: r.left + "px", top: r.top + "px", pointerEvents: "none" });
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener("resize", resize); resize(); view("front");

// ---------------------------------------------------------------- ws
let simMs = 0, rt = 0, sps = 0, frames = 0, lastFpsT = performance.now();
const scRate = new Float32Array(meta.superclasses.length);
const roRate = new Float32Array(meta.readouts.length);
let retinaLum = new Float32Array(meta.retina);
const raster = []; // [simMs, neuronIndexInSelection]
let rasterSet = new Map(meta.readouts.map((r, i) => [r.index, i]));
let rasterLabel = "readouts";
const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
let ws;
function connect() {
  ws = new WebSocket(wsUrl); ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.op === "state") { syncUI(m.state, m.paused); if (m.speed !== undefined) setSpeedUI(m.speed); }
      else if (m.op === "body") onBody(m);
      return;
    }
    const dv = new DataView(ev.data);
    const magic = dv.getUint32(0, true);
    if (magic === 0x42424f4b) { onBodyBinary(ev.data, dv); return; }   // 'KOBB'
    if (magic !== MAGIC) return;
    simMs = dv.getFloat32(4, true); rt = dv.getFloat32(8, true);
    const total = dv.getUint32(12, true), nIds = dv.getUint32(16, true), nSc = dv.getUint32(20, true), nRo = dv.getUint32(24, true), nRet = dv.getUint32(28, true);
    const frameSimS = dv.getFloat32(32, true);
    let o = 36;
    const ids = new Uint32Array(ev.data, o, nIds); o += nIds * 4;
    scRate.set(new Float32Array(ev.data, o, nSc)); o += nSc * 4;
    roRate.set(new Float32Array(ev.data, o, nRo)); o += nRo * 4;
    retinaLum = new Float32Array(ev.data, o, nRet);
    for (let k = 0; k < nIds; k++) {
      const i = ids[k]; lastSpike[i] = simMs;
      const r = rasterSet.get(i); if (r !== undefined) raster.push(simMs, r);
    }
    if (nIds && mode !== "flight") lastAttr.needsUpdate = true;
    sps = 0.8 * sps + 0.2 * (total / Math.max(frameSimS, 1e-6));
    while (raster.length > 60000) raster.splice(0, 2);
  };
  ws.onopen = () => { ws.send(JSON.stringify({ op: "body_sub", value: true })); while (pending.length) ws.send(JSON.stringify(pending.shift())); };
  ws.onclose = () => setTimeout(connect, 1000);
}
connect();
const pending = [];
const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); else pending.push(o); };

// ---------------------------------------------------------------- UI
const state = { gain: 1, sugar: false, visual: "off", orn: {}, buttons: {}, custom: [] };
function stim(patch) { Object.assign(state, patch); send({ op: "stim", patch }); syncUI(state); }
function syncUI(s, paused) {
  Object.assign(state, s);
  document.querySelectorAll("[data-toggle]").forEach(b => b.classList.toggle("on", !!state[b.dataset.toggle]));
  document.querySelectorAll("[data-visual]").forEach(b => b.classList.toggle("on", state.visual === b.dataset.visual));
  document.querySelectorAll("[data-button]").forEach(b => b.classList.toggle("on", !!(state.buttons || {})[b.dataset.button]));
  document.querySelectorAll("[data-orn]").forEach(b => b.classList.toggle("on", !!(state.orn || {})[b.dataset.orn]));
  $("#gain").value = state.gain; $("#gainv").textContent = Number(state.gain).toFixed(2);
  if (paused !== undefined) $("#pause").classList.toggle("on", paused);
}
$("#gain").oninput = (e) => stim({ gain: +e.target.value });
const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 0];
function setSpeedUI(v) { $("#speed").value = SPEEDS.indexOf(v) >= 0 ? SPEEDS.indexOf(v) : 3; $("#speedv").textContent = v > 0 ? `${v}× 実時間` : "最速"; }
$("#speed").oninput = (e) => { const v = SPEEDS[+e.target.value]; setSpeedUI(v); send({ op: "speed", value: v }); };
setSpeedUI(meta.speed ?? 1);
document.querySelectorAll("[data-toggle]").forEach(b => b.onclick = () => stim({ [b.dataset.toggle]: !state[b.dataset.toggle] }));
document.querySelectorAll("[data-visual]").forEach(b => b.onclick = () => stim({ visual: b.dataset.visual }));
let paused = false;
$("#pause").onclick = () => { paused = !paused; send({ op: "pause", value: paused }); $("#pause").classList.toggle("on", paused); };
$("#reset").onclick = () => { send({ op: "reset" }); lastSpike.fill(-1e9); lastAttr.needsUpdate = true; raster.length = 0; };
for (const [name, desc] of Object.entries(meta.buttons)) {
  const b = document.createElement("button"); b.textContent = BTN_JA[name] || name; b.title = desc; b.dataset.button = name;
  b.onclick = () => stim({ buttons: { ...state.buttons, [name]: !(state.buttons || {})[name] } });
  $("#buttons").appendChild(b);
}
for (const [g, n] of Object.entries(meta.orn)) {
  const b = document.createElement("button"); b.textContent = g; b.title = `${n} ORNs`; b.dataset.orn = g;
  b.onclick = () => stim({ orn: { ...state.orn, [g]: !(state.orn || {})[g] } });
  $("#orn").appendChild(b);
}
$("#inject").onclick = async () => {
  const glob = $("#glob").value.trim(); if (!glob) return;
  const r = await (await fetch(`/api/search?q=${encodeURIComponent(glob)}`)).json();
  $("#globinfo").textContent = `${r.n} cells: ` + r.types.slice(0, 12).map(t => `${t.type}×${t.n}`).join(" ");
  stim({ custom: [...state.custom.filter(c => c.glob !== glob), { glob, amp: +$("#amp").value }] });
  if (r.ids.length) { rasterSet = new Map(r.ids.slice(0, 200).map((i, k) => [i, k])); rasterLabel = glob; raster.length = 0; }
};
$("#clear").onclick = () => { stim({ custom: [] }); rasterSet = new Map(meta.readouts.map((r, i) => [r.index, i])); rasterLabel = "readouts"; raster.length = 0; };

// bars
const scBars = meta.superclasses.map((s, i) => {
  const d = document.createElement("div"); d.className = "bar";
  d.innerHTML = `<span title="${s}: ${meta.superclass_counts[i]} 細胞" style="color:${PALETTE[i % PALETTE.length]}">${scJa(s)}</span><div class="track"><div class="fill"></div></div><span class="val"></span>`;
  $("#scbars").appendChild(d); return d;
});
const roOrder = meta.readouts.map((r, i) => i).sort((a, b) => {
  const ka = Object.keys(RO_JA).indexOf(meta.readouts[a].type), kb = Object.keys(RO_JA).indexOf(meta.readouts[b].type);
  return ka - kb || meta.readouts[a].side.localeCompare(meta.readouts[b].side);
});
const roBars = new Array(meta.readouts.length);
for (const i of roOrder) {
  const r = meta.readouts[i]; const d = document.createElement("div"); d.className = "bar";
  d.innerHTML = `<span title="${r.type} bodyId ${r.id}">${RO_JA[r.type] || r.type} ${r.side === "L" ? "左" : r.side === "R" ? "右" : ""}</span><div class="track"><div class="fill"></div></div><span class="val"></span>`;
  $("#readouts").appendChild(d); roBars[i] = d;
}

// retina map
const rc = $("#retina").getContext("2d");
const uv = meta.uv;
function drawRetina() {
  rc.fillStyle = "#000"; rc.fillRect(0, 0, 240, 120);
  for (let i = 0; i < meta.retina; i++) {
    const l = retinaLum[i] || 0; if (l <= 0) continue;
    rc.fillStyle = `rgba(255,230,120,${0.25 + 0.75 * l})`;
    rc.fillRect(uv[2*i] * 238, uv[2*i+1] * 118, 2, 2);
  }
  rc.strokeStyle = "#2b3648"; rc.beginPath(); rc.moveTo(120, 0); rc.lineTo(120, 120); rc.stroke();
  rc.fillStyle = "#8a93a3"; rc.font = "11px ui-monospace, monospace";
  rc.fillText("左目", 4, 12); rc.fillText("右目", 124, 12);
}
// raster
const rr = $("#raster").getContext("2d");
function drawRaster() {
  rr.fillStyle = "#000"; rr.fillRect(0, 0, 360, 160);
  const win = 2000, t0 = simMs - win, rows = Math.max(1, rasterSet.size);
  rr.fillStyle = "#9ad7ff";
  for (let k = 0; k < raster.length; k += 2) {
    const t = raster[k]; if (t < t0) continue;
    rr.fillRect((t - t0) / win * 360, raster[k+1] / rows * 158, 1.5, Math.max(1, 158 / rows - 1));
  }
  $("#rasterinfo").textContent = `${rasterLabel === "readouts" ? "行動の出力ニューロン" : rasterLabel}（${rasterSet.size} 細胞）`;
}

// click -> nearest projected point
const proj = new THREE.Vector3();
canvas.addEventListener("pointerdown", (e) => { canvas._down = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", async (e) => {
  const d = canvas._down; if (!d || Math.hypot(e.clientX - d[0], e.clientY - d[1]) > 4) return;
  const r = canvas.getBoundingClientRect();
  const x = ((e.clientX - r.left) / r.width) * 2 - 1, y = -((e.clientY - r.top) / r.height) * 2 + 1;
  let best = -1, bestD = 0.02 * 0.02;
  for (let i = 0; i < N; i++) {
    proj.set(pos[3*i], pos[3*i+1], pos[3*i+2]).project(camera);
    if (proj.z > 1) continue;
    const dd = (proj.x - x) ** 2 + ((proj.y - y) * r.height / r.width) ** 2;
    if (dd < bestD) { bestD = dd; best = i; }
  }
  if (best < 0) return;
  const c = await (await fetch(`/api/cell/${best}`)).json();
  $("#cellinfo").innerHTML = `<b>${c.type || "(型なし)"}</b> ${c.side === "L" ? "左" : c.side === "R" ? "右" : ""} · ${scJa(c.superclass)}<br>bodyId ${c.bodyId} · 伝達物質 ${c.nt || "不明"}（${c.sign > 0 ? "興奮性 +" : "抑制性 −"}）<br>出力先 ${c.out_degree} 細胞 · 入力元 ${c.in_degree} 細胞 · いま ${c.rate_hz.toFixed(1)} Hz · 位置は${c.measured_soma ? "実測" : "推定"}`;
  rasterSet = new Map([[best, 0]]); rasterLabel = c.type || String(best); raster.length = 0;
});

// body (flybody) relay
const path = [];
let mode = "brain", bodySeen = false;
function setMode(m) {
  mode = m;
  $("#flight").hidden = m !== "flight"; $("#gl").style.visibility = m === "flight" ? "hidden" : "visible";
  document.querySelectorAll("[data-mode]").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
}
document.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => setMode(b.dataset.mode));
function drawPath(canvas, m, W, H) {
  const c = canvas.getContext("2d"); c.fillStyle = "rgba(0,0,0,0.6)"; c.clearRect(0, 0, W, H); c.fillRect(0, 0, W, H);
  if (path.length < 4) return;
  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  for (let i = 0; i < path.length; i += 2) { minx = Math.min(minx, path[i]); maxx = Math.max(maxx, path[i]); miny = Math.min(miny, path[i+1]); maxy = Math.max(maxy, path[i+1]); }
  const span = Math.max(maxx - minx, maxy - miny, 2), sx = (Math.min(W, H) - 20) / span, ox = (minx + maxx) / 2, oy = (miny + maxy) / 2;
  c.strokeStyle = "#7cc7ff"; c.beginPath();
  for (let i = 0; i < path.length; i += 2) { const x = W / 2 + (path[i] - ox) * sx, y = H / 2 - (path[i+1] - oy) * sx; i ? c.lineTo(x, y) : c.moveTo(x, y); }
  c.stroke();
  const x = W / 2 + (m.pos[0] - ox) * sx, y = H / 2 - (m.pos[1] - oy) * sx;
  c.fillStyle = "#ffe36b"; c.beginPath(); c.arc(x, y, 4, 0, 6.283); c.fill();
  if (m.yaw !== undefined) { c.strokeStyle = "#ffe36b"; c.beginPath(); c.moveTo(x, y); c.lineTo(x + 12 * Math.cos(m.yaw), y - 12 * Math.sin(m.yaw)); c.stroke(); }
  c.fillStyle = "#cfd6e2"; c.font = "10px ui-monospace"; c.fillText(`飛行経路 ${span.toFixed(0)} cm`, 6, H - 6);
}
let chaseUrl = null, eyesUrl = null;
function onBodyBinary(buf, dv) {
  const t = dv.getFloat32(4, true), pos = [dv.getFloat32(8, true), dv.getFloat32(12, true), dv.getFloat32(16, true)];
  const yaw = dv.getFloat32(20, true), cmd = [dv.getFloat32(24, true), dv.getFloat32(28, true)];
  const jl = dv.getUint32(32, true), el = dv.getUint32(36, true);
  const jpeg = new Blob([new Uint8Array(buf, 40, jl)], { type: "image/jpeg" });
  const eyes = new Blob([new Uint8Array(buf, 40 + jl, el)], { type: "image/jpeg" });
  if (chaseUrl) URL.revokeObjectURL(chaseUrl); if (eyesUrl) URL.revokeObjectURL(eyesUrl);
  chaseUrl = URL.createObjectURL(jpeg); eyesUrl = URL.createObjectURL(eyes);
  onBody({ t, pos, yaw, cmd, chaseUrl, eyesUrl });
}
function onBody(m) {
  const sec = $("#bodysec"); if (sec.hidden) sec.hidden = false;
  if (!bodySeen) { bodySeen = true; setMode("flight"); }
  if (m.chaseUrl) { if (mode === "flight") $("#chase").src = m.chaseUrl; else $("#bodycam").src = m.chaseUrl; }
  if (m.eyesUrl) { if (mode === "flight") $("#eyesbig").src = m.eyesUrl; else $("#bodyeyes").src = m.eyesUrl; }
  if (m.pos) { path.push(m.pos[0], m.pos[1]); while (path.length > 4000) path.splice(0, 2); }
  drawPath($("#bodypath"), m, 360, 200); drawPath($("#minimap"), m, 220, 160);
  if (m.cmd) {
    const txt = `体の時間 ${(m.t ?? 0).toFixed(2)} s · 前進 ${m.cmd[0].toFixed(1)} cm/s · 旋回 ${m.cmd[1].toFixed(2)} rad/s · 高さ ${(m.pos?.[2] ?? 0).toFixed(1)} cm`;
    $("#bodyinfo").textContent = txt; $("#flightinfo").textContent = txt;
  }
}

// ---------------------------------------------------------------- loop
function tick() {
  requestAnimationFrame(tick);
  if (mode !== "flight") {          // the point cloud is hidden in flight mode: do not render it
    controls.update();
    mat.uniforms.uNow.value = simMs;
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
  frames++;
  const now = performance.now();
  if (now - lastFpsT > 500) {
    $("#fps").textContent = Math.round(frames * 1000 / (now - lastFpsT)); frames = 0; lastFpsT = now;
    $("#simt").textContent = (simMs / 1000).toFixed(2); $("#rt").textContent = rt.toFixed(3); $("#sps").textContent = Math.round(sps).toLocaleString();
    const scMax = Math.max(1, ...scRate);
    scBars.forEach((d, i) => { d.querySelector(".fill").style.width = `${100 * scRate[i] / scMax}%`; d.querySelector(".val").textContent = scRate[i].toFixed(2); });
    const roMax = Math.max(5, ...roRate);
    roBars.forEach((d, i) => { d.querySelector(".fill").style.width = `${100 * roRate[i] / roMax}%`; d.querySelector(".val").textContent = roRate[i].toFixed(1); });
    drawRetina(); drawRaster();
  }
}
tick();
