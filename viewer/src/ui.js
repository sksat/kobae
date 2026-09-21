// Layout-only helpers: folding side panels, "n switched on" badges on collapsible sections,
// a toggle for the minor region rows, and remembered open/closed state. No protocol logic here.
const $ = (s) => document.querySelector(s);
const store = {
  get: (k) => { try { return localStorage.getItem("kobae." + k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem("kobae." + k, v); } catch {} },
};

// side panels fold away so the stage can take the whole width; main.js listens for "resize"
const app = $("#app");
for (const b of document.querySelectorAll("[data-fold]")) {
  const cls = "fold-" + b.dataset.fold;
  const apply = (on) => { app.classList.toggle(cls, on); b.classList.toggle("on", on); dispatchEvent(new Event("resize")); };
  b.onclick = () => { const on = !app.classList.contains(cls); store.set(cls, on ? "1" : ""); apply(on); };
  if (store.get(cls)) apply(true);
}

// collapsible sections remember whether they were open
for (const d of document.querySelectorAll("details[id]")) {
  const v = store.get(d.id); if (v !== null) d.open = v === "1";
  d.addEventListener("toggle", () => store.set(d.id, d.open ? "1" : ""));
}

// badge = how many buttons inside the section are switched on (visible even when collapsed)
for (const badge of document.querySelectorAll("[data-count]")) {
  const box = $(badge.dataset.count);
  const upd = () => { const n = box.querySelectorAll("button.on").length; badge.textContent = n ? String(n) : ""; };
  new MutationObserver(upd).observe(box, { attributes: true, attributeFilter: ["class"], subtree: true, childList: true });
  upd();
}

// region rows for tiny classes (.minor, set by main.js) stay hidden behind one toggle
const sc = $("#scbars");
new MutationObserver((_, obs) => {
  const n = sc.querySelectorAll(".bar.minor").length; if (!n) return;
  obs.disconnect();
  const b = document.createElement("button"); b.className = "more";
  const label = () => { b.textContent = sc.classList.contains("all") ? "少数のクラスを隠す" : `少数のクラス ${n} 件も表示`; };
  b.onclick = () => { sc.classList.toggle("all"); label(); };
  label(); sc.after(b);
}).observe(sc, { childList: true });
