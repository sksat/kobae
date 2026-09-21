import { chromium } from "playwright";
const url = process.argv[2] || "http://localhost:8765/";
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on("console", m => { if (m.type() === "error") console.log("console:", m.text().slice(0, 200)); });
page.on("pageerror", e => console.log("pageerror:", String(e).slice(0, 300)));
await page.goto(url, { waitUntil: "load" });
await page.waitForTimeout(6000);
const info = async () => page.evaluate(() => ({ fps: document.querySelector("#fps").textContent, mode: document.querySelector("#flight").hidden ? "brain" : "flight", parent: document.querySelector("#gl").parentElement.id }));
console.log("after load:", await info());
// switch to brain mode, measure long tasks over 4 s
await page.click("[data-mode=brain]");
await page.waitForTimeout(500);
const longTasks = await page.evaluate(() => new Promise(res => { const out = []; const o = new PerformanceObserver(l => out.push(...l.getEntries().map(e => Math.round(e.duration)))); o.observe({ entryTypes: ["longtask"] }); setTimeout(() => { o.disconnect(); res(out); }, 4000); }));
console.log("brain mode:", await info(), "long tasks (ms) in 4 s:", longTasks.length, longTasks.slice(0, 12));
// drag test: camera position before/after
const camBefore = await page.evaluate(() => JSON.stringify(window.__cam ? window.__cam() : null));
const box = await page.locator("#gl").boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 200, box.y + box.height / 2 + 60, { steps: 12 }); await page.mouse.up();
await page.waitForTimeout(800);
const camAfter = await page.evaluate(() => JSON.stringify(window.__cam ? window.__cam() : null));
console.log("camera before:", camBefore, "after:", camAfter);
await page.screenshot({ path: process.argv[3] || "/tmp/claude-1000/-home-sksat/3b79048e-29b9-4884-80d3-019470acc91f/scratchpad/probe.png" });
await browser.close();
