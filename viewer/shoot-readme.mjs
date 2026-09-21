// README screenshots against a running kobae server (default: BC-250).
//   node shoot-readme.mjs [brain|flight] [base-url]
// "brain": sugar on -> brain view lit; leaves the sim as found (sugar off, reset).
// "flight": expects the body loop to be running; captures the flight view from the chase camera.
// Rendering is software GL (swiftshader) on purpose: the workstation GPU stays out of it.
import { chromium } from "playwright";
const what = process.argv[2] || "brain";
const base = process.argv[3] || "http://localhost:8765/";
const out = new URL("../docs/", import.meta.url).pathname;
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on("pageerror", e => console.log("pageerror:", String(e).slice(0, 300)));
page.on("console", m => { if (m.type() === "error") console.log("console:", m.text().slice(0, 200)); });
await page.goto(base, { waitUntil: "load" });
await page.waitForTimeout(8000);
const state = () => page.evaluate(() => ({
  mode: document.querySelector("#flight").hidden ? "brain" : "flight",
  fps: document.querySelector("#fps").textContent,
  info: document.querySelector("#flightinfo")?.textContent,
}));
if (what === "brain") {
  await page.click("[data-mode=brain]"); await page.waitForTimeout(1500);
  await page.click("[data-toggle=sugar]");                // LB3c -> MN9: the README's flagship phenomenon
  await page.waitForTimeout(4000);
  console.log("sugar", await state());
  await page.screenshot({ path: `${out}viewer-brain-sugar.png` });
  await page.click("[data-toggle=sugar]");
  await page.click("#reset"); await page.waitForTimeout(500);   // leave the sugar attractor
} else {
  await page.click("[data-mode=flight]"); await page.waitForTimeout(6000);
  console.log("flight", await state());
  await page.screenshot({ path: `${out}viewer-flight.png` });
}
await browser.close();
