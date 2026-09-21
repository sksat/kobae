// UI probe: screenshots of brain and flight mode at two sizes against a dev/preview server.
import { chromium } from "playwright";
const base = process.argv[2] || "http://localhost:5173/";
const out = "/tmp/claude-1000/-home-sksat/3b79048e-29b9-4884-80d3-019470acc91f/scratchpad/";
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
for (const [w, h] of [[1600, 900], [1920, 1080]]) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  page.on("pageerror", e => console.log("pageerror:", String(e).slice(0, 300)));
  page.on("console", m => { if (m.type() === "error") console.log("console:", m.text().slice(0, 200)); });
  await page.goto(base, { waitUntil: "load" });
  await page.waitForTimeout(9000);
  console.log(w, await page.evaluate(() => ({ mode: document.querySelector("#flight").hidden ? "brain" : "flight", fps: document.querySelector("#fps").textContent })));
  await page.screenshot({ path: `${out}ui-flight-${w}.png` });
  await page.click("[data-mode=brain]"); await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}ui-brain-${w}.png` });
  if (w === 1600) {
    await page.click("#sec-odor summary"); await page.click("[data-orn=DM2]"); await page.click("[data-orn=VA1v]"); await page.click("[data-button=WIND]");
    await page.click("button.more"); await page.waitForTimeout(800);
    await page.screenshot({ path: `${out}ui-brain-${w}-open.png` });
    await page.click("[data-orn=DM2]"); await page.click("[data-orn=VA1v]"); await page.click("[data-button=WIND]");   // leave the sim as found
    await page.click("[data-fold=left]"); await page.click("[data-mode=flight]"); await page.waitForTimeout(1500);
    await page.screenshot({ path: `${out}ui-flight-${w}-fold.png` });
    await page.click("[data-fold=left]"); await page.click("[data-fold=right]"); await page.click("[data-mode=brain]"); await page.waitForTimeout(1500);
    await page.screenshot({ path: `${out}ui-brain-${w}-foldr.png` });
    await page.click("[data-fold=right]");
  }
  await page.close();
}
await browser.close();
