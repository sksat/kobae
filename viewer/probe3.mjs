import { chromium } from "playwright";
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on("pageerror", e => console.log("pageerror:", String(e).slice(0, 300)));
await page.goto("http://localhost:8765/", { waitUntil: "load" });
await page.waitForTimeout(8000);
for (let i = 0; i < 12; i++) {
  const info = await page.evaluate(() => document.querySelector("#flightinfo").textContent);
  if (info.startsWith("歩行") || info.startsWith("着地")) { console.log(info); await page.click("[data-fview=side]"); await page.waitForTimeout(600); await page.screenshot({ path: "/tmp/claude-1000/-home-sksat/3b79048e-29b9-4884-80d3-019470acc91f/scratchpad/perch.png" }); break; }
  await page.waitForTimeout(2000);
}
console.log(await page.evaluate(() => document.querySelector("#flightinfo").textContent));
await browser.close();
