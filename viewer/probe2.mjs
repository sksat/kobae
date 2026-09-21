import { chromium } from "playwright";
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on("pageerror", e => console.log("pageerror:", String(e).slice(0, 300)));
page.on("console", m => { if (m.type() === "error") console.log("console:", m.text().slice(0, 200)); });
await page.goto("http://localhost:8765/", { waitUntil: "load" });
await page.waitForTimeout(9000);
console.log(await page.evaluate(() => ({ mode: document.querySelector("#flight").hidden ? "brain" : "flight", fps: document.querySelector("#fps").textContent, info: document.querySelector("#flightinfo").textContent })));
await page.screenshot({ path: "/tmp/claude-1000/-home-sksat/3b79048e-29b9-4884-80d3-019470acc91f/scratchpad/flightview.png" });
// drag in the flight view
const box = await page.locator("#fly").boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down();
await page.mouse.move(box.x + box.width / 2 + 250, box.y + box.height / 2 + 80, { steps: 10 }); await page.mouse.up();
await page.waitForTimeout(700);
await page.screenshot({ path: "/tmp/claude-1000/-home-sksat/3b79048e-29b9-4884-80d3-019470acc91f/scratchpad/flightview2.png" });
await browser.close();
