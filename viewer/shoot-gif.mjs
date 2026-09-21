// Video capture for the README GIF: the flight view with a body loop running, recorded by Playwright
// (software GL: the workstation GPU stays out of it). Writes <outdir>/flight.webm and <outdir>/hud.tsv (t_ms, HUD text).
//   node shoot-gif.mjs <outdir> [seconds] [base-url]
// The page paints ~5 fps under software GL, so the GIF is made from the distinct frames played at 10 fps (~2x):
//   ffmpeg -ss <start> -t <len> -i flight.webm -vf "mpdecimate=hi=512:lo=192:frac=0.2,setpts=N/10/TB" -r 10 -c:v libx264 -crf 12 distinct.mp4
//   ffmpeg -i distinct.mp4 -vf "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -r 10 docs/viewer-flight.gif
// Pick <start> from hud.tsv (video time = hud t + ~16 s): a stretch with turning / climbing, flight at both ends.
import { chromium } from "playwright";
import { appendFileSync, writeFileSync, renameSync } from "node:fs";
const out = process.argv[2], secs = Number(process.argv[3] || 40), base = process.argv[4] || "http://localhost:8765/";
const size = { width: 1280, height: 720 };
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const ctx = await browser.newContext({ viewport: size, deviceScaleFactor: 1, recordVideo: { dir: out, size } });
const page = await ctx.newPage();
page.on("pageerror", e => console.log("pageerror:", String(e).slice(0, 300)));
await page.goto(base, { waitUntil: "load" });
await page.waitForTimeout(8000);
await page.click("[data-mode=flight]"); await page.waitForTimeout(5000);
writeFileSync(`${out}/hud.tsv`, "");
const t0 = Date.now();
while (Date.now() - t0 < secs * 1000) {
  const info = await page.evaluate(() => `${document.querySelector("#flightinfo").textContent}\t${document.querySelector("#fps").textContent}`);
  appendFileSync(`${out}/hud.tsv`, `${Date.now() - t0}\t${info}\n`);
  await page.waitForTimeout(500);
}
const video = page.video();
await ctx.close();
renameSync(await video.path(), `${out}/flight.webm`);
await browser.close();
console.log("done");
