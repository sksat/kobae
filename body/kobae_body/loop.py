"""Closed loop: kobae brain server <-> flybody flight body.

  brain read-outs (descending neurons, Hz) --decode--> Command(speed, yaw, climb)
  body camera (fly's eye view) --luminance at R1-R6 uv--> brain retina (op "retina")

Decoder (engineered mapping, same spirit as DoomFly's BCI decoder; not biology):
  yaw   = k_turn * (DNa02_R - DNa02_L)      DNa02 drives ipsilateral turning
  speed = base + k_fwd * DNp09 - k_back * MDN
  climb = 0 (flight height is held by the policy's reference)

Usage: uv run python -m kobae_body.loop --brain ws://localhost:8765/ws --policy policies/flight.npz
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time

import numpy as np
import websockets

from .flight import FlightBody

MAGIC = 0x4B4F4241


def luminance_from_frame(rgb: np.ndarray, uv: np.ndarray, fov_split: float = 0.6) -> np.ndarray:
    """Sample a camera frame at each photoreceptor's uv (left 60 % / right 60 % overlap, DoomFly convention)."""
    h, w, _ = rgb.shape
    gray = rgb[..., :3].astype(np.float32).mean(-1) / 255.0
    gray = gray ** 2.2  # sRGB -> linear-ish
    x = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    y = np.clip((uv[:, 1] * (h - 1)).astype(int), 0, h - 1)
    return gray[y, x]


class Decoder:
    def __init__(self, readouts: list[dict], k_turn=0.06, k_fwd=0.3, k_back=0.3, base_speed=15.0, tau=0.1):
        self.readouts = readouts
        self.k_turn, self.k_fwd, self.k_back, self.base = k_turn, k_fwd, k_back, base_speed
        self.tau = tau
        self.rates = np.zeros(len(readouts))

    def update(self, rates, dt):
        a = 1 - np.exp(-dt / self.tau)
        self.rates += a * (np.asarray(rates) - self.rates)

    def rate(self, typ, side=None):
        return sum(r for r, ro in zip(self.rates, self.readouts) if ro["type"] == typ and (side is None or ro["side"] == side))

    def command(self):
        yaw = float(np.clip((self.rate("DNa02", "R") - self.rate("DNa02", "L")) * self.k_turn, -6, 6))
        speed = float(np.clip(self.base + self.rate("DNp09") * self.k_fwd - self.rate("MDN") * self.k_back, 2, 40))
        return speed, yaw


async def run(brain: str, policy: str, wpg: str | None, seconds: float, video: str | None, verbose: bool):
    body = FlightBody(policy, wpg)
    import aiohttp
    async with aiohttp.ClientSession() as s:
        meta = await (await s.get(brain.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "/api/meta"))).json()
    uv = np.asarray(meta["uv"], np.float32).reshape(-1, 2)
    dec = Decoder(meta["readouts"])
    frames = []
    async with websockets.connect(brain, max_size=1 << 26) as ws:
        await ws.send(json.dumps({"op": "stim", "patch": {"visual": "external"}}))
        t_start = time.perf_counter(); last = t_start; sim_body = 0.0
        # one loop iteration = 10 ms of body time (50 control steps) ~ 5.5 brain batches
        while time.perf_counter() - t_start < seconds:
            body.step(50); sim_body += 0.01
            rgb = body.render(camera_id=1, width=160, height=120)
            lum = luminance_from_frame(rgb, uv)
            await ws.send(json.dumps({"op": "retina", "lum": lum.round(3).tolist()}))
            await ws.send(json.dumps({"op": "readouts"}))
            # drain messages until the readouts reply
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("op") == "readouts":
                        dec.update(m["rates"], 0.01); break
            speed, yaw = dec.command()
            body.command.speed, body.command.yaw = speed, yaw
            if video is not None and len(frames) < 3000:
                frames.append(body.render(camera_id=1, width=320, height=240))
            if verbose and int(sim_body * 100) % 50 == 0:
                p, _ = body.body_pose()
                print(f"body {sim_body:5.2f}s  pos {p.round(2)}  cmd speed {speed:5.1f} yaw {yaw:+5.2f}  "
                      f"DNa02 L/R {dec.rate('DNa02','L'):.0f}/{dec.rate('DNa02','R'):.0f}  wall/body {(time.perf_counter()-t_start)/sim_body:.1f}x")
    if video and frames:
        import mediapy
        mediapy.write_video(video, frames, fps=100)
        print("wrote", video)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="ws://localhost:8765/ws")
    ap.add_argument("--policy", default="policies/flight.npz")
    ap.add_argument("--wpg", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a.brain, a.policy, a.wpg, a.seconds, a.video, a.v))


if __name__ == "__main__":
    main()
