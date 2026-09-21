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

import base64
import io

import numpy as np
import websockets
from PIL import Image

from .flight import FlightBody

MAGIC = 0x4B4F4241


def luminance_from_eyes(left: np.ndarray, right: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample the two eye images at each photoreceptor's uv.

    The uv map (DoomFly) puts the left eye's columns in x in [0, 0.6] and the right eye's in
    [0.4, 1.0]; each is stretched over its own eye camera image."""
    def gray(rgb):
        g = rgb[..., :3].astype(np.float32).mean(-1) / 255.0
        return g ** 2.2  # sRGB -> linear
    gl, gr = gray(left), gray(right)
    h, w = gl.shape
    out = np.zeros(len(uv), np.float32)
    x, y = uv[:, 0], uv[:, 1]
    L = x < 0.5
    xl = np.clip((x[L] / 0.6 * (w - 1)).astype(int), 0, w - 1); yl = np.clip((y[L] * (h - 1)).astype(int), 0, h - 1)
    out[L] = gl[yl, xl]
    R = ~L
    xr = np.clip(((x[R] - 0.4) / 0.6 * (w - 1)).astype(int), 0, w - 1); yr = np.clip((y[R] * (h - 1)).astype(int), 0, h - 1)
    out[R] = gr[yr, xr]
    return out


class Decoder:
    """Two read-out modes, both engineered mappings (DoomFly's decoders):
    biological: DNa02 R-L -> turn, DNp09 -> forward, MDN -> backward (silent under pure visual input)
    bci       : DNp20 R-L -> turn, DNpe017 -> forward (visual descending neurons; the default here)"""

    def __init__(self, readouts: list[dict], mode="bci", k_turn=0.12, k_fwd=0.4, k_back=0.3, base_speed=12.0, tau=0.1):
        self.readouts = readouts
        self.mode = mode
        self.k_turn, self.k_fwd, self.k_back, self.base = k_turn, k_fwd, k_back, base_speed
        self.tau = tau
        self.rates = np.zeros(len(readouts))

    def update(self, rates, dt):
        a = 1 - np.exp(-dt / self.tau)
        self.rates += a * (np.asarray(rates) - self.rates)

    def rate(self, typ, side=None):
        return sum(r for r, ro in zip(self.rates, self.readouts) if ro["type"] == typ and (side is None or ro["side"] == side))

    def command(self):
        if self.mode == "bci":
            # DNp20 projects ipsilaterally; stronger right -> turn right (negative yaw)
            yaw = float(np.clip((self.rate("DNp20", "L") - self.rate("DNp20", "R")) * self.k_turn, -6, 6))
            speed = float(np.clip(self.base + self.rate("DNpe017") * self.k_fwd, 2, 40))
        else:
            yaw = float(np.clip((self.rate("DNa02", "R") - self.rate("DNa02", "L")) * self.k_turn, -6, 6))
            speed = float(np.clip(self.base + self.rate("DNp09") * self.k_fwd - self.rate("MDN") * self.k_back, 2, 40))
        return speed, yaw


async def run(brain: str, policy: str, wpg: str | None, seconds: float, video: str | None, verbose: bool, mode: str = "bci"):
    body = FlightBody(policy, wpg)
    import aiohttp
    async with aiohttp.ClientSession() as s:
        meta = await (await s.get(brain.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "/api/meta"))).json()
    uv = np.asarray(meta["uv"], np.float32).reshape(-1, 2)
    dec = Decoder(meta["readouts"], mode=mode)
    frames = []
    async with websockets.connect(brain, max_size=1 << 26) as ws:
        await ws.send(json.dumps({"op": "stim", "patch": {"visual": "external"}}))
        t_start = time.perf_counter(); last = t_start; sim_body = 0.0
        # one loop iteration = 10 ms of body time (50 control steps) ~ 5.5 brain batches
        while time.perf_counter() - t_start < seconds:
            body.step(50); sim_body += 0.01
            left, right = body.eyes(64, 48)
            lum = luminance_from_eyes(left, right, uv)
            rgb = np.concatenate([left, right], axis=1)
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
            p, q = body.body_pose()
            if int(round(sim_body * 100)) % 5 == 0:   # 20 Hz relay to the viewer
                buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=70)
                await ws.send(json.dumps({"op": "body", "jpeg": base64.b64encode(buf.getvalue()).decode(),
                                          "pos": [float(p[0]), float(p[1]), float(p[2])],
                                          "yaw": float(2 * np.arctan2(q[3], q[0])), "cmd": [speed, yaw], "t": sim_body}))
            if video is not None and len(frames) < 3000:
                frames.append(body.render(camera_id=1, width=320, height=240))
            if verbose and int(sim_body * 100) % 50 == 0:
                p, _ = body.body_pose()
                print(f"body {sim_body:5.2f}s  pos {p.round(2)}  cmd speed {speed:5.1f} yaw {yaw:+5.2f}  "
                      f"DNp20 L/R {dec.rate('DNp20','L'):.0f}/{dec.rate('DNp20','R'):.0f} DNpe017 {dec.rate('DNpe017'):.0f} "
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
    ap.add_argument("--mode", choices=["bci", "biological"], default="bci")
    a = ap.parse_args()
    asyncio.run(run(a.brain, a.policy, a.wpg, a.seconds, a.video, a.v, a.mode))


if __name__ == "__main__":
    main()
