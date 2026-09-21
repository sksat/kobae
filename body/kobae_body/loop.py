"""Closed loop: kobae brain server <-> flybody flight body.

  brain read-outs (descending neurons, Hz) --decode--> Command(speed, yaw, climb)
  body camera (fly's eye view) --luminance at R1-R6 uv--> brain retina (op "retina")

The brain runs in lockstep with the body (op "step"), so brain time == body time. To viewers the
body process relays 'KOBP' binary frames: pose of every fly body (for the browser rig) + eye images.

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

import io

import numpy as np
import websockets
from PIL import Image

from .flight import FlightBody
from .behavior import Behaviour

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
    bci       : DNp20 R-L -> turn, DNpe017 -> forward (visual descending neurons; the default here)

    Calibration: the connectome is not mirror-symmetric. Under uniform illumination the right DNp20
    fires ~11 Hz more than the left (44 vs 33 Hz; 21 vs 10 Hz in the dark), which as a raw command
    is a permanent right turn. ``calibrate`` records per-readout baselines under uniform light and
    the decoder uses rate - baseline (DoomFly calibrated its BCI read-outs the same way)."""

    def __init__(self, readouts: list[dict], mode="bci", k_turn=0.12, k_fwd=0.4, k_back=0.3, base_speed=12.0, tau=0.1):
        self.readouts = readouts
        self.mode = mode
        self.k_turn, self.k_fwd, self.k_back, self.base = k_turn, k_fwd, k_back, base_speed
        self.tau = tau
        self.rates = np.zeros(len(readouts))
        self.baseline = np.zeros(len(readouts))

    def update(self, rates, dt, baseline_tau: float = 20.0):
        a = 1 - np.exp(-dt / self.tau)
        self.rates += a * (np.asarray(rates) - self.rates)
        # slow adaptation of the baseline (high-pass): the network has several activity regimes
        # (e.g. the odour/sugar-driven high-activity attractor persists after the stimulus), and
        # a baseline measured in one regime saturates the command in another
        b = 1 - np.exp(-dt / baseline_tau)
        self.baseline += b * (self.rates - self.baseline)

    def rate(self, typ, side=None):
        return sum(r - b for r, b, ro in zip(self.rates, self.baseline, self.readouts)
                   if ro["type"] == typ and (side is None or ro["side"] == side))

    def raw(self, typ, side=None):
        return sum(r for r, ro in zip(self.rates, self.readouts) if ro["type"] == typ and (side is None or ro["side"] == side))

    def command(self):
        if self.mode == "bci":
            # DNp20 projects ipsilaterally; stronger right -> turn right (negative yaw)
            # kept inside the imitation policy's envelope (it was trained on real flight: <= ~30 cm/s)
            yaw = float(np.clip((self.rate("DNp20", "L") - self.rate("DNp20", "R")) * self.k_turn, -3, 3))
            speed = float(np.clip(self.base + self.rate("DNpe017") * self.k_fwd, 5, 25))
        else:
            yaw = float(np.clip((self.rate("DNa02", "R") - self.rate("DNa02", "L")) * self.k_turn, -6, 6))
            speed = float(np.clip(self.base + self.rate("DNp09") * self.k_fwd - self.rate("MDN") * self.k_back, 2, 40))
        return speed, yaw


async def run(brain: str, policy: str, wpg: str | None, seconds: float, video: str | None, verbose: bool, mode: str = "bci",
              camera: str = "walker/track2", video_fps: int = 30, body_mode: str = "kinematic", realtime: bool = True):
    body = FlightBody(policy, wpg, kinematic=(body_mode == "kinematic"), legs=(body_mode == "kinematic"))
    behaviour = Behaviour(body) if body_mode == "kinematic" else None
    import mujoco
    m = body.physics.model.ptr
    walker_ids = [i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("walker/")]
    walker_ids = np.asarray(walker_ids, dtype=np.int64)

    import aiohttp
    async with aiohttp.ClientSession() as s:
        meta = await (await s.get(brain.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "/api/meta"))).json()
    uv = np.asarray(meta["uv"], np.float32).reshape(-1, 2)
    dec = Decoder(meta["readouts"], mode=mode)
    frames = []
    async with websockets.connect(brain, max_size=1 << 26) as ws:
        await ws.send(json.dumps({"op": "stim", "patch": {"visual": "external"}}))
        # calibration: 3 s of uniform illumination, record baselines
        await ws.send(json.dumps({"op": "retina", "lum": [0.5] * len(uv)}))
        acc = []
        for _ in range(33):
            await ws.send(json.dumps({"op": "step", "ms": 90}))
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("op") == "readouts":
                        acc.append(m["rates"]); break
        dec.baseline = np.mean(acc[-11:], axis=0)
        if verbose:
            print("calibrated baselines (Hz):", {ro["type"] + ro["side"]: round(b, 1) for ro, b in zip(dec.readouts, dec.baseline)})
        t_start = time.perf_counter(); last_relay = 0.0; sim_body = 0.0
        # one loop iteration = 9 ms of body time (45 control steps) = 5 brain batches, in LOCKSTEP:
        # the brain advances exactly the body's elapsed time, so brain time == body time
        STEP_S = 0.009
        while time.perf_counter() - t_start < seconds:
            flying = behaviour.update(STEP_S) if behaviour else True
            if flying:
                body.step(45)
            sim_body += STEP_S
            left, right = body.eyes(64, 48)
            lum = luminance_from_eyes(left, right, uv)
            rgb = np.concatenate([left, right], axis=1)
            await ws.send(json.dumps({"op": "retina", "lum": lum.round(3).tolist()}))
            await ws.send(json.dumps({"op": "step", "ms": STEP_S * 1000}))
            while True:   # wait for the brain to finish this step (its reply carries the read-outs)
                msg = await ws.recv()
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("op") == "readouts":
                        dec.update(m["rates"], STEP_S); break
            speed, yaw = dec.command()
            p, q = body.body_pose()
            if flying:
                body.command.speed, body.command.yaw = speed, yaw
                # hold a cruising height (the perches walk up the pillars): gentle descent/climb toward 1.5 cm
                body.command.climb = float(np.clip(0.4 * (1.5 - p[2]), -3.0, 3.0))
            it = int(round(sim_body / STEP_S))
            now = time.perf_counter()
            if now - last_relay >= 1 / 20:   # 20 Hz wall-clock relay to the viewer: poses of all fly bodies + both eyes
                last_relay = now
                poses = body.walker_poses(walker_ids)                      # (n,7) f32
                eyes = np.concatenate([left, right], axis=1)
                ebuf = io.BytesIO(); Image.fromarray(eyes).save(ebuf, format="JPEG", quality=60)
                eb = ebuf.getvalue()
                st = {"FLY": 0, "LAND": 1, "PERCH": 2, "TAKEOFF": 3}.get(behaviour.state if behaviour else "FLY", 0)
                hdr = b"KOBP" + struct.pack("<7fII", sim_body, float(p[0]), float(p[1]), float(p[2]),
                                            float(2 * np.arctan2(q[3], q[0])), speed, yaw, len(walker_ids), len(eb) | (st << 28))
                await ws.send(hdr + walker_ids.astype(np.uint16).tobytes() + poses.tobytes() + eb)
            if realtime:   # live view: do not run the body faster than real time
                lag = sim_body - (time.perf_counter() - t_start)
                if lag > 0:
                    await asyncio.sleep(min(lag, 0.05))
            if video is not None and (sim_body * video_fps) // 1 != ((sim_body - STEP_S) * video_fps) // 1 and len(frames) < 6000:
                fr = body.render(camera_id=camera, width=640, height=480)
                # picture-in-picture: the two eyes (what the brain sees) top-left
                pip = np.concatenate([left, right], axis=1)
                pip = np.asarray(Image.fromarray(pip).resize((256, 96)))
                fr = fr.copy(); fr[8:104, 8:264] = pip
                frames.append(fr)
            if verbose and it % 56 == 0:
                p, _ = body.body_pose()
                print(f"body {sim_body:5.2f}s {behaviour.state if behaviour else 'MUJOCO':7s} pos {p.round(2)}  cmd speed {speed:5.1f} yaw {yaw:+5.2f} crashes {body.relaunches}  "
                      f"DNp20 L/R {dec.rate('DNp20','L'):.0f}/{dec.rate('DNp20','R'):.0f} DNpe017 {dec.rate('DNpe017'):.0f} "
                      f"DNa02 L/R {dec.rate('DNa02','L'):.0f}/{dec.rate('DNa02','R'):.0f}  brain {m['sim_ms']/1000:.2f}s  wall/body {(time.perf_counter()-t_start)/sim_body:.1f}x")
    if video and frames:
        import mediapy
        mediapy.write_video(video, frames, fps=video_fps)   # real-time playback
        print("wrote", video, len(frames), "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="ws://localhost:8765/ws")
    ap.add_argument("--policy", default="policies/flight.npz")
    ap.add_argument("--wpg", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--mode", choices=["bci", "biological"], default="bci")
    ap.add_argument("--camera", default="walker/track2", help="walker/track1|track2|track3|back|side|hero, top_camera")
    ap.add_argument("--body", choices=["kinematic", "mujoco"], default="kinematic",
                    help="kinematic: pose follows the commanded trajectory (real time); mujoco: full flight dynamics (~1/13 real time)")
    ap.add_argument("--no-realtime", action="store_true", help="run the body as fast as it can (recordings)")
    a = ap.parse_args()
    asyncio.run(run(a.brain, a.policy, a.wpg, a.seconds, a.video, a.v, a.mode, a.camera, body_mode=a.body, realtime=not a.no_realtime))


if __name__ == "__main__":
    main()
