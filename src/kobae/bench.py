"""Realtime-factor benchmark of the GPU (and optionally CPU) backend.

Protocols: idle, sugar (LB3c 30 mV), visual (all R1-R6 at luminance 1 + lamina 12 mV).
Realtime factor = simulated seconds / wall seconds, model already loaded on the device,
measured over ``sim_s`` seconds after ``warm_s`` seconds of warm-up (the sugar protocol
is measured in its high-activity attractor, i.e. after the bistable jump).
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np

from . import model as M
from .validate import protocol_drive


def bench_gpu(G, protocols=("idle", "sugar", "visual"), sim_s: float = 2.0, warm_s: float = 2.0, verbose=True) -> list[dict]:
    from .gpu import GpuBrain
    gpu = GpuBrain(G, verbose=verbose)
    out = []
    for proto in protocols:
        gpu.reset_state()
        gpu.set_drive(protocol_drive(G, proto))
        gpu.run(int(warm_s * 1000 / (M.DELAY_STEPS * M.DT_MS)), sync=True)
        gpu.read_counts(clear=True)
        batches = int(sim_s * 1000 / (M.DELAY_STEPS * M.DT_MS))
        t = time.perf_counter()
        gpu.run(batches, sync=True)
        wall = time.perf_counter() - t
        counts = gpu.read_counts(clear=True)
        sim = batches * M.DELAY_STEPS * M.DT_MS / 1000
        r = {"backend": "gpu", "device": gpu.info.get("device"), "protocol": proto, "sim_s": sim, "wall_s": round(wall, 3),
             "realtime_x": round(sim / wall, 3), "spikes_per_s": int(counts.sum() / sim), "active_cells": int((counts > 0).sum())}
        out.append(r)
        if verbose:
            print(json.dumps(r))
    return out


def bench_cpu(G, protocols=("idle", "sugar", "visual"), sim_s: float = 1.0, warm_s: float = 2.0, verbose=True) -> list[dict]:
    from .cpu import CpuBrain
    out = []
    for proto in protocols:
        b = CpuBrain(G); b.set_drive(protocol_drive(G, proto))
        b.run(int(warm_s * 1000 / M.DT_MS))
        steps = int(sim_s * 1000 / M.DT_MS)
        t = time.perf_counter()
        counts, _, _ = b.run(steps)
        wall = time.perf_counter() - t
        sim = steps * M.DT_MS / 1000
        r = {"backend": "cpu-numba", "device": platform.processor() or platform.machine(), "protocol": proto, "sim_s": sim,
             "wall_s": round(wall, 3), "realtime_x": round(sim / wall, 4), "spikes_per_s": int(counts.sum() / sim),
             "active_cells": int((counts > 0).sum())}
        out.append(r)
        if verbose:
            print(json.dumps(r))
    return out
