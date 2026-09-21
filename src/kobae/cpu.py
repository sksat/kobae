"""CPU reference: a faithful port of DOOMFLY's ``doom/engine.py`` ``advance`` (numba).

Event-driven over an "active" set; every edge retained; 1.8 ms delay queue;
arrivals dropped while refractory. Used to validate the GPU kernels, not for speed.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from . import model as M


@njit(cache=True)
def advance(ptr, post, weight, v, g, refractory, drive, queue, queue_count, cursor, steps, dt,
            counts, active, active_flag, nactive, log, log_count, log_cap):
    av = math.exp(-dt / 20); ag = math.exp(-dt / 5)
    coupling = (av - ag) / 3
    delay = int(round(1.8 / dt)); rfc = int(round(2.2 / dt))
    delay_slots = queue.shape[0]
    for step in range(steps):
        slot = cursor % delay_slots
        future = (cursor + delay) % delay_slots
        for k in range(nactive[0]):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                v[i] = -52 + (v[i] + 52) * av + drive[i] * (1 - av) + g[i] * coupling
                g[i] *= ag
                if v[i] > -45:
                    counts[i] += 1
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1
                    if log_count[0] < log_cap:
                        log[log_count[0], 0] = step
                        log[log_count[0], 1] = i
                    log_count[0] += 1
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                g[j] += weight[e]
                if active_flag[j] == 0:
                    active_flag[j] = 1; active[nactive[0]] = j; nactive[0] += 1
        queue_count[slot] = 0
        for q in range(queue_count[future]):
            i = queue[future, q]; v[i] = -52; g[i] = 0; refractory[i] = rfc
        cursor += 1
    return cursor


class CpuBrain:
    def __init__(self, graph):
        self.G = graph
        n = graph.n
        self.n = n
        self.ptr = graph.ptr.astype(np.int64); self.post = graph.post.astype(np.int32)
        self.weight = graph.weight.astype(np.float32)
        self.v = np.full(n, M.V_REST, dtype=np.float32); self.g = np.zeros(n, dtype=np.float32)
        self.drive = np.zeros(n, dtype=np.float32); self.refractory = np.zeros(n, dtype=np.int16)
        self.queue = np.zeros((M.DELAY_STEPS + 1, n), dtype=np.int32)
        self.queue_count = np.zeros(self.queue.shape[0], dtype=np.int32)
        self.counts = np.zeros(n, dtype=np.int32)
        self.active = np.zeros(n, dtype=np.int32); self.active_flag = np.zeros(n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        self.cursor = 0
        self.sim_steps = 0
        self.total_spikes = 0

    def set_drive(self, drive: np.ndarray):
        drive = np.asarray(drive, dtype=np.float32)
        if drive.shape != (self.n,):
            raise ValueError("drive shape")
        self.drive[:] = drive
        # neurons with nonzero drive must be in the active set (DOOMFLY seeds retina/lamina/sugar)
        for i in np.flatnonzero(drive != 0):
            if self.active_flag[i] == 0:
                self.active_flag[i] = 1; self.active[self.nactive[0]] = i; self.nactive[0] += 1

    def run(self, steps: int, log_cap: int = 0):
        """Advance ``steps`` ticks; returns (counts, elapsed_s, spike_log[(step,neuron)])."""
        self.counts.fill(0)
        log = np.zeros((max(log_cap, 1), 2), dtype=np.int64); log_count = np.zeros(1, dtype=np.int64)
        t = time.perf_counter()
        self.cursor = advance(self.ptr, self.post, self.weight, self.v, self.g, self.refractory, self.drive,
                              self.queue, self.queue_count, self.cursor, steps, M.DT_MS, self.counts,
                              self.active, self.active_flag, self.nactive, log, log_count, log_cap)
        el = time.perf_counter() - t
        self.sim_steps += steps
        self.total_spikes += int(self.counts.sum())
        return self.counts.copy(), el, log[:min(log_count[0], log_cap)]
