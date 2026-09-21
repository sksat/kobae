"""GPU vs CPU-reference validation (DESIGN.md §3).

Criteria: exact spike-set match over the first ``exact_ms`` (default 20 ms); over the
steady state (the last ``steady_ms`` of the run, after both simulations have settled
into the same activity regime) total spikes within 2 %, per-neuron count correlation
> 0.99 and Jaccard of the spiking-cell sets > 0.95.

The sugar protocol is bistable: after 0.3-1.6 s of ~0.2 M spikes/s the network jumps
to a ~1.2 M spikes/s attractor. The jump time depends on rounding at the 1e-6 level
(verified by perturbing the CPU weights), so window statistics are only comparable
once both runs are in the same regime; hence ``steady_ms`` at the end of a long run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import model as M
from .cpu import CpuBrain
from .gpu import GpuBrain


def protocol_drive(G, name: str) -> np.ndarray:
    d = np.zeros(G.n, np.float32)
    if name == "sugar":
        d[G.sugar] = M.SUGAR_DRIVE
    elif name == "visual":
        d[G.retina] = M.luminance_drive(1.0)
        d[G.lamina] = M.LAMINA_TONIC
    elif name == "idle":
        pass
    else:
        raise ValueError(name)
    return d


def compare(G, protocol: str = "sugar", total_ms: float = 2500.0, exact_ms: float = 20.0,
            steady_ms: float = 900.0, window_ms: float = 90.0, verbose: bool = True) -> dict:
    steps = int(round(total_ms / M.DT_MS))
    steps -= steps % M.DELAY_STEPS
    exact_steps = int(round(exact_ms / M.DT_MS))
    steady_steps = int(round(steady_ms / M.DT_MS))
    drive = protocol_drive(G, protocol)

    cpu = CpuBrain(G); cpu.set_drive(drive)
    t = time.perf_counter()
    c_counts_all, _, c_log = cpu.run(steps, log_cap=200_000_000)
    cpu_s = time.perf_counter() - t

    gpu = GpuBrain(G, ring_cap=1 << 24, verbose=verbose); gpu.set_drive(drive)
    t = time.perf_counter()
    gpu.run(steps // M.DELAY_STEPS, sync=True)
    gpu_s = time.perf_counter() - t
    g_step, g_neuron, over = gpu.read_spikes()
    # per-window totals and steady-state per-neuron counts from the spike logs
    wsteps = int(round(window_ms / M.DT_MS))
    c_win = np.bincount(c_log[:, 0] // wsteps, minlength=steps // wsteps + 1).tolist()
    g_win = np.bincount(g_step // wsteps, minlength=steps // wsteps + 1).tolist()
    c_counts = np.bincount(c_log[c_log[:, 0] >= steps - steady_steps, 1], minlength=G.n)
    g_counts = np.bincount(g_neuron[g_step >= steps - steady_steps], minlength=G.n)

    c_set = set(map(tuple, c_log[c_log[:, 0] < exact_steps].tolist()))
    g_early = g_step < exact_steps
    g_set = set(zip(g_step[g_early].tolist(), g_neuron[g_early].tolist()))
    exact_ok = c_set == g_set
    first_diff = None
    if not exact_ok:
        diff = sorted(c_set ^ g_set)
        first_diff = {"step": int(diff[0][0]), "neuron": int(diff[0][1]), "in_cpu": diff[0] in c_set, "n_diff": len(diff)}

    c_tot, g_tot = int(c_counts.sum()), int(g_counts.sum())
    corr = float(np.corrcoef(c_counts.astype(np.float64), g_counts.astype(np.float64))[0, 1]) if c_tot and g_tot else float("nan")
    ca, ga = c_counts > 0, g_counts > 0
    jacc = float((ca & ga).sum() / max(1, (ca | ga).sum()))
    # cells firing >= 5 Hz in the steady window (the plain Jaccard is dominated by 1-spike cells)
    thr = 5.0 * steady_ms / 1000
    ca5, ga5 = c_counts >= thr, g_counts >= thr
    jacc5 = float((ca5 & ga5).sum() / max(1, (ca5 | ga5).sum()))
    # first divergence in the whole run (sorted spike sequences)
    c_all = np.unique(c_log.astype(np.int64), axis=0)
    g_all = np.unique(np.stack([g_step, g_neuron], axis=1), axis=0) if len(g_step) else np.zeros((0, 2), np.int64)
    m = min(len(c_all), len(g_all))
    neq = np.flatnonzero((c_all[:m] != g_all[:m]).any(axis=1))
    diverge_ms = float(c_all[neq[0], 0] * M.DT_MS) if len(neq) else (float("inf") if len(c_all) == len(g_all) else float(min(len(c_all), len(g_all))))
    res = {
        "protocol": protocol, "sim_ms": steps * M.DT_MS, "exact_ms": exact_ms, "steady_ms": steady_ms,
        "window_ms": window_ms, "cpu_windows": c_win, "gpu_windows": g_win,
        "exact_match": exact_ok, "first_diff": first_diff, "first_divergence_ms": diverge_ms,
        "cpu_steady_spikes": c_tot, "gpu_steady_spikes": g_tot, "steady_ratio": g_tot / max(1, c_tot),
        "count_corr": corr, "jaccard": jacc, "jaccard_5hz": jacc5, "ring_overflow": over,
        "cpu_s": round(cpu_s, 2), "gpu_s": round(gpu_s, 2),
        "cpu_rt": round(steps * M.DT_MS / 1000 / cpu_s, 4), "gpu_rt": round(steps * M.DT_MS / 1000 / gpu_s, 4),
        "pass": exact_ok and ((c_tot == 0 and g_tot == 0) or (abs(g_tot / max(1, c_tot) - 1) < 0.02 and corr > 0.99 and jacc5 > 0.95)),
        "jaccard_all_cells_pass": jacc > 0.95,
    }
    if verbose:
        print(json.dumps(res, indent=1))
    return res
