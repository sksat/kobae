"""Display positions for every neuron.

84 % of retained neurons have a measured soma (``somaLocation`` / ``tosomaLocation``,
EM voxel units of 8 nm). The rest (mostly sensory cells whose soma lies outside the
imaged volume, and some optic-lobe cells) get a fallback: the mean position of their
synaptic partners, iterated, and finally the centroid of their superclass with jitter.
Positions are returned in micrometres, centred on the brain.
"""
from __future__ import annotations

import numpy as np

VOXEL_NM = 8.0


def compute_positions(G, seed: int = 0):
    n = G.n
    soma = np.asarray(G.soma, dtype=np.float64)
    if hasattr(G, "tosoma"):
        miss = ~np.isfinite(soma[:, 0])
        soma[miss] = G.tosoma[miss]
    measured = np.isfinite(soma[:, 0])
    pos = soma.copy()
    pos[~measured] = np.nan
    ptr, post = G.ptr, G.post
    # partner-mean fallback, two passes (targets then sources)
    pre = np.repeat(np.arange(n), np.diff(ptr))
    for _ in range(3):
        have = np.isfinite(pos[:, 0])
        need = ~have
        if not need.any():
            break
        # edges where the missing cell is pre and the target has a position
        m = need[pre] & have[post]
        acc = np.zeros((n, 3)); cnt = np.zeros(n)
        for k in range(3):
            acc[:, k] += np.bincount(pre[m], weights=pos[post[m], k], minlength=n)
        cnt += np.bincount(pre[m], minlength=n)
        m2 = need[post] & have[pre]
        for k in range(3):
            acc[:, k] += np.bincount(post[m2], weights=pos[pre[m2], k], minlength=n)
        cnt += np.bincount(post[m2], minlength=n)
        fill = need & (cnt > 0)
        pos[fill] = acc[fill] / cnt[fill, None]
    have = np.isfinite(pos[:, 0])
    rng = np.random.default_rng(seed)
    if (~have).any():
        sc = G.superclass
        for s in np.unique(sc[~have]):
            sel = (~have) & (sc == s)
            ref = have & (sc == s)
            c = pos[ref].mean(axis=0) if ref.any() else np.nanmean(pos[have], axis=0)
            spread = pos[ref].std(axis=0) if ref.sum() > 10 else np.array([2000.0] * 3)
            pos[sel] = c + rng.normal(size=(sel.sum(), 3)) * spread * 0.5
    um = pos * VOXEL_NM / 1000.0
    um -= np.median(um[measured], axis=0)
    # body-aligned frame for display: brain->VNC axis is -y (brain up), the widest
    # brain direction orthogonal to it is x (animal's left at -x), z = depth
    brain = np.isin(G.superclass, ["cb_intrinsic", "ol_intrinsic"]) & measured
    vnc = np.isin(G.superclass, ["vnc_intrinsic", "vnc_motor"]) & measured
    b = um[vnc].mean(axis=0) - um[brain].mean(axis=0)
    y = -b / np.linalg.norm(b)
    P = um[brain] - um[brain].mean(axis=0)
    P -= np.outer(P @ y, y)
    w, v = np.linalg.eigh(np.cov(P.T))
    x = v[:, -1]; x -= (x @ y) * y; x /= np.linalg.norm(x)
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    out = um @ R
    left = (G.soma_side == "L") & measured
    if left.any() and out[left, 0].mean() > out[measured, 0].mean():
        out[:, 0] *= -1; out[:, 2] *= -1
    return out.astype(np.float32), measured
