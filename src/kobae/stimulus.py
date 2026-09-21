"""Stimulus state -> per-neuron drive current (mV-equivalent), DOOMFLY conventions.

A stimulus is a dict (JSON from the viewer):
  gain      : global multiplier (0..2)
  sugar     : bool                     LB3c at 30 mV
  visual    : "off" | "flash" | "left" | "right" | "bar" | "grating" | "loom" | "external"
              ("external": luminance per mapped photoreceptor is pushed by a client, e.g. the body)
  lamina    : bool                     L1/L2/L3/L5 tonic 12 mV (auto-on with visual)
  orn       : {glomerulus: bool}       ORN_<glom> cells at 30 mV
  buttons   : {name: bool}             named cell sets from presets (FLYBOARD-style selectors)
  custom    : [{"glob": "DNa02*", "amp": 30.0, "side": "L"|"R"|""}]
Photoreceptor current = 30 * L / (0.02 + L) with L the luminance at the cell's uv.
"""
from __future__ import annotations

import fnmatch
import math

import numpy as np

from . import model as M

BUTTONS = {
    # name: (description, selector list; a cell matches if ANY selector matches)
    "FOOD":  ("味覚 (LB3c 糖 + SNta*/tpGRN* 咽頭)", [{"type_glob": "LB3c"}, {"type_glob": "SNta*"}, {"type_glob": "tpGRN*"}]),
    "BITTER": ("苦味 GRN (LB1*/LB2*)", [{"type_glob": "LB1*"}, {"type_glob": "LB2*"}]),
    "PAIN":  ("侵害受容 (ppk25 推定)", [{"receptor": "putative_ppk25"}]),
    "PHEROMONE": ("接触化学受容 (ppk23、フェロモン)", [{"receptor": "putative_ppk23"}]),
    "MATE":  ("求愛回路 (fru/dsx 高発現)", [{"fru": "fru_high"}, {"fru": "dsx_high"}, {"fru": "coexpress_high"}]),
    "DOPAMINE": ("ドーパミン細胞 (PAM*/PPL*)", [{"type_glob": "PAM*"}, {"type_glob": "PPL*"}]),
    "WIND":  ("風・聴覚 (ジョンストン器官 JO-*)", [{"type_glob": "JO-*"}]),
    "GIANT": ("巨大線維 (逃避)", [{"type_glob": "GF"}, {"type_glob": "Giant Fiber*"}]),
}


def _match(G, sel: dict) -> np.ndarray:
    ct = G.cell_type
    m = np.ones(G.n, dtype=bool)
    if "type_glob" in sel:
        pat = sel["type_glob"]
        m &= np.array([fnmatch.fnmatchcase(t, pat) for t in ct])
    if "receptor" in sel:
        m &= getattr(G, "receptor_type", np.full(G.n, "")) == sel["receptor"]
    if "fru" in sel:
        m &= getattr(G, "fru_dsx", np.full(G.n, "")) == sel["fru"]
    if "side" in sel and sel["side"]:
        m &= G.soma_side == sel["side"]
    return m


def button_indices(G, name: str) -> np.ndarray:
    _, sels = BUTTONS[name]
    m = np.zeros(G.n, dtype=bool)
    for s in sels:
        m |= _match(G, s)
    return np.flatnonzero(m)


def custom_indices(G, glob: str, side: str = "") -> np.ndarray:
    return np.flatnonzero(_match(G, {"type_glob": glob, "side": side}))


def orn_groups(G) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i in G.orn:
        groups.setdefault(str(G.cell_type[i])[4:], []).append(int(i))
    return {k: np.asarray(v, dtype=np.int32) for k, v in sorted(groups.items())}


def luminance_pattern(uv: np.ndarray, mode: str, t_s: float) -> np.ndarray:
    """Luminance in [0,1] per mapped photoreceptor for a visual mode at sim time t."""
    x, y = uv[:, 0], uv[:, 1]
    if mode == "flash":
        return np.ones(len(uv), np.float32) if (t_s % 1.0) < 0.5 else np.zeros(len(uv), np.float32)
    if mode == "left":
        return (x < 0.5).astype(np.float32)
    if mode == "right":
        return (x >= 0.5).astype(np.float32)
    if mode == "bar":
        c = (t_s * 0.5) % 1.2 - 0.1
        return (np.abs(x - c) < 0.05).astype(np.float32)
    if mode == "grating":
        return (0.5 + 0.5 * np.sign(np.sin(2 * math.pi * (x * 6 - t_s * 1.0)))).astype(np.float32)
    if mode == "loom":
        r = 0.05 + 0.6 * ((t_s % 2.0) / 2.0) ** 2
        return ((x - 0.5) ** 2 + (y - 0.5) ** 2 < r * r).astype(np.float32)
    return np.zeros(len(uv), np.float32)


class Stimulator:
    def __init__(self, G):
        self.G = G
        self.orn = orn_groups(G)
        self._button_cache: dict[str, np.ndarray] = {}
        self.state = {"gain": 1.0, "sugar": False, "visual": "off", "lamina": False,
                      "orn": {}, "buttons": {}, "custom": []}
        self.retina_lum = np.zeros(len(G.retina), np.float32)
        self.external_lum: np.ndarray | None = None   # set by the body process (visual="external")

    def set(self, patch: dict):
        for k, v in patch.items():
            if k in self.state:
                self.state[k] = v

    def button(self, name):
        if name not in self._button_cache:
            self._button_cache[name] = button_indices(self.G, name)
        return self._button_cache[name]

    def drive(self, t_s: float) -> np.ndarray:
        G, s = self.G, self.state
        d = np.zeros(G.n, np.float32)
        g = float(s.get("gain", 1.0))
        if s.get("sugar"):
            d[G.sugar] = M.SUGAR_DRIVE
        vis = s.get("visual", "off")
        if vis == "external" and self.external_lum is not None:
            lum = self.external_lum
            self.retina_lum = lum
            d[G.retina] = 30.0 * lum / (0.02 + lum)
            d[G.lamina] = M.LAMINA_TONIC
        elif vis != "off":
            lum = luminance_pattern(G.uv, vis, t_s)
            self.retina_lum = lum
            d[G.retina] = 30.0 * lum / (0.02 + lum)
            d[G.lamina] = M.LAMINA_TONIC
        elif s.get("lamina"):
            self.retina_lum[:] = 0
            d[G.lamina] = M.LAMINA_TONIC
        else:
            self.retina_lum[:] = 0
        for glom, on in (s.get("orn") or {}).items():
            if on and glom in self.orn:
                d[self.orn[glom]] = 30.0
        for name, on in (s.get("buttons") or {}).items():
            if on and name in BUTTONS:
                d[self.button(name)] = 30.0
        for c in s.get("custom") or []:
            idx = custom_indices(G, c.get("glob", ""), c.get("side", ""))
            d[idx] = float(c.get("amp", 30.0))
        return d * g
