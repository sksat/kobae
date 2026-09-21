"""Shared LIF model constants (DOOMFLY / Shiu et al. 2024 style)."""
import math

DT_MS = 0.1
V_REST = -52.0
V_THRESH = -45.0
TAU_M_MS = 20.0
TAU_G_MS = 5.0
DELAY_MS = 1.8
REFRACTORY_MS = 2.2

DELAY_STEPS = int(round(DELAY_MS / DT_MS))          # 18
REFRACTORY_STEPS = int(round(REFRACTORY_MS / DT_MS))  # 22
AV = math.exp(-DT_MS / TAU_M_MS)
AG = math.exp(-DT_MS / TAU_G_MS)
COUPLING = (AV - AG) / 3.0

# Fixed-point scale for synaptic arrivals accumulated with integer atomics on the GPU.
G_SCALE = 65536.0

# Stimulus currents (mV-equivalent), as in DOOMFLY doom/engine.py
SUGAR_DRIVE = 30.0
LAMINA_TONIC = 12.0


def luminance_drive(lum: float) -> float:
    """Photoreceptor current for a [0,1] luminance (saturating, DOOMFLY)."""
    return 30.0 * lum / (0.02 + lum)
