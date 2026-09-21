"""flybody flight body driven by a command (speed, yaw rate, climb) with the trained flight policy.

The trained policy (Vaxenburg et al. 2025, imitation flight) tracks a reference
centre-of-mass trajectory using a wingbeat pattern generator. Here the reference
trajectory is not a recorded fly but is generated on the fly from a *command*:
forward speed, yaw rate and climb rate. Whoever sets the command steers the fly;
in kobae that is the brain's descending-neuron read-out (see ``loop.py``).

Runs without TensorFlow: the policy MLP is loaded from the .npz made by
``extract_policy.py`` and evaluated in numpy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dm_control import mujoco as dm_mujoco

from flybody.fly_envs import flight_imitation
from flybody.quaternions import mult_quat
from flybody.tasks.task_utils import real2canonical  # noqa: F401  (kept for reference)

CONTROL_DT = 2e-4  # flybody flight control timestep, s (matches _FLY_CONTROL_TIMESTEP)


class NumpyPolicy:
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.n_hidden = int(z["n_hidden"])
        self.w = [z[f"w{k}"] for k in range(self.n_hidden)]
        self.b = [z[f"b{k}"] for k in range(self.n_hidden)]
        self.ln_offset, self.ln_scale = z["ln_offset"], z["ln_scale"]
        self.mean_w, self.mean_b = z["mean_w"], z["mean_b"]
        self.obs_dim = self.w[0].shape[0]
        self.act_dim = self.mean_w.shape[1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        h = x @ self.w[0] + self.b[0]
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        h = (h - mu) / np.sqrt(var + 1e-5) * self.ln_scale + self.ln_offset
        h = np.tanh(h)
        for k in range(1, self.n_hidden):
            h = np.tanh(h @ self.w[k] + self.b[k])
        return np.tanh(h @ self.mean_w + self.mean_b)  # DMPO head: tanh-squashed mean in [-1, 1]


def flatten_obs(obs: dict) -> np.ndarray:
    """Acme batch_concat order: dict keys in their (sorted OrderedDict) order, each flattened."""
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in obs.keys()])


def canonical_to_real(a: np.ndarray, spec) -> np.ndarray:
    lo, hi = spec.minimum, spec.maximum
    return lo + (np.clip(a, -1, 1) + 1) * 0.5 * (hi - lo)


@dataclass
class Command:
    speed: float = 20.0      # cm/s forward
    yaw: float = 0.0         # rad/s, + = counter-clockwise (left turn)
    climb: float = 0.0       # cm/s vertical


class FlightBody:
    """A flying fly whose reference trajectory is regenerated from ``command`` every ``horizon`` steps."""

    def __init__(self, policy_path: str | Path, wpg_pattern_path: str | None = None,
                 horizon_steps: int = 50, future_steps: int = 5, seed: int = 0):
        self.env = flight_imitation(None, wpg_pattern_path, future_steps=future_steps,
                                    terminal_com_dist=float("inf"),
                                    random_state=np.random.RandomState(seed))
        self.env.task._time_limit = float("inf")
        self.policy = NumpyPolicy(policy_path)
        self.spec = self.env.action_spec()
        self.horizon = horizon_steps
        self.future = future_steps
        self.command = Command()
        self.pos = np.array([0.0, 0.0, 1.0]); self.quat = np.array([1.0, 0, 0, 0])
        self._new_segment(reset=True)
        self.timestep = self.env.reset()
        self.t = 0
        self.wall = 0.0

    # reference trajectory segment from the current pose and command
    def _segment(self, n: int):
        c = self.command
        qpos = np.zeros((n, 7)); qvel = np.zeros((n, 6))
        qpos[0, :3] = self.pos; qpos[0, 3:] = self.quat
        yaw0 = 2 * np.arctan2(self.quat[3], self.quat[0])
        dth = c.yaw * CONTROL_DT
        dq = np.array([np.cos(dth / 2), 0, 0, np.sin(dth / 2)])
        w = np.zeros(3); dm_mujoco.mju_quat2Vel(w, dq, 1)
        for i in range(n):
            yaw = yaw0 + c.yaw * i * CONTROL_DT
            qvel[i, :3] = [c.speed * np.cos(yaw), c.speed * np.sin(yaw), c.climb]
            qvel[i, 3:] = w
            if i:
                qpos[i, :3] = qpos[i - 1, :3] + qvel[i, :3] * CONTROL_DT
                qpos[i, 3:] = mult_quat(dq, qpos[i - 1, 3:])
        return qpos, qvel

    def _new_segment(self, reset=False):
        n = self.horizon + self.future + 2
        qpos, qvel = self._segment(n)
        self.env.task._traj_generator.set_next_trajectory(qpos, qvel)
        self._seg_step = 0

    def step(self, n: int = 1):
        """Advance n control steps (0.2 ms each). Returns the last timestep."""
        t0 = time.perf_counter()
        for _ in range(n):
            if self._seg_step >= self.horizon:
                # continue the reference from where the *reference* currently is (not the body),
                # so the target stays smooth; the body keeps tracking it
                task = self.env.task
                k = min(self._seg_step, len(task._ref_qpos) - 1)
                self.pos = task._ref_qpos[k, :3].copy(); self.quat = task._ref_qpos[k, 3:].copy()
                self._new_segment()
                self.timestep = self.env.reset()
            obs = flatten_obs(self.timestep.observation)
            a = self.policy(obs)
            self.timestep = self.env.step(canonical_to_real(a, self.spec))
            self._seg_step += 1; self.t += 1
        self.wall += time.perf_counter() - t0
        return self.timestep

    # ---- readouts for the coupling
    @property
    def physics(self):
        return self.env.physics

    def body_pose(self):
        w = self.env.task._walker
        p = self.physics.bind(w.root_body)
        return p.xpos.copy(), p.xquat.copy()

    def render(self, camera_id=1, width=320, height=240):
        return self.physics.render(camera_id=camera_id, width=width, height=height)
