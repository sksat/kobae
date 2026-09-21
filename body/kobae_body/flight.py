"""flybody flight body driven by a command (speed, yaw rate, climb) with the trained flight policy.

The trained policy (Vaxenburg et al. 2025, imitation flight) tracks a reference
root trajectory using a wingbeat pattern generator. Here the reference is not a
recorded fly: it is extended on the fly from a *command* (forward speed, yaw
rate, climb rate). Whoever sets the command steers the fly; in kobae that is the
brain's descending-neuron read-out (``loop.py``).

Runs without TensorFlow: the policy MLP is loaded from the .npz made by
``extract_policy.py`` and evaluated in numpy.

Performance notes: flybody's task resets recompile the MJCF (0.4 s each) and its
``before_step`` re-resolves mjcf bindings every step (6 ms). We run ONE episode
of unbounded length, extend the reference arrays in place ahead of the task's
step counter, and patch ``before_step`` to use cached bindings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dm_control import mjcf
from dm_control import mujoco as dm_mujoco

from flybody.fly_envs import flight_imitation
from flybody.quaternions import mult_quat

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
        """Acme LayerNormMLP: Linear -> LayerNorm -> tanh, then Linear -> ELU (activate_final=True),
        then MultivariateNormalDiagHead with tanh_mean=False: the mean is a plain Linear output,
        clipped to [-1, 1] by the environment's CanonicalSpecWrapper."""
        x = obs.astype(np.float32)
        h = x @ self.w[0] + self.b[0]
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        h = (h - mu) / np.sqrt(var + 1e-5) * self.ln_scale + self.ln_offset
        h = np.tanh(h)
        for k in range(1, self.n_hidden):
            z = h @ self.w[k] + self.b[k]
            h = np.where(z > 0, z, np.expm1(z))   # ELU
        return np.clip(h @ self.mean_w + self.mean_b, -1.0, 1.0)


def flatten_obs(obs: dict) -> np.ndarray:
    """Acme batch_concat uses tree.flatten, which visits dict keys in SORTED order."""
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(obs.keys())])


def canonical_to_real(a: np.ndarray, spec) -> np.ndarray:
    lo, hi = spec.minimum, spec.maximum
    return lo + (np.clip(a, -1, 1) + 1) * 0.5 * (hi - lo)


@dataclass
class Command:
    speed: float = 20.0      # cm/s forward
    yaw: float = 0.0         # rad/s, + = counter-clockwise (left turn)
    climb: float = 0.0       # cm/s vertical


class FlightBody:
    """A flying fly whose reference trajectory is extended from ``command`` as time advances."""

    def __init__(self, policy_path: str | Path, wpg_pattern_path: str | None = None,
                 horizon_steps: int = 50, future_steps: int = 5, seed: int = 0, buffer_s: float = 30.0):
        self.env = flight_imitation(None, wpg_pattern_path, future_steps=future_steps,
                                    terminal_com_dist=float("inf"),
                                    random_state=np.random.RandomState(seed))
        self.env._time_limit = float("inf")
        self.policy = NumpyPolicy(policy_path)
        self.spec = self.env.action_spec()
        self.horizon = horizon_steps
        self.future = future_steps
        self.command = Command()
        self.t = 0
        self.wall = 0.0
        self.timestep = self.env.reset()          # the only reset: compiles the model
        task = self.env.task
        task._traj_timesteps = 1 << 30            # never "reach the end" of the reference
        # long reference buffers (root frame), seeded with the row the reset used
        n = int(buffer_s / CONTROL_DT)
        self.ref_qpos = np.zeros((n, 7)); self.ref_qvel = np.zeros((n, 6))
        self.ref_qpos[0] = task._ref_qpos[0]; self.ref_qvel[0] = task._ref_qvel[0]
        self.filled = 1
        task._ref_qpos = self.ref_qpos; task._ref_qvel = self.ref_qvel
        self._patch_before_step()
        self._extend(self.horizon + self.future + 2)

    # ---- reference trajectory -------------------------------------------------
    def _extend(self, n: int):
        """Append n rows continuing from the last row with the current command."""
        c = self.command
        if self.filled + n > len(self.ref_qpos):   # grow the buffer
            grow = len(self.ref_qpos)
            self.ref_qpos = np.concatenate([self.ref_qpos, np.zeros((grow, 7))])
            self.ref_qvel = np.concatenate([self.ref_qvel, np.zeros((grow, 6))])
            self.env.task._ref_qpos = self.ref_qpos; self.env.task._ref_qvel = self.ref_qvel
        q0 = self.ref_qpos[self.filled - 1]
        yaw0 = 2 * np.arctan2(q0[6], q0[3])
        dth = c.yaw * CONTROL_DT
        dq = np.array([np.cos(dth / 2), 0, 0, np.sin(dth / 2)])
        w = np.zeros(3); dm_mujoco.mju_quat2Vel(w, dq, 1)
        pos = q0[:3].copy(); quat = q0[3:].copy()
        for i in range(n):
            yaw = yaw0 + c.yaw * (i + 1) * CONTROL_DT
            vel = np.array([c.speed * np.cos(yaw), c.speed * np.sin(yaw), c.climb])
            pos = pos + vel * CONTROL_DT
            quat = mult_quat(dq, quat)
            k = self.filled + i
            self.ref_qpos[k, :3] = pos; self.ref_qpos[k, 3:] = quat
            self.ref_qvel[k, :3] = vel; self.ref_qvel[k, 3:] = w
        self.filled += n

    def _patch_before_step(self):
        """Same semantics as FlightImitationWBPG.before_step, with bindings resolved once."""
        task = self.env.task
        physics = self.env.physics
        # resolve indices once; then read/write the raw mujoco arrays directly
        wing_qpos_idx = np.asarray(physics.bind(task._wing_joints).qposadr)
        gj = physics.bind(mjcf.get_frame_freejoint(task._ghost.mjcf_model))
        gq, gv = int(gj.qposadr), int(gj.dofadr)
        offset = np.hstack((task._ghost_offset, 4 * [0]))
        wbpg = task._wbpg
        parent_before = type(task).__mro__[1].before_step  # Flying.before_step
        data = physics.data

        def before_step(physics, action, random_state):
            base_freq, rel_range = wbpg.base_beat_freq, wbpg.rel_freq_range
            act = action[task._user_idx_action]
            ctrl = wbpg.step(ctrl_freq=base_freq * (1 + rel_range * act))
            action[task._wing_inds_action] += (ctrl - data.qpos[wing_qpos_idx])
            step = int(np.round(data.time / task.control_timestep))
            g = task._ref_qpos[step] + offset
            data.qpos[gq:gq + 3] = g[:3]; data.qpos[gq + 3:gq + 7] = g[3:] / np.linalg.norm(g[3:])
            data.qvel[gv:gv + 6] = task._ref_qvel[step]
            parent_before(task, physics, action, random_state)

        task.before_step = before_step

    # ---- stepping ---------------------------------------------------------------
    def step(self, n: int = 1):
        """Advance n control steps (0.2 ms each). Returns the last timestep."""
        t0 = time.perf_counter()
        for _ in range(n):
            need = self.env.task._step_counter + self.future + 2 + self.horizon
            if self.filled < need:
                self._extend(self.horizon)
            obs = flatten_obs(self.timestep.observation)
            a = self.policy(obs)
            self.timestep = self.env.step(canonical_to_real(a, self.spec))
            if self.timestep.last():   # fell below the terminal height etc.: relaunch from the reference
                self.timestep = self.env.reset()
            self.t += 1
        self.wall += time.perf_counter() - t0
        return self.timestep

    # ---- readouts for the coupling ----------------------------------------------
    @property
    def physics(self):
        return self.env.physics

    def body_pose(self):
        p, q = self.env.task._walker.get_pose(self.physics)
        return np.asarray(p).copy(), np.asarray(q).copy()

    def ref_pose(self):
        k = min(self.env.task._step_counter, self.filled - 1)
        return self.ref_qpos[k, :3].copy(), self.ref_qpos[k, 3:].copy()

    def render(self, camera_id=1, width=320, height=240):
        return self.physics.render(camera_id=camera_id, width=width, height=height)
