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
EYE_CAMERAS = ("walker/eye_left", "walker/eye_right")


ARENA_RADIUS = 150.0   # cm; a wall ring keeps the world finite and the fly's eyes always see something


def add_scenery(arena_mjcf, seed: int = 0, n_pillars: int = 60, radius: float = 120.0, floor: float = 3000.0):
    """Make the world worth looking at: a large lit floor (flybody's arena floor is 8x8 cm, the fly
    leaves it in half a second), a sky, coloured pillars and a few big landmarks."""
    rng = np.random.default_rng(seed)
    root = arena_mjcf
    # floor: enlarge and brighten the existing checker plane
    g = root.find("geom", "groundplane")
    g.size = [floor, floor, 0.5]
    mat = root.find("material", "groundplane")
    mat.texuniform = False
    mat.texrepeat = [floor / 4, floor / 4]      # one repeat (2 squares) per 8 cm -> 4 cm squares
    mat.reflectance = 0.0
    tex = root.find("texture", "groundplane")
    tex.rgb1 = [0.42, 0.5, 0.42]; tex.rgb2 = [0.22, 0.3, 0.26]
    # sky and light
    root.asset.add("texture", name="sky", type="skybox", builtin="gradient",
                   rgb1=[0.62, 0.78, 0.95], rgb2=[0.25, 0.4, 0.7], width=256, height=256)
    root.visual.headlight.ambient = [0.25, 0.25, 0.25]
    root.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    root.visual.headlight.specular = [0.05, 0.05, 0.05]
    root.visual.map.zfar = 4000
    wb = root.worldbody
    wb.add("light", name="sun", directional=True, pos=[0, 0, 200], dir=[0.3, 0.2, -1], diffuse=[0.5, 0.5, 0.45],
           specular=[0.05, 0.05, 0.05], castshadow=True)
    palette = [[0.85, 0.3, 0.25], [0.95, 0.75, 0.2], [0.3, 0.6, 0.9], [0.4, 0.75, 0.4], [0.8, 0.5, 0.8], [0.95, 0.95, 0.95]]
    for k in range(n_pillars):
        r = rng.uniform(8, radius); a = rng.uniform(0, 2 * np.pi)
        h = rng.uniform(3, 12); c = palette[k % len(palette)]
        wb.add("geom", name=f"pillar{k}", type="cylinder", size=[rng.uniform(0.4, 1.2), h],
               pos=[r * np.cos(a), r * np.sin(a), h], rgba=c + [1], contype=0, conaffinity=0)
    # a few big landmarks inside the ring
    for k, c in enumerate(palette):
        a = 2 * np.pi * k / len(palette)
        wb.add("geom", name=f"landmark{k}", type="box", size=[6, 6, 25],
               pos=[radius * 1.15 * np.cos(a), radius * 1.15 * np.sin(a), 25], rgba=c + [1], contype=0, conaffinity=0)
    # the arena wall: a ring of tall panels in alternating tones (a horizon the eyes can see from anywhere)
    n_wall = 48
    for k in range(n_wall):
        a = 2 * np.pi * (k + 0.5) / n_wall
        tone = [0.82, 0.8, 0.74] if k % 2 == 0 else [0.36, 0.34, 0.3]
        wb.add("geom", name=f"wall{k}", type="box", size=[ARENA_RADIUS * np.pi / n_wall, 1.0, 40],
               pos=[ARENA_RADIUS * np.cos(a), ARENA_RADIUS * np.sin(a), 40], euler=[0, 0, np.degrees(a) + 90],
               rgba=tone + [1], contype=0, conaffinity=0)


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
                 horizon_steps: int = 50, future_steps: int = 5, seed: int = 0, buffer_s: float = 30.0,
                 kinematic: bool = False, legs: bool = False):
        """``kinematic``: no dynamics; the fly is placed on the reference trajectory every step and the
        wings follow the pattern generator. ~100x cheaper than the MuJoCo flight (real-time live view).
        ``legs``: keep the leg joints (kinematic only; the flight policy was trained without them)."""
        self.kinematic = kinematic
        self.legs = legs and kinematic
        self.env = flight_imitation(None, wpg_pattern_path, future_steps=future_steps,
                                    terminal_com_dist=float("inf"), disable_legs=not self.legs,
                                    random_state=np.random.RandomState(seed))
        self.env._time_limit = float("inf")
        add_scenery(self.env.task._arena.mjcf_model, seed=seed)
        self.policy = NumpyPolicy(policy_path)
        self.spec = self.env.action_spec()
        self.horizon = horizon_steps
        self.future = future_steps
        self.command = Command()
        self.t = 0
        self.wall = 0.0
        self.buffer_s = buffer_s
        self.relaunches = 0
        self._eye_cams = None
        self._launch()

    def _launch(self):
        """(Re)start an episode: compiles the model, installs the long reference buffers and the fast hooks."""
        self.timestep = self.env.reset()
        task = self.env.task
        task._traj_timesteps = 1 << 30            # never "reach the end" of the reference
        n = int(self.buffer_s / CONTROL_DT)
        self.ref_qpos = np.zeros((n, 7)); self.ref_qvel = np.zeros((n, 6))
        self.ref_qpos[0] = task._ref_qpos[0]; self.ref_qvel[0] = task._ref_qvel[0]
        self.filled = 1
        task._ref_qpos = self.ref_qpos; task._ref_qvel = self.ref_qvel
        self._patch_before_step()
        self._hide_helpers()
        self._extend(self.horizon + self.future + 2)
        # kinematic mode: resolve the root free joint and wing joints once per (re)launch
        rj = self.physics.bind(mjcf.get_frame_freejoint(task._walker.mjcf_model))
        self._root_q, self._root_v = int(rj.qposadr), int(rj.dofadr)
        self._wing_q = np.asarray(self.physics.bind(task._wing_joints).qposadr)
        self._wing_folded = self.physics.data.qpos[self._wing_q].copy()
        if self.legs:
            lj = self.physics.bind(task._leg_joints)
            self._leg_q = np.asarray(lj.qposadr)
            self._leg_names = [j.name for j in task._leg_joints]
            self._leg_retracted = np.asarray(task._leg_springrefs, dtype=np.float64)
            self.physics.data.qpos[self._leg_q] = self._leg_retracted
        self._renderer = None   # the model may have been recompiled

    def _hide_helpers(self):
        """The ghost fly, trajectory dots and crosshair are visualisation aids: keep them out of the eyes."""
        import mujoco
        m = self.physics.model
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m.ptr, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("ghost"):
                m.geom_rgba[i, 3] = 0.0
        for i in range(m.nsite):
            m.site_rgba[i, 3] = 0.0

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
        if self.kinematic:
            return self._step_kinematic(n)
        t0 = time.perf_counter()
        for _ in range(n):
            need = self.env.task._step_counter + self.future + 2 + self.horizon
            if self.filled < need:
                self._extend(self.horizon)
            obs = flatten_obs(self.timestep.observation)
            a = self.policy(obs)
            self.timestep = self.env.step(canonical_to_real(a, self.spec))
            if self.timestep.last():   # crashed (below the terminal height): relaunch a fresh episode
                self.relaunches += 1
                self._launch()
            self.t += 1
        self.wall += time.perf_counter() - t0
        return self.timestep

    def _step_kinematic(self, n: int):
        import mujoco
        t0 = time.perf_counter()
        task = self.env.task; d = self.physics.data
        for _ in range(n):
            k = task._step_counter + 1
            if self.filled < k + self.future + 2 + self.horizon:
                self._extend(self.horizon)
            q = self.ref_qpos[k]
            d.qpos[self._root_q:self._root_q + 3] = q[:3]; d.qpos[self._root_q + 3:self._root_q + 7] = q[3:]
            d.qvel[self._root_v:self._root_v + 6] = self.ref_qvel[k]
            d.qpos[self._wing_q] = task._wbpg.step(ctrl_freq=task._wbpg.base_beat_freq)
            d.time += CONTROL_DT
            task._step_counter = k
            self.t += 1
        mujoco.mj_forward(self.physics.model.ptr, d.ptr)
        self.wall += time.perf_counter() - t0
        return self.timestep

    # ---- direct pose control (kinematic behaviours: landing, perching, take-off) ------------
    def place(self, pos, quat, wings="folded", legs=None, advance_s: float = 0.0):
        """Put the fly at pos/quat, set wings (folded | beat) and leg joint angles; advance time."""
        import mujoco
        d = self.physics.data; task = self.env.task
        d.qpos[self._root_q:self._root_q + 3] = pos; d.qpos[self._root_q + 3:self._root_q + 7] = quat
        d.qvel[self._root_v:self._root_v + 6] = 0
        if wings == "beat":
            d.qpos[self._wing_q] = task._wbpg.step(ctrl_freq=task._wbpg.base_beat_freq)
        else:
            d.qpos[self._wing_q] = self._wing_folded
        if self.legs and legs is not None:
            d.qpos[self._leg_q] = legs
        if advance_s:
            n = int(round(advance_s / CONTROL_DT)); d.time += n * CONTROL_DT; task._step_counter += n; self.t += n
        mujoco.mj_forward(self.physics.model.ptr, d.ptr)

    def reseed_reference(self, pos, quat):
        """Restart the reference trajectory from pos/quat (after a perch); keeps the step counter."""
        k = self.env.task._step_counter
        if k + self.horizon + self.future + 4 > len(self.ref_qpos):
            grow = len(self.ref_qpos)
            self.ref_qpos = np.concatenate([self.ref_qpos, np.zeros((grow, 7))]); self.ref_qvel = np.concatenate([self.ref_qvel, np.zeros((grow, 6))])
            self.env.task._ref_qpos = self.ref_qpos; self.env.task._ref_qvel = self.ref_qvel
        self.ref_qpos[k, :3] = pos; self.ref_qpos[k, 3:] = quat; self.ref_qvel[k] = 0
        self.filled = k + 1
        self._extend(self.horizon + self.future + 2)

    def pillars(self):
        """[(x, y, radius, half_height)] of the arena pillars, from the compiled model."""
        import mujoco
        m = self.physics.model.ptr; out = []
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if n.startswith("pillar"):
                out.append((float(m.geom_pos[g][0]), float(m.geom_pos[g][1]), float(m.geom_size[g][0]), float(m.geom_size[g][1])))
        return out

    def leg_gait(self, phase: float, amp: float = 0.35):
        """Procedural tripod gait around the standing pose (joint angle 0): returns leg qpos.
        Not a measured gait; an animation so that a perched fly visibly walks."""
        q = np.zeros(len(self._leg_q))
        for i, name in enumerate(self._leg_names):
            leg = name.split("_")[-2]; side = name.split("_")[-1]
            tripod = 0.0 if (leg, side) in (("T1", "left"), ("T2", "right"), ("T3", "left")) else np.pi
            ph = phase + tripod
            if name.startswith("coxa_T") or name.startswith("coxa_twist"):
                q[i] = amp * np.sin(ph) * (1 if side == "left" else -1)
            elif name.startswith("femur_T") or name.startswith("femur_twist"):
                q[i] = -0.5 * amp * max(0.0, np.sin(ph + np.pi / 2))
            elif name.startswith("tibia"):
                q[i] = 0.6 * amp * max(0.0, np.sin(ph + np.pi / 2))
        return q

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

    def eyes(self, width=64, height=48):
        """(left, right) egocentric eye images, via a persistent mujoco.Renderer (4x faster than physics.render)."""
        import mujoco
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.physics.model.ptr, height=height, width=width)
            self._eye_cams = [mujoco.mj_name2id(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in EYE_CAMERAS]
            # shadows and reflections cost ~25 ms per render on these GPUs; the eyes do not need them.
            # The fly's own body (geom groups 1,3,4,5) is hidden from its eyes as well.
            for fl in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX,
                       mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_HAZE):
                self._renderer.scene.flags[fl] = False
            self._eye_opt = mujoco.MjvOption()
            for gi in (1, 3, 4, 5):
                self._eye_opt.geomgroup[gi] = 0
        out = []
        for c in self._eye_cams:
            self._renderer.update_scene(self.physics.data.ptr, camera=c, scene_option=self._eye_opt)
            out.append(self._renderer.render().copy())
        return tuple(out)

    def walker_poses(self, body_ids):
        """(n,7) world pos+quat of the given body ids (for the browser rig)."""
        d = self.physics.data
        return np.concatenate([d.xpos[body_ids], d.xquat[body_ids]], axis=1).astype(np.float32)
