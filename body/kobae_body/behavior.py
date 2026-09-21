"""Landing / perching / take-off behaviour for the kinematic body.

This is an engineered state machine layered on top of the brain-driven flight, not a
model of fly motor control: the connectome model, under purely visual input, keeps
the walking/landing descending neurons silent, so without this the fly would fly
for ever. States:

  FLY      brain command -> reference trajectory (as in loop.py)
  LAND     a pillar is close and ahead: glide onto its surface in 0.4 s, wings fold
  PERCH    walk up the pillar for a few seconds (procedural tripod gait)
  TAKEOFF  push off backwards, unfold the wings, restart the reference trajectory

Quaternions are (w, x, y, z), MuJoCo convention.
"""
from __future__ import annotations

import math
import numpy as np

from .flight import CONTROL_DT, FlightBody


def quat_from_axes(x, z):
    """Rotation whose body x-axis is `x` and body z-axis is `z` (both unit, orthogonal) -> (w,x,y,z)."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; w = s / 4; qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s; qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; qx = s / 4; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s; qy = s / 4; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = s / 4
    q = np.array([w, qx, qy, qz]); return q / np.linalg.norm(q)


def slerp(a, b, t):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(np.dot(a, b))
    if d < 0: b = -b; d = -d
    if d > 0.9995:
        q = a + t * (b - a); return q / np.linalg.norm(q)
    th = math.acos(d); s = math.sin(th)
    return (math.sin((1 - t) * th) / s) * a + (math.sin(t * th) / s) * b


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


class Behaviour:
    def __init__(self, body: FlightBody, land_dist: float = 2.5, min_fly_s: float = 4.0, perch_s=(3.0, 6.0), seed: int = 0):
        self.body = body
        self.pillars = body.pillars()
        self.state = "FLY"
        self.t_state = 0.0
        self.land_dist = land_dist
        self.min_fly_s = min_fly_s
        self.perch_s = perch_s
        self.rng = np.random.default_rng(seed)
        self._perch_len = 4.0
        self._p0 = self._q0 = self._p1 = self._q1 = None
        self._normal = None
        self._gait_phase = 0.0

    # ---- helpers
    def _pose(self):
        p, q = self.body.body_pose(); return np.asarray(p, float), np.asarray(q, float)

    def _pillar_ahead(self, p, yaw):
        """Nearest pillar surface within land_dist and within +-50 deg of the heading."""
        best = None
        for (x, y, r, h) in self.pillars:
            dx, dy = x - p[0], y - p[1]
            d = math.hypot(dx, dy) - r
            if d > self.land_dist or p[2] > 2 * h - 0.5:
                continue
            ang = abs((math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi)
            if ang > math.radians(50):
                continue
            if best is None or d < best[0]:
                best = (d, x, y, r)
        return best

    # ---- step: called once per loop iteration (dt of body time). Returns True if the brain command should drive flight.
    def update(self, dt: float, gf_rate: float = 0.0) -> bool:
        self.t_state += dt
        p, q = self._pose()
        yaw = 2 * math.atan2(q[3], q[0])
        if self.state == "FLY":
            if self.t_state > self.min_fly_s:
                hit = self._pillar_ahead(p, yaw)
                if hit is not None:
                    d, x, y, r = hit
                    n = np.array([p[0] - x, p[1] - y, 0.0]); n /= np.linalg.norm(n)   # outward normal at the contact
                    contact = np.array([x, y, p[2]]) + n * (r + 0.12)
                    self._p0, self._q0 = p.copy(), q.copy()
                    self._p1 = contact
                    self._q1 = quat_from_axes(x=[0, 0, 1], z=n)      # head up, feet on the wall
                    self._normal = n
                    self.state, self.t_state = "LAND", 0.0
                    return False
            return True
        if self.state == "LAND":
            a = min(1.0, self.t_state / 0.4)
            s = a * a * (3 - 2 * a)
            pos = self._p0 + (self._p1 - self._p0) * s
            self.body.place(pos, slerp(self._q0, self._q1, s), wings="beat" if a < 0.8 else "folded",
                            legs=self.body._leg_retracted if a < 0.5 else self.body.leg_gait(0.0, 0.0), advance_s=dt)
            if a >= 1.0:
                self.state, self.t_state = "PERCH", 0.0
                self._perch_len = float(self.rng.uniform(*self.perch_s))
            return False
        if self.state == "PERCH":
            # walk up the pillar: 0.6 cm/s, tripod gait at 8 Hz
            self._gait_phase += 2 * math.pi * 8 * dt
            self._p1 = self._p1 + np.array([0, 0, 0.6 * dt])
            self.body.place(self._p1, self._q1, wings="folded", legs=self.body.leg_gait(self._gait_phase), advance_s=dt)
            if self.t_state >= self._perch_len or gf_rate > 50:
                self.state, self.t_state = "TAKEOFF", 0.0
                self._p0, self._q0 = self._p1.copy(), self._q1.copy()
                n = self._normal
                self._p1 = self._p0 + n * 1.2 + np.array([0, 0, 0.3])
                self._q1 = yaw_quat(math.atan2(n[1], n[0]))      # level, heading away from the pillar
            return False
        if self.state == "TAKEOFF":
            a = min(1.0, self.t_state / 0.3)
            s = a * a * (3 - 2 * a)
            pos = self._p0 + (self._p1 - self._p0) * s
            self.body.place(pos, slerp(self._q0, self._q1, s), wings="beat",
                            legs=self.body._leg_retracted if a > 0.5 else self.body.leg_gait(self._gait_phase, 0.1), advance_s=dt)
            if a >= 1.0:
                self.body.reseed_reference(self._p1, self._q1)
                self.body.command.speed = 12.0; self.body.command.yaw = 0.0
                self.state, self.t_state = "FLY", 0.0
            return False
        return True
