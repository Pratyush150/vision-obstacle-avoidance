"""Dynamic Window Approach over body-frame obstacle points."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DWA:
    v_max: float = 0.5
    w_max: float = 1.5
    a_v: float = 1.0          # m/s^2
    a_w: float = 3.0          # rad/s^2
    dt: float = 0.1           # control period the window is computed for
    horizon: float = 2.0
    nv: int = 9
    nw: int = 21
    robot_radius: float = 0.2
    margin: float = 0.05
    w_heading: float = 1.0
    w_clear: float = 0.6
    w_speed: float = 0.4
    clear_cap: float = 0.10    # clearance beyond this earns nothing, else stopping short wins

    def rollout(self, v: np.ndarray, w: np.ndarray, steps: int, h: float) -> np.ndarray:
        """Body-frame trajectories (K, steps, 3) for constant (v, w) pairs."""
        k = np.arange(1, steps + 1) * h
        th = np.outer(w, k)
        ws = np.where(np.abs(w) < 1e-6, 1e-6, w)[:, None]
        x = np.where(np.abs(w)[:, None] < 1e-6, np.outer(v, k), (v[:, None] / ws) * np.sin(th))
        y = np.where(np.abs(w)[:, None] < 1e-6, 0.0, (v[:, None] / ws) * (1 - np.cos(th)))
        return np.stack([x, y, th], axis=2)

    def plan(self, v0: float, w0: float, goal_body: np.ndarray, obs: np.ndarray):
        """Return (v, w, info). goal_body is the goal in the body frame."""
        vs = np.linspace(max(0.0, v0 - self.a_v * self.dt), min(self.v_max, v0 + self.a_v * self.dt), self.nv)
        wl = np.linspace(max(-self.w_max, w0 - self.a_w * self.dt), min(self.w_max, w0 + self.a_w * self.dt), self.nw)
        V, W = (a.ravel() for a in np.meshgrid(vs, wl))
        steps = int(self.horizon / 0.1)
        tr = self.rollout(V, W, steps, 0.1)

        reach = self.v_max * self.horizon + self.robot_radius + 1.0
        if len(obs):
            obs = obs[np.hypot(obs[:, 0], obs[:, 1]) < reach]
        if len(obs):
            d = np.linalg.norm(tr[:, :, None, :2] - obs[None, None, :, :], axis=3).min(axis=2)
            # only the part of the rollout the robot can still stop before matters
            clear = d.min(axis=1) - self.robot_radius
            stop_d = V ** 2 / (2 * self.a_v)
            travelled = np.abs(V)[:, None] * np.arange(1, steps + 1) * 0.1
            reach = travelled <= stop_d[:, None] + 0.3
            first = np.where(reach, d, np.inf).min(axis=1) - self.robot_radius
            admissible = first > self.margin
        else:
            clear = np.full(len(V), self.clear_cap)
            admissible = np.ones(len(V), bool)

        end = tr[:, -1]
        bearing = np.arctan2(goal_body[1] - end[:, 1], goal_body[0] - end[:, 0]) - end[:, 2]
        heading = 1 - np.abs(np.angle(np.exp(1j * bearing))) / np.pi
        score = (self.w_heading * heading
                 + self.w_clear * np.minimum(clear, self.clear_cap) / self.clear_cap
                 + self.w_speed * V / self.v_max)
        score[~admissible] = -np.inf
        if not np.isfinite(score).any():
            # recovery: rotate in place toward the side with more room
            side = 1.0 if (len(obs) == 0 or obs[:, 1].mean() < 0) else -1.0
            return 0.0, side * self.w_max * 0.6, {"recovery": True, "traj": None}
        i = int(np.argmax(score))
        return float(V[i]), float(W[i]), {"recovery": False, "traj": tr[i]}


@dataclass
class GapSelector:
    """Free-space steering: pick the unobstructed heading closest to the goal.

    Plain DWA scores trajectories by where they end up after a two-second
    horizon. Facing an obstacle squarely, "stop and keep facing the goal"
    beats "turn away", and the robot deadlocks a few centimetres short. This
    looks further: a heading is free if a robot-width corridor of length
    `lookahead` along it contains no obstacle point. The chosen heading
    becomes a subgoal that DWA then tracks with its dynamic limits.
    """
    lookahead: float = 2.0
    half_width: float = 0.28        # robot radius + margin
    max_angle_deg: float = 120.0
    step_deg: float = 3.0
    hysteresis: float = 0.3
    subgoal_dist: float = 1.5
    prev: float = 0.0

    def free_mask(self, obs: np.ndarray, angles: np.ndarray, length: float) -> np.ndarray:
        if len(obs) == 0:
            return np.ones(len(angles), bool)
        r = np.hypot(obs[:, 0], obs[:, 1])
        phi = np.arctan2(obs[:, 1], obs[:, 0])
        rel = angles[:, None] - phi[None, :]
        along, lateral = r * np.cos(rel), np.abs(r * np.sin(rel))
        hit = (along > -self.half_width) & (along < length) & (lateral < self.half_width)
        return ~hit.any(axis=1)

    def select(self, obs: np.ndarray, goal_body: np.ndarray) -> np.ndarray:
        goal_dist = float(np.hypot(*goal_body))
        goal_bearing = float(np.arctan2(goal_body[1], goal_body[0]))
        length = min(self.lookahead, goal_dist)
        if self.free_mask(obs, np.array([goal_bearing]), length)[0]:
            self.prev = goal_bearing
            return goal_body
        m = np.radians(self.max_angle_deg)
        angles = np.arange(-m, m + 1e-9, np.radians(self.step_deg))
        free = self.free_mask(obs, angles, length)
        if not free.any():
            return goal_body
        dg = np.abs(np.angle(np.exp(1j * (angles - goal_bearing))))
        cost = dg + self.hysteresis * np.abs(np.angle(np.exp(1j * (angles - self.prev))))
        a = float(angles[free][np.argmin(cost[free])])
        self.prev = a
        d = min(self.subgoal_dist, goal_dist)
        return np.array([d * np.cos(a), d * np.sin(a)])
