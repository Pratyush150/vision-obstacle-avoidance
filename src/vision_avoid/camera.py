"""Pinhole depth camera, rendered by exact ray casting against the world.

Frames
------
optical : x right, y down, z forward   (what a depth image is expressed in)
body    : x forward, y left, z up      (origin on the ground under the robot centre)
world   : x, y on the floor, z up

Depth is z-depth along the optical axis, as RealSense / ZED / OAK publish it,
not range along the ray. Invalid pixels are 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import World

# optical -> body (before tilt): optical z -> body x, optical x -> -body y, optical y -> -body z
R_BODY_OPT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class DepthCamera:
    width: int = 96
    height: int = 72
    hfov_deg: float = 87.0          # RealSense D435 depth HFOV
    mount_height: float = 0.30
    mount_x: float = 0.10           # forward of robot centre
    tilt_deg: float = 15.0          # pitched down
    min_range: float = 0.2
    max_range: float = 6.0
    noise_k: float = 0.004          # sigma_z = k * z^2  (stereo-style quadratic error)
    dropout: float = 0.02           # fraction of pixels returned invalid

    @property
    def K(self) -> np.ndarray:
        fx = (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)
        return np.array([[fx, 0, (self.width - 1) / 2],
                         [0, fx, (self.height - 1) / 2],
                         [0, 0, 1.0]])

    @property
    def R_body_cam(self) -> np.ndarray:
        """Rotation optical -> body. Positive tilt pitches the view down."""
        return rot_y(np.radians(self.tilt_deg)) @ R_BODY_OPT

    @property
    def t_body_cam(self) -> np.ndarray:
        return np.array([self.mount_x, 0.0, self.mount_height])

    def pixel_rays(self) -> np.ndarray:
        """(H*W, 3) optical-frame rays with z = 1."""
        K = self.K
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return np.stack([(u.ravel() - K[0, 2]) / K[0, 0],
                         (v.ravel() - K[1, 2]) / K[1, 1],
                         np.ones(u.size)], axis=1)

    def render(self, world: World, pose, rng=None) -> np.ndarray:
        """Depth image (H, W) from robot pose (x, y, yaw)."""
        x, y, yaw = pose
        Rwb = rot_z(yaw)
        C = Rwb @ self.t_body_cam + np.array([x, y, 0.0])
        D = (Rwb @ self.R_body_cam @ self.pixel_rays().T).T   # t is z-depth
        t = np.full(len(D), np.inf)

        down = D[:, 2] < -1e-9
        t[down] = -C[2] / D[down, 2]                           # ground plane

        a = D[:, 0] ** 2 + D[:, 1] ** 2
        for cx, cy, r, h in world.obstacles:
            ox, oy = C[0] - cx, C[1] - cy
            b = 2 * (ox * D[:, 0] + oy * D[:, 1])
            c = ox * ox + oy * oy - r * r
            disc = b * b - 4 * a * c
            ok = (disc >= 0) & (a > 1e-12)
            sq = np.sqrt(np.where(ok, disc, 0.0))
            t1 = (-b - sq) / (2 * np.where(ok, a, 1.0))
            z = C[2] + t1 * D[:, 2]
            side = ok & (t1 > 0) & (z >= 0) & (z <= h)
            t = np.where(side & (t1 < t), t1, t)
            if C[2] > h:                                       # top cap seen from above
                tc = np.where(down, (h - C[2]) / np.where(down, D[:, 2], -1.0), 1e9)
                px, py = C[0] + tc * D[:, 0] - cx, C[1] + tc * D[:, 1] - cy
                cap = down & (px * px + py * py <= r * r) & (tc > 0)
                t = np.where(cap & (tc < t), tc, t)

        if rng is not None:
            finite = np.isfinite(t)
            t[finite] += rng.normal(0, 1, finite.sum()) * self.noise_k * t[finite] ** 2
            t[rng.random(len(t)) < self.dropout] = 0.0
        t[~np.isfinite(t) | (t < self.min_range) | (t > self.max_range)] = 0.0
        return t.reshape(self.height, self.width)
