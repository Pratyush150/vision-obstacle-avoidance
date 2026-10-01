"""Depth image -> obstacle points on the floor plane, in the robot body frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import DepthCamera


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Valid pixels of a z-depth image -> (N, 3) optical-frame points."""
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def fit_ground_ransac(pts: np.ndarray, rng, iters: int = 60, tol: float = 0.03,
                      max_tilt_deg: float = 20.0):
    """Fit the floor as the largest near-horizontal plane.

    Returns (n, d) with n . p + d = 0, n pointing up, refined by least squares
    on the inliers, or None if no plausible floor is visible. The tilt gate
    stops a large wall from being taken as the floor.
    """
    if len(pts) < 30:
        return None
    cos_max = np.cos(np.radians(max_tilt_deg))
    # score hypotheses on a fixed-size subsample; all hypotheses at once
    sub = pts[rng.integers(0, len(pts), min(len(pts), 1500))]
    tri = pts[rng.integers(0, len(pts), (iters, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n = n[ok] / norm[ok, None]
    n *= np.sign(n[:, 2:3] + 1e-12)
    keep = n[:, 2] >= cos_max
    if not keep.any():
        return None
    n, p0 = n[keep], tri[ok][keep, 0]
    d = -(n * p0).sum(axis=1)
    counts = (np.abs(sub @ n.T + d) < tol).sum(axis=0)
    i = int(np.argmax(counts))
    if counts[i] < 0.05 * len(sub):
        return None
    best = (n[i], d[i])
    n, d = best
    inl = pts[np.abs(pts @ n + d) < tol]
    c = inl.mean(axis=0)
    _, vecs = np.linalg.eigh((inl - c).T @ (inl - c))   # smallest eigenvector = normal
    n = vecs[:, 0] if vecs[2, 0] > 0 else -vecs[:, 0]
    return n, -n @ c


def voxel_xy(xy: np.ndarray, cell: float) -> np.ndarray:
    """One representative point per occupied cell."""
    if len(xy) == 0:
        return xy
    _, keep = np.unique(np.floor(xy / cell).astype(np.int64), axis=0, return_index=True)
    return xy[keep]


@dataclass
class Perception:
    """Configurable obstacle extraction.

    mode="depth"  : everything between min_h and max_h above the fitted floor.
    mode="planar" : only a thin band at camera height -- what a 2D lidar at the
                    same mount would report. The baseline that misses low obstacles.
    """
    camera: DepthCamera
    mode: str = "depth"
    min_h: float = 0.05
    max_h: float = 0.60
    band: float = 0.04
    cell: float = 0.05
    use_ransac: bool = True
    max_range: float = 4.0       # beyond this, stereo depth noise exceeds obstacle heights
    noise_sigmas: float = 3.0    # height threshold grows with expected depth noise

    def obstacles(self, depth: np.ndarray, rng) -> tuple[np.ndarray, dict]:
        cam = self.camera
        pc = backproject(depth, cam.K)
        pc = pc[pc[:, 2] < self.max_range]
        p = pc @ cam.R_body_cam.T + cam.t_body_cam
        info = {"n_points": len(p), "plane": None}
        if self.mode == "planar":
            m = np.abs(p[:, 2] - cam.mount_height) < self.band
            return voxel_xy(p[m, :2], self.cell), info
        plane = fit_ground_ransac(p, rng) if self.use_ransac else None
        if plane is None:            # fall back to the nominal mount geometry
            n, d = np.array([0.0, 0.0, 1.0]), 0.0
        else:
            n, d = plane
            info["plane"] = plane
        h = p @ n + d
        thresh = self.min_h + self.noise_sigmas * cam.noise_k * pc[:, 2] ** 2
        m = (h > thresh) & (h < self.max_h)
        return voxel_xy(p[m, :2], self.cell), info


class EgoMemory:
    """Short-term obstacle memory carried by odometry.

    A 87 degree camera sees nothing beside the robot. Without memory, an
    obstacle that has just left the field of view stops existing, and the
    planner will happily turn into it. Points are stored in the odometry
    frame and expire by age and distance.
    """

    def __init__(self, horizon_s: float = 6.0, radius: float = 3.0, cell: float = 0.05,
                 min_hits: int = 2):
        self.horizon_s, self.radius, self.cell, self.min_hits = horizon_s, radius, cell, min_hits
        self.pts = np.zeros((0, 2))
        self.stamp = np.zeros(0)
        self.hits = np.zeros(0, int)

    def update(self, body_xy: np.ndarray, odom_pose, t: float) -> np.ndarray:
        """Add this frame's points; return confirmed points in the body frame.

        A voxel is confirmed once it has been observed in `min_hits` frames.
        Single-frame depth noise rarely lands in the same 5 cm cell twice, so
        this keeps the memory from integrating speckle into phantom walls.
        Points in the current frame are always returned: the newest view is
        never withheld from the planner.
        """
        x, y, yaw = odom_pose
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        new = voxel_xy(body_xy @ R.T + [x, y], self.cell)
        pts = np.vstack([self.pts, new])
        stamp = np.concatenate([self.stamp, np.full(len(new), t)])
        hits = np.concatenate([self.hits, np.ones(len(new), int)])
        keep = (t - stamp <= self.horizon_s) & (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < self.radius)
        pts, stamp, hits = pts[keep], stamp[keep], hits[keep]
        order = np.argsort(-stamp, kind="stable")              # newest first
        pts, stamp, hits = pts[order], stamp[order], hits[order]
        _, first, inv = np.unique(np.floor(pts / self.cell).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
        total = np.zeros(len(first), int)
        np.add.at(total, inv.ravel(), hits)
        self.pts, self.stamp, self.hits = pts[first], stamp[first], total
        out = (self.hits >= self.min_hits) | (self.stamp == t)
        return (self.pts[out] - [x, y]) @ R                    # back to body frame
