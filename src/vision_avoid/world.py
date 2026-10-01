"""A flat arena populated with vertical cylinders of varying height.

Low cylinders (shorter than the camera mount) are the interesting ones: a
planar scan taken at camera height passes straight over them, but the robot
still cannot drive through them.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class World:
    obstacles: np.ndarray            # (N, 4): cx, cy, radius, height
    size: tuple = (10.0, 10.0)
    start: tuple = (0.6, 5.0, 0.0)   # x, y, yaw
    goal: tuple = (9.4, 5.0)
    meta: dict = field(default_factory=dict)

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        """Distance from each point to the nearest obstacle surface (2D)."""
        xy = np.atleast_2d(xy)
        if len(self.obstacles) == 0:
            return np.full(len(xy), np.inf)
        d = np.linalg.norm(xy[:, None, :] - self.obstacles[None, :, :2], axis=2)
        return (d - self.obstacles[None, :, 2]).min(axis=1)


def _reachable(world: World, robot_radius: float, cell: float = 0.05) -> bool:
    """Grid BFS over the obstacle map inflated by the robot radius."""
    nx, ny = int(world.size[0] / cell), int(world.size[1] / cell)
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    free = (world.clearance(pts) > robot_radius + 0.02).reshape(nx, ny)

    def idx(p):
        return min(nx - 1, int(p[0] / cell)), min(ny - 1, int(p[1] / cell))

    s, g = idx(world.start), idx(world.goal)
    if not (free[s] and free[g]):
        return False
    seen = np.zeros_like(free)
    seen[s] = True
    q = deque([s])
    while q:
        i, j = q.popleft()
        if (i, j) == g:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return False


def random_world(seed: int, n_tall: int = 14, n_low: int = 6,
                 robot_radius: float = 0.2, low_height: float = 0.12) -> World:
    """Random arena guaranteed to have a collision-free path start -> goal.

    Rejection-samples until the inflated free space connects start and goal,
    so every failure in a benchmark is the planner's, not the map's.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(200):
        obs = []
        for k in range(n_tall + n_low):
            r = rng.uniform(0.15, 0.45) if k < n_tall else rng.uniform(0.12, 0.25)
            h = rng.uniform(0.6, 1.5) if k < n_tall else low_height
            for _ in range(50):
                c = rng.uniform([1.6, 0.6], [8.4, 9.4])
                ok = all(np.hypot(*(c - o[:2])) > r + o[2] + 0.1 for o in obs)
                if ok:
                    obs.append([c[0], c[1], r, h])
                    break
        w = World(np.array(obs, dtype=float), meta={"seed": seed, "attempt": attempt})
        if _reachable(w, robot_radius):
            return w
    raise RuntimeError(f"no solvable world for seed {seed}")
