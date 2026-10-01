"""Closed-loop episode: camera -> perception -> (memory) -> DWA -> unicycle."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import DepthCamera
from .perception import EgoMemory, Perception
from .planner import DWA, GapSelector
from .world import World


@dataclass
class Config:
    name: str = "depth+memory"
    mode: str = "depth"               # "depth" | "planar"
    memory: bool = True
    gaps: bool = True                 # False = plain DWA straight at the goal
    camera: DepthCamera = field(default_factory=DepthCamera)
    planner: DWA = field(default_factory=DWA)
    odom_noise: float = 0.02          # fractional std on wheel odometry increments
    dt: float = 0.1
    t_max: float = 60.0
    goal_tol: float = 0.25


def run_episode(world: World, cfg: Config, seed: int = 0, record: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    cam, dwa = cfg.camera, cfg.planner
    per = Perception(cam, mode=cfg.mode)
    mem = EgoMemory() if cfg.memory else None
    gaps = GapSelector() if cfg.gaps else None
    pose = np.array(world.start, float)
    odom = pose.copy()
    v = w = 0.0
    traj, min_clear, t = [pose.copy()], np.inf, 0.0
    frames = []
    outcome, hit = "timeout", None
    while t < cfg.t_max:
        depth = cam.render(world, pose, rng)
        obs, info = per.obstacles(depth, rng)
        if mem is not None:
            obs = mem.update(obs, odom, t)
        c, s = np.cos(odom[2]), np.sin(odom[2])
        g = np.array(world.goal) - odom[:2]
        goal_body = np.array([c * g[0] + s * g[1], -s * g[0] + c * g[1]])
        target = gaps.select(obs, goal_body) if gaps is not None else goal_body
        v, w, pinfo = dwa.plan(v, w, target, obs)
        if record:
            frames.append({"t": t, "pose": pose.copy(), "depth": depth, "obs": obs,
                           "plan": pinfo["traj"], "plane": info["plane"],
                           "target": target})

        # true motion, then noisy odometry of the same motion
        for p, noise in ((pose, 0.0), (odom, cfg.odom_noise)):
            dv = v * (1 + noise * rng.normal())
            dw = w * (1 + noise * rng.normal())
            p[0] += dv * np.cos(p[2] + dw * cfg.dt / 2) * cfg.dt
            p[1] += dv * np.sin(p[2] + dw * cfg.dt / 2) * cfg.dt
            p[2] += dw * cfg.dt
        t += cfg.dt
        traj.append(pose.copy())

        clear = world.clearance(pose[:2])[0] - dwa.robot_radius
        min_clear = min(min_clear, clear)
        if clear < 0:
            outcome = "collision"
            d = np.hypot(*(world.obstacles[:, :2] - pose[:2]).T) - world.obstacles[:, 2]
            ob = world.obstacles[int(np.argmin(d))]
            # where the obstacle was relative to the heading at impact
            bearing = np.arctan2(ob[1] - pose[1], ob[0] - pose[0]) - pose[2]
            hit = {"height": float(ob[3]),
                   "bearing_deg": float(np.degrees(np.angle(np.exp(1j * bearing))))}
            break
        if np.hypot(*(pose[:2] - world.goal)) < cfg.goal_tol:
            outcome = "goal"
            break

    traj = np.array(traj)
    out = {"outcome": outcome, "time": t, "min_clearance": float(min_clear),
           "path_length": float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()),
           "hit": hit, "traj": traj}
    if record:
        out["frames"] = frames
    return out
