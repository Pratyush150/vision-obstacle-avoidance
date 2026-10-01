"""Regenerate docs/*.png from the simulator. Every pixel is real output.

    python examples/make_figures.py --seed 6
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from vision_avoid import Config, DepthCamera, Perception, backproject, random_world, run_episode  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import CONFIGS  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
INK, MUTED, LOW, TALL = "#1f2328", "#8c959f", "#d4760a", "#57606a"
COLORS = {"planar-slice": "#cf222e", "depth, plain DWA": "#8250df",
          "depth, no memory": "#bf8700", "depth + memory": "#1a7f37"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def draw_world(ax, w):
    for cx, cy, r, h in w.obstacles:
        low = h < 0.3
        ax.add_patch(Circle((cx, cy), r, fc=LOW if low else TALL, ec="none", alpha=0.9 if low else 0.55))
    ax.plot(*w.start[:2], "o", c=INK, ms=5)
    ax.plot(*w.goal, "*", c=INK, ms=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")


def trajectories(seed):
    w = random_world(seed)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    draw_world(ax, w)
    for name, kw in CONFIGS.items():
        r = run_episode(w, Config(name=name, **kw), seed=seed)
        tr = r["traj"]
        ax.plot(tr[:, 0], tr[:, 1], c=COLORS[name], lw=1.6, label=f"{name}: {r['outcome']}")
        if r["outcome"] == "collision":
            ax.plot(tr[-1, 0], tr[-1, 1], "x", c=COLORS[name], ms=9, mew=2)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title(f"Same world, same seed ({seed}). Orange = 12 cm obstacles, grey = tall.",
                 loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "trajectories.png", dpi=150)


def perception(seed):
    """One frame: depth image, side view with fitted floor, top view depth vs planar."""
    w = random_world(seed)
    cam = DepthCamera()
    rng = np.random.default_rng(0)
    # first frame with a low obstacle dead ahead, taken from the planar run,
    # which (not seeing it) drives straight at one
    r = run_episode(w, Config(mode="planar", memory=False), seed=seed, record=True)
    low = w.obstacles[w.obstacles[:, 3] < 0.3]
    frame = None
    for f in r["frames"]:
        x, y, yaw = f["pose"]
        rel = low[:, :2] - [x, y]
        fwd = rel @ [np.cos(yaw), np.sin(yaw)]
        lat = rel @ [-np.sin(yaw), np.cos(yaw)]
        if np.any((fwd > 1.0) & (fwd < 2.0) & (np.abs(lat) < 0.6 * fwd)):
            frame = f
            break
    if frame is None:
        raise SystemExit(f"seed {seed}: no frame with a low obstacle ahead")
    depth = cam.render(w, frame["pose"], rng)
    p = backproject(depth, cam.K) @ cam.R_body_cam.T + cam.t_body_cam
    dep, info = Perception(cam).obstacles(depth, rng)
    pla, _ = Perception(cam, mode="planar").obstacles(depth, rng)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.2, 1.3, 1]})
    im = axs[0].imshow(np.where(depth > 0, depth, np.nan), cmap="viridis")
    axs[0].set_title("Depth image (z-depth, m; black = invalid)", loc="left", fontsize=9)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_facecolor("black")
    fig.colorbar(im, ax=axs[0], fraction=0.035)

    near = p[:, 0] < 4
    axs[1].scatter(p[near, 0], p[near, 2], s=1, c=MUTED)
    n, d = info["plane"] if info["plane"] is not None else (np.array([0, 0, 1.0]), 0.0)
    xs = np.linspace(0, 4, 2)
    floor = -(n[0] * xs + d) / n[2]
    axs[1].plot(xs, floor, c=INK, lw=1, label="RANSAC floor")
    thresh = Perception(cam)
    zc = xs  # approximate optical depth by forward distance for the drawn band
    axs[1].plot(xs, floor + thresh.min_h + thresh.noise_sigmas * cam.noise_k * zc ** 2,
                c=LOW, lw=1, ls="--", label="obstacle threshold")
    axs[1].axhspan(cam.mount_height - thresh.band, cam.mount_height + thresh.band,
                   color=COLORS["planar-slice"], alpha=0.15, label="planar slice")
    axs[1].set_xlabel("forward (m)")
    axs[1].set_ylabel("height (m)")
    axs[1].set_ylim(-0.15, 0.9)
    axs[1].legend(frameon=False, fontsize=7, loc="upper left")
    axs[1].set_title("Side view of the back-projected points", loc="left", fontsize=9)

    axs[2].scatter(dep[:, 1], dep[:, 0], s=6, c=COLORS["depth + memory"], label="depth: above floor")
    axs[2].scatter(pla[:, 1], pla[:, 0], s=6, c=COLORS["planar-slice"], label="planar slice")
    axs[2].plot(0, 0, "^", c=INK)
    axs[2].set_xlim(2, -2)
    axs[2].set_ylim(0, 4)
    axs[2].set_aspect("equal")
    axs[2].set_xlabel("left (m)")
    axs[2].set_ylabel("forward (m)")
    axs[2].legend(frameon=False, fontsize=7, loc="upper left")
    axs[2].set_title("Obstacle points, body frame", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "perception.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    trajectories(a.seed)
    perception(a.seed)
