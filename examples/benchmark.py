"""Run every configuration on the same seeded worlds and tabulate outcomes.

    python examples/benchmark.py --worlds 60 --jobs 8

Every world is checked solvable before use (grid search on the true map,
inflated by the robot radius), so every failure is the system's.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from vision_avoid import Config, random_world, run_episode  # noqa: E402

CONFIGS = {
    "planar-slice": dict(mode="planar", memory=False),
    "depth, plain DWA": dict(mode="depth", memory=True, gaps=False),
    "depth, no memory": dict(mode="depth", memory=False),
    "depth + memory": dict(mode="depth", memory=True),
}


def job(args):
    name, seed = args
    r = run_episode(random_world(seed), Config(name=name, **CONFIGS[name]), seed=seed)
    return {"config": name, "seed": seed, "outcome": r["outcome"], "time": r["time"],
            "min_clearance": r["min_clearance"], "path_length": r["path_length"], "hit": r["hit"]}


def summarise(rows):
    out = {}
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        ok = [r for r in rs if r["outcome"] == "goal"]
        out[name] = {
            "n": len(rs),
            "goal": len(ok),
            "collision": sum(r["outcome"] == "collision" for r in rs),
            "timeout": sum(r["outcome"] == "timeout" for r in rs),
            "collisions_with_low_obstacles": sum(r["hit"] is not None and r["hit"]["height"] < 0.3 for r in rs),
            "collisions_side_on": sum(r["hit"] is not None and abs(r["hit"]["bearing_deg"]) > 45 for r in rs),
            "median_time_s": float(np.median([r["time"] for r in ok])) if ok else None,
            "median_path_m": float(np.median([r["path_length"] for r in ok])) if ok else None,
            "p5_clearance_m": float(np.percentile([r["min_clearance"] for r in ok], 5)) if ok else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    tasks = [(n, s) for s in range(a.worlds) for n in CONFIGS]
    with Pool(a.jobs) as p:
        rows = p.map(job, tasks, chunksize=1)
    Path(a.out).mkdir(exist_ok=True)
    summary = summarise(rows)
    json.dump({"rows": rows, "summary": summary}, open(Path(a.out) / "benchmark.json", "w"), indent=1)
    print(f"{'config':<20}{'goal':>6}{'coll':>6}{'low':>5}{'side':>5}{'tmo':>6}{'t_med':>8}{'path':>7}{'clr_p5':>8}")
    for name, s in summary.items():
        f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        print(f"{name:<20}{s['goal']:>6}{s['collision']:>6}{s['collisions_with_low_obstacles']:>5}"
              f"{s['collisions_side_on']:>5}{s['timeout']:>6}"
              f"{f(s['median_time_s']):>8}{f(s['median_path_m']):>7}{f(s['p5_clearance_m'], 3):>8}")


if __name__ == "__main__":
    main()
