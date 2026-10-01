import numpy as np
import pytest

from vision_avoid import (DWA, Config, GapSelector, DepthCamera, EgoMemory, Perception, World,
                          backproject, fit_ground_ransac, random_world, run_episode)
from vision_avoid.camera import rot_z
from vision_avoid.world import _reachable

EMPTY = World(np.zeros((0, 4)))


def one(cx, cy, r, h):
    return World(np.array([[cx, cy, r, h]], float))


# ---------------------------------------------------------------- camera
def test_floor_depth_on_optical_axis_matches_geometry():
    cam = DepthCamera(width=97, height=73, tilt_deg=20)
    d = cam.render(EMPTY, (0, 0, 0))
    expected = cam.mount_height / np.sin(np.radians(20))
    assert d[36, 48] == pytest.approx(expected, rel=1e-9)


def test_cylinder_depth_level_camera():
    cam = DepthCamera(width=97, height=73, tilt_deg=0)
    d = cam.render(one(3.0, 0.0, 0.3, 1.0), (0, 0, 0))
    assert d[36, 48] == pytest.approx(3.0 - 0.3 - cam.mount_x, abs=1e-9)


def test_low_obstacle_top_cap_is_visible_from_above():
    cam = DepthCamera(width=97, height=73, tilt_deg=30)
    w = one(1.0, 0.0, 0.25, 0.12)
    d = cam.render(w, (0, 0, 0))
    p = backproject(d, cam.K) @ cam.R_body_cam.T + cam.t_body_cam
    on_cap = np.abs(p[:, 2] - 0.12) < 1e-6
    assert on_cap.sum() > 20
    assert np.all(np.hypot(p[on_cap, 0] - 1.0, p[on_cap, 1]) <= 0.25 + 1e-6)


def test_backprojection_round_trip_lands_on_scene_surfaces():
    cam = DepthCamera()
    w = random_world(3)
    pose = (2.0, 4.0, 0.4)
    d = cam.render(w, pose)
    p = backproject(d, cam.K) @ cam.R_body_cam.T + cam.t_body_cam
    p = p @ rot_z(pose[2]).T + [pose[0], pose[1], 0]
    floor = np.abs(p[:, 2]) < 1e-6
    surf = np.abs(w.clearance(p[:, :2])) < 1e-6
    cap = np.zeros(len(p), bool)
    for cx, cy, r, h in w.obstacles:
        cap |= (np.abs(p[:, 2] - h) < 1e-6) & (np.hypot(p[:, 0] - cx, p[:, 1] - cy) <= r + 1e-6)
    assert np.all(floor | surf | cap)


def test_noise_and_dropout_are_applied():
    cam = DepthCamera(dropout=0.1)
    d0 = cam.render(EMPTY, (0, 0, 0))
    d1 = cam.render(EMPTY, (0, 0, 0), np.random.default_rng(0))
    valid = (d0 > 0) & (d1 > 0)
    assert 0.05 < ((d0 > 0) & (d1 == 0)).mean() < 0.15
    assert np.std(d1[valid] - d0[valid]) > 0


# ---------------------------------------------------------------- perception
def test_ransac_recovers_floor_under_noise():
    cam = DepthCamera()
    rng = np.random.default_rng(1)
    d = cam.render(random_world(2), (0.6, 5, 0), rng)
    p = backproject(d, cam.K) @ cam.R_body_cam.T + cam.t_body_cam
    n, dd = fit_ground_ransac(p[p[:, 0] < 4], rng)
    assert np.degrees(np.arccos(n[2])) < 1.0
    assert abs(dd) < 0.02


def test_ransac_rejects_a_wall():
    rng = np.random.default_rng(0)
    yz = rng.uniform(-1, 1, (500, 2))
    wall = np.column_stack([np.full(500, 2.0), yz])
    assert fit_ground_ransac(wall, rng) is None


def test_ransac_absorbs_camera_tilt_calibration_error():
    """Mount believed to be 12 deg, actually 15 deg: without a floor fit the
    floor rises with range and reads as an obstacle wall."""
    true_cam, believed = DepthCamera(tilt_deg=15), DepthCamera(tilt_deg=12)
    rng = np.random.default_rng(0)
    d = true_cam.render(EMPTY, (0, 0, 0))
    fixed, _ = Perception(believed, use_ransac=False).obstacles(d, rng)
    fitted, _ = Perception(believed).obstacles(d, rng)
    assert len(fixed) > 50
    assert len(fitted) == 0


def test_planar_slice_misses_low_obstacle_depth_does_not():
    cam = DepthCamera()
    rng = np.random.default_rng(0)
    d = cam.render(one(1.5, 0.0, 0.2, 0.12), (0, 0, 0))
    planar, _ = Perception(cam, mode="planar").obstacles(d, rng)
    depth, _ = Perception(cam).obstacles(d, rng)
    assert len(planar) == 0
    assert len(depth) > 5
    assert np.all(np.hypot(depth[:, 0] - 1.5, depth[:, 1]) < 0.2 + 0.08)


def test_far_floor_noise_is_not_reported_as_obstacles():
    cam = DepthCamera()
    rng = np.random.default_rng(4)
    hits = sum(len(Perception(cam).obstacles(cam.render(EMPTY, (0, 0, 0), rng), rng)[0])
               for _ in range(20))
    assert hits <= 5


# ---------------------------------------------------------------- memory
def test_memory_keeps_points_after_they_leave_view_and_follows_odometry():
    m = EgoMemory(horizon_s=5, min_hits=1)
    m.update(np.array([[1.0, 0.0]]), (0, 0, 0), 0.0)
    body = m.update(np.zeros((0, 2)), (0, 0, np.pi / 2), 1.0)   # turned left 90 deg
    assert body == pytest.approx(np.array([[0.0, -1.0]]), abs=1e-9)


def test_memory_expires_by_age_and_distance():
    m = EgoMemory(horizon_s=2, radius=3, min_hits=1)
    m.update(np.array([[1.0, 0.0]]), (0, 0, 0), 0.0)
    assert len(m.update(np.zeros((0, 2)), (0, 0, 0), 2.5)) == 0
    m.update(np.array([[1.0, 0.0]]), (0, 0, 0), 3.0)
    assert len(m.update(np.zeros((0, 2)), (-5, 0, 0), 3.1)) == 0


def test_memory_confirms_repeated_points_and_drops_one_off_speckle():
    m = EgoMemory(min_hits=2)
    m.update(np.array([[1.0, 0.0], [2.0, 1.0]]), (0, 0, 0), 0.0)   # 2.0,1.0 is speckle
    m.update(np.array([[1.0, 0.0]]), (0, 0, 0), 0.1)
    out = m.update(np.zeros((0, 2)), (0, 0, 0), 0.2)
    assert out == pytest.approx(np.array([[1.0, 0.0]]))


def test_memory_always_returns_the_current_frame():
    m = EgoMemory(min_hits=3)
    out = m.update(np.array([[1.0, 0.0]]), (0, 0, 0), 0.0)
    assert len(out) == 1


# ---------------------------------------------------------------- planner
def test_dwa_rollout_matches_unicycle_integration():
    dwa = DWA()
    tr = dwa.rollout(np.array([0.4]), np.array([0.8]), 20, 0.1)[0]
    x = y = th = 0.0
    for _ in range(2000):
        x += 0.4 * np.cos(th) * 0.001
        y += 0.4 * np.sin(th) * 0.001
        th += 0.8 * 0.001
    assert tr[-1] == pytest.approx([x, y, th], abs=1e-3)


def test_dwa_drives_straight_at_goal_in_free_space():
    v, w, _ = DWA().plan(0.5, 0.0, np.array([5.0, 0.0]), np.zeros((0, 2)))
    assert v == pytest.approx(0.5) and abs(w) < 1e-9


def test_dwa_never_picks_a_trajectory_it_cannot_stop_on():
    dwa = DWA()
    wall = np.column_stack([np.full(41, 0.6), np.linspace(-1, 1, 41)])
    v, w, info = dwa.plan(0.5, 0.0, np.array([5.0, 0.0]), wall)
    if info["traj"] is not None:
        stop = v ** 2 / (2 * dwa.a_v) + 0.3
        k = int(np.ceil(stop / max(v, 1e-6) / 0.1))
        seg = info["traj"][:k, :2]
        d = np.linalg.norm(seg[:, None] - wall[None], axis=2).min()
        assert d - dwa.robot_radius > dwa.margin


def test_gap_selector_passes_goal_through_when_corridor_is_free():
    g = GapSelector()
    assert g.select(np.zeros((0, 2)), np.array([4.0, 1.0])) == pytest.approx([4.0, 1.0])


def test_gap_selector_steers_around_a_blocking_pole_toward_the_open_side():
    g = GapSelector()
    a = np.linspace(0, 2 * np.pi, 40)
    pole = np.column_stack([1.0 + 0.3 * np.cos(a), -0.1 + 0.3 * np.sin(a)])
    tgt = g.select(pole, np.array([5.0, 0.0]))
    assert tgt[1] > 0                                    # pole sits right of centre: go left
    assert g.free_mask(pole, np.array([np.arctan2(tgt[1], tgt[0])]), 2.0)[0]


# ---------------------------------------------------------------- world / sim
@pytest.mark.parametrize("seed", range(5))
def test_random_worlds_are_solvable(seed):
    w = random_world(seed)
    assert _reachable(w, 0.2)
    assert (w.obstacles[:, 3] < 0.2).sum() > 0


def test_episode_reaches_goal_in_empty_world():
    r = run_episode(EMPTY, Config(), seed=0)
    assert r["outcome"] == "goal"
    assert r["path_length"] < 9.4


def test_planar_baseline_drives_into_a_low_obstacle_depth_avoids_it():
    w = World(np.array([[5.0, 5.0, 0.25, 0.12]]))
    planar = run_episode(w, Config(mode="planar", memory=False), seed=0)
    depth = run_episode(w, Config(), seed=0)
    assert planar["outcome"] == "collision"
    assert depth["outcome"] == "goal"
