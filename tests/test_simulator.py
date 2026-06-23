"""シミュレーターのテスト (simulator.py)"""

import pytest
import numpy as np
from simulator import Simulator, SimParams
from obstacles import SphereObstacle, MovingObstacle


# ── ユーティリティ ──────────────────────────────────────────────────────────────

def _default_sim(**overrides) -> Simulator:
    return Simulator(SimParams(nav_mode='gps', **overrides))


def _short_run(profile='standard', dist_m=1500.0) -> dict:
    """短距離シミュレーションを実行して履歴を返す。"""
    sim = _default_sim(flight_profile=profile)
    wp0 = np.array([0.0, 0.0, 200.0])
    wp1 = np.array([dist_m, 0.0, 200.0])
    return sim.run([wp0, wp1], profile=profile)


# ── turning_radius ─────────────────────────────────────────────────────────────

def test_turning_radius_formula():
    sim = _default_sim()
    v = 150.0
    assert sim.turning_radius(v) == pytest.approx(v ** 2 / sim.p.max_accel)


def test_turning_radius_zero_speed():
    sim = _default_sim()
    assert sim.turning_radius(0.0) == pytest.approx(0.0)


def test_turning_radius_increases_with_speed():
    sim = _default_sim()
    assert sim.turning_radius(100.0) < sim.turning_radius(200.0)


# ── _auto_terminal_time ────────────────────────────────────────────────────────

def test_auto_terminal_time_positive():
    sim = _default_sim()
    assert sim._auto_terminal_time() > 0.0


def test_auto_terminal_time_reasonable_range():
    # ダイブ高度 500m / 速度 272 m/s → 少なくとも数秒以上
    sim = _default_sim()
    t = sim._auto_terminal_time()
    assert 5.0 < t < 600.0


# ── _vz_command ────────────────────────────────────────────────────────────────

def test_vz_command_positive_error_climbs():
    sim = _default_sim()
    vz = sim._vz_command(alt_error=200.0, current_vz=0.0, desired_speed=272.0)
    assert vz > 0.0


def test_vz_command_negative_error_descends():
    sim = _default_sim()
    vz = sim._vz_command(alt_error=-200.0, current_vz=0.0, desired_speed=272.0)
    assert vz < 0.0


def test_vz_command_zero_error():
    sim = _default_sim()
    vz = sim._vz_command(alt_error=0.0, current_vz=0.0, desired_speed=272.0)
    assert vz == pytest.approx(0.0)


def test_vz_command_capped_at_half_speed():
    sim = _default_sim()
    # 非常に大きな高度誤差でも上昇速度は 速度×0.5 に制限
    vz = sim._vz_command(alt_error=1e9, current_vz=0.0, desired_speed=100.0)
    assert vz <= 100.0 * 0.5 + 1e-6


# ── _avoidance_steer ───────────────────────────────────────────────────────────

def test_avoidance_no_change_when_no_obstacles():
    sim = _default_sim()
    desired = np.array([10.0, 0.0, 0.0])
    result = sim._avoidance_steer(np.zeros(3), desired, 10.0, 0.0, [], [])
    assert np.allclose(result, desired)


def test_avoidance_no_change_far_from_obstacle():
    sim = _default_sim()
    obs = SphereObstacle(pos=np.array([1e6, 0.0, 0.0]), radius=50.0)
    desired = np.array([10.0, 0.0, 0.0])
    pos = np.zeros(3)
    result = sim._avoidance_steer(pos, desired, 10.0, 0.0, [obs], [])
    assert np.allclose(result, desired)


def test_avoidance_fires_inside_obstacle():
    # Bug #2 regression: dist < 0（障害物内部）でも回避が発動する
    sim = _default_sim()
    obs = SphereObstacle(pos=np.zeros(3), radius=50.0)
    inside_pos  = np.array([10.0, 0.0, 0.0])       # 内部（dist = -40）
    desired_vel = np.array([0.0, 10.0, 0.0])        # 直交方向に進行中
    result = sim._avoidance_steer(inside_pos, desired_vel, 10.0, 0.0, [obs], [])
    assert not np.allclose(result, desired_vel, atol=0.1), "内部侵入時に回避が発動しなかった"


def test_avoidance_result_preserves_speed():
    sim = _default_sim()
    obs = SphereObstacle(pos=np.array([0.0, 20.0, 0.0]), radius=50.0)
    pos = np.array([0.0, -10.0, 0.0])   # zone 内に入る
    desired_vel = np.array([10.0, 5.0, 0.0])
    desired_speed = float(np.linalg.norm(desired_vel))
    result = sim._avoidance_steer(pos, desired_vel, desired_speed, 0.0, [obs], [])
    assert abs(np.linalg.norm(result) - desired_speed) < 0.1


def test_avoidance_deflects_heading_near_obstacle():
    sim = _default_sim()
    # 右前方に障害物（avoidance zone 内）→ Y 負方向（左）に曲げられる
    # radius=10, zone=30: pos=(0,0,0), obs=(20,15,0), dist=25-10=15 < 30 → fires
    obs = SphereObstacle(pos=np.array([20.0, 15.0, 0.0]), radius=10.0)
    pos = np.array([0.0, 0.0, 0.0])
    desired = np.array([10.0, 0.0, 0.0])
    result = sim._avoidance_steer(pos, desired, 10.0, 0.0, [obs], [])
    # repulsion dir は pos→obs の逆方向 → Y 成分が負になる（右前方障害物を左に回避）
    assert result[1] < 0.0


# ── run() 基本テスト ────────────────────────────────────────────────────────────

def test_run_returns_required_keys():
    hist = _short_run()
    required = ['pos', 'vel', 'accel', 'time', 'speed', 'elevation', 'azimuth', 'hit_ground']
    for key in required:
        assert key in hist, f"履歴に '{key}' が含まれていない"


def test_run_shapes_consistent():
    hist = _short_run()
    n = len(hist['time'])
    assert hist['pos'].shape    == (n, 3)
    assert hist['vel'].shape    == (n, 3)
    assert hist['accel'].shape  == (n, 3)
    assert hist['speed'].shape  == (n,)


def test_run_speed_within_max():
    sim = _default_sim(max_speed=272.0)
    hist = sim.run([np.array([0.0, 0.0, 200.0]), np.array([2000.0, 0.0, 200.0])],
                   profile='standard')
    assert float(hist['speed'].max()) <= 272.0 + 2.0  # 微小な dt オーバーシュートを許容


def test_run_trajectory_reaches_vicinity_of_waypoint():
    hist = _short_run(dist_m=2000.0)
    final_pos = hist['pos'][-1]
    wp1 = np.array([2000.0, 0.0, 200.0])
    # 最終位置は目標の近くか、通過済み
    dist_start = np.linalg.norm(hist['pos'][0] - wp1)
    dist_end   = np.linalg.norm(final_pos - wp1)
    assert dist_end < dist_start, "シミュレーション後の方が目標から遠い"


def test_run_time_monotonically_increasing():
    hist = _short_run()
    diffs = np.diff(hist['time'])
    assert np.all(diffs > 0), "時刻が単調増加していない"


def test_run_hit_ground_is_bool():
    hist = _short_run()
    assert isinstance(hist['hit_ground'], bool)


def test_run_parallel_returns_multiple_results():
    sim0 = Simulator(SimParams(max_speed=272.0, nav_mode='gps'))
    sim1 = Simulator(SimParams(max_speed=200.0, nav_mode='gps'))
    wps = [np.array([0.0, 0.0, 200.0]), np.array([2000.0, 0.0, 200.0])]
    results = Simulator.run_parallel([(wps, sim0), (wps, sim1)])
    assert len(results) == 2
    assert 'pos' in results[0]
    assert 'pos' in results[1]
