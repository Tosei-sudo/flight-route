"""
GridPlanner / expand_waypoints のテスト

障害物がない場合・直線が通れる場合・障害物で遮られる場合の
各シナリオで A* 経路生成と WP 展開が正しく動作することを確認する。
"""

import numpy as np
import pytest

from planner import GridPlanner, expand_waypoints
from obstacles import SphereObstacle, AreaObstacle


# ── GridPlanner 基本動作 ──────────────────────────────────────────────────────

class TestGridPlannerBasics:
    def test_plan_returns_start_and_goal(self):
        """plan() の先頭が start、末尾が goal と一致する。"""
        planner = GridPlanner(obstacles=[], resolution=100.0)
        start = np.array([0.0, 0.0, 0.0])
        goal  = np.array([1000.0, 0.0, 0.0])
        path  = planner.plan(start, goal)
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)

    def test_plan_no_obstacles_two_points(self):
        """障害物なしでは [start, goal] の 2 点のみ返す。"""
        planner = GridPlanner(obstacles=[], resolution=100.0)
        start = np.array([0.0, 0.0, 0.0])
        goal  = np.array([2000.0, 0.0, 0.0])
        path  = planner.plan(start, goal)
        assert len(path) == 2

    def test_plan_clear_path_two_points(self):
        """直線が障害物ゾーン外なら 2 点のみ。"""
        obs     = SphereObstacle(pos=np.array([0.0, 5000.0, 0.0]), radius=500.0)
        planner = GridPlanner(obstacles=[obs], resolution=100.0)
        start   = np.array([0.0, 0.0, 0.0])
        goal    = np.array([2000.0, 0.0, 0.0])
        path    = planner.plan(start, goal)
        assert len(path) == 2

    def test_plan_blocked_inserts_waypoints(self):
        """直線が障害物に遮られた場合、中間点が挿入される。"""
        # radius=200m → zone=600m; start/goal は (0,0,0),(2000,0,0) でそれぞれ 1000m 離れておりゾーン外
        obs     = SphereObstacle(pos=np.array([1000.0, 0.0, 0.0]), radius=200.0)
        planner = GridPlanner(obstacles=[obs], resolution=100.0)
        start   = np.array([0.0, 0.0, 0.0])
        goal    = np.array([2000.0, 0.0, 0.0])
        path    = planner.plan(start, goal, pad=2000.0)
        assert len(path) > 2, "障害物があるのに中間点が挿入されなかった"

    def test_plan_path_outside_obstacle_zone(self):
        """生成経路の全点が障害物ゾーン外にある。"""
        obs     = SphereObstacle(pos=np.array([1000.0, 0.0, 0.0]), radius=300.0)
        planner = GridPlanner(obstacles=[obs], resolution=100.0)
        start   = np.array([0.0, 0.0, 0.0])
        goal    = np.array([2000.0, 0.0, 0.0])
        path    = planner.plan(start, goal, pad=2000.0)
        for pt in path:
            dist = obs.dist_from_surface(pt)
            assert dist >= -1.0, f"経路点 {pt[:2]} が障害物ゾーン内 (dist={dist:.1f})"

    def test_plan_z_interpolation(self):
        """高度差がある場合、start/goal の高度が保持される。"""
        planner = GridPlanner(obstacles=[], resolution=200.0)
        start   = np.array([0.0, 0.0, 100.0])
        goal    = np.array([2000.0, 0.0, 500.0])
        path    = planner.plan(start, goal)
        assert abs(path[0][2] - 100.0) < 1e-6
        assert abs(path[-1][2] - 500.0) < 1e-6

    def test_plan_minimum_two_points(self):
        """start と goal が同じグリッドセルに落ちても最低 2 点を返す。"""
        planner = GridPlanner(obstacles=[], resolution=10000.0)
        start   = np.array([0.0, 0.0, 0.0])
        goal    = np.array([1.0, 0.0, 0.0])  # 同じセル
        path    = planner.plan(start, goal)
        assert len(path) >= 2
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)


# ── 複数障害物 ────────────────────────────────────────────────────────────────

class TestGridPlannerMultiObstacle:
    def test_two_obstacles_side_by_side(self):
        """左右の障害物の間を通り抜けられる場合は経路が生成される。"""
        obs_top = SphereObstacle(pos=np.array([1000.0,  600.0, 0.0]), radius=200.0)
        obs_bot = SphereObstacle(pos=np.array([1000.0, -600.0, 0.0]), radius=200.0)
        planner = GridPlanner(obstacles=[obs_top, obs_bot], resolution=100.0)
        start   = np.array([0.0, 0.0, 0.0])
        goal    = np.array([2000.0, 0.0, 0.0])
        path    = planner.plan(start, goal, pad=2000.0)
        # 経路が見つかり、両障害物ゾーン外を通る
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)
        for pt in path:
            assert obs_top.dist_from_surface(pt) >= -1.0
            assert obs_bot.dist_from_surface(pt) >= -1.0

    def test_area_obstacle_blocked(self):
        """AreaObstacle が直線を塞いでいる場合に迂回する。"""
        # X=500〜1500, Y=-500〜500 の矩形エリア
        verts = np.array([[500.0, -500.0],
                          [1500.0, -500.0],
                          [1500.0,  500.0],
                          [500.0,  500.0]])
        obs     = AreaObstacle(vertices_enu=verts, height_msl=1000.0,
                               label='禁止空域', zone_m=200.0)
        planner = GridPlanner(obstacles=[obs], resolution=100.0)
        start   = np.array([0.0, 0.0, 0.0])
        goal    = np.array([2000.0, 0.0, 0.0])
        path    = planner.plan(start, goal, pad=1500.0)
        assert len(path) > 2, "AreaObstacle を回避するはずが中間点なし"


# ── LOS チェック ──────────────────────────────────────────────────────────────

class TestLosCheck:
    def test_los_clear_when_no_obstacles(self):
        planner = GridPlanner(obstacles=[], resolution=100.0)
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1000.0, 0.0, 0.0])
        assert planner._los_clear(a, b)

    def test_los_blocked_by_obstacle(self):
        obs     = SphereObstacle(pos=np.array([500.0, 0.0, 0.0]), radius=200.0)
        planner = GridPlanner(obstacles=[obs], resolution=100.0)
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1000.0, 0.0, 0.0])
        assert not planner._los_clear(a, b)

    def test_los_clear_when_passing_beside_obstacle(self):
        obs     = SphereObstacle(pos=np.array([500.0, 1000.0, 0.0]), radius=200.0)
        planner = GridPlanner(obstacles=[obs], resolution=100.0)
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1000.0, 0.0, 0.0])
        assert planner._los_clear(a, b)


# ── expand_waypoints ──────────────────────────────────────────────────────────

class TestExpandWaypoints:
    def test_no_obstacles_unchanged(self):
        """障害物なしでは WP 列が変わらない。"""
        wps    = [np.array([0.0, 0.0, 0.0]), np.array([1000.0, 0.0, 0.0])]
        result = expand_waypoints(wps, obstacles=[], resolution=100.0)
        assert len(result) == 2
        assert np.allclose(result[0], wps[0])
        assert np.allclose(result[-1], wps[-1])

    def test_clear_path_unchanged(self):
        """直線が障害物ゾーン外なら WP 列は変わらない。"""
        obs    = SphereObstacle(pos=np.array([500.0, 5000.0, 0.0]), radius=100.0)
        wps    = [np.array([0.0, 0.0, 0.0]), np.array([1000.0, 0.0, 0.0])]
        result = expand_waypoints(wps, [obs], resolution=100.0)
        assert len(result) == 2

    def test_blocked_path_expands(self):
        """障害物で遮られた経路に中間点が挿入される。"""
        # radius=200m → zone=600m; start/goal は 1000m 離れておりゾーン外
        obs    = SphereObstacle(pos=np.array([1000.0, 0.0, 0.0]), radius=200.0)
        wps    = [np.array([0.0, 0.0, 0.0]), np.array([2000.0, 0.0, 0.0])]
        result = expand_waypoints(wps, [obs], resolution=100.0)
        assert len(result) > 2

    def test_start_and_goal_preserved(self):
        """展開後も先頭・末尾 WP は変わらない。"""
        obs    = SphereObstacle(pos=np.array([1000.0, 0.0, 0.0]), radius=200.0)
        wps    = [np.array([0.0, 0.0, 0.0]), np.array([2000.0, 0.0, 0.0])]
        result = expand_waypoints(wps, [obs], resolution=100.0)
        assert np.allclose(result[0], wps[0])
        assert np.allclose(result[-1], wps[-1])

    def test_callable_wp_skipped_and_preserved(self):
        """callable（移動目標）を含む区間は A* をスキップし callable を保持する。"""
        moving = lambda t: np.array([500.0, 0.0, 0.0])
        obs    = SphereObstacle(pos=np.array([500.0, 0.0, 0.0]), radius=200.0)
        wps    = [np.array([0.0, 0.0, 0.0]), moving]
        result = expand_waypoints(wps, [obs], resolution=100.0)
        assert callable(result[-1])
        assert len(result) == 2  # callable 区間は展開しない

    def test_multiple_segments_all_expanded(self):
        """複数区間でそれぞれ A* が適用され、先頭・末尾は維持される。"""
        obs  = SphereObstacle(pos=np.array([1000.0, 0.0, 0.0]), radius=400.0)
        wps  = [np.array([0.0,    0.0, 0.0]),
                np.array([2000.0, 0.0, 0.0]),
                np.array([3000.0, 0.0, 0.0])]
        result = expand_waypoints(wps, [obs], resolution=100.0)
        assert len(result) >= 3
        assert np.allclose(result[0], wps[0])
        assert np.allclose(result[-1], wps[-1])

    def test_single_waypoint_unchanged(self):
        """WP が 1 点以下の場合はそのまま返す。"""
        wps    = [np.array([0.0, 0.0, 0.0])]
        result = expand_waypoints(wps, [], resolution=100.0)
        assert result is wps


# ── SimParams との統合 ─────────────────────────────────────────────────────────

class TestSimParamsIntegration:
    def test_sim_params_has_planner_fields(self):
        """SimParams に use_global_planner と planner_resolution がある。"""
        from simulator import SimParams
        p = SimParams()
        assert hasattr(p, 'use_global_planner')
        assert hasattr(p, 'planner_resolution')
        assert isinstance(p.use_global_planner, bool)
        assert isinstance(p.planner_resolution, float)

    def test_sim_params_planner_disabled(self):
        """use_global_planner=False でも SimParams が正常生成される。"""
        from simulator import SimParams
        p = SimParams(use_global_planner=False)
        assert p.use_global_planner is False

    def test_expand_waypoints_called_in_run(self):
        """use_global_planner=True のとき run() が expand_waypoints を呼ぶ。"""
        from unittest.mock import patch
        from simulator import Simulator, SimParams
        import simulator as sim_module

        sim   = Simulator(SimParams(use_global_planner=True, nav_mode='gps'))
        start = np.array([0.0, 0.0, 0.0])
        goal  = np.array([500.0, 0.0, 0.0])

        with patch.object(sim_module, 'expand_waypoints', wraps=sim_module.expand_waypoints) as mock_expand:
            try:
                sim.run([start, goal])
            except Exception:
                pass
            mock_expand.assert_called_once()

    def test_expand_waypoints_not_called_when_disabled(self):
        """use_global_planner=False のとき expand_waypoints を呼ばない。"""
        from unittest.mock import patch
        from simulator import Simulator, SimParams
        import simulator as sim_module

        sim   = Simulator(SimParams(use_global_planner=False, nav_mode='gps'))
        start = np.array([0.0, 0.0, 0.0])
        goal  = np.array([500.0, 0.0, 0.0])

        with patch.object(sim_module, 'expand_waypoints') as mock_expand:
            try:
                sim.run([start, goal])
            except Exception:
                pass
            mock_expand.assert_not_called()
