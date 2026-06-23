"""
バグ修正リグレッションテスト

Issue #2〜#10 および #20 で修正されたバグが再発していないことを確認する。
各テストは静的コード検査 + 実行時動作確認の二段構成。
"""

import re
import inspect
import threading
import pytest
import numpy as np

from simulator import Simulator, SimParams
from obstacles import SphereObstacle
from nav import INSSensor
import geo
import gdb_export
import config as cfg
import wind
import terrain


# ── Issue #2: 障害物内部侵入時の回避ゾーン判定 ────────────────────────────────

class TestIssue2ObstaclePenetration:
    def test_static_no_guard_0(self):
        """`if 0 < dist < obs.zone` の誤ったガードがない"""
        src = inspect.getsource(Simulator._avoidance_steer)
        assert not re.search(r'if\s+0\s*<\s*dist\s*<\s*obs\.zone', src), \
            "古いガード `if 0 < dist < obs.zone` が残っている"

    def test_static_correct_check(self):
        """`if dist < obs.zone` が存在する"""
        src = inspect.getsource(Simulator._avoidance_steer)
        assert re.search(r'if\s+dist\s*<\s*obs\.zone', src), \
            "`if dist < obs.zone` が見つからない"

    def test_runtime_fires_inside_obstacle(self):
        """pos が障害物内部（dist < 0）でも回避が発動する"""
        sim = Simulator(SimParams(nav_mode='gps'))
        obs = SphereObstacle(pos=np.zeros(3), radius=50.0)
        inside_pos = np.array([10.0, 0.0, 0.0])         # dist = -40
        desired    = np.array([0.0, 10.0, 0.0])          # 直交方向
        assert obs.dist_from_surface(inside_pos) < 0
        result = sim._avoidance_steer(inside_pos, desired, 10.0, 0.0, [obs], [])
        assert not np.allclose(result, desired, atol=0.1)


# ── Issue #3: 回避ブレーキの符号 ────────────────────────────────────────────────

class TestIssue3BrakeSign:
    def test_static_minus_equals(self):
        """`repulse[:] -= heading_h * max(-dot, ...)` になっている"""
        src = inspect.getsource(Simulator._avoidance_steer)
        assert re.search(r'repulse\[:\]\s*-=\s*heading_h\s*\*\s*max\(-dot', src), \
            "ブレーキ符号が += のまま（誤った正方向加速）"
        assert not re.search(r'repulse\[:\]\s*\+=\s*heading_h\s*\*\s*max\(-dot', src), \
            "誤ったブレーキ行 `+= heading_h * max(-dot` が残っている"


# ── Issue #4: 終末誘導の残り時間算出 ──────────────────────────────────────────

class TestIssue4TerminalTimeToGo:
    def test_static_uses_dist_3d(self):
        """`time_to_go = dist_3d / speed` を使っている（dist_horiz ではない）"""
        src = inspect.getsource(Simulator.run)
        assert not re.search(r'time_to_go\s*=\s*dist_horiz\s*/\s*speed', src), \
            "`time_to_go = dist_horiz / speed` が残っている"
        assert re.search(r'time_to_go\s*=\s*dist_3d\s*/\s*speed', src), \
            "`time_to_go = dist_3d / speed` が見つからない"


# ── Issue #5: 艦船移動速度コメント ────────────────────────────────────────────

class TestIssue5MovingShipComment:
    def test_comment_says_50_not_100(self):
        """_moving_ship のコメントが 100m/s ではなく 50m/s"""
        src = inspect.getsource(cfg._moving_ship)
        assert '100m/s' not in src and '100 m/s' not in src, \
            "古いコメント '100m/s' が残っている"


# ── Issue #6: GeoJSON フォールバックパス ─────────────────────────────────────

class TestIssue6GeoJsonPath:
    def test_static_uses_config_geojson_path(self):
        """_fallback() が config.GEOJSON_PATH を使っている"""
        src = inspect.getsource(gdb_export._fallback)
        assert 'GEOJSON_PATH' in src, \
            "_fallback が config.GEOJSON_PATH を参照していない"
        assert not (re.search(r"gdb_path[^)]*replace|replace[^)]*gdb_path", src)
                    and 'GEOJSON_PATH' not in src), \
            "GEOJSON_PATH を gdb_path から文字列生成している（修正前の実装）"


# ── Issue #7: INS 前進 Euler 積分 ────────────────────────────────────────────

class TestIssue7INSVelOld:
    def test_static_vel_old_present(self):
        """`vel_old = self._vel.copy()` が INSSensor.update にある"""
        src = inspect.getsource(INSSensor.update)
        assert 'vel_old' in src, "vel_old が INSSensor.update に見つからない"

    def test_runtime_uses_pre_update_velocity(self):
        """vel=0, accel=1, dt=1 のとき pos は 0 のまま（vel_old=0）"""
        ins = INSSensor(np.zeros(3), np.zeros(3))
        pos = ins.update(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0)
        assert abs(float(pos[0])) < 0.01, \
            f"vel_old=0 なのに pos が {pos[0]:.3f} になった（誤った積分）"


# ── Issue #8: callable 第1ウェイポイントのアサーション ──────────────────────

class TestIssue8CallableFirstWp:
    def test_static_callable_check_exists(self):
        """`geo_to_local` に callable チェックがある"""
        src = inspect.getsource(geo.geo_to_local)
        assert 'callable' in src

    def test_runtime_raises_assertion_error(self):
        """callable を第1 WP に渡すと AssertionError が発生する"""
        with pytest.raises(AssertionError, match="GEO_WAYPOINTS\\[0\\]"):
            geo.geo_to_local([lambda t: (45.0, 141.0, 0), (41.0, 143.0, 0)])


# ── Issue #9: 終末誘導時間の誤った ×2 ─────────────────────────────────────────

class TestIssue9TerminalTimeNoX2:
    def test_static_no_spurious_multiplication_by_2(self):
        """`_auto_terminal_time` に不要な *2 がない"""
        src = inspect.getsource(Simulator._auto_terminal_time)
        suspicious = [
            line.strip() for line in src.splitlines()
            if re.search(r'(?<!\*)\*\s*2(?!\*)', line)
            and '1.3' not in line       # *1.3 のマージン係数は正常
            and 'sqrt' not in line      # sqrt の中の **2 は正常
            and '**' not in line        # 累乗は除外
        ]
        assert not suspicious, f"不要な *2 が疑われる行: {suspicious}"

    def test_runtime_returns_positive_value(self):
        t = Simulator()._auto_terminal_time()
        assert t > 0.0


# ── Issue #10: スレッドセーフなシングルトン ───────────────────────────────────

class TestIssue10ThreadSafeSingletons:
    @pytest.mark.parametrize('module,src_fn', [
        ('wind',    lambda: inspect.getsource(wind)),
        ('terrain', lambda: inspect.getsource(terrain)),
    ])
    def test_has_threading_import(self, module, src_fn):
        assert 'threading' in src_fn(), f"{module}.py に threading がない"

    @pytest.mark.parametrize('module,src_fn', [
        ('wind',    lambda: inspect.getsource(wind)),
        ('terrain', lambda: inspect.getsource(terrain)),
    ])
    def test_has_lock(self, module, src_fn):
        src = src_fn()
        assert '_lock' in src, f"{module}.py に _lock がない"
        assert 'with _lock' in src, f"{module}.py に `with _lock:` がない"

    @pytest.mark.parametrize('module,src_fn', [
        ('wind',    lambda: inspect.getsource(wind.get_wind_field)),
        ('terrain', lambda: inspect.getsource(terrain._ensure_init)),
    ])
    def test_double_checked_locking(self, module, src_fn):
        """ダブルチェックロッキングパターンがある"""
        src = src_fn()
        # パターン A: if X is None → lock → if X is None  (wind.py)
        pattern_a = re.search(
            r'if\s+\w+\s+is\s+None.*\n.*with\s+_lock.*\n.*if\s+\w+\s+is\s+None',
            src, re.DOTALL)
        # パターン B: if X is not None: return → lock → if X is not None  (terrain.py)
        pattern_b = re.search(
            r'if\s+\w+\s+is\s+not\s+None.*return.*\n.*with\s+_lock.*\n.*if\s+\w+\s+is\s+not\s+None',
            src, re.DOTALL)
        assert pattern_a or pattern_b, \
            f"{module}.py にダブルチェックロッキングが見つからない"


# ── Issue #20: 複数障害物の球面射影 ──────────────────────────────────────────

class TestIssue20MultiObstacleSphericalProjection:
    def test_threat_dirs_second_pass_present(self):
        """`threat_dirs` 二次パス + 球面射影が _avoidance_steer にある"""
        src = inspect.getsource(Simulator._avoidance_steer)
        assert 'threat_dirs' in src, "`threat_dirs` 二次パスが見つからない"
        assert 'sin_theta' in src or 'v_perp' in src, "球面射影コードが見つからない"

    def test_two_obstacles_no_oscillation(self):
        """2障害物を同時に避ける際、どちらも侵入しない方向に誘導される"""
        sim = Simulator(SimParams(nav_mode='gps'))
        # 左右に障害物を配置、正面方向に進行
        obs_left  = SphereObstacle(pos=np.array([100.0,  80.0, 0.0]), radius=20.0)
        obs_right = SphereObstacle(pos=np.array([100.0, -80.0, 0.0]), radius=20.0)
        pos     = np.array([0.0, 0.0, 0.0])
        desired = np.array([10.0, 0.0, 0.0])
        result  = sim._avoidance_steer(pos, desired, 10.0, 0.0,
                                       [obs_left, obs_right], [])
        # 結果がどちらの障害物にも向かっていないこと
        # （Y=0 の対称位置から進んでいれば、対称破れが起きても dist は増加するはず）
        dist_left  = obs_left.dist_from_surface(result)
        dist_right = obs_right.dist_from_surface(result)
        assert dist_left  > obs_left.dist_from_surface(desired)  - 1.0
        assert dist_right > obs_right.dist_from_surface(desired) - 1.0
