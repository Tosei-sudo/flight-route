"""
グローバルパスプランナー  (2-D grid A*)

ウェイポイント間を障害物ゾーンを避けた A* 経路で展開し、
シミュレーター実行前に中間 WP を挿入する前処理モジュール。

障害物のローカル反発（_avoidance_steer）では対応しにくい
大型障害物・広域禁止空域を事前に迂回するための経路計画レイヤ。

典型的な使い方（simulator.py が内部で呼ぶ）:
    SimParams(use_global_planner=True)
"""

from __future__ import annotations

import heapq
import logging

import numpy as np

from obstacles import FixedObstacle

logger = logging.getLogger(__name__)


class GridPlanner:
    """2-D グリッド A* プランナー。

    水平面 (XY) でグリッドを構築し A* で経路を求める。
    高度 (Z) は始点・終点間を線形補間し、スムーシング時に 3-D チェックをかける。

    Args:
        obstacles  : 固定障害物リスト（zone を回避ゾーンとして使用）
        resolution : グリッドセルの一辺長 [m]
    """

    def __init__(self, obstacles: list[FixedObstacle], resolution: float = 300.0):
        self._obs = obstacles
        self._res = float(resolution)

    # ── 障害物チェック ────────────────────────────────────────────────────────

    def _is_blocked(self, x: float, y: float, z: float = 0.0) -> bool:
        """座標 (x, y, z) がいずれかの回避ゾーン内なら True。"""
        pos = np.array([x, y, z])
        return any(obs.dist_from_surface(pos) < obs.zone for obs in self._obs)

    # ── 直線視通チェック ──────────────────────────────────────────────────────

    def _los_clear(self, a: np.ndarray, b: np.ndarray, samples: int = 20) -> bool:
        """a → b の直線が障害物ゾーンを通過しないなら True。"""
        for t in np.linspace(0.0, 1.0, samples)[1:-1]:
            p = a + t * (b - a)
            if self._is_blocked(float(p[0]), float(p[1]), float(p[2])):
                return False
        return True

    # ── スムーシング ──────────────────────────────────────────────────────────

    def _smooth(self, pts: list[np.ndarray]) -> list[np.ndarray]:
        """視通チェックで冗長な中間点を除去（グリーディ・ショートカット）。"""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if self._los_clear(pts[i], pts[j]):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        return out

    # ── A* ───────────────────────────────────────────────────────────────────

    def plan(self, start: np.ndarray, goal: np.ndarray,
             pad: float = 5000.0) -> list[np.ndarray]:
        """start → goal の衝突回避 3-D ウェイポイント列を返す。

        グリッドは XY 平面 (z=0) で構築し障害物ゾーンを保守的にマーク。
        スムーシング後の経路は 3-D LOS チェックで冗長点を除去する。

        Returns
        -------
        list[np.ndarray]
            障害物を避けた 3-D 座標列。start と goal を含む。
            経路が見つからない場合は ``[start, goal]`` を返す。
        """
        res = self._res

        # グリッド範囲（start-goal AABB + パディング）
        x_min = min(float(start[0]), float(goal[0])) - pad
        x_max = max(float(start[0]), float(goal[0])) + pad
        y_min = min(float(start[1]), float(goal[1])) - pad
        y_max = max(float(start[1]), float(goal[1])) + pad

        nx = max(int((x_max - x_min) / res) + 1, 2)
        ny = max(int((y_max - y_min) / res) + 1, 2)

        def to_cell(pt: np.ndarray) -> tuple[int, int]:
            i = int((float(pt[0]) - x_min) / res)
            j = int((float(pt[1]) - y_min) / res)
            return int(np.clip(i, 0, nx - 1)), int(np.clip(j, 0, ny - 1))

        def cell_center(ci: int, cj: int) -> tuple[float, float]:
            return x_min + ci * res, y_min + cj * res

        # ブロックマップ構築（z=0 で保守的にチェック）
        blocked = np.zeros((nx, ny), dtype=bool)
        for ci in range(nx):
            for cj in range(ny):
                wx, wy = cell_center(ci, cj)
                if self._is_blocked(wx, wy):
                    blocked[ci, cj] = True

        logger.info("A* グリッド %d×%d (解像度 %.0fm), ブロック %d セル",
                    nx, ny, res, int(blocked.sum()))

        s = to_cell(start)
        g = to_cell(goal)

        if s == g:
            return [start.copy(), goal.copy()]

        # A*（8 方向移動）
        def h(ci: int, cj: int) -> float:
            return float(np.hypot(ci - g[0], cj - g[1]))

        heap: list = [(h(*s), 0.0, s, None)]
        came_from: dict[tuple, tuple | None] = {}
        g_cost: dict[tuple, float] = {s: 0.0}

        found = False
        while heap:
            _, gc, cur, par = heapq.heappop(heap)
            if cur in came_from:
                continue
            came_from[cur] = par
            if cur == g:
                found = True
                break
            ci, cj = cur
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = ci + di, cj + dj
                    if not (0 <= ni < nx and 0 <= nj < ny):
                        continue
                    if blocked[ni, nj]:
                        continue
                    nxt = (ni, nj)
                    ng = gc + float(np.hypot(di, dj))
                    if ng < g_cost.get(nxt, float('inf')):
                        g_cost[nxt] = ng
                        heapq.heappush(heap, (ng + h(ni, nj), ng, nxt, cur))

        if not found:
            logger.warning("A* 経路未発見: start=%s goal=%s → 直線経路にフォールバック",
                           start[:2], goal[:2])
            return [start.copy(), goal.copy()]

        # 経路復元
        path_cells: list[tuple[int, int]] = []
        node: tuple | None = g
        while node is not None:
            path_cells.append(node)
            node = came_from[node]
        path_cells.reverse()

        # グリッド座標 → 3-D（高度は線形補間）
        total = max(len(path_cells) - 1, 1)
        dz = float(goal[2]) - float(start[2])
        pts_3d: list[np.ndarray] = []
        for k, (ci, cj) in enumerate(path_cells):
            wx, wy = cell_center(ci, cj)
            wz = float(start[2]) + dz * k / total
            pts_3d.append(np.array([wx, wy, wz]))

        # start / goal を正確な値で上書き
        pts_3d[0] = start.copy()
        pts_3d[-1] = goal.copy()

        smoothed = self._smooth(pts_3d)
        logger.info("A* 完了: %d セル → スムーシング後 %d 点 (中間点 %d 個)",
                    len(path_cells), len(smoothed), len(smoothed) - 2)
        return smoothed


# ── 公開 API ──────────────────────────────────────────────────────────────────

def expand_waypoints(
    waypoints: list,
    obstacles: list[FixedObstacle],
    resolution: float = 300.0,
) -> list:
    """ウェイポイント列を A* で障害物回避済みの列に展開する。

    連続する固定 WP ペアごとに A* を実行し、直線経路が障害物ゾーンを
    通過する区間にのみ中間点を挿入する。

    callable（移動目標）エントリは A* をスキップし、そのまま保持する
    （移動目標への終末誘導はリアクティブ回避に任せる）。

    Args:
        waypoints  : np.ndarray（固定）または callable（移動目標）の混合リスト
        obstacles  : 固定障害物リスト
        resolution : A* グリッド解像度 [m]（小さいほど精密・低速）

    Returns
    -------
    list
        中間点を挿入した展開済みウェイポイント列
    """
    if len(waypoints) < 2 or not obstacles:
        return waypoints

    planner = GridPlanner(obstacles, resolution=resolution)
    result: list = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]

        # callable を含む区間はスキップ
        if callable(a) or callable(b):
            result.append(b)
            continue

        # LOS が通れば A* 不要
        if planner._los_clear(a, b):
            result.append(b)
            continue

        path = planner.plan(a, b)
        result.extend(path[1:])  # start はすでに result に含まれる

    return result
