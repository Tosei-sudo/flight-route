"""
障害物定義モジュール

クラス階層:
  FixedObstacle (ABC)
  ├── SphereObstacle  … レーダーサイト・禁止空域など（球状）
  └── BoxObstacle     … 建物・施設など（軸平行直方体 AABB）
  MovingObstacle      … 航空機・艦船など移動体

ユーザー編集箇所:
  get_fixed_obstacles()  — 固定障害物リスト
  get_moving_obstacles() — 移動体リスト

座標は _geo_pos() を使うと緯度経度+AGL で指定できる。
回避ゾーン = zone プロパティ (config.py の AVOIDANCE_MARGIN を参照)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from config import AVOIDANCE_MARGIN
from geo import geo_to_local, GEO_WAYPOINTS
from terrain import terrain_height_at

GDB_PATH = r'C:\Users\tosei\work\flight-route\MD.gdb'


# ── ヘルパー ────────────────────────────────────────────────────────────────────

def _geo_pos(lat: float, lon: float, alt_agl: float = 0.0) -> np.ndarray:
    """緯度経度 + AGL高度 → ローカル ENU 座標 [m]（出発点原点）"""
    pts = geo_to_local([GEO_WAYPOINTS[0], (lat, lon, 0.0)])
    pos = pts[1].copy()
    pos[2] = terrain_height_at(lat, lon) + alt_agl
    return pos


# ── 固定障害物 基底クラス ────────────────────────────────────────────────────────

class FixedObstacle(ABC):
    """固定障害物の抽象基底クラス。

    サブクラスは zone / dist_from_surface / repulsion_dir を実装する。
    """

    label: str

    @property
    @abstractmethod
    def zone(self) -> float:
        """回避行動を開始する距離 [m]（表面からの距離）"""

    @abstractmethod
    def dist_from_surface(self, pos: np.ndarray) -> float:
        """pos から障害物表面までの符号付き距離 [m]。
        正値 = 外部, 負値 = 内部（侵入済み）。
        """

    @abstractmethod
    def repulsion_dir(self, pos: np.ndarray) -> np.ndarray:
        """pos から見た障害物表面の外向き法線（単位ベクトル）。
        回避方向の基準として使用。
        """


# ── 球状障害物 ────────────────────────────────────────────────────────────────

@dataclass
class SphereObstacle(FixedObstacle):
    """球状障害物（レーダーサイト・禁止空域など）。

    Args:
        pos    : 中心位置（ローカル ENU [m]）
        radius : 回避半径 [m]
        label  : 表示ラベル
    """
    pos: np.ndarray
    radius: float
    label: str = ''

    @property
    def zone(self) -> float:
        return self.radius * AVOIDANCE_MARGIN

    def dist_from_surface(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.pos)) - self.radius

    def repulsion_dir(self, pos: np.ndarray) -> np.ndarray:
        to_pos = pos - self.pos
        n = float(np.linalg.norm(to_pos))
        return to_pos / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


# ── 直方体障害物 ─────────────────────────────────────────────────────────────

@dataclass
class BoxObstacle(FixedObstacle):
    """軸平行直方体障害物（建物・施設など）。

    Args:
        pos          : 直方体の中心位置（ローカル ENU [m]）
        half_extents : 各軸方向の半サイズ [dx, dy, dz] [m]
        label        : 表示ラベル

    Notes:
        AGL での中心指定例（地上50m建物、高さ100m）:
            pos = _geo_pos(lat, lon, alt_agl=50.0)   ← 中心を地上50m
            half_extents = np.array([width/2, depth/2, 50.0])
    """
    pos: np.ndarray
    half_extents: np.ndarray
    label: str = ''

    @property
    def zone(self) -> float:
        # 外接球半径 × マージン（直方体全体をカバーする球）
        return float(np.linalg.norm(self.half_extents)) * AVOIDANCE_MARGIN

    def dist_from_surface(self, pos: np.ndarray) -> float:
        """AABB の符号付き距離関数 (SDF)"""
        q = np.abs(pos - self.pos) - self.half_extents
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))

    def repulsion_dir(self, pos: np.ndarray) -> np.ndarray:
        # 直方体表面上の最近点を求め、そこから pos へのベクトルを返す
        nearest = np.clip(pos - self.pos, -self.half_extents, self.half_extents) + self.pos
        to_pos = pos - nearest
        n = float(np.linalg.norm(to_pos))
        if n > 1e-9:
            return to_pos / n
        # pos が直方体内部: 最近い面の外向き法線
        penetration = self.half_extents - np.abs(pos - self.pos)
        axis = int(np.argmin(penetration))
        normal = np.zeros(3)
        normal[axis] = float(np.sign(pos[axis] - self.pos[axis]))
        return normal


# ── 移動体障害物 ─────────────────────────────────────────────────────────────

@dataclass
class MovingObstacle:
    """移動体（航空機・艦船など）。

    Args:
        pos_init : t=0 のローカル ENU 位置 [m]
        vel      : 速度ベクトル [m/s]
        radius   : 回避半径 [m]
        label    : 表示ラベル
    """
    pos_init: np.ndarray
    vel: np.ndarray
    radius: float
    label: str = ''

    def pos_at(self, t: float) -> np.ndarray:
        """時刻 t [s] における予測位置"""
        return self.pos_init + self.vel * t

    @property
    def zone(self) -> float:
        return self.radius * AVOIDANCE_MARGIN


# ── 障害物定義（ここを編集する）────────────────────────────────────────────────

_HARDCODED_SPHERES: list[SphereObstacle] = [
    SphereObstacle(
        pos=_geo_pos(34.88, 135.280, alt_agl=0.0),
        radius=2000.0,
        label='防空レーダーA',
    ),
    SphereObstacle(
        pos=_geo_pos(35.261049, 135.147330, alt_agl=0.0),
        radius=6000.0,
        label='防空レーダーC',
    ),
    SphereObstacle(
        pos=_geo_pos(35.35, 135.240, alt_agl=0.0),
        radius=2500.0,
        label='防空レーダーB',
    ),
]

_HARDCODED_BOXES: list[BoxObstacle] = [
    BoxObstacle(
        pos=_geo_pos(35.12, 135.265, alt_agl=100.0),
        half_extents=np.array([300.0, 150.0, 100.0]),
        label='通信施設C',
    ),
]


def _load_sphere_obstacles_from_gdb() -> list[SphereObstacle] | None:
    """GDB の SPHERE_OBSTACLE フィーチャクラスから球状障害物を読み込む。

    arcpy が使用できない場合や GDB が見つからない場合は None を返す。
    """
    try:
        import arcpy  # noqa: PLC0415
    except ImportError:
        return None

    import logging
    logger = logging.getLogger(__name__)
    fc = rf'{GDB_PATH}\SPHERE_OBSTACLE'
    try:
        result = []
        with arcpy.da.SearchCursor(fc, ['NAME', 'RADIUS', 'SHAPE@XY', 'SHAPE@Z']) as cur:
            for name, radius, (lon, lat), z in cur:
                result.append(SphereObstacle(
                    pos=_geo_pos(lat, lon, alt_agl=float(z or 0.0)),
                    radius=float(radius),
                    label=str(name or ''),
                ))
        logger.info("GDB SPHERE_OBSTACLE: %d 件読み込み (%s)", len(result), GDB_PATH)
        return result
    except Exception as exc:
        logging.getLogger(__name__).warning("GDB 読み込み失敗: %s", exc)
        return None


def get_fixed_obstacles() -> list[FixedObstacle]:
    """固定障害物リストを返す。

    球状障害物は GDB (SPHERE_OBSTACLE) から読み込む。
    arcpy が使えない場合はハードコードにフォールバックする。
    BoxObstacle はハードコードで管理する。
    """
    spheres = _load_sphere_obstacles_from_gdb()
    if spheres is None:
        spheres = _HARDCODED_SPHERES
    return spheres + _HARDCODED_BOXES


def get_moving_obstacles() -> list[MovingObstacle]:
    """移動体リストを返す。位置は t=0 時点。"""
    return [
        # t≈245s に経路を東向きに横断する想定
        # 出発点から西25km (lon≈134.98°) に配置し、120m/s で東進
        MovingObstacle(
            pos_init=_geo_pos(35.13, 134.98, alt_agl=250.0),
            vel=np.array([120.0, -10.0, 0.0]),   # 東向き 120m/s、わずかに南下
            radius=1500.0,
            label='移動体α',
        ),
    ]
