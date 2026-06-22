"""
GDB エクスポートモジュール

arcpy が使える環境では ArcGIS File GDB (FGDB) に書き出す。
使えない場合は GeoJSON にフォールバックする。

出力フィーチャクラス:
  FLIGHT_TRAJECTORY   … Polyline Z  飛翔軌道
  WAYPOINTS           … Point Z     ウェイポイント
  FIXED_OBSTACLES     … Polygon     固定障害物フットプリント (sphere / box / area)
  MOVING_OBS_TRACKS   … Polyline Z  移動体軌跡
"""

import os
import logging
import numpy as np

from geo import local_to_geo
from obstacles import (get_fixed_obstacles, get_moving_obstacles,
                       SphereObstacle, BoxObstacle, AreaObstacle)

logger = logging.getLogger(__name__)


# ── 座標変換ヘルパー ──────────────────────────────────────────────────────────

def _lonlatz(pos: np.ndarray) -> tuple[float, float, float]:
    """ENU → (lon, lat, alt_msl)"""
    lat, lon = local_to_geo(pos)
    return float(lon), float(lat), float(pos[2])


def _circle_ring(center_enu: np.ndarray, radius_m: float, n: int = 64) -> list[tuple]:
    """ENU 中心・半径 [m] → [(lon, lat), ...] の円近似リング（閉じている）"""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = []
    for t in theta:
        pt = center_enu.copy()
        pt[0] += radius_m * np.cos(t)
        pt[1] += radius_m * np.sin(t)
        lat, lon = local_to_geo(pt)
        ring.append((float(lon), float(lat)))
    ring.append(ring[0])
    return ring


def _box_ring(center_enu: np.ndarray, half_extents: np.ndarray) -> list[tuple]:
    """ENU 中心・半サイズ → [(lon, lat), ...] の矩形リング（閉じている）"""
    hx, hy = half_extents[0], half_extents[1]
    ring = []
    for dx, dy in [(-hx, -hy), (+hx, -hy), (+hx, +hy), (-hx, +hy)]:
        pt = center_enu.copy()
        pt[0] += dx
        pt[1] += dy
        lat, lon = local_to_geo(pt)
        ring.append((float(lon), float(lat)))
    ring.append(ring[0])
    return ring


def _area_ring(vertices_enu: np.ndarray) -> list[tuple]:
    """ENU XY 頂点列 → [(lon, lat), ...] リング（閉じていなければ閉じる）"""
    ring = []
    for xy in vertices_enu:
        lat, lon = local_to_geo(np.array([xy[0], xy[1], 0.0]))
        ring.append((float(lon), float(lat)))
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


# ── arcpy を使った GDB 書き出し ───────────────────────────────────────────────

# 各実行回でレコードを区別するための RUN_ID フォーマット
_RUN_ID_FMT = '%Y%m%d_%H%M%S'

# FC 定義: (FC名, ジオメトリ種別, has_z, [(フィールド名, 型, 長さ), ...])
_FC_DEFS = {
    'FLIGHT_TRAJECTORY': ('POLYLINE', 'ENABLED', [
        ('RUN_ID',        'TEXT',   20),
        ('NAME',          'TEXT',   100),
        ('FLIGHT_TIME_S', 'DOUBLE', None),
        ('TOTAL_DIST_M',  'DOUBLE', None),
        ('MAX_SPEED_MS',  'DOUBLE', None),
        ('HIT_GROUND',    'SHORT',  None),
    ]),
    'WAYPOINTS': ('POINT', 'ENABLED', [
        ('RUN_ID',     'TEXT',   20),
        ('NAME',       'TEXT',   100),
        ('ALT_MSL_M',  'DOUBLE', None),
    ]),
    'FIXED_OBSTACLES': ('POLYGON', 'DISABLED', [
        ('RUN_ID',       'TEXT',   20),
        ('NAME',         'TEXT',   100),
        ('OBS_TYPE',     'TEXT',   20),
        ('RADIUS_M',     'DOUBLE', None),
        ('HEIGHT_MSL_M', 'DOUBLE', None),
        ('ZONE_M',       'DOUBLE', None),
    ]),
    'MOVING_OBS_TRACKS': ('POLYLINE', 'ENABLED', [
        ('RUN_ID',   'TEXT',   20),
        ('NAME',     'TEXT',   100),
        ('RADIUS_M', 'DOUBLE', None),
        ('SPEED_MS', 'DOUBLE', None),
    ]),
}


def _ensure_fc(arcpy, gdb_path: str, fc_name: str, sr) -> str:
    """FC が存在しなければ作成し、不足フィールドがあれば追加する。パスを返す。"""
    fc = f'{gdb_path}\\{fc_name}'
    geom, has_z, fields = _FC_DEFS[fc_name]
    if not arcpy.Exists(fc):
        arcpy.management.CreateFeatureclass(
            gdb_path, fc_name, geom, has_z=has_z, spatial_reference=sr)
        for fname, ftype, length in fields:
            kw = {'field_length': length} if length else {}
            arcpy.management.AddField(fc, fname, ftype, **kw)
    else:
        # 既存 FC に不足フィールドがあれば追加（スキーマ移行対応）
        existing = {f.name.upper() for f in arcpy.ListFields(fc)}
        for fname, ftype, length in fields:
            if fname.upper() not in existing:
                kw = {'field_length': length} if length else {}
                arcpy.management.AddField(fc, fname, ftype, **kw)
    return fc


def _save_gdb_arcpy(hist: dict, waypoints_msl: np.ndarray, gdb_path: str) -> None:
    import arcpy
    from datetime import datetime

    sr      = arcpy.SpatialReference(4326)
    t_end   = float(hist['time'][-1])
    pos_arr = hist['pos']
    run_id  = datetime.now().strftime(_RUN_ID_FMT)

    folder   = os.path.dirname(os.path.abspath(gdb_path)) or '.'
    gdb_name = os.path.basename(gdb_path)
    os.makedirs(folder, exist_ok=True)
    if not arcpy.Exists(gdb_path):
        arcpy.management.CreateFileGDB(folder, gdb_name)

    fixed_obs  = get_fixed_obstacles()
    moving_obs = get_moving_obstacles()

    # ── FLIGHT_TRAJECTORY ────────────────────────────────────────────────────
    fc = _ensure_fc(arcpy, gdb_path, 'FLIGHT_TRAJECTORY', sr)
    total_dist = float(np.sum(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)))
    line = arcpy.Polyline(
        arcpy.Array([arcpy.Point(*_lonlatz(p)) for p in pos_arr]), sr, True)
    with arcpy.da.InsertCursor(
            fc, ['SHAPE@', 'RUN_ID', 'NAME', 'FLIGHT_TIME_S',
                 'TOTAL_DIST_M', 'MAX_SPEED_MS', 'HIT_GROUND']) as cur:
        cur.insertRow([line, run_id, '飛翔軌道', t_end,
                       round(total_dist, 1),
                       round(float(hist['speed'].max()), 1),
                       int(hist['hit_ground'])])

    # ── WAYPOINTS ────────────────────────────────────────────────────────────
    fc = _ensure_fc(arcpy, gdb_path, 'WAYPOINTS', sr)
    n_wp     = len(waypoints_msl)
    wp_names = ['出発点'] + [f'経由点{i}' for i in range(1, n_wp - 1)] + ['目的地']
    with arcpy.da.InsertCursor(fc, ['SHAPE@', 'RUN_ID', 'NAME', 'ALT_MSL_M']) as cur:
        for name, wp in zip(wp_names, waypoints_msl):
            lon, lat, alt = _lonlatz(wp)
            cur.insertRow([arcpy.PointGeometry(arcpy.Point(lon, lat, alt), sr, True),
                           run_id, name, round(alt, 1)])

    # ── FIXED_OBSTACLES ──────────────────────────────────────────────────────
    fc = _ensure_fc(arcpy, gdb_path, 'FIXED_OBSTACLES', sr)
    with arcpy.da.InsertCursor(
            fc, ['SHAPE@', 'RUN_ID', 'NAME', 'OBS_TYPE',
                 'RADIUS_M', 'HEIGHT_MSL_M', 'ZONE_M']) as cur:
        for obs in fixed_obs:
            if isinstance(obs, SphereObstacle):
                ring = _circle_ring(obs.pos, obs.radius)
                poly = arcpy.Polygon(
                    arcpy.Array([arcpy.Point(lon, lat) for lon, lat in ring]), sr)
                cur.insertRow([poly, run_id, obs.label, 'sphere',
                               obs.radius, None, round(obs.zone, 1)])
            elif isinstance(obs, BoxObstacle):
                ring = _box_ring(obs.pos, obs.half_extents)
                poly = arcpy.Polygon(
                    arcpy.Array([arcpy.Point(lon, lat) for lon, lat in ring]), sr)
                cur.insertRow([poly, run_id, obs.label, 'box',
                               None, float(obs.pos[2] + obs.half_extents[2]),
                               round(obs.zone, 1)])
            elif isinstance(obs, AreaObstacle):
                ring = _area_ring(obs.vertices_enu)
                poly = arcpy.Polygon(
                    arcpy.Array([arcpy.Point(lon, lat) for lon, lat in ring]), sr)
                cur.insertRow([poly, run_id, obs.label, 'area',
                               None, obs.height_msl, obs.zone_m])

    # ── MOVING_OBS_TRACKS ────────────────────────────────────────────────────
    fc = _ensure_fc(arcpy, gdb_path, 'MOVING_OBS_TRACKS', sr)
    with arcpy.da.InsertCursor(
            fc, ['SHAPE@', 'RUN_ID', 'NAME', 'RADIUS_M', 'SPEED_MS']) as cur:
        for obs in moving_obs:
            n_steps = max(2, int(t_end / 5))
            times   = np.linspace(0, t_end, n_steps)
            line    = arcpy.Polyline(
                arcpy.Array([arcpy.Point(*_lonlatz(obs.pos_at(t))) for t in times]),
                sr, True)
            cur.insertRow([line, run_id, obs.label,
                           obs.radius,
                           round(float(np.linalg.norm(obs.vel)), 1)])

    logger.info("追記: %s  RUN_ID=%s", gdb_path, run_id)
    print(f"追記: {gdb_path}  RUN_ID={run_id}")


# ── 公開 API ─────────────────────────────────────────────────────────────────

def save_gdb(hist: dict, waypoints_msl: np.ndarray,
             path: str = 'result/flight_route.gdb') -> None:
    """シミュレーション結果を ArcGIS File GDB で保存する。

    arcpy が使えない場合・書き出しに失敗した場合は GeoJSON にフォールバックする。
    """
    try:
        import arcpy  # noqa: F401
        _save_gdb_arcpy(hist, waypoints_msl, path)
    except ImportError:
        _fallback(hist, waypoints_msl, path, reason='arcpy が見つかりません')
    except Exception as exc:
        _fallback(hist, waypoints_msl, path, reason=str(exc))


def _fallback(hist, waypoints_msl, gdb_path, reason):
    from geojson_export import save_geojson
    from config import GEOJSON_PATH
    logger.warning("GDB 書き出しをスキップ → GeoJSON にフォールバック (%s)", reason)
    save_geojson(hist, waypoints_msl, GEOJSON_PATH)
