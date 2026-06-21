"""
GeoJSON エクスポートモジュール  (EPSG:4326 / WGS84)

出力フィーチャー:
  - 飛翔軌道     : LineString  [lon, lat, alt_msl]
  - ウェイポイント: Point
  - 固定障害物
      SphereObstacle → Polygon (64点近似円)
      BoxObstacle    → Polygon (フットプリント矩形) + height プロパティ
  - 移動体       : Point (初期位置) + LineString (飛翔中の軌跡)
"""

import json
import numpy as np

from config import EARTH_R
from geo import local_to_geo
from obstacles import get_fixed_obstacles, get_moving_obstacles, SphereObstacle, BoxObstacle


def _local_to_geojson_coord(pos: np.ndarray) -> list:
    """ローカル ENU → GeoJSON 座標 [lon, lat, alt]"""
    lat, lon = local_to_geo(pos)
    return [round(lon, 8), round(lat, 8), round(float(pos[2]), 2)]


def _circle_polygon(center_local: np.ndarray, radius_m: float, n: int = 64) -> list:
    """中心（ローカル ENU）と半径から GeoJSON Polygon の座標リストを生成する。"""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = []
    for t in theta:
        offset = center_local.copy()
        offset[0] += radius_m * np.cos(t)
        offset[1] += radius_m * np.sin(t)
        ring.append(_local_to_geojson_coord(offset)[:2])  # 平面投影なので alt 省略
    ring.append(ring[0])  # 閉じる
    return ring


def _box_polygon(center_local: np.ndarray, half_extents: np.ndarray) -> list:
    """AABB の XY フットプリントを GeoJSON Polygon 座標リストで返す。"""
    hx, hy = half_extents[0], half_extents[1]
    offsets = [(-hx, -hy), (+hx, -hy), (+hx, +hy), (-hx, +hy)]
    ring = []
    for dx, dy in offsets:
        pt = center_local.copy()
        pt[0] += dx
        pt[1] += dy
        ring.append(_local_to_geojson_coord(pt)[:2])
    ring.append(ring[0])
    return ring


def save_geojson(hist: dict, waypoints_msl: np.ndarray,
                 path: str = 'flight_route.geojson') -> None:
    """シミュレーション結果を GeoJSON (EPSG:4326) で保存する。

    Args:
        hist          : simulate() の返り値 dict
        waypoints_msl : MSL 高度に変換済みのウェイポイント配列
        path          : 出力ファイルパス
    """
    features: list[dict] = []
    fixed_obs  = get_fixed_obstacles()
    moving_obs = get_moving_obstacles()

    # ── 飛翔軌道 (LineString) ─────────────────────────────────────────────────
    traj_coords = [_local_to_geojson_coord(p) for p in hist['pos']]
    total_dist  = float(np.sum(np.linalg.norm(np.diff(hist['pos'], axis=0), axis=1)))
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": traj_coords,
        },
        "properties": {
            "name":             "飛翔軌道",
            "flight_time_s":    float(hist['time'][-1]),
            "total_distance_m": round(total_dist, 1),
            "max_speed_ms":     round(float(hist['speed'].max()), 1),
            "hit_ground":       bool(hist['hit_ground']),
        },
    })

    # ── ウェイポイント (Point) ────────────────────────────────────────────────
    n_wp = len(waypoints_msl)
    wp_names = (["出発点"]
                + [f"経由点{i}" for i in range(1, n_wp - 1)]
                + ["目的地"])
    for name, wp in zip(wp_names, waypoints_msl):
        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": _local_to_geojson_coord(wp),
            },
            "properties": {
                "name":           name,
                "altitude_msl_m": round(float(wp[2]), 1),
            },
        })

    # ── 固定障害物 ────────────────────────────────────────────────────────────
    for obs in fixed_obs:
        if isinstance(obs, SphereObstacle):
            # 物理半径の円ポリゴン
            features.append({
                "type": "Feature",
                "geometry": {
                    "type":        "Polygon",
                    "coordinates": [_circle_polygon(obs.pos, obs.radius)],
                },
                "properties": {
                    "name":      obs.label,
                    "type":      "sphere_obstacle",
                    "radius_m":  obs.radius,
                    "zone_m":    round(obs.zone, 1),
                },
            })
            # 回避ゾーン円ポリゴン
            features.append({
                "type": "Feature",
                "geometry": {
                    "type":        "Polygon",
                    "coordinates": [_circle_polygon(obs.pos, obs.zone)],
                },
                "properties": {
                    "name": f"{obs.label}（回避ゾーン）",
                    "type": "avoidance_zone",
                },
            })

        elif isinstance(obs, BoxObstacle):
            # フットプリント矩形ポリゴン
            features.append({
                "type": "Feature",
                "geometry": {
                    "type":        "Polygon",
                    "coordinates": [_box_polygon(obs.pos, obs.half_extents)],
                },
                "properties": {
                    "name":      obs.label,
                    "type":      "box_obstacle",
                    "width_m":   float(obs.half_extents[0] * 2),
                    "depth_m":   float(obs.half_extents[1] * 2),
                    "height_m":  float(obs.half_extents[2] * 2),
                    "zone_m":    round(obs.zone, 1),
                },
            })

    # ── 移動体 ────────────────────────────────────────────────────────────────
    t_end = float(hist['time'][-1])
    for obs in moving_obs:
        # 初期位置
        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": _local_to_geojson_coord(obs.pos_init),
            },
            "properties": {
                "name":      obs.label,
                "type":      "moving_obstacle",
                "radius_m":  obs.radius,
                "speed_ms":  round(float(np.linalg.norm(obs.vel)), 1),
            },
        })
        # 飛翔中の軌跡 LineString
        n_steps = max(2, int(t_end / 5))
        times   = np.linspace(0, t_end, n_steps)
        traj    = [_local_to_geojson_coord(obs.pos_at(t)) for t in times]
        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "LineString",
                "coordinates": traj,
            },
            "properties": {
                "name": f"{obs.label}（軌跡）",
                "type": "moving_obstacle_track",
            },
        })

    # ── GeoJSON 出力 ─────────────────────────────────────────────────────────
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"保存: {path}  ({len(features)} フィーチャー)")
