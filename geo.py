import numpy as np
from config import GEO_WAYPOINTS, EARTH_R


def geo_to_local(geo_wps: list) -> np.ndarray:
    """緯度経度リスト → ローカル ENU 直交座標 [m] に変換。
    先頭点が原点 (x=0, y=0)。x=東, y=北, z=高度(MSL)。
    callable エントリ（移動WP）はスキップする。"""
    lat0 = np.radians(geo_wps[0][0])
    lon0 = np.radians(geo_wps[0][1])
    pts = []
    for entry in geo_wps:
        if callable(entry):
            continue
        lat_deg, lon_deg, alt = entry
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        pts.append([
            (lon - lon0) * EARTH_R * np.cos(lat0),
            (lat - lat0) * EARTH_R,
            alt,
        ])
    return np.array(pts, dtype=float)


def geo_to_local_pt(lat_deg: float, lon_deg: float, alt: float = 0.0) -> np.ndarray:
    """単一の緯度経度を ENU ローカル座標に変換。原点は GEO_WAYPOINTS[0]。"""
    lat0 = np.radians(GEO_WAYPOINTS[0][0])
    lon0 = np.radians(GEO_WAYPOINTS[0][1])
    return np.array([
        (np.radians(lon_deg) - lon0) * EARTH_R * np.cos(lat0),
        (np.radians(lat_deg) - lat0) * EARTH_R,
        float(alt),
    ])


def local_to_geo(pos_m: np.ndarray) -> tuple[float, float]:
    """ローカル ENU 座標 [m] → (緯度°, 経度°) に逆変換。"""
    lat0 = np.radians(GEO_WAYPOINTS[0][0])
    lon0 = np.radians(GEO_WAYPOINTS[0][1])
    lat = lat0 + pos_m[1] / EARTH_R
    lon = lon0 + pos_m[0] / (EARTH_R * np.cos(lat0))
    return float(np.degrees(lat)), float(np.degrees(lon))


WAYPOINTS = geo_to_local(GEO_WAYPOINTS)
