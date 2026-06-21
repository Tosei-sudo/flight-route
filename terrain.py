import glob
import logging
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

from config import DEM_WIDE_DIR, DEM_DETAIL_DIR, TERRAIN_CLEARANCE, TERRAIN_LOOKAHEAD
from geo import local_to_geo


class _TileIndex:
    """フォルダ内の DEM タイル群を管理し lat/lon → 高度クエリを提供する。

    EPSG:4326 (Copernicus dt2) と EPSG:6668 (JGD2011 tif) のどちらも、
    座標軸が lon=x / lat=y の地理座標系なので変換なしに直接クエリできる。
    WGS84 と JGD2011 の差は最大 ~1m のため 5m DEM スケールでは無視できる。
    """

    def __init__(self, folder: str, glob_pattern: str):
        self._tiles: list[rasterio.DatasetReader] = []
        for p in sorted(glob.glob(f"{folder}/{glob_pattern}")):
            try:
                self._tiles.append(rasterio.open(p))
            except Exception as e:
                logger.warning("DEM スキップ: %s  (%s)", p, e)

    @property
    def count(self) -> int:
        return len(self._tiles)

    def height_at(self, lat: float, lon: float) -> float | None:
        """lat/lon の高度 [m] を返す。タイル未検出なら None。"""
        for ds in self._tiles:
            b = ds.bounds
            if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                try:
                    val = float(next(ds.sample([(lon, lat)]))[0])
                    nd  = ds.nodata
                    if nd is None or not np.isclose(val, nd, atol=1.0):
                        return max(val, 0.0)
                except Exception:
                    pass
        return None


# ── モジュールレベルの初期化（初回クエリ時に実行）────────────────────────────
_wide:   _TileIndex | None = None
_detail: _TileIndex | None = None


def _ensure_init() -> None:
    global _wide, _detail
    if _wide is not None:
        return
    _wide   = _TileIndex(DEM_WIDE_DIR,   '*')
    _detail = _TileIndex(DEM_DETAIL_DIR, '*.tif')
    w, d    = _wide.count, _detail.count
    if w == 0 and d == 0:
        logger.warning("DEM タイルが見つかりません。高度 = 0 m で演算します。")
    else:
        logger.info("DEM 読み込み: WIDE %d タイル (30m) / DETAIL %d タイル (5m)", w, d)


def terrain_height_at(lat: float, lon: float) -> float:
    """指定緯度経度の地形標高 [m MSL]。

    DETAIL (5m) を優先。カバー外なら WIDE (30m)、両方なければ 0.0。
    """
    _ensure_init()
    h = _detail.height_at(lat, lon)
    if h is not None:
        return h
    h = _wide.height_at(lat, lon)
    return h if h is not None else 0.0


def terrain_floor(pos: np.ndarray, vel: np.ndarray, speed: float) -> float:
    """現在位置〜先読み範囲の最大地形高さ + クリアランスを返す。

    速度方向の直線上を 20 点サンプリングしつつ、
    各点で左右にも横幅 spread_m をチェックすることで、
    旋回中の死角や狭い稜線ピークを検知する。
    """
    lat, lon = local_to_geo(pos)
    floor = terrain_height_at(lat, lon) + TERRAIN_CLEARANCE

    if speed <= 1.0:
        return floor

    unit   = vel / speed
    perp_h = np.array([-unit[1], unit[0], 0.0])
    pn     = float(np.linalg.norm(perp_h))
    perp   = perp_h / pn if pn > 1e-9 else np.array([0.0, 1.0, 0.0])

    for t_ahead in np.linspace(0, TERRAIN_LOOKAHEAD, 20)[1:]:
        center   = pos + unit * speed * t_ahead
        spread_m = speed * t_ahead * 0.1
        for lateral in (-spread_m, 0.0, spread_m):
            sp = center + perp * lateral
            la_lat, la_lon = local_to_geo(sp)
            floor = max(floor, terrain_height_at(la_lat, la_lon) + TERRAIN_CLEARANCE)
    return floor
