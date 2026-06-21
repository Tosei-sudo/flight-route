"""
風場モジュール

MD.gdb 内の WIND_{ALT}M ラスターデータセット（3バンド, WGS84）から
高度補間付きの 3D 風ベクトルを提供する。

バンド定義:
  Band 1: U [m/s]  東西成分（正 = 東向き）
  Band 2: V [m/s]  南北成分（正 = 北向き）
  Band 3: W [m/s]  鉛直成分（正 = 上昇気流）

ラスターが存在しない場合はゼロ風にフォールバックする。

テスト用ラスター作成:
  python wind.py  または  from wind import create_test_wind_rasters
"""

import os
import re
import shutil
import logging
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

_ZERO3        = np.zeros(3, dtype=np.float64)
_WIND_PATTERN = re.compile(r'^WIND_(\d+)M$', re.IGNORECASE)


# ── 双線形補間ヘルパー ────────────────────────────────────────────────────────

def _bilinear(data: np.ndarray, row_f: float, col_f: float) -> float:
    nrows, ncols = data.shape
    r0 = int(np.clip(int(row_f), 0, nrows - 2))
    c0 = int(np.clip(int(col_f), 0, ncols - 2))
    dr = row_f - r0
    dc = col_f - c0
    return float(
        data[r0,     c0    ] * (1 - dr) * (1 - dc) +
        data[r0,     c0 + 1] * (1 - dr) *      dc  +
        data[r0 + 1, c0    ] *      dr  * (1 - dc) +
        data[r0 + 1, c0 + 1] *      dr  *      dc
    )


# ── WindField クラス ─────────────────────────────────────────────────────────

class WindField:
    """高度別 2D ラスター群から 3D 風ベクトルを提供するクラス。

    GDB 内の WIND_0M / WIND_1000M / WIND_3000M … を起動時に一括ロードし、
    wind_enu(lat, lon, alt_msl) で高度線形補間 + XY 双線形補間した
    ENU 風速ベクトル [m/s] を返す。
    """

    def __init__(self, gdb_path: str):
        # {alt_msl: {'uvw': ndarray(3, rows, cols), 'xmin', 'ymax', 'cw', 'ch'}}
        self._layers: dict[int, dict] = {}
        self._load(gdb_path)

    # ── 読み込み ─────────────────────────────────────────────────────────────

    def _load(self, gdb_path: str) -> None:
        try:
            import arcpy  # noqa: PLC0415
        except ImportError:
            logger.warning("WindField: arcpy なし → ゼロ風にフォールバック")
            return

        try:
            arcpy.env.workspace = gdb_path
            names = arcpy.ListRasters('WIND_*M') or []
        except Exception as exc:
            logger.warning("WindField: ラスター一覧取得失敗 %s", exc)
            return

        for name in sorted(names):
            m = _WIND_PATTERN.match(name)
            if not m:
                continue
            alt  = int(m.group(1))
            path = f'{gdb_path}\\{name}'
            try:
                robj = arcpy.Raster(path)
                xmin = robj.extent.XMin
                ymax = robj.extent.YMax
                cw   = robj.meanCellWidth
                ch   = robj.meanCellHeight

                bands = []
                for b in range(1, 4):
                    arr = arcpy.RasterToNumPyArray(
                        f'{path}\\Band_{b}', nodata_to_value=0.0)
                    bands.append(arr.astype(np.float32))
                uvw = np.stack(bands, axis=0)  # (3, rows, cols)

                self._layers[alt] = {
                    'uvw': uvw, 'xmin': xmin, 'ymax': ymax, 'cw': cw, 'ch': ch,
                }
                logger.info("WindField: WIND_%dM (%d×%d px) 読み込み完了",
                            alt, uvw.shape[2], uvw.shape[1])
            except Exception as exc:
                logger.warning("WindField: %s 読み込み失敗 %s", name, exc)

        if self._layers:
            logger.info("WindField: %d 高度レイヤー %s m",
                        len(self._layers), sorted(self._layers))
        else:
            logger.info("WindField: WIND_*M ラスターなし → ゼロ風")

    # ── クエリ ───────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self._layers)

    def wind_enu(self, lat: float, lon: float, alt_msl: float) -> np.ndarray:
        """指定座標・高度の ENU 風速ベクトル [U, V, W] m/s を返す。"""
        if not self._layers:
            return _ZERO3.copy()

        alts = sorted(self._layers)

        # 高度ブラケット選択
        if alt_msl <= alts[0]:
            lo, hi, t = alts[0], alts[0], 0.0
        elif alt_msl >= alts[-1]:
            lo, hi, t = alts[-1], alts[-1], 0.0
        else:
            for i in range(len(alts) - 1):
                if alts[i] <= alt_msl <= alts[i + 1]:
                    lo, hi = alts[i], alts[i + 1]
                    span = hi - lo
                    t = (alt_msl - lo) / span if span > 0 else 0.0
                    break

        def _sample(layer_alt: int) -> np.ndarray:
            lyr   = self._layers[layer_alt]
            col_f = (lon - lyr['xmin']) / lyr['cw']
            row_f = (lyr['ymax'] - lat) / lyr['ch']
            uvw   = lyr['uvw']
            return np.array([_bilinear(uvw[b], row_f, col_f) for b in range(3)])

        w0 = _sample(lo)
        w1 = _sample(hi) if lo != hi else w0
        return w0 * (1.0 - t) + w1 * t


# ── シングルトン ──────────────────────────────────────────────────────────────

_instance: WindField | None = None


def get_wind_field(gdb_path: str) -> WindField:
    """WindField のシングルトンを返す（初回のみロード）。"""
    global _instance
    if _instance is None:
        _instance = WindField(gdb_path)
    return _instance


# ── テスト用ラスター作成 ──────────────────────────────────────────────────────

def create_test_wind_rasters(
        gdb_path: str,
        altitudes: list[tuple[int, float, float, float]] | None = None,
) -> None:
    """テスト用の一様風ラスターを GDB に作成する（既存は上書き）。

    Args:
        gdb_path  : 出力先 GDB パス
        altitudes : [(高度m, U m/s, V m/s, W m/s), ...] のリスト
                    省略時は 0m と 3000m の 2 層を作成
    """
    import arcpy  # noqa: PLC0415

    if altitudes is None:
        altitudes = [
            (0,    10.0,  5.0, 0.0),   # 地表: 東10 + 北5 m/s
            (3000, 20.0,  5.0, 0.5),   # 3000m: 東20 + 弱い上昇気流
        ]

    sr       = arcpy.SpatialReference(4326)
    xmin, ymin, xmax, ymax = 128.0, 30.0, 148.0, 47.0
    cell_deg = 0.1
    ncols    = int(round((xmax - xmin) / cell_deg))
    nrows    = int(round((ymax - ymin) / cell_deg))

    tmp = tempfile.mkdtemp()
    try:
        for alt, u_val, v_val, w_val in altitudes:
            raster_name = f'WIND_{alt}M'
            raster_path = f'{gdb_path}\\{raster_name}'

            if arcpy.Exists(raster_path):
                arcpy.management.Delete(raster_path)

            lower_left = arcpy.Point(xmin, ymin)
            band_paths = []
            for bname, val in [('u', u_val), ('v', v_val), ('w', w_val)]:
                arr = np.full((nrows, ncols), val, dtype=np.float32)
                r   = arcpy.NumPyArrayToRaster(arr, lower_left, cell_deg, cell_deg)
                bp  = os.path.join(tmp, f'{bname}_{alt}.tif')
                r.save(bp)
                arcpy.management.DefineProjection(bp, sr)
                band_paths.append(bp)

            arcpy.management.CompositeBands(band_paths, raster_path)
            logger.info("テストラスター作成: %s  U=%.1f V=%.1f W=%.1f m/s",
                        raster_name, u_val, v_val, w_val)
            print(f"作成: {raster_name}  U={u_val} V={v_val} W={w_val} m/s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── 単体実行: テストラスター作成 ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    gdb = r'C:\Users\tosei\work\flight-route\MD.gdb'
    if len(sys.argv) > 1:
        gdb = sys.argv[1]
    create_test_wind_rasters(gdb)
    print("完了")
