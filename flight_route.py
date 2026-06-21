"""
Flight Route Simulator  —  エントリポイント

物理モデル:
  MAX_ACCEL = 5.2 m/s²   最大加速度（揚力が重力を相殺した後のマニューバ加速度）
  MAX_SPEED = 275 m/s    最高速度
  旋回半径  = v² / MAX_ACCEL

設定変更は config.py を編集してください。

GEO_WAYPOINTS の各エントリは以下のどちらかで記述できます:
  tuple  (lat, lon, alt_agl)        — 固定ウェイポイント
  callable(t) -> (lat, lon, alt_agl) — 移動ウェイポイント（t は秒）
"""

import logging
import os
import sys
import argparse
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import (OUTPUT_DIR, SAVE_CSV, CSV_PATH,
                    SAVE_GEOJSON, GEOJSON_PATH, PLOT_PATH, LOG_PATH,
                    MAX_ACCEL, MAX_SPEED, G)

logger = logging.getLogger(__name__)


def _setup_logging(log_path: str) -> None:
    """ファイル(DEBUG全量) + コンソール(INFO のみ) の二重ハンドラを設定する。"""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt_file    = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s',
        datefmt='%H:%M:%S')
    fmt_console = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt_file)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    root.addHandler(fh)
    root.addHandler(ch)
from geo import GEO_WAYPOINTS, local_to_geo, geo_to_local_pt
from terrain import terrain_height_at
from simulator import simulate, turning_radius
from exporter import save_csv
from geojson_export import save_geojson
from plotter import plot


def _make_wp_resolvers() -> list:
    """GEO_WAYPOINTS エントリ (tuple | callable) を MSL ENU リゾルバのリストに変換する。

    - tuple (lat, lon, alt_agl)        → np.ndarray  固定 ENU MSL座標
    - callable(t) -> (lat, lon, alt_agl) → Callable[[float], np.ndarray]  動的解決
    """
    resolvers = []
    for entry in GEO_WAYPOINTS:
        if callable(entry):
            def _wrap(f):
                def _resolve(t: float) -> np.ndarray:
                    lat, lon, alt_agl = f(t)
                    pos = geo_to_local_pt(lat, lon, alt_agl)
                    pos[2] += terrain_height_at(lat, lon)
                    return pos
                return _resolve
            resolvers.append(_wrap(entry))
        else:
            lat_deg, lon_deg, alt_agl = entry
            pos = geo_to_local_pt(lat_deg, lon_deg, alt_agl)
            pos[2] += terrain_height_at(lat_deg, lon_deg)
            resolvers.append(pos)
    return resolvers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plot', action='store_true', help='プロット画面を表示しない（PNG保存は継続）')
    parser.add_argument('--profile', default=None, choices=['standard', 'low', 'auto'],
                        help='飛行プロファイル: standard=直線巡航 / low=地形密着低空飛行 / auto=目標種別で自動選択')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _setup_logging(LOG_PATH)
    logger.info("ログ出力: %s", os.path.abspath(LOG_PATH))

    logger.info("=== 飛翔ルートシミュレーション ===")
    logger.info("  最大加速度  : %s m/s^2", MAX_ACCEL)
    logger.info("  最高速度    : %s m/s", MAX_SPEED)
    logger.info("  重力加速度  : %s m/s^2", G)
    _tr = turning_radius(MAX_SPEED)
    logger.info("  最小旋回半径: %s m (%.1f km)", f"{_tr:,.0f}", _tr / 1000)
    logger.info("")

    # WPリゾルバ構築（固定 or 移動、AGL→MSL変換済み）
    WP_RESOLVERS = _make_wp_resolvers()

    logger.info("ウェイポイント (高度=地上高 AGL):")
    wp_labels = (['出発点']
                 + [f'経由点{i}' for i in range(1, len(GEO_WAYPOINTS) - 1)]
                 + ['目的地'])
    for lbl, entry in zip(wp_labels, GEO_WAYPOINTS):
        if callable(entry):
            lat, lon, alt_agl = entry(0.0)
            terrain_h = terrain_height_at(lat, lon)
            logger.info("  %s: (%.6f°, %.6f°)  ★移動目標 (t=0)  AGL %.0fm  地形 %.0fm  → MSL %.0fm",
                        lbl, lat, lon, alt_agl, terrain_h, alt_agl + terrain_h)
        else:
            lat, lon, alt_agl = entry
            terrain_h = terrain_height_at(lat, lon)
            logger.info("  %s: (%.6f°, %.6f°)  AGL %.0fm  地形 %.0fm  → MSL %.0fm",
                        lbl, lat, lon, alt_agl, terrain_h, alt_agl + terrain_h)
    logger.info("")
    logger.info("シミュレーション中...")
    try:
        hist = simulate(WP_RESOLVERS, profile=args.profile)
    except Exception:
        logger.exception("シミュレーション中に例外が発生しました")
        raise

    total_dist = float(np.sum(np.linalg.norm(np.diff(hist['pos'], axis=0), axis=1)))
    total_time = float(hist['time'][-1])
    avg_speed  = total_dist / total_time if total_time > 0 else 0.0

    status = "地面衝突" if hist['hit_ground'] else "正常終了"

    logger.info("")
    logger.info("─── 結果 ───────────────────────────────")
    logger.info("  終了状態    : %s", status)
    logger.info("  総飛翔距離  : %.2f km", total_dist / 1000)
    logger.info("  飛翔時間    : %.1f s (%.1f min)", total_time, total_time / 60)
    logger.info("  平均速度    : %.1f m/s", avg_speed)
    logger.info("  最高到達速度: %.1f m/s", hist['speed'].max())
    logger.info("─────────────────────────────────────────")

    # プロット・エクスポート用: callable WPを最終時刻で解決した ndarray
    t_end = total_time
    WP_MSL_FINAL = np.array([r(t_end) if callable(r) else r for r in WP_RESOLVERS])

    if SAVE_CSV:
        save_csv(hist, CSV_PATH)
        logger.info("保存: %s", CSV_PATH)

    if SAVE_GEOJSON:
        save_geojson(hist, WP_MSL_FINAL, GEOJSON_PATH)

    plot(hist, WP_MSL_FINAL, PLOT_PATH, show=not args.no_plot)
    logger.info("保存: %s", PLOT_PATH)
    logger.info("ログ: %s", os.path.abspath(LOG_PATH))
