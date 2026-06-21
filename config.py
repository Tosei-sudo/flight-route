# ── 表示オプション ────────────────────────────────────────────────────────────
PLOT_3D_ONLY = False   # True: 3D軌道のみ表示 / False: 全グラフ表示
OUTPUT_DIR    = 'result'            # 出力先フォルダ（存在しなければ自動作成）
SAVE_CSV      = True                # True: テレメトリーCSVを保存
CSV_PATH      = 'result/telemetry.csv'
SAVE_GDB      = True                # True: ArcGIS File GDB を保存（arcpy不可時はGeoJSONにフォールバック）
GDB_OUTPUT_PATH = r'C:\Users\tosei\work\flight-route\MD.gdb'
GEOJSON_PATH  = 'result/flight_route.geojson'  # GDB不可時のフォールバックパス
PLOT_PATH     = 'result/flight_route.png'
LOG_PATH      = 'result/simulation.log'

# ── 出発時の姿勢・初速 ───────────────────────────────────────────────────────
# 方角: +X軸から反時計回り (0°=東, 90°=北)
# 仰角: 水平面からの角度   (0°=水平, 90°=真上)
INIT_AZIMUTH_DEG   =  90.0   # 方角 (度)
INIT_ELEVATION_DEG = 45.0   # 仰角 (度)
INIT_SPEED         = 10.0   # m/s  初速

# ── 物理定数 ─────────────────────────────────────────────────────────────────
MAX_ACCEL = 80    # m/s²  最大加速度
MAX_SPEED = 272.0  # m/s   最高速度
G         = 9.81   # m/s²  重力加速度
DT        = 0.5    # s     シミュレーション時間刻み

# 空気抵抗係数  a_drag = DRAG_K * v²  (速度の逆方向)
# 参考: 275 m/s で 3.8 m/s²、端速度 ≈ 322 m/s
DRAG_K = 5e-5   # m⁻¹

# ── 飛行プロファイル ──────────────────────────────────────────────────────────
# 'standard': 直線巡航しつつ地形回避（WP高度 or 地形クリアランスの高い方を維持）
# 'low'     : 地形密着低空飛行（谷を探して遠回りしてでも低い経路を選択）
# 'auto'    : 最終WPが移動目標(callable)→ standard、固定目標(tuple)→ low を自動選択
FLIGHT_PROFILE = 'low'

# low プロファイル専用: 谷探索パラメータ
LOW_VALLEY_FAN_DEG = 60.0   # 谷スキャンの角度幅 ±度（大きいほど大回りを許容）
LOW_VALLEY_RAYS    = 61     # スキャン本数（奇数推奨、正面を含む）
LOW_VALLEY_COST    = 10.0  # 迂回コスト [m]（小さいほど積極的に遠回り）

# ── 地形追従 ─────────────────────────────────────────────────────────────────
DEM_WIDE_DIR   = 'DEM/WIDE'    # 広域DEM（30m）フォルダ。*.dt2 / *.tif を自動検索
DEM_DETAIL_DIR = 'DEM/DETAIL'  # 詳細DEM（5m）フォルダ。WIDE より優先して使用
TERRAIN_CLEARANCE   = 75.0   # m  地形上面からの最低クリアランス
TERRAIN_LOOKAHEAD   = 15.0   # s  先読み時間
TERRAIN_TIME_CONST  =  3.0   # s  地形追従の垂直制御時定数（小さいほど急峻に反応）

# ── 離陸フェーズ ─────────────────────────────────────────────────────────────
# 地形クリアランス高度に達するまで垂直優先で上昇する離陸フェーズ。
# LAUNCH_CLIMB_RATIO: 上昇速度 / 全速度 (0〜1)。大きいほど急上昇。
LAUNCH_CLIMB_RATIO = 0.8   # 離陸中の上昇速度割合

# ── 終末誘導 ─────────────────────────────────────────────────────────────────
# 'auto'     : callable 最終WP → ARH（比例航法）、tuple 最終WP → popup_dive を自動選択
# 'popup_dive': 常にポップアップ→ダイブ（固定目標向け）
# 'arh'      : 常にアクティブレーダーホーミング・比例航法（移動目標向け）
TERMINAL_GUIDANCE = 'auto'

from dataclasses import dataclass

@dataclass
class PopupDiveParams:
    """ポップアップ→ダイブ型終末誘導のパラメータ。"""
    guidance_time: float | None = None  # s  None=自動算出
    popup_height: float         = 500.0  # m  ポップアップ目標高度（目標MSL +）
    dive_angle_deg: float       = -20.0  # 度  ダイブ突入角（負=降下）

@dataclass
class ARHParams:
    """アクティブレーダーホーミング（比例航法）のパラメータ。"""
    nav_constant: float  = 4.0   # PN ゲイン N（推奨 3〜5）
    engage_time: float   = 30.0  # s  終末誘導移行の残り時間閾値

POPUP_DIVE = PopupDiveParams()
ARH        = ARHParams()

# ── ナビゲーション（位置取得）────────────────────────────────────────────────
NAV_MODE = 'fused'          # 'gps': 理想GPS / 'ins': 慣性航法 / 'fused': GPS/INS融合
INS_ACCEL_BIAS   = (0.05, 0.05, 0.02)  # m/s²  INS加速度バイアス (x, y, z)
INS_GPS_INTERVAL = 60                   # steps  fused時のGPS補正間隔（60×0.5s = 30s）

GPS_SAT_RANGE        = (4, 12)  # (最小, 最大) 衛星数範囲（4未満はフィックス不能）
GPS_BASE_ACCURACY    = 3.0      # m   基準測位精度（水平, 8衛星時）
GPS_SAT_CHANGE_STEPS = 20       # steps  衛星数ランダムウォークの更新間隔

# ── 障害物回避 ───────────────────────────────────────────────────────────────
AVOIDANCE_MARGIN    = 3.0   # 回避ゾーン = 障害物半径 × MARGIN
COLLISION_LOOKAHEAD = 8.0   # s  移動体の衝突予測先読み時間

# ── 地理座標ウェイポイント [(緯度°, 経度°, 高度m), ...] ──────────────────────
import numpy as np
def _moving_ship(t):
    # 艦船が東に100m/s で移動
    lat0, lon0 = 35.987488, 135.006355
    lon = lon0 + (t * 100.0) / (6371000 * np.cos(np.radians(lat0))) * (180 / np.pi)
    return (lat0, lon, 0)

GEO_WAYPOINTS = [
    (45.39521, 141.76507, 0),  # 出発点（固定）
    (41.944442, 143.231490, 0),  # 出発点（固定）
    # _moving_ship,               # 目的地（移動）
]

CAPTURE_R      = 50.0   # m   ウェイポイント到達判定半径（水平距離、巡航フェーズ）
HIT_RADIUS     = 10.0   # m   終末ダイブ時の命中判定半径（3D距離）
EARTH_R   = 6371000.0   # m   地球半径（球体近似）