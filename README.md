# Flight Route Simulator

出発点から複数の経由地点を経て目標地点まで飛翔するルートをシミュレーションするプログラムです。

## 機能

- 複数ウェイポイント（固定・移動目標）を経由する3D飛翔軌道の計算
- 地形データ（DEM）に基づく地形追従・地形回避
- 障害物回避（固定・移動）
- 飛行フェーズ管理（離陸 → 巡航 → 終末誘導）
- 終末誘導方式: ポップアップ→ダイブ / アクティブレーダーホーミング（比例航法）
- 燃料・バッテリー消費モデル（スラスト → 滑空 → 弾道の3段階）
- GPS/INS融合ナビゲーション
- 大気密度・風の影響を考慮した空気抵抗計算
- 複数インスタンスの並列実行・比較
- 結果の3Dプロット（matplotlib）、CSV保存、ArcGIS File GDB出力

## 物理モデル

| パラメータ | 値 |
|---|---|
| 最大加速度 | 設定可能（デフォルト: `config.py` 参照） |
| 最高速度 | 272 m/s（設定可能） |
| 旋回半径 | v² / MAX_ACCEL（速度に応じて変化） |
| 重力加速度 | 9.81 m/s² |
| 空気抵抗 | a_drag = DRAG_K × ρ比 × v²（速度の逆方向） |

## ファイル構成

```
flight_route.py   - エントリポイント
config.py         - 全設定パラメータ（ウェイポイント含む）
simulator.py      - シミュレーションコア（Simulator クラス）
geo.py            - 地理座標↔ローカルENU座標変換、ウェイポイント定義
terrain.py        - DEMファイル読み込み・地形高度取得
nav.py            - GPS/INS ナビゲーションセンサー
atmosphere.py     - 大気密度モデル
wind.py           - 風場モデル
obstacles.py      - 障害物定義・回避計算
plotter.py        - matplotlib による3Dプロット
exporter.py       - CSV出力
gdb_export.py     - ArcGIS File GDB出力
geojson_export.py - GeoJSON出力（GDB不可時のフォールバック）
```

## セットアップ

```bash
pip install numpy matplotlib
```

地形追従を使用する場合は `DEM/WIDE/`（広域 30m DEM）または `DEM/DETAIL/`（詳細 5m DEM）に `.dt2` / `.tif` ファイルを配置してください。

## 使い方

### 基本実行

```bash
python flight_route.py
```

### オプション

```bash
# プロット画面を表示せずPNGのみ保存
python flight_route.py --no-plot

# 飛行プロファイルを指定
python flight_route.py --profile standard   # 直線巡航
python flight_route.py --profile low        # 地形密着低空飛行
python flight_route.py --profile auto       # 目標種別で自動選択

# 2インスタンスを並列実行して結果を比較
python flight_route.py --compare
```

## 設定

`config.py` を編集して動作をカスタマイズできます。

### ウェイポイント

```python
# 固定ウェイポイント: (緯度°, 経度°, 地上高m)
GEO_WAYPOINTS = [
    (45.395, 141.765, 0),   # 出発点
    (41.944, 143.231, 0),   # 目的地
]

# 移動目標（callable）も指定可能
def _moving_ship(t):
    lat0, lon0 = 41.944, 143.231
    lon = lon0 + (t * 50.0) / (6371000 * np.cos(np.radians(lat0))) * (180 / np.pi)
    return (lat0, lon, 0)

GEO_WAYPOINTS = [
    (45.395, 141.765, 0),
    _moving_ship,   # 移動目標（ARH終末誘導が自動選択される）
]
```

### 主要パラメータ

| 設定項目 | 変数名 | 説明 |
|---|---|---|
| 飛行プロファイル | `FLIGHT_PROFILE` | `'standard'` / `'low'` / `'auto'` |
| 終末誘導 | `TERMINAL_GUIDANCE` | `'popup_dive'` / `'arh'` / `'auto'` |
| 地形クリアランス | `TERRAIN_CLEARANCE` | 地形上面からの最低高度 [m] |
| ナビゲーション | `NAV_MODE` | `'gps'` / `'ins'` / `'fused'` |
| 燃料容量 | `FUEL_CAPACITY` | [kg]（任意単位） |

## 出力

実行後、`result/` フォルダに以下のファイルが生成されます。

| ファイル | 内容 |
|---|---|
| `flight_route.png` | 3D軌道・速度・加速度・高度プロット |
| `telemetry.csv` | 時系列テレメトリーデータ |
| `simulation.log` | 実行ログ |

## プログラムからの利用

```python
from geo import GEO_WAYPOINTS
from simulator import Simulator

# シングル実行
sim = Simulator()
result = sim.run(waypoints)

# パラメータ変更
sim = Simulator(max_speed=250.0, max_accel=60.0)
result = sim.run(waypoints)

# 並列実行
results = Simulator.run_parallel([
    (waypoints_a,),
    (waypoints_b, Simulator(max_speed=250.0)),
])
```

結果の `dict` には `pos`（位置）、`vel`（速度）、`accel`（加速度）、`time`、`speed`、`phase`、`fuel`、`battery` などが含まれます。
