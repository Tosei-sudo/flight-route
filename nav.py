"""
位置取得モジュール（ナビゲーションセンサー抽象化）

GPS / INS など異なる位置取得方式を共通インターフェースで扱う。
実センサーがない場合でも、シミュレーターの真位置・加速度を使って
各センサーの推定アルゴリズムをエミュレートする。

将来の拡張例:
  GPSSensor  : 更新レートを下げる / 白色雑音を加える
  INSSensor  : 加速度バイアス・ドリフトを加える
  FusedSensor: カルマンフィルタで GPS と INS を統合
"""

import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class PositionSensor(ABC):
    """位置センサーの抽象基底クラス。"""

    @abstractmethod
    def update(self, true_pos: np.ndarray, true_vel: np.ndarray,
               accel: np.ndarray, dt: float) -> np.ndarray:
        """センサー推定位置を更新して返す。

        Args:
            true_pos : 真の位置 [m]（物理エンジンが知る実座標）
            true_vel : 真の速度 [m/s]
            accel    : 今ステップの加速度 [m/s²]
            dt       : 時間刻み [s]
        Returns:
            推定位置 [m]
        """

    @property
    @abstractmethod
    def pos(self) -> np.ndarray:
        """最後に更新した推定位置 [m]。"""


class GPSSensor(PositionSensor):
    """GPS センサー。

    衛星数をランダムウォークで変化させ、衛星数に応じた測位誤差を付加する。
    衛星数 < 4 でフィックス不能（前回位置を保持）。
    誤差モデル: σ = base_accuracy × √(8 / n_sats)  （8衛星で基準精度）
                高度誤差は水平の 1.5 倍（GPS の幾何学的特性）
    """

    def __init__(self, init_pos: np.ndarray,
                 sat_range: tuple[int, int] = (4, 12),
                 base_accuracy: float = 3.0,
                 sat_change_steps: int = 20):
        self._pos              = init_pos.copy()
        self._sat_min          = sat_range[0]
        self._sat_max          = sat_range[1]
        self._base_accuracy    = base_accuracy
        self._sat_change_steps = sat_change_steps
        self._step             = 0
        self._n_sats           = int(np.random.randint(sat_range[0], sat_range[1] + 1))

    @property
    def n_sats(self) -> int:
        return self._n_sats

    def update(self, true_pos: np.ndarray, true_vel: np.ndarray,
               accel: np.ndarray, dt: float) -> np.ndarray:
        self._step += 1

        # 衛星数のランダムウォーク（±1〜2 を緩やかに変動）
        if self._step % self._sat_change_steps == 0:
            delta = int(np.random.choice([-2, -1, -1, 0, 1, 1, 2]))
            self._n_sats = int(np.clip(self._n_sats + delta,
                                       self._sat_min, self._sat_max))

        if self._n_sats < 4:
            # フィックス不能: 前回位置を保持
            return self._pos

        # 衛星数に応じた測位誤差（σ ∝ 1/√n_sats）
        sigma_h = self._base_accuracy * np.sqrt(8.0 / self._n_sats)
        sigma_v = sigma_h * 1.5   # 高度誤差は水平より大きい
        noise   = np.array([np.random.normal(0, sigma_h),
                            np.random.normal(0, sigma_h),
                            np.random.normal(0, sigma_v)])
        self._pos = true_pos + noise
        return self._pos

    @property
    def pos(self) -> np.ndarray:
        return self._pos


class INSSensor(PositionSensor):
    """INS（慣性航法装置）センサー。

    初期位置と初期速度を起点に、加速度を累積積分して推定位置を算出する。
    加速度バイアス (accel_bias) が二重積分されるため、位置誤差は t² で増大する。
    """

    def __init__(self, init_pos: np.ndarray, init_vel: np.ndarray,
                 accel_bias: np.ndarray | None = None):
        self._pos  = init_pos.copy()
        self._vel  = init_vel.copy()
        self._bias = np.asarray(accel_bias, dtype=float) if accel_bias is not None else np.zeros(3)

    def update(self, true_pos: np.ndarray, true_vel: np.ndarray,
               accel: np.ndarray, dt: float) -> np.ndarray:
        # 真位置は参照しない — バイアス入り加速度を積分して推定
        biased_accel = accel + self._bias
        vel_old      = self._vel.copy()
        self._vel   += biased_accel * dt
        self._pos   += vel_old * dt
        return self._pos

    @property
    def pos(self) -> np.ndarray:
        return self._pos


class FusedSensor(PositionSensor):
    """GPS/INS 融合センサー。

    平常時は INS（加速度積分）で推定し、gps_interval ステップごとに
    GPS 真値で位置・速度を補正（リセット）する。
    これにより INS ドリフトが一定間隔でリセットされ、誤差の蓄積を抑制できる。

    将来: ハードリセットの代わりにカルマンフィルタで滑らかに融合可能。
    """

    def __init__(self, init_pos: np.ndarray, init_vel: np.ndarray,
                 accel_bias: np.ndarray | None = None, gps_interval: int = 60,
                 gps: GPSSensor | None = None):
        self._ins          = INSSensor(init_pos, init_vel, accel_bias)
        self._gps          = gps or GPSSensor(init_pos)
        self._gps_interval = gps_interval
        self._step         = 0

    def update(self, true_pos: np.ndarray, true_vel: np.ndarray,
               accel: np.ndarray, dt: float) -> np.ndarray:
        self._step += 1
        self._ins.update(true_pos, true_vel, accel, dt)

        if self._step % self._gps_interval == 0:
            # GPS 補正: GPS の測定位置（ノイズあり）で INS をリセット
            gps_pos = self._gps.update(true_pos, true_vel, accel, dt)
            drift   = self._ins._pos - true_pos
            if self._gps.n_sats >= 4:
                self._ins._pos = gps_pos.copy()
                self._ins._vel = true_vel.copy()  # 速度はGPS真値で補正
                gps_err = float(np.linalg.norm(gps_pos - true_pos))
                logger.info(
                    "[GPS補正 t=%ds]  衛星%d機  ドリフト (%.1f, %.1f, %.1f) m  GPS誤差 %.1f m",
                    self._step * dt, self._gps.n_sats,
                    drift[0], drift[1], drift[2], gps_err)
                logger.debug(
                    "GPS補正詳細: INS位置=%s  GPS位置=%s  真位置=%s",
                    np.round(self._ins._pos, 1),
                    np.round(gps_pos, 1),
                    np.round(true_pos, 1))
            else:
                logger.info(
                    "[GPS補正 t=%ds]  衛星%d機 → フィックス不能、INS継続",
                    self._step * dt, self._gps.n_sats)
        else:
            self._gps.update(true_pos, true_vel, accel, dt)  # 衛星数を進める

        return self._ins.pos

    @property
    def pos(self) -> np.ndarray:
        return self._ins.pos


def make_sensor(mode: str, init_pos: np.ndarray,
                init_vel: np.ndarray) -> PositionSensor:
    """config の NAV_MODE から適切なセンサーを生成する。"""
    from config import (INS_ACCEL_BIAS, INS_GPS_INTERVAL,
                        GPS_SAT_RANGE, GPS_BASE_ACCURACY, GPS_SAT_CHANGE_STEPS)
    bias = np.asarray(INS_ACCEL_BIAS, dtype=float)
    gps  = GPSSensor(init_pos,
                     sat_range=GPS_SAT_RANGE,
                     base_accuracy=GPS_BASE_ACCURACY,
                     sat_change_steps=GPS_SAT_CHANGE_STEPS)
    if mode == 'ins':
        return INSSensor(init_pos, init_vel, accel_bias=bias)
    if mode == 'fused':
        return FusedSensor(init_pos, init_vel, accel_bias=bias,
                           gps_interval=INS_GPS_INTERVAL, gps=gps)
    return gps  # 'gps'
