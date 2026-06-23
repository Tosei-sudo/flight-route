"""ナビゲーションセンサーのテスト (nav.py)"""

import pytest
import numpy as np
from nav import GPSSensor, INSSensor, FusedSensor, make_sensor


class TestINSSensor:
    def test_initial_position_stored(self):
        init_pos = np.array([10.0, 20.0, 30.0])
        ins = INSSensor(init_pos, np.zeros(3))
        assert np.allclose(ins.pos, init_pos)

    def test_zero_accel_no_movement(self):
        ins = INSSensor(np.array([5.0, 5.0, 5.0]), np.zeros(3))
        pos = ins.update(np.zeros(3), np.zeros(3), np.zeros(3), 1.0)
        assert np.allclose(pos, [5.0, 5.0, 5.0])

    def test_vel_old_used_for_position_update(self):
        # Bug #7 regression:
        # 正しい実装: pos += vel_old * dt  (更新前の速度を使う)
        # 誤った実装: pos += vel_new * dt  (更新後の速度を使う)
        # vel=0, accel=1, dt=1 → vel_old=0 なので pos は変わらない
        ins = INSSensor(np.zeros(3), np.zeros(3))
        pos = ins.update(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0)
        assert abs(float(pos[0])) < 0.01, (
            f"vel_old=0 なのに pos が動いた: {pos[0]:.3f} (誤った Euler 積分の疑い)"
        )

    def test_two_steps_correct_euler(self):
        # Step1: vel_old=0 → pos=0,  vel_new=1
        # Step2: vel_old=1 → pos=1,  vel_new=1 (accel=0)
        ins = INSSensor(np.zeros(3), np.zeros(3))
        ins.update(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0)
        pos = ins.update(np.zeros(3), np.zeros(3), np.zeros(3), 1.0)
        assert abs(float(pos[0]) - 1.0) < 0.01

    def test_bias_accumulates_over_time(self):
        bias = np.array([0.1, 0.0, 0.0])
        ins = INSSensor(np.zeros(3), np.zeros(3), accel_bias=bias)
        for _ in range(20):
            ins.update(np.zeros(3), np.zeros(3), np.zeros(3), 0.5)
        # バイアスが蓄積されて正 X 方向にドリフトする
        assert ins.pos[0] > 0.0

    def test_pos_property(self):
        ins = INSSensor(np.array([1.0, 2.0, 3.0]), np.zeros(3))
        assert np.allclose(ins.pos, [1.0, 2.0, 3.0])


class TestGPSSensor:
    def test_initial_pos_stored(self):
        init = np.array([100.0, 200.0, 50.0])
        gps = GPSSensor(init)
        assert np.allclose(gps.pos, init)

    def test_returns_near_true_pos_with_many_satellites(self):
        np.random.seed(42)
        gps = GPSSensor(np.zeros(3), sat_range=(10, 12), base_accuracy=3.0)
        true_pos = np.array([1000.0, 2000.0, 100.0])
        pos = gps.update(true_pos, np.zeros(3), np.zeros(3), 0.5)
        assert np.linalg.norm(pos - true_pos) < 30.0  # 30 m 以内

    def test_n_sats_within_range(self):
        gps = GPSSensor(np.zeros(3), sat_range=(4, 12))
        for _ in range(100):
            gps.update(np.zeros(3), np.zeros(3), np.zeros(3), 0.5)
        assert 4 <= gps.n_sats <= 12


class TestFusedSensor:
    def test_initial_pos(self):
        init = np.array([1.0, 2.0, 3.0])
        sensor = FusedSensor(init, np.zeros(3))
        assert np.allclose(sensor.pos, init)

    def test_gps_correction_resets_drift(self):
        # INS はドリフトするが GPS 補正でリセットされる
        bias = np.array([1.0, 0.0, 0.0])  # 大きめのバイアス
        true_pos = np.array([100.0, 0.0, 0.0])
        sensor = FusedSensor(true_pos.copy(), np.zeros(3),
                             accel_bias=bias, gps_interval=10,
                             gps=GPSSensor(true_pos, sat_range=(8, 12)))
        # 10 ステップ = GPS 補正タイミング
        for _ in range(10):
            sensor.update(true_pos, np.zeros(3), np.zeros(3), 0.5)
        # 補正後は真位置の近くにいるはず
        assert np.linalg.norm(sensor.pos - true_pos) < 50.0


class TestMakeSensor:
    def test_gps_mode(self):
        s = make_sensor('gps', np.zeros(3), np.zeros(3))
        assert isinstance(s, GPSSensor)

    def test_ins_mode(self):
        s = make_sensor('ins', np.zeros(3), np.zeros(3))
        assert isinstance(s, INSSensor)

    def test_fused_mode(self):
        s = make_sensor('fused', np.zeros(3), np.zeros(3))
        assert isinstance(s, FusedSensor)
