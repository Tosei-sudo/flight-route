"""CSV エクスポートのテスト (exporter.py)"""

import csv
import pytest
import numpy as np
from exporter import save_csv


def _make_hist(n: int = 5) -> dict:
    return {
        'pos':        np.zeros((n, 3)),
        'vel':        np.zeros((n, 3)),
        'accel':      np.zeros((n, 3)),
        'time':       np.arange(n, dtype=float) * 0.5,
        'speed':      np.ones(n) * 10.0,
        'elevation':  np.zeros(n),
        'azimuth':    np.zeros(n),
        'hit_ground': False,
    }


def test_save_csv_creates_file(tmp_path):
    path = str(tmp_path / 'telem.csv')
    save_csv(_make_hist(), path)
    assert (tmp_path / 'telem.csv').exists()


def test_save_csv_row_count(tmp_path):
    n = 7
    path = str(tmp_path / 'telem.csv')
    save_csv(_make_hist(n), path)
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    assert len(rows) == n + 1  # ヘッダー + データ行


def test_save_csv_header_columns(tmp_path):
    path = str(tmp_path / 'telem.csv')
    save_csv(_make_hist(), path)
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    expected = [
        'time_s', 'x_m', 'y_m', 'z_m',
        'vx_ms', 'vy_ms', 'vz_ms', 'speed_ms',
        'ax_ms2', 'ay_ms2', 'az_ms2', 'accel_mag_ms2',
        'elevation_deg', 'azimuth_deg',
    ]
    assert header == expected


def test_save_csv_time_values(tmp_path):
    n = 4
    hist = _make_hist(n)
    path = str(tmp_path / 'telem.csv')
    save_csv(hist, path)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        times = [float(row['time_s']) for row in reader]
    expected = list(np.arange(n) * 0.5)
    assert times == pytest.approx(expected, abs=1e-3)


def test_save_csv_speed_values(tmp_path):
    hist = _make_hist(3)
    hist['speed'] = np.array([10.0, 20.0, 30.0])
    path = str(tmp_path / 'telem.csv')
    save_csv(hist, path)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        speeds = [float(row['speed_ms']) for row in reader]
    assert speeds == pytest.approx([10.0, 20.0, 30.0], abs=1e-3)
