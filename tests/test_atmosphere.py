"""ISA 標準大気モデルのテスト"""

import pytest
from atmosphere import air_density_ratio


def test_sea_level_is_one():
    assert air_density_ratio(0.0) == pytest.approx(1.0, rel=1e-4)


def test_below_sea_level_clamped_to_surface():
    assert air_density_ratio(-500.0) == pytest.approx(air_density_ratio(0.0))


def test_tropopause_known_value():
    # 対流圏界面 11000 m ≈ 0.2971
    assert air_density_ratio(11_000.0) == pytest.approx(0.2971, rel=0.005)


def test_stratosphere_lower_than_tropopause():
    assert air_density_ratio(15_000.0) < air_density_ratio(11_000.0)


def test_high_altitude_low_density():
    # 20000 m ≈ 0.072
    ratio = air_density_ratio(20_000.0)
    assert 0.06 < ratio < 0.09


def test_monotonically_decreasing():
    altitudes = [0, 500, 1000, 3000, 5000, 8000, 11_000, 15_000, 20_000]
    ratios = [air_density_ratio(h) for h in altitudes]
    for i in range(len(ratios) - 1):
        assert ratios[i] > ratios[i + 1], f"高度 {altitudes[i]}→{altitudes[i+1]} m で密度が増加"


def test_always_positive():
    for h in [0, 5_000, 11_000, 20_000, 50_000]:
        assert air_density_ratio(h) > 0
