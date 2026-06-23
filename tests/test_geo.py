"""座標変換モジュールのテスト (geo.py)"""

import pytest
import numpy as np
from geo import geo_to_local, geo_to_local_pt, local_to_geo
from config import GEO_WAYPOINTS, EARTH_R


def test_first_waypoint_is_origin():
    pts = geo_to_local(GEO_WAYPOINTS[:2])
    assert np.allclose(pts[0], [0.0, 0.0, 0.0], atol=0.1)


def test_north_is_positive_y():
    # 北に移動すると y が増加
    pt_south = geo_to_local_pt(45.0, 141.76, 0.0)
    pt_north = geo_to_local_pt(45.5, 141.76, 0.0)
    assert pt_north[1] > pt_south[1]


def test_east_is_positive_x():
    # 東に移動すると x が増加
    pt_west = geo_to_local_pt(45.39, 141.5, 0.0)
    pt_east = geo_to_local_pt(45.39, 142.0, 0.0)
    assert pt_east[0] > pt_west[0]


def test_altitude_is_z():
    pt = geo_to_local_pt(45.39521, 141.76507, 500.0)
    assert pt[2] == pytest.approx(500.0)


def test_geo_to_local_pt_distance_accuracy():
    # 1度の緯度差 ≈ 111 km
    lat0, lon0 = GEO_WAYPOINTS[0][0], GEO_WAYPOINTS[0][1]
    pt = geo_to_local_pt(lat0 + 1.0, lon0, 0.0)
    expected_y = np.radians(1.0) * EARTH_R
    assert abs(pt[1] - expected_y) < 10.0  # 10 m 以内


def test_local_to_geo_roundtrip():
    lat_orig, lon_orig = 45.5, 141.9
    local = geo_to_local_pt(lat_orig, lon_orig, 0.0)
    lat_back, lon_back = local_to_geo(local)
    assert abs(lat_back - lat_orig) < 1e-5
    assert abs(lon_back - lon_orig) < 1e-5


def test_geo_to_local_skips_callables():
    # callable エントリはスキップされる（固定 WP のみ変換）
    wps = [GEO_WAYPOINTS[0], lambda t: (45.5, 141.9, 0), GEO_WAYPOINTS[-1]]
    pts = geo_to_local(wps)
    assert pts.shape == (2, 3)  # callable を除いた2点


def test_callable_first_wp_raises_assertion():
    # Bug #8 regression: 第1 WP が callable の場合 AssertionError を発生させる
    with pytest.raises(AssertionError, match="GEO_WAYPOINTS\\[0\\]"):
        geo_to_local([lambda t: (45.0, 141.0, 0), (41.0, 143.0, 0)])
