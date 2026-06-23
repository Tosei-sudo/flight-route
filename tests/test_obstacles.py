"""障害物クラスのテスト (obstacles.py)"""

import pytest
import numpy as np
from obstacles import SphereObstacle, BoxObstacle, MovingObstacle, AreaObstacle
from config import AVOIDANCE_MARGIN


class TestSphereObstacle:
    def _make(self, radius=50.0):
        return SphereObstacle(pos=np.zeros(3), radius=radius)

    def test_dist_outside(self):
        obs = self._make(radius=10.0)
        dist = obs.dist_from_surface(np.array([15.0, 0.0, 0.0]))
        assert dist == pytest.approx(5.0)

    def test_dist_on_surface(self):
        obs = self._make(radius=10.0)
        dist = obs.dist_from_surface(np.array([10.0, 0.0, 0.0]))
        assert abs(dist) < 1e-6

    def test_dist_inside_is_negative(self):
        obs = self._make(radius=50.0)
        dist = obs.dist_from_surface(np.array([10.0, 0.0, 0.0]))
        assert dist == pytest.approx(-40.0)

    def test_zone_equals_radius_times_margin(self):
        obs = SphereObstacle(pos=np.zeros(3), radius=100.0)
        assert obs.zone == pytest.approx(100.0 * AVOIDANCE_MARGIN)

    def test_repulsion_dir_points_outward(self):
        obs = self._make(radius=10.0)
        d = obs.repulsion_dir(np.array([5.0, 0.0, 0.0]))
        assert np.allclose(d, [1.0, 0.0, 0.0], atol=1e-6)

    def test_repulsion_dir_is_unit_vector(self):
        obs = self._make(radius=10.0)
        pos = np.array([3.0, 4.0, 0.0])  # 斜め方向
        d = obs.repulsion_dir(pos)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-6

    def test_repulsion_dir_at_center_fallback(self):
        obs = self._make(radius=10.0)
        d = obs.repulsion_dir(np.zeros(3))  # 中心: ゼロ除算回避
        assert abs(np.linalg.norm(d) - 1.0) < 1e-6


class TestBoxObstacle:
    def _make(self, half=10.0):
        return BoxObstacle(pos=np.zeros(3), half_extents=np.array([half, half, half]))

    def test_dist_outside(self):
        obs = self._make(half=10.0)
        dist = obs.dist_from_surface(np.array([20.0, 0.0, 0.0]))
        assert dist == pytest.approx(10.0)

    def test_dist_inside_is_negative(self):
        obs = self._make(half=10.0)
        dist = obs.dist_from_surface(np.zeros(3))  # 中心
        assert dist < 0

    def test_dist_on_face(self):
        obs = self._make(half=10.0)
        dist = obs.dist_from_surface(np.array([10.0, 0.0, 0.0]))
        assert abs(dist) < 1e-6

    def test_zone_covers_entire_box(self):
        extents = np.array([5.0, 10.0, 3.0])
        obs = BoxObstacle(pos=np.zeros(3), half_extents=extents)
        expected = float(np.linalg.norm(extents)) * AVOIDANCE_MARGIN
        assert obs.zone == pytest.approx(expected)

    def test_repulsion_outside_points_away(self):
        obs = self._make(half=10.0)
        pos = np.array([20.0, 0.0, 0.0])
        d = obs.repulsion_dir(pos)
        assert d[0] > 0  # 外側に向く


class TestMovingObstacle:
    def test_pos_at_zero(self):
        obs = MovingObstacle(
            pos_init=np.array([10.0, 20.0, 30.0]),
            vel=np.array([5.0, 0.0, 0.0]),
            radius=50.0,
        )
        assert np.allclose(obs.pos_at(0.0), [10.0, 20.0, 30.0])

    def test_pos_at_t(self):
        obs = MovingObstacle(
            pos_init=np.zeros(3),
            vel=np.array([10.0, 0.0, 0.0]),
            radius=50.0,
        )
        assert np.allclose(obs.pos_at(5.0), [50.0, 0.0, 0.0])

    def test_pos_at_negative_t(self):
        obs = MovingObstacle(
            pos_init=np.zeros(3),
            vel=np.array([10.0, 0.0, 0.0]),
            radius=50.0,
        )
        assert np.allclose(obs.pos_at(-2.0), [-20.0, 0.0, 0.0])

    def test_zone(self):
        obs = MovingObstacle(pos_init=np.zeros(3), vel=np.zeros(3), radius=100.0)
        assert obs.zone == pytest.approx(100.0 * AVOIDANCE_MARGIN)


class TestAreaObstacle:
    def _make_square(self, size=100.0, height=500.0):
        verts = np.array([
            [0.0,    0.0   ],
            [size,   0.0   ],
            [size,   size  ],
            [0.0,    size  ],
        ])
        return AreaObstacle(vertices_enu=verts, height_msl=height)

    def test_inside_below_ceiling(self):
        obs = self._make_square(size=100.0, height=500.0)
        pos = np.array([50.0, 50.0, 100.0])  # 内部、天井以下
        dist = obs.dist_from_surface(pos)
        assert dist < 0  # 内部は負

    def test_outside_polygon(self):
        obs = self._make_square(size=100.0, height=500.0)
        pos = np.array([200.0, 50.0, 100.0])  # 外部
        dist = obs.dist_from_surface(pos)
        assert dist > 0  # 外部は正

    def test_above_ceiling_is_clear(self):
        obs = self._make_square(size=100.0, height=500.0)
        pos = np.array([50.0, 50.0, 600.0])  # 天井 (500m) より上
        dist = obs.dist_from_surface(pos)
        assert dist == float('inf')

    def test_zone_property(self):
        obs = self._make_square()
        assert obs.zone == pytest.approx(2000.0)  # デフォルト zone_m
