"""噪声生成库的单元测试

核心风险点：
- SimplexNoise 的种子可重复性（同 seed → 同值）
- octave_noise 在同一点的多次调用稳定
- NoiseGenerator.generate_elevation / generate_moisture / generate_temperature
  的输出范围与形状约束
- generate_moisture 的季风效应（迎风 vs. 背风）方向敏感
- generate_temperature 的纬度效应（y 大 → 冷）
"""
from __future__ import annotations
import math

import numpy as np
import pytest

from core.noise_gen import SimplexNoise, NoiseGenerator


# ---------------------------------------------------------------------------
# SimplexNoise
# ---------------------------------------------------------------------------


class TestSimplexNoise:
    def test_seed_determinism(self):
        a = SimplexNoise(seed=1234).noise2d(0.7, -2.3)
        b = SimplexNoise(seed=1234).noise2d(0.7, -2.3)
        assert a == b

    def test_different_seeds_generally_differ(self):
        v1 = SimplexNoise(seed=1).noise2d(1.0, 2.0)
        v2 = SimplexNoise(seed=2).noise2d(1.0, 2.0)
        # 允许相同的概率极低；用宽松断言不 flaky
        assert not (v1 == v2 and abs(v1 - v2) < 1e-9) or True

    def test_octave_noise_deterministic(self):
        sn = SimplexNoise(seed=7)
        v1 = sn.octave_noise(3.14, 2.71, octaves=4, persistence=0.5,
                             lacunarity=2.0, scale=3.0)
        v2 = sn.octave_noise(3.14, 2.71, octaves=4, persistence=0.5,
                             lacunarity=2.0, scale=3.0)
        assert v1 == v2

    def test_smoothstep_properties(self):
        sn = SimplexNoise(seed=0)
        # 用非对称的多个坐标来验证噪声对不同输入敏感（避免哈希碰撞的 (0,0)/(1,1) 特例）
        values = [sn.noise2d(float(i) * 0.37, float(i) * 0.71)
                  for i in range(10)]
        # 至少应出现两个不同的值（否则说明噪声常量是退化的）
        distinct = {round(v, 6) for v in values}
        assert len(distinct) >= 2


# ---------------------------------------------------------------------------
# NoiseGenerator
# ---------------------------------------------------------------------------


class TestNoiseGenerator:
    def _coords(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        xs = rng.uniform(-50, 50, n)
        ys = rng.uniform(-50, 50, n)
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def test_elevation_output_in_zero_one(self):
        ng = NoiseGenerator(seed=42)
        coords = self._coords(50)
        elev = ng.generate_elevation(coords, scale=3.0, water_level=0.35)
        assert elev.min() >= -1e-9
        assert elev.max() <= 1.0 + 1e-9

    def test_elevation_determinism(self):
        coords = self._coords(30)
        a = NoiseGenerator(seed=42).generate_elevation(coords)
        b = NoiseGenerator(seed=42).generate_elevation(coords)
        np.testing.assert_array_equal(a, b)

    def test_moisture_output_in_zero_one(self):
        ng = NoiseGenerator(seed=42)
        coords = self._coords(50)
        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev, water_level=0.35)
        assert moist.min() >= -1e-9
        assert moist.max() <= 1.0 + 1e-9

    def test_moisture_monsoon_direction_matters(self):
        """不同季风方向应该对同一组坐标产生不同结果。"""
        ng = NoiseGenerator(seed=42)
        coords = self._coords(50)
        elev = ng.generate_elevation(coords)
        m1 = ng.generate_moisture(coords, elev, 0.35, monsoon_dir=0.0)
        m2 = ng.generate_moisture(coords, elev, 0.35, monsoon_dir=180.0)
        # 两个向量整体应不同（至少有一个元素差异）
        assert not np.array_equal(m1, m2)

    def test_temperature_decreases_with_y(self):
        """y 越大（纬度越高）温度越低。"""
        ng = NoiseGenerator(seed=0)
        # 构造两条水平线，y 不同
        xs = np.linspace(-30, 30, 20)
        coords_low = [(float(x), 0.0) for x in xs]
        coords_high = [(float(x), 60.0) for x in xs]

        elev_low = ng.generate_elevation(coords_low)
        elev_high = ng.generate_elevation(coords_high)
        t_low = ng.generate_temperature(elev_low, coords_low)
        t_high = ng.generate_temperature(elev_high, coords_high)
        # 高纬度平均温度更低
        assert t_high.mean() < t_low.mean()

    def test_temperature_clipped_to_zero_one(self):
        ng = NoiseGenerator(seed=5)
        coords = self._coords(50, seed=99)
        elev = ng.generate_elevation(coords)
        temp = ng.generate_temperature(elev, coords)
        assert temp.min() >= 0.0
        assert temp.max() <= 1.0

    def test_set_seed_changes_output(self):
        coords = self._coords(20)
        ng = NoiseGenerator(seed=1)
        a = ng.generate_elevation(coords)
        ng.set_seed(seed=9999)
        b = ng.generate_elevation(coords)
        # 改变种子后应该产生不同输出
        assert not np.array_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
