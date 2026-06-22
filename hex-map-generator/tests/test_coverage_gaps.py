"""
测试覆盖缺口补充 — 针对核心模块的边界条件和未覆盖功能
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.feature_gen import FeatureGenerator
from core.hex_grid import HexCoord, HexGrid
from core.noise_gen import NoiseGenerator, PerlinNoise
from core.terrain_gen import (
    BIOME_OCEAN,
    BIOME_PLAINS,
    BIOME_DESERT,
    BIOME_FOREST,
    BIOME_SNOW,
    BIOME_VOLCANO,
    BIOME_SWAMP,
    BIOME_MOUNTAINS,
    BIOME_HIGH_MOUNTAINS,
    BIOME_TAIGA,
    TerrainData,
    TerrainGenerator,
)


# ============================================================
# 1) core/hex_grid.py — 几何计算方法测试
# ============================================================
class TestHexCoordOperations:
    def test_hex_coord_add(self):
        h1 = HexCoord(2, 3)
        h2 = HexCoord(-1, 4)
        result = h1 + h2
        assert result.q == 1 and result.r == 7

    def test_hex_coord_sub(self):
        h1 = HexCoord(5, 2)
        h2 = HexCoord(3, 1)
        result = h1 - h2
        assert result.q == 2 and result.r == 1

    def test_hex_coord_mul(self):
        h = HexCoord(3, -2)
        result = h * 3
        assert result.q == 9 and result.r == -6

    def test_distance_to(self):
        h1 = HexCoord(0, 0)
        h2 = HexCoord(3, -2)
        assert h1.distance_to(h2) == 3

    def test_neighbors_count(self):
        h = HexCoord(0, 0)
        neighbors = h.neighbors()
        assert len(neighbors) == 6

    def test_range_returns_correct_count(self):
        h = HexCoord(0, 0)
        assert len(h.range(0)) == 1
        assert len(h.range(1)) == 7
        assert len(h.range(2)) == 19

    def test_to_cube(self):
        h = HexCoord(2, 3)
        q, r, s = h.to_cube()
        assert q == 2 and r == 3 and s == -5
        assert q + r + s == 0


class TestHexGridGeometry:
    def test_hex_center_origin(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(0, 0), 10)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_hex_center_offset(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(2, 3), 10)
        expected_x = 10 * (3.0 / 2.0 * 2)
        expected_y = 10 * (math.sqrt(3.0) / 2.0 * 2 + math.sqrt(3.0) * 3)
        assert cx == pytest.approx(expected_x)
        assert cy == pytest.approx(expected_y)

    def test_hex_corners_count(self):
        grid = HexGrid(size=5)
        corners = grid.hex_corners(HexCoord(0, 0), 10)
        assert len(corners) == 6

    def test_hex_at_pixel_roundtrip(self):
        grid = HexGrid(size=5)
        hex_size = 20
        for hc in grid.hexes[:20]:
            cx, cy = grid.hex_center(hc, hex_size)
            hc_back = grid.hex_at_pixel(cx, cy, hex_size)
            assert hc_back == hc

    def test_get_random_hex(self):
        grid = HexGrid(size=3)
        rng = np.random.Generator(np.random.PCG64(42))
        hexes_set = set()
        for _ in range(50):
            hc = grid.get_random_hex(rng)
            hexes_set.add((hc.q, hc.r))
        assert len(hexes_set) > 1


# ============================================================
# 2) core/noise_gen.py — PerlinNoise 核心算法测试
# ============================================================
class TestPerlinNoise:
    def test_noise2d_range(self):
        pn = PerlinNoise(seed=42)
        values = []
        for x in np.linspace(-10, 10, 20):
            for y in np.linspace(-10, 10, 20):
                values.append(pn.noise2d(x, y))
        assert min(values) >= -1.1
        assert max(values) <= 1.1

    def test_noise2d_seed_reproducibility(self):
        pn1 = PerlinNoise(seed=123)
        pn2 = PerlinNoise(seed=123)
        for x in np.linspace(0, 1, 10):
            for y in np.linspace(0, 1, 10):
                assert pn1.noise2d(x, y) == pytest.approx(pn2.noise2d(x, y))

    def test_octave_noise_range(self):
        pn = PerlinNoise(seed=42)
        values = []
        for _ in range(50):
            x, y = np.random.uniform(-10, 10, 2)
            values.append(pn.octave_noise(x, y, octaves=4))
        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_generate_elevation_empty(self):
        ng = NoiseGenerator(seed=42)
        result = ng.generate_elevation([])
        assert len(result) == 0

    def test_generate_elevation_single(self):
        ng = NoiseGenerator(seed=42)
        result = ng.generate_elevation([(0.0, 0.0)])
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0


# ============================================================
# 3) core/terrain_gen.py — TerrainData 和边界条件
# ============================================================
class TestTerrainData:
    def test_default_values(self):
        td = TerrainData()
        assert td.biome == BIOME_OCEAN
        assert td.is_water is True
        assert td.elevation == 0.0
        assert td.moisture == 0.0
        assert td.temperature == 0.5
        assert td.resource is None
        assert td.settlement == 0

    def test_repr(self):
        td = TerrainData()
        td.biome = BIOME_PLAINS
        td.elevation = 0.5
        repr_str = repr(td)
        assert "Terrain" in repr_str
        assert "plains" in repr_str


class TestTerrainGeneratorBiomes:
    def test_classify_biome_ocean(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.1, 0.5, 0.5, False)
        assert result == BIOME_OCEAN

    def test_classify_biome_lake(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.25, 0.5, 0.5, False)
        assert result == "lake"

    def test_classify_biome_beach(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.38, 0.5, 0.5, True)
        assert result == "beach"

    def test_classify_biome_desert(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.45, 0.2, 0.8, False)
        assert result == BIOME_DESERT

    def test_classify_biome_forest(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.4, 0.6, 0.6, False)
        assert result == BIOME_FOREST

    def test_classify_biome_snow(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.9, 0.5, 0.1, False)
        assert result == BIOME_SNOW

    def test_classify_biome_swamp(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.38, 0.9, 0.35, False)
        assert result == BIOME_SWAMP

    def test_classify_biome_mountains(self):
        tg = TerrainGenerator()
        result = tg.classify_biome(0.75, 0.5, 0.5, False)
        assert result == BIOME_MOUNTAINS


# ============================================================
# 4) core/feature_gen.py — 特性生成测试
# ============================================================
class TestFeatureGenerator:
    def test_generate_rivers_empty(self):
        tg = TerrainGenerator()
        rng = np.random.default_rng(42)
        td = {}
        fg = FeatureGenerator(td, [])
        rivers = fg.generate_rivers(rng, num_rivers=5)
        assert rivers == []

    def test_generate_settlements_empty(self):
        td = {}
        fg = FeatureGenerator(td, [])
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(rng)
        assert settlements == []

    def test_generate_resources_empty(self):
        td = {}
        fg = FeatureGenerator(td, [])
        rng = np.random.default_rng(42)
        fg.generate_resources(rng)
        assert len(td) == 0

    def test_generate_shipping_routes_empty(self):
        td = {}
        fg = FeatureGenerator(td, [])
        rng = np.random.default_rng(42)
        routes = fg.generate_shipping_routes(rng)
        assert routes == []


# ============================================================
# 5) 端到端：完整地形生成流程
# ============================================================
class TestFullPipelineRobustness:
    def test_empty_inputs(self):
        ng = NoiseGenerator(seed=42)
        tg = TerrainGenerator()

        result = ng.generate_elevation([])
        assert len(result) == 0

        result = ng.generate_moisture([], np.array([]))
        assert len(result) == 0

        result = ng.generate_temperature(np.array([]), [])
        assert len(result) == 0

        result = tg.generate(np.array([]), np.array([]), np.array([]), [])
        assert result == {}

    def test_single_hex(self):
        ng = NoiseGenerator(seed=42)
        tg = TerrainGenerator()
        fg = FeatureGenerator({}, [])

        coords = [(0.0, 0.0)]
        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev)
        temp = ng.generate_temperature(elev, coords)

        hex_list = [(HexCoord(0, 0), 0.0, 0.0)]
        td = tg.generate(elev, moist, temp, hex_list)
        assert len(td) == 1

        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=1)
        assert isinstance(rivers, list)

    def test_seed_consistency(self):
        ng1 = NoiseGenerator(seed=123)
        ng2 = NoiseGenerator(seed=123)

        coords = [(i * 0.1, i * 0.2) for i in range(20)]
        e1 = ng1.generate_elevation(coords)
        e2 = ng2.generate_elevation(coords)

        np.testing.assert_array_almost_equal(e1, e2)

    def test_edge_biomes(self):
        tg = TerrainGenerator()

        assert tg.classify_biome(0.35, 0.5, 0.5, False) == BIOME_TAIGA
        assert tg.classify_biome(0.70, 0.5, 0.5, False) == BIOME_MOUNTAINS
        assert tg.classify_biome(0.85, 0.5, 0.5, False) == BIOME_HIGH_MOUNTAINS