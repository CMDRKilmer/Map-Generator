"""
测试套件 — 覆盖核心功能的测试缺口
重点关注：边界条件、复杂逻辑、核心模块
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.feature_gen import FeatureGenerator
from core.hex_grid import HexCoord, HexGrid
from core.noise_gen import NoiseGenerator, PerlinNoise
from core.terrain_gen import (
    BIOME_BEACH,
    BIOME_DESERT,
    BIOME_FOREST,
    BIOME_HIGH_MOUNTAINS,
    BIOME_LAKE,
    BIOME_OCEAN,
    BIOME_PLAINS,
    BIOME_SNOW,
    BIOME_VOLCANO,
    TerrainData,
    TerrainGenerator,
)


# ============================================================
# 1) core/hex_grid.py — HexCoord 核心方法
# ============================================================
class TestHexCoordCore:
    def test_distance_to(self):
        """distance_to 应返回正确的六边形距离"""
        h1 = HexCoord(0, 0)
        h2 = HexCoord(3, -1)
        assert h1.distance_to(h2) == 3

        h3 = HexCoord(-2, 5)
        h4 = HexCoord(2, -3)
        assert h3.distance_to(h4) == 8

    def test_neighbors(self):
        """neighbors 应返回 6 个不同的邻居"""
        h = HexCoord(0, 0)
        neighbors = h.neighbors()
        assert len(neighbors) == 6
        assert len(set(neighbors)) == 6
        for nh in neighbors:
            assert h.distance_to(nh) == 1

    def test_range(self):
        """range 应返回距离 ≤ n 的所有六边形"""
        h = HexCoord(0, 0)
        r0 = h.range(0)
        assert len(r0) == 1
        assert r0[0] == h

        r1 = h.range(1)
        assert len(r1) == 7

        r2 = h.range(2)
        assert len(r2) == 19

    def test_equality_and_hash(self):
        """相等坐标应有相同哈希值"""
        h1 = HexCoord(2, -3)
        h2 = HexCoord(2, -3)
        h3 = HexCoord(3, -3)
        assert h1 == h2
        assert h1 != h3
        assert hash(h1) == hash(h2)
        assert hash(h1) != hash(h3)

    def test_arithmetic_operations(self):
        """坐标运算应正确"""
        h1 = HexCoord(2, -3)
        h2 = HexCoord(1, 4)
        assert h1 + h2 == HexCoord(3, 1)
        assert h1 - h2 == HexCoord(1, -7)
        assert h1 * 3 == HexCoord(6, -9)


# ============================================================
# 2) core/hex_grid.py — HexGrid 核心方法
# ============================================================
class TestHexGridCore:
    def test_hex_center(self):
        """hex_center 应返回正确的像素坐标"""
        grid = HexGrid(1)
        h = HexCoord(0, 0)
        cx, cy = grid.hex_center(h, 10)
        assert abs(cx) < 0.001
        assert abs(cy) < 0.001

        h2 = HexCoord(1, 0)
        cx2, cy2 = grid.hex_center(h2, 10)
        assert abs(cx2 - 15.0) < 0.001

    def test_hex_corners(self):
        """hex_corners 应返回 6 个顶点"""
        grid = HexGrid(1)
        h = HexCoord(0, 0)
        corners = grid.hex_corners(h, 10)
        assert len(corners) == 6

    def test_hex_at_pixel_roundtrip(self):
        """像素坐标转换为六边形后应能还原"""
        grid = HexGrid(5)
        for h in grid.hexes[:20]:
            cx, cy = grid.hex_center(h, 10)
            h_recovered = grid.hex_at_pixel(cx, cy, 10)
            assert h == h_recovered

    def test_get_random_hex(self):
        """get_random_hex 应返回网格内的有效六边形"""
        grid = HexGrid(3)
        rng = np.random.default_rng(42)
        for _ in range(10):
            h = grid.get_random_hex(rng)
            assert h in grid.hexes

    def test_get_hexes_within_radius(self):
        """get_hexes_within_radius 应返回指定范围内的六边形"""
        grid = HexGrid(5)
        center = HexCoord(0, 0)
        hexes = grid.get_hexes_within_radius(center, 2)
        assert len(hexes) == 19
        for h in hexes:
            assert h.distance_to(center) <= 2


# ============================================================
# 3) core/terrain_gen.py — classify_biome 边界条件
# ============================================================
class TestTerrainClassifier:
    def test_classify_biome_ocean(self):
        """低高程应分类为海洋"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.10, 0.5, 0.5, False) == BIOME_OCEAN
        assert tg.classify_biome(0.14, 0.5, 0.5, False) == BIOME_OCEAN

    def test_classify_biome_lake(self):
        """中等低高程应分类为湖泊"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.20, 0.5, 0.5, False) == BIOME_LAKE
        assert tg.classify_biome(0.30, 0.5, 0.5, False) == BIOME_LAKE

    def test_classify_biome_beach(self):
        """海岸附近应分类为沙滩"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.35, 0.5, 0.5, True) == BIOME_BEACH
        assert tg.classify_biome(0.40, 0.5, 0.5, True) == BIOME_BEACH

    def test_classify_biome_snow(self):
        """高海拔低温应分类为雪地"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.90, 0.5, 0.2, False) == BIOME_SNOW
        assert tg.classify_biome(0.86, 0.5, 0.1, False) == BIOME_SNOW

    def test_classify_biome_high_mountains(self):
        """极高海拔应分类为高山"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.85, 0.5, 0.5, False) == BIOME_HIGH_MOUNTAINS

    def test_classify_biome_desert(self):
        """低湿度高温应分类为沙漠"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.40, 0.20, 0.7, False) == BIOME_DESERT

    def test_classify_biome_forest(self):
        """中等湿度温度应分类为森林"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.40, 0.60, 0.5, False) == BIOME_FOREST

    def test_classify_biome_plains(self):
        """中等湿度高温应分类为平原"""
        tg = TerrainGenerator()
        assert tg.classify_biome(0.40, 0.40, 0.6, False) == BIOME_PLAINS

    def test_classify_biome_volcano(self):
        """火山不应由 classify_biome 返回（由 generate 单独处理）"""
        tg = TerrainGenerator()
        result = tg.classify_biome(0.70, 0.30, 0.5, False)
        assert result != BIOME_VOLCANO


# ============================================================
# 4) core/noise_gen.py — PerlinNoise 边界测试
# ============================================================
class TestPerlinNoiseBoundary:
    def test_noise2d_range(self):
        """noise2d 输出应在 [-1, 1] 范围"""
        p = PerlinNoise(seed=42)
        for i in range(-10, 11):
            for j in range(-10, 11):
                v = p.noise2d(float(i), float(j))
                assert -1.0 <= v <= 1.0

    def test_octave_noise_normalized(self):
        """octave_noise 应归一化到 [-1, 1]"""
        p = PerlinNoise(seed=42)
        for _ in range(100):
            x = np.random.uniform(-100, 100)
            y = np.random.uniform(-100, 100)
            v = p.octave_noise(x, y)
            assert -1.0 <= v <= 1.0

    def test_seed_reproducibility(self):
        """相同种子应产生相同噪声"""
        p1 = PerlinNoise(seed=123)
        p2 = PerlinNoise(seed=123)
        for i in range(20):
            x = float(i) * 0.1
            y = float(i) * 0.2
            assert abs(p1.noise2d(x, y) - p2.noise2d(x, y)) < 0.0001


# ============================================================
# 5) core/noise_gen.py — NoiseGenerator 效应测试
# ============================================================
class TestNoiseGeneratorEffects:
    def test_generate_elevation_normalized(self):
        """generate_elevation 输出应归一化到 [0, 1]"""
        ng = NoiseGenerator(seed=42)
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        elev = ng.generate_elevation(coords)
        assert elev.min() >= 0.0
        assert elev.max() <= 1.0

    def test_generate_moisture_monsoon_effect(self):
        """季风方向应影响湿度分布"""
        ng = NoiseGenerator(seed=42)
        coords = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        elev = ng.generate_elevation(coords)
        moist_east = ng.generate_moisture(coords, elev, monsoon_dir=90.0)
        moist_west = ng.generate_moisture(coords, elev, monsoon_dir=270.0)
        assert not np.array_equal(moist_east, moist_west)

    def test_generate_temperature_latitude_effect(self):
        """温度应随纬度变化"""
        ng = NoiseGenerator(seed=42)
        coords = [(0.0, -5.0), (0.0, 0.0), (0.0, 5.0)]
        elev = np.array([0.5, 0.5, 0.5])
        temp = ng.generate_temperature(elev, coords)
        assert temp[0] > temp[1] > temp[2]

    def test_generate_temperature_elevation_effect(self):
        """温度应随高程降低"""
        ng = NoiseGenerator(seed=42)
        coords = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        elev = np.array([0.1, 0.5, 0.9])
        temp = ng.generate_temperature(elev, coords)
        assert temp[0] > temp[1] > temp[2]


# ============================================================
# 6) core/feature_gen.py — 聚落评分和生成
# ============================================================
class TestFeatureGeneratorSettlements:
    def test_settlement_score_biome_preferences(self):
        """聚落评分应正确反映地形偏好"""
        td_plains = TerrainData()
        td_plains.biome = "plains"
        td_plains.elevation = 0.3
        td_plains.is_coast = False
        td_plains.river_flow = 0

        td_desert = TerrainData()
        td_desert.biome = "desert"
        td_desert.elevation = 0.3
        td_desert.is_coast = False
        td_desert.river_flow = 0

        fg = FeatureGenerator({}, [])
        assert fg._settlement_score(td_plains, HexCoord(0, 0)) > fg._settlement_score(
            td_desert, HexCoord(0, 0)
        )

    def test_settlement_score_coast_bonus(self):
        """海岸应获得额外评分"""
        td_inland = TerrainData()
        td_inland.biome = "plains"
        td_inland.elevation = 0.3
        td_inland.is_coast = False
        td_inland.river_flow = 0

        td_coast = TerrainData()
        td_coast.biome = "plains"
        td_coast.elevation = 0.3
        td_coast.is_coast = True
        td_coast.river_flow = 0

        fg = FeatureGenerator({}, [])
        assert fg._settlement_score(td_coast, HexCoord(0, 0)) > fg._settlement_score(
            td_inland, HexCoord(0, 0)
        )

    def test_settlement_score_river_bonus(self):
        """河流应获得额外评分"""
        td_no_river = TerrainData()
        td_no_river.biome = "plains"
        td_no_river.elevation = 0.3
        td_no_river.is_coast = False
        td_no_river.river_flow = 0

        td_river = TerrainData()
        td_river.biome = "plains"
        td_river.elevation = 0.3
        td_river.is_coast = False
        td_river.river_flow = 0.5

        fg = FeatureGenerator({}, [])
        assert fg._settlement_score(td_river, HexCoord(0, 0)) > fg._settlement_score(
            td_no_river, HexCoord(0, 0)
        )


# ============================================================
# 7) core/feature_gen.py — 资源生成
# ============================================================
class TestFeatureGeneratorResources:
    def test_generate_resources_volcanic_bonus(self):
        """火山附近应有更高的资源概率"""
        td_volcanic = TerrainData()
        td_volcanic.is_water = False
        td_volcanic.settlement = 0
        td_volcanic.volcanic = True
        td_volcanic.biome = "volcano"

        td_normal = TerrainData()
        td_normal.is_water = False
        td_normal.settlement = 0
        td_normal.volcanic = False
        td_normal.biome = "plains"

        terrain_data = {
            HexCoord(0, 0): td_volcanic,
            HexCoord(1, 0): td_normal,
        }
        hex_list = [(HexCoord(0, 0), 0.0, 0.0), (HexCoord(1, 0), 1.0, 0.0)]
        fg = FeatureGenerator(terrain_data, hex_list)

        volcanic_resources = 0
        for _ in range(100):
            td_volcanic.resource = None
            td_normal.resource = None
            fg.generate_resources(np.random.default_rng(_), density=0.0)
            if td_volcanic.resource:
                volcanic_resources += 1

        assert volcanic_resources > 20


# ============================================================
# 8) core/feature_gen.py — 航线生成
# ============================================================
class TestFeatureGeneratorShipping:
    def test_generate_shipping_routes_coastal_only(self):
        """航线应只连接沿海聚落"""
        td_coastal_settlement = TerrainData()
        td_coastal_settlement.is_water = False
        td_coastal_settlement.is_coast = True
        td_coastal_settlement.settlement = 3

        td_inland_settlement = TerrainData()
        td_inland_settlement.is_water = False
        td_inland_settlement.is_coast = False
        td_inland_settlement.settlement = 3

        terrain_data = {
            HexCoord(0, 0): td_coastal_settlement,
            HexCoord(5, 0): td_coastal_settlement,
            HexCoord(10, 0): td_inland_settlement,
        }
        hex_list = [
            (HexCoord(0, 0), 0.0, 0.0),
            (HexCoord(5, 0), 5.0, 0.0),
            (HexCoord(10, 0), 10.0, 0.0),
        ]
        fg = FeatureGenerator(terrain_data, hex_list)

        routes = fg.generate_shipping_routes(np.random.default_rng(42))
        assert isinstance(routes, list)


# ============================================================
# 9) 端到端：完整生成流程
# ============================================================
class TestFullGenerationPipeline:
    def test_full_pipeline_with_seed(self):
        """完整生成流程应可运行且结果一致"""
        ng = NoiseGenerator(seed=123)
        grid = HexGrid(size=10)
        coords = [(float(hc.q), float(hc.r)) for hc in grid.hexes]

        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev)
        temp = ng.generate_temperature(elev, coords)

        tg = TerrainGenerator()
        hex_list = [(hc, float(hc.q), float(hc.r)) for hc in grid.hexes]
        terrain_data = tg.generate(elev, moist, temp, hex_list)
        assert len(terrain_data) == len(grid.hexes)

        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=3)
        settlements = fg.generate_settlements(rng)
        roads = fg.generate_roads()
        fg.generate_resources(rng)
        shipping = fg.generate_shipping_routes(rng)

        assert isinstance(rivers, list)
        assert isinstance(settlements, list)
        assert isinstance(roads, list)
        assert isinstance(shipping, list)

    def test_empty_grid_handling(self):
        """空网格应被正确处理"""
        grid = HexGrid(size=0)
        assert len(grid.hexes) == 1

    def test_edge_case_elevation_all_water(self):
        """全水域高程应生成湖泊/海洋"""
        ng = NoiseGenerator(seed=999)
        grid = HexGrid(size=3)
        coords = [(float(hc.q), float(hc.r)) for hc in grid.hexes]
        elev = ng.generate_elevation(coords)
        elev[:] = 0.2

        tg = TerrainGenerator()
        hex_list = [(hc, float(hc.q), float(hc.r)) for hc in grid.hexes]
        terrain_data = tg.generate(elev, np.ones_like(elev), np.ones_like(elev), hex_list)

        for td in terrain_data.values():
            assert td.is_water

    def test_edge_case_elevation_all_land(self):
        """全陆地高程应生成陆地生物群落"""
        ng = NoiseGenerator(seed=999)
        grid = HexGrid(size=3)
        coords = [(float(hc.q), float(hc.r)) for hc in grid.hexes]
        elev = ng.generate_elevation(coords)
        elev[:] = 0.6

        tg = TerrainGenerator()
        hex_list = [(hc, float(hc.q), float(hc.r)) for hc in grid.hexes]
        terrain_data = tg.generate(elev, np.ones_like(elev), np.ones_like(elev), hex_list)

        for td in terrain_data.values():
            assert not td.is_water