"""
核心逻辑单元测试 — 补充回归测试安全网
覆盖关键算法和边界条件
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.hex_grid import HexCoord, HexGrid
from core.noise_gen import PerlinNoise
from core.terrain_gen import (
    BIOME_OCEAN,
    BIOME_LAKE,
    BIOME_BEACH,
    BIOME_PLAINS,
    BIOME_FOREST,
    BIOME_DESERT,
    BIOME_SNOW,
    BIOME_VOLCANO,
    BIOME_HILLS,
    BIOME_MOUNTAINS,
    BIOME_HIGH_MOUNTAINS,
    BIOME_SWAMP,
    BIOME_TUNDRA,
    BIOME_SAVANNA,
    BIOME_TAIGA,
    BIOME_DENSE_FOREST,
    BIOME_RAINFOREST,
    TerrainGenerator,
)
from core.feature_gen import FeatureGenerator
from core.terrain_gen import TerrainData


# ============================================================
# 1) HexGrid 几何计算测试
# ============================================================
class TestHexGridGeometry:
    def test_hex_center_origin(self):
        """原点六边形中心应在 (0, 0)"""
        grid = HexGrid(size=5)
        hc = HexCoord(0, 0)
        cx, cy = grid.hex_center(hc, 10.0)
        assert abs(cx) < 0.001 and abs(cy) < 0.001

    def test_hex_center_offset(self):
        """偏移六边形的中心计算"""
        grid = HexGrid(size=5)
        hc = HexCoord(2, -1)
        cx, cy = grid.hex_center(hc, 2.0)
        expected_x = 2.0 * (3.0 / 2.0 * 2)
        expected_y = 2.0 * (math.sqrt(3.0) / 2.0 * 2 + math.sqrt(3.0) * (-1))
        assert abs(cx - expected_x) < 0.001
        assert abs(cy - expected_y) < 0.001

    def test_hex_corners_count(self):
        """每个六边形应有6个顶点"""
        grid = HexGrid(size=5)
        corners = grid.hex_corners(HexCoord(0, 0), 10.0)
        assert len(corners) == 6

    def test_hex_corners_equidistant(self):
        """所有顶点应等距于中心"""
        grid = HexGrid(size=5)
        corners = grid.hex_corners(HexCoord(0, 0), 10.0)
        cx, cy = grid.hex_center(HexCoord(0, 0), 10.0)
        for x, y in corners:
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            assert abs(dist - 10.0) < 0.001

    def test_hex_at_pixel_roundtrip(self):
        """像素坐标转换应为可逆的"""
        grid = HexGrid(size=5)
        hex_size = 20.0
        for hc in grid.hexes[:20]:
            cx, cy = grid.hex_center(hc, hex_size)
            hc_back = grid.hex_at_pixel(cx, cy, hex_size)
            assert hc == hc_back

    def test_hex_round_boundary(self):
        """浮点坐标四舍五入边界测试"""
        grid = HexGrid(size=5)
        hc = grid._hex_round(0.2, 0.2)
        assert hc == HexCoord(0, 0)
        hc = grid._hex_round(0.8, 0.8)
        assert hc != HexCoord(0, 0)

    def test_get_hexes_within_radius(self):
        """获取指定半径内的六边形"""
        center = HexCoord(0, 0)
        hexes = center.range(2)
        # 半径为2的六边形数量 = 1 + 6 + 12 = 19
        assert len(hexes) == 19
        # 验证距离限制
        for h in hexes:
            assert h.distance_to(center) <= 2


# ============================================================
# 2) TerrainGenerator.classify_biome 边界条件测试
# ============================================================
class TestBiomeClassification:
    def test_biome_ocean_deep(self):
        """低高程应为海洋"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.10, 0.5, 0.5, False)
        assert biome == BIOME_OCEAN

    def test_biome_lake(self):
        """浅水区域应为湖泊"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.25, 0.5, 0.5, False)
        assert biome == BIOME_LAKE

    def test_biome_beach(self):
        """海岸区域应为沙滩"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.38, 0.5, 0.5, True)
        assert biome == BIOME_BEACH

    def test_biome_snow_high(self):
        """高海拔低温应为雪地"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.90, 0.5, 0.2, False)
        assert biome == BIOME_SNOW

    def test_biome_high_mountains(self):
        """极高海拔应为高山"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.85, 0.5, 0.6, False)
        assert biome == BIOME_HIGH_MOUNTAINS

    def test_biome_mountains(self):
        """高海拔应为山地"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.75, 0.5, 0.5, False)
        assert biome == BIOME_MOUNTAINS

    def test_biome_hills(self):
        """中高海拔应为丘陵"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.60, 0.5, 0.5, False)
        assert biome == BIOME_HILLS

    def test_biome_desert(self):
        """低湿度高温应为沙漠"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.20, 0.70, False)
        assert biome == BIOME_DESERT

    def test_biome_tundra(self):
        """低湿度低温应为苔原"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.20, 0.30, False)
        assert biome == BIOME_TUNDRA

    def test_biome_savanna(self):
        """中低湿度高温应为草原"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.30, 0.70, False)
        assert biome == BIOME_SAVANNA

    def test_biome_plains(self):
        """中湿度应为平原"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.45, 0.55, False)
        assert biome == BIOME_PLAINS

    def test_biome_forest(self):
        """中高湿度应为森林"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.60, 0.60, False)
        assert biome == BIOME_FOREST

    def test_biome_taiga(self):
        """中高湿度低温应为泰加林"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.60, 0.25, False)
        assert biome == BIOME_TAIGA

    def test_biome_dense_forest(self):
        """高湿度适中温度应为密林"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.80, 0.60, False)
        assert biome == BIOME_DENSE_FOREST

    def test_biome_rainforest(self):
        """高湿度高温应为雨林"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.85, 0.80, False)
        assert biome == BIOME_RAINFOREST

    def test_biome_swamp(self):
        """高湿度低海拔应为沼泽"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.35, 0.85, 0.40, False)
        assert biome == BIOME_SWAMP

    def test_biome_boundary_water_level(self):
        """恰好在 water_level 的边界值"""
        tg = TerrainGenerator()
        # 刚好低于水面
        biome_below = tg.classify_biome(tg.water_level - 0.001, 0.5, 0.5, False)
        assert biome_below == BIOME_LAKE
        # 刚好高于水面且非海岸
        biome_above = tg.classify_biome(tg.water_level + 0.001, 0.5, 0.5, False)
        assert biome_above != BIOME_LAKE


# ============================================================
# 3) PerlinNoise 噪声测试
# ============================================================
class TestPerlinNoise:
    def test_noise_range(self):
        """噪声值应在合理范围"""
        pn = PerlinNoise(seed=42)
        values = [pn.noise2d(x * 0.1, x * 0.1) for x in range(100)]
        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_noise_determinism(self):
        """相同种子应产生相同结果"""
        pn1 = PerlinNoise(seed=123)
        pn2 = PerlinNoise(seed=123)
        for x in range(10):
            for y in range(10):
                assert pn1.noise2d(x * 0.5, y * 0.5) == pn2.noise2d(x * 0.5, y * 0.5)

    def test_octave_noise_range(self):
        """多八度噪声应归一化到合理范围"""
        pn = PerlinNoise(seed=42)
        values = [pn.octave_noise(x * 0.1, x * 0.1) for x in range(100)]
        assert min(values) >= -1.2
        assert max(values) <= 1.2

    def test_smoothstep_monotonic(self):
        """平滑步进函数应单调递增"""
        pn = PerlinNoise(seed=42)
        prev = -1.0
        for t in np.linspace(0, 1, 50):
            val = pn._smoothstep(t)
            assert val >= prev
            prev = val

    def test_lerp_basic(self):
        """线性插值应正确工作"""
        pn = PerlinNoise(seed=42)
        assert pn._lerp(0, 10, 0) == 0
        assert pn._lerp(0, 10, 1) == 10
        assert pn._lerp(0, 10, 0.5) == 5


# ============================================================
# 4) FeatureGenerator 核心算法测试
# ============================================================
class TestFeatureGenerator:
    def test_settlement_score_biome(self):
        """聚落评分应正确反映生物群落偏好"""
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.3
        td.is_coast = False
        td.river_flow = 0

        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score == 10.0

    def test_settlement_score_coast_bonus(self):
        """海岸应有加成"""
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.3
        td.is_coast = True
        td.river_flow = 0

        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score == 15.0  # 10 (plains) + 5 (coast)

    def test_settlement_score_river_bonus(self):
        """河流应有加成"""
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.3
        td.is_coast = False
        td.river_flow = 0.5

        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score == 14.0  # 10 (plains) + 4 (river)

    def test_settlement_score_high_elevation_penalty(self):
        """高海拔应有惩罚"""
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.7  # > 0.55
        td.is_coast = False
        td.river_flow = 0

        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        # 10 - (0.7-0.55)*15 = 10 - 2.25 = 7.75
        assert abs(score - 7.75) < 0.01

    def test_find_road_path_simple(self):
        """简单路径查找"""
        terrain_data = {}
        hex_coords_list = []

        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2:
                    hc = HexCoord(q, r)
                    td = TerrainData()
                    td.biome = "plains"
                    td.is_water = False
                    terrain_data[hc] = td
                    hex_coords_list.append((hc, float(q), float(r)))

        fg = FeatureGenerator(terrain_data, hex_coords_list)
        path = fg._find_road_path(HexCoord(-2, 0), HexCoord(2, 0))
        assert len(path) >= 5  # 至少需要5步

    def test_find_road_path_blocked_by_water(self):
        """路径应避开水域"""
        terrain_data = {}
        hex_coords_list = []

        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2:
                    hc = HexCoord(q, r)
                    td = TerrainData()
                    # 在中间设置水域障碍
                    if q == 0 and abs(r) <= 1:
                        td.biome = "ocean"
                        td.is_water = True
                    else:
                        td.biome = "plains"
                        td.is_water = False
                    terrain_data[hc] = td
                    hex_coords_list.append((hc, float(q), float(r)))

        fg = FeatureGenerator(terrain_data, hex_coords_list)
        path = fg._find_road_path(HexCoord(-2, 0), HexCoord(2, 0))
        # 应能绕开水域找到路径
        assert path is not None
        for hc in path:
            assert not terrain_data[hc].is_water


# ============================================================
# 5) HexCoord 坐标运算测试
# ============================================================
class TestHexCoordOperations:
    def test_distance(self):
        """距离计算应正确"""
        h1 = HexCoord(0, 0)
        h2 = HexCoord(3, -1)
        assert h1.distance_to(h2) == 3

    def test_addition(self):
        """坐标加法"""
        h1 = HexCoord(2, 3)
        h2 = HexCoord(-1, 4)
        result = h1 + h2
        assert result == HexCoord(1, 7)

    def test_subtraction(self):
        """坐标减法"""
        h1 = HexCoord(5, 2)
        h2 = HexCoord(3, -1)
        result = h1 - h2
        assert result == HexCoord(2, 3)

    def test_multiplication(self):
        """坐标标量乘法"""
        h = HexCoord(2, -3)
        result = h * 3
        assert result == HexCoord(6, -9)

    def test_neighbors_count(self):
        """邻居数量应为6"""
        h = HexCoord(0, 0)
        neighbors = h.neighbors()
        assert len(neighbors) == 6

    def test_neighbors_unique(self):
        """邻居应互不相同"""
        h = HexCoord(0, 0)
        neighbors = h.neighbors()
        assert len(set(neighbors)) == 6


# ============================================================
# 6) 极端边界条件测试
# ============================================================
class TestEdgeCases:
    def test_empty_hex_list(self):
        """空六边形列表不应崩溃"""
        tg = TerrainGenerator()
        result = tg.generate(
            np.array([]), np.array([]), np.array([]), []
        )
        assert result == {}

    def test_single_hex(self):
        """单个六边形应正常处理"""
        tg = TerrainGenerator()
        hc = HexCoord(0, 0)
        result = tg.generate(
            np.array([0.5]), np.array([0.5]), np.array([0.5]), [(hc, 0.0, 0.0)]
        )
        assert len(result) == 1
        assert hc in result

    def test_minimal_grid(self):
        """最小网格应正常工作"""
        grid = HexGrid(size=0)
        assert len(grid.hexes) == 1
        assert HexCoord(0, 0) in grid.hexes

    def test_large_grid(self):
        """较大网格应能生成"""
        grid = HexGrid(size=5)
        # 六边形数量 = 1 + 6*(1+2+3+4+5) = 1 + 6*15 = 91
        assert len(grid.hexes) == 91

    def test_noise_seed_zero(self):
        """种子为0应正常工作"""
        pn = PerlinNoise(seed=0)
        val = pn.noise2d(0.5, 0.5)
        assert isinstance(val, float)

    def test_generate_rivers_no_candidates(self):
        """没有候选起点时不应崩溃"""
        terrain_data = {}
        hex_coords_list = []
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        rivers = fg.generate_rivers(np.random.default_rng(42), num_rivers=5)
        assert rivers == []

    def test_generate_settlements_no_land(self):
        """没有陆地时不应崩溃"""
        terrain_data = {}
        hex_coords_list = []
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        settlements = fg.generate_settlements(np.random.default_rng(42))
        assert settlements == []