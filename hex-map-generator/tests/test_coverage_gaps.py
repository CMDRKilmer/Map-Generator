"""
测试覆盖缺口补充 — 针对核心模块的边界条件和未覆盖功能
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.hex_grid import HexCoord, HexGrid
from core.noise_gen import PerlinNoise, NoiseGenerator
from core.terrain_gen import (
    BIOME_OCEAN,
    BIOME_PLAINS,
    BIOME_DESERT,
    BIOME_FOREST,
    BIOME_SNOW,
    BIOME_VOLCANO,
    TerrainGenerator,
)
from core.feature_gen import FeatureGenerator


# ============================================================
# 1) core/hex_grid.py — 核心方法边界条件测试
# ============================================================
class TestHexGridCoreMethods:
    def test_hex_center_origin(self):
        """原点坐标的六边形中心应为 (0, 0)"""
        grid = HexGrid(size=5)
        hc = HexCoord(0, 0)
        cx, cy = grid.hex_center(hc, 10)
        assert abs(cx) < 0.001 and abs(cy) < 0.001

    def test_hex_center_offset(self):
        """偏移坐标的六边形中心计算"""
        grid = HexGrid(size=5)
        hc = HexCoord(2, -1)
        cx, cy = grid.hex_center(hc, 10)
        # 平顶六边形: x = size * 3/2 * q, y = size * (sqrt(3)/2 * q + sqrt(3) * r)
        expected_x = 10 * 1.5 * 2
        expected_y = 10 * (math.sqrt(3) / 2 * 2 + math.sqrt(3) * (-1))
        assert abs(cx - expected_x) < 0.001
        assert abs(cy - expected_y) < 0.001

    def test_hex_corners_count(self):
        """六边形应有6个顶点"""
        grid = HexGrid(size=5)
        hc = HexCoord(0, 0)
        corners = grid.hex_corners(hc, 10)
        assert len(corners) == 6

    def test_hex_corners_symmetry(self):
        """六边形顶点应关于中心对称"""
        grid = HexGrid(size=5)
        hc = HexCoord(0, 0)
        corners = grid.hex_corners(hc, 10)
        cx, cy = grid.hex_center(hc, 10)
        for i, (x, y) in enumerate(corners):
            opposite_i = (i + 3) % 6
            ox, oy = corners[opposite_i]
            # 对角点应关于中心对称
            assert abs((x + ox) / 2 - cx) < 0.001
            assert abs((y + oy) / 2 - cy) < 0.001

    def test_hex_round_identity(self):
        """整数坐标四舍五入应保持不变"""
        grid = HexGrid(size=5)
        for q, r in [(0, 0), (1, 0), (0, 1), (-1, 0), (1, -1)]:
            result = grid._hex_round(float(q), float(r))
            assert result.q == q and result.r == r

    def test_hex_round_edge_cases(self):
        """边界坐标的四舍五入"""
        grid = HexGrid(size=5)
        # 边界情况：应舍入到最近的六边形
        hc = grid._hex_round(0.5, 0.5)
        assert hc in [HexCoord(0, 0), HexCoord(1, 0), HexCoord(0, 1)]

    def test_hex_at_pixel_reverse(self):
        """像素坐标转换应为 hex_center 的逆运算"""
        grid = HexGrid(size=5)
        hex_size = 20
        for hc in [HexCoord(0, 0), HexCoord(1, 0), HexCoord(2, -1), HexCoord(-1, 1)]:
            cx, cy = grid.hex_center(hc, hex_size)
            result = grid.hex_at_pixel(cx, cy, hex_size)
            assert result == hc

    def test_hex_distance(self):
        """距离计算应正确"""
        h1 = HexCoord(0, 0)
        h2 = HexCoord(3, -1)
        assert h1.distance_to(h2) == 3

    def test_hex_range(self):
        """range 方法应返回正确数量的六边形"""
        hc = HexCoord(0, 0)
        # 距离 ≤ 0: 只有自身
        assert len(hc.range(0)) == 1
        # 距离 ≤ 1: 1 + 6 = 7
        assert len(hc.range(1)) == 7
        # 距离 ≤ 2: 1 + 6 + 12 = 19
        assert len(hc.range(2)) == 19


# ============================================================
# 2) core/noise_gen.py — PerlinNoise 边界条件与稳定性
# ============================================================
class TestPerlinNoiseEdgeCases:
    def test_noise_zero_coords(self):
        """原点坐标的噪声值"""
        pn = PerlinNoise(seed=42)
        v = pn.noise2d(0.0, 0.0)
        assert -1.0 <= v <= 1.0

    def test_noise_extreme_coords(self):
        """极端坐标的噪声值应保持在范围内"""
        pn = PerlinNoise(seed=42)
        for x, y in [(1e6, 1e6), (-1e6, 1e6), (0, 1e6), (-1000, -1000)]:
            v = pn.noise2d(x, y)
            assert -1.0 <= v <= 1.0

    def test_noise_determinism(self):
        """相同种子应产生相同结果"""
        pn1 = PerlinNoise(seed=42)
        pn2 = PerlinNoise(seed=42)
        for x, y in [(0.5, 0.5), (1.23, 4.56), (-0.78, 9.01)]:
            assert abs(pn1.noise2d(x, y) - pn2.noise2d(x, y)) < 1e-10

    def test_octave_noise_range(self):
        """多八度噪声应在 [-1, 1] 范围内"""
        pn = PerlinNoise(seed=42)
        for _ in range(50):
            x = np.random.uniform(-100, 100)
            y = np.random.uniform(-100, 100)
            v = pn.octave_noise(x, y, octaves=6)
            assert -1.0 <= v <= 1.0

    def test_generate_elevation_empty(self):
        """空坐标列表应返回空数组"""
        ng = NoiseGenerator(seed=42)
        elev = ng.generate_elevation([])
        assert elev.shape == (0,)

    def test_generate_elevation_single(self):
        """单个坐标应返回单个值"""
        ng = NoiseGenerator(seed=42)
        elev = ng.generate_elevation([(0.0, 0.0)])
        assert elev.shape == (1,)
        assert 0.0 <= elev[0] <= 1.0

    def test_generate_moisture_monsoon_effect(self):
        """季风效应应影响湿度分布"""
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.5, 0) for i in range(20)]
        elev = np.ones(20) * 0.3  # 低高程，减少高程效应
        moist_east = ng.generate_moisture(coords, elev, monsoon_dir=90.0)  # 东风
        moist_west = ng.generate_moisture(coords, elev, monsoon_dir=270.0)  # 西风
        # 东风时东边更湿润，西风时西边更湿润
        assert moist_east[-1] > moist_east[0]
        assert moist_west[0] > moist_west[-1]


# ============================================================
# 3) core/terrain_gen.py — classify_biome 边界条件
# ============================================================
class TestTerrainClassification:
    def test_classify_ocean(self):
        """低高程应为海洋"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.1, 0.5, 0.5, False)
        assert biome == BIOME_OCEAN

    def test_classify_lake(self):
        """中等低高程应为湖泊"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.25, 0.5, 0.5, False)
        assert biome == BIOME_OCEAN or biome == "lake"

    def test_classify_beach(self):
        """海岸附近应为沙滩"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.38, 0.5, 0.5, True)
        assert biome == "beach"

    def test_classify_desert(self):
        """低湿度高温应为沙漠"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.4, 0.1, 0.8, False)
        assert biome == BIOME_DESERT

    def test_classify_snow(self):
        """高海拔低温应为雪地"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.9, 0.5, 0.1, False)
        assert biome == BIOME_SNOW

    def test_classify_forest(self):
        """中等高程湿度应为森林"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.4, 0.6, 0.5, False)
        assert biome == BIOME_FOREST

    def test_classify_plains(self):
        """中等高程中等湿度应为平原"""
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.4, 0.4, 0.6, False)
        assert biome == BIOME_PLAINS

    def test_classify_swamp(self):
        """低高程高湿度应为沼泽"""
        tg = TerrainGenerator()
        # elevation 需要 >= water_level(0.35) 才能进入沼泽判断分支
        biome = tg.classify_biome(0.38, 0.9, 0.4, False)
        assert biome == "swamp"

    def test_classify_volcano_condition(self):
        """火山需要特定条件"""
        tg = TerrainGenerator()
        # 高海拔低湿度才可能成为火山
        biome = tg.classify_biome(0.75, 0.1, 0.5, False)
        # classify_biome 本身不处理火山，火山是在 generate 中单独判断的
        assert biome in ("mountains", "high_mountains")


# ============================================================
# 4) core/feature_gen.py — 核心算法测试
# ============================================================
class TestFeatureGeneratorCore:
    def test_settlement_score_biome_preference(self):
        """聚落评分应正确反映生物群落偏好"""
        from core.terrain_gen import TerrainData

        td1 = TerrainData()
        td1.biome = "plains"
        td1.elevation = 0.3
        td1.is_coast = False
        td1.river_flow = 0

        td2 = TerrainData()
        td2.biome = "desert"
        td2.elevation = 0.3
        td2.is_coast = False
        td2.river_flow = 0

        fg = FeatureGenerator({}, [])
        score1 = fg._settlement_score(td1, HexCoord(0, 0))
        score2 = fg._settlement_score(td2, HexCoord(1, 0))

        assert score1 > score2  # 平原应比沙漠更适合建聚落

    def test_settlement_score_coast_bonus(self):
        """海岸应获得额外评分"""
        from core.terrain_gen import TerrainData

        td1 = TerrainData()
        td1.biome = "plains"
        td1.elevation = 0.3
        td1.is_coast = True
        td1.river_flow = 0

        td2 = TerrainData()
        td2.biome = "plains"
        td2.elevation = 0.3
        td2.is_coast = False
        td2.river_flow = 0

        fg = FeatureGenerator({}, [])
        score1 = fg._settlement_score(td1, HexCoord(0, 0))
        score2 = fg._settlement_score(td2, HexCoord(1, 0))

        assert score1 > score2  # 海岸平原应比内陆平原得分高

    def test_generate_rivers_no_candidates(self):
        """没有候选高地时应返回空列表"""
        from core.terrain_gen import TerrainData

        terrain_data = {}
        hex_list = []
        fg = FeatureGenerator(terrain_data, hex_list)
        rivers = fg.generate_rivers(np.random.default_rng(42), num_rivers=5)
        assert rivers == []

    def test_generate_roads_no_settlements(self):
        """没有聚落时应返回空列表"""
        from core.terrain_gen import TerrainData

        td = TerrainData()
        td.settlement = 0  # SETTLEMENT_NONE
        terrain_data = {HexCoord(0, 0): td}
        hex_list = [(HexCoord(0, 0), 0.0, 0.0)]
        fg = FeatureGenerator(terrain_data, hex_list)
        roads = fg.generate_roads()
        assert roads == []

    def test_find_road_path_simple(self):
        """简单路径查找"""
        from core.terrain_gen import TerrainData

        # 创建简单地形：平原连通
        terrain_data = {}
        hex_list = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                hc = HexCoord(q, r)
                td = TerrainData()
                td.biome = "plains"
                td.is_water = False
                terrain_data[hc] = td
                hex_list.append((hc, float(q), float(r)))

        fg = FeatureGenerator(terrain_data, hex_list)
        path = fg._find_road_path(HexCoord(-2, 0), HexCoord(2, 0))
        assert len(path) >= 5  # 至少需要5步

    def test_generate_name_format(self):
        """生成的聚落名称格式应正确"""
        fg = FeatureGenerator({}, [])
        rng = np.random.default_rng(42)
        for typ in ["capital", "city", "town", "village"]:
            name = fg._generate_name(rng, typ)
            assert len(name) == 2  # 应是两字符名称


# ============================================================
# 5) 端到端：完整流程稳定性
# ============================================================
class TestEndToEndStability:
    def test_full_pipeline_determinism(self):
        """相同种子应产生相同的地形数据"""
        from core.hex_grid import HexGrid

        seed = 42
        grid_size = 10

        # 第一次运行
        ng1 = NoiseGenerator(seed=seed)
        grid1 = HexGrid(size=grid_size)
        coords1 = [(float(hc.q), float(hc.r)) for hc in grid1.hexes]
        elev1 = ng1.generate_elevation(coords1)
        moist1 = ng1.generate_moisture(coords1, elev1)
        temp1 = ng1.generate_temperature(elev1, coords1)

        tg1 = TerrainGenerator()
        hex_list1 = [(hc, float(hc.q), float(hc.r)) for hc in grid1.hexes]
        terrain1 = tg1.generate(elev1, moist1, temp1, hex_list1)

        # 第二次运行
        ng2 = NoiseGenerator(seed=seed)
        grid2 = HexGrid(size=grid_size)
        coords2 = [(float(hc.q), float(hc.r)) for hc in grid2.hexes]
        elev2 = ng2.generate_elevation(coords2)
        moist2 = ng2.generate_moisture(coords2, elev2)
        temp2 = ng2.generate_temperature(elev2, coords2)

        tg2 = TerrainGenerator()
        hex_list2 = [(hc, float(hc.q), float(hc.r)) for hc in grid2.hexes]
        terrain2 = tg2.generate(elev2, moist2, temp2, hex_list2)

        # 验证结果相同
        np.testing.assert_array_equal(elev1, elev2)
        np.testing.assert_array_equal(moist1, moist2)
        np.testing.assert_array_equal(temp1, temp2)

        # 验证地形数据相同
        for hc in grid1.hexes:
            assert terrain1[hc].biome == terrain2[hc].biome
            assert abs(terrain1[hc].elevation - terrain2[hc].elevation) < 1e-10

    def test_different_seeds_different_results(self):
        """不同种子应产生不同的高程数据"""
        ng1 = NoiseGenerator(seed=42)
        ng2 = NoiseGenerator(seed=43)
        coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        elev1 = ng1.generate_elevation(coords)
        elev2 = ng2.generate_elevation(coords)
        # 至少有一个值不同
        assert not np.array_equal(elev1, elev2)