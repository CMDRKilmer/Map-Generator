"""
测试缺口分析补充套件 — 覆盖核心业务逻辑中未被测试的高风险路径
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pytest

from core.feature_gen import FeatureGenerator
from core.hex_grid import HEX_DIRECTIONS, HexCoord, HexGrid
from core.noise_gen import NoiseGenerator, PerlinNoise
from core.terrain_gen import (
    BIOME_BEACH,
    BIOME_DESERT,
    BIOME_DENSE_FOREST,
    BIOME_FOREST,
    BIOME_HIGH_MOUNTAINS,
    BIOME_HILLS,
    BIOME_LAKE,
    BIOME_MOUNTAINS,
    BIOME_OCEAN,
    BIOME_PLAINS,
    BIOME_RAINFOREST,
    BIOME_SAVANNA,
    BIOME_SNOW,
    BIOME_SWAMP,
    BIOME_TAIGA,
    BIOME_TUNDRA,
    BIOME_VOLCANO,
    RESOURCE_FOOD,
    RESOURCE_GOLD,
    RESOURCE_IRON,
    RESOURCE_STONE,
    RESOURCE_WOOD,
    SETTLEMENT_CAPITAL,
    SETTLEMENT_CITY,
    SETTLEMENT_NONE,
    SETTLEMENT_TOWN,
    SETTLEMENT_VILLAGE,
    TerrainData,
    TerrainGenerator,
)


# ============================================================
# 1) HexCoord — 核心坐标运算 (下游广泛使用的共享工具)
# ============================================================
class TestHexCoordOperations:
    def test_equality_and_hash(self):
        """相同坐标应相等且哈希一致"""
        a = HexCoord(2, 3)
        b = HexCoord(2, 3)
        c = HexCoord(3, 2)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert a != "not a hex"

    def test_s_property(self):
        """s = -q - r"""
        hc = HexCoord(3, -5)
        assert hc.s == 2
        assert HexCoord(0, 0).s == 0

    def test_addition(self):
        a = HexCoord(1, 2)
        b = HexCoord(3, 4)
        result = a + b
        assert result.q == 4 and result.r == 6

    def test_subtraction(self):
        a = HexCoord(5, 3)
        b = HexCoord(2, 1)
        result = a - b
        assert result.q == 3 and result.r == 2

    def test_multiplication(self):
        hc = HexCoord(2, -1)
        result = hc * 3
        assert result.q == 6 and result.r == -3

    def test_distance_to(self):
        """六边形距离 = max(|dq|, |dr|, |ds|)"""
        assert HexCoord(0, 0).distance_to(HexCoord(0, 0)) == 0
        assert HexCoord(0, 0).distance_to(HexCoord(1, 0)) == 1
        assert HexCoord(0, 0).distance_to(HexCoord(2, -1)) == 2
        assert HexCoord(3, -2).distance_to(HexCoord(-1, 4)) == 6

    def test_neighbors_count_and_identity(self):
        """6 个邻居，且距离都为 1"""
        hc = HexCoord(0, 0)
        neighbors = hc.neighbors()
        assert len(neighbors) == 6
        for n in neighbors:
            assert hc.distance_to(n) == 1

    def test_neighbors_match_constants(self):
        """neighbors() 结果与 HEX_DIRECTIONS 一致"""
        hc = HexCoord(5, 7)
        neighbors = hc.neighbors()
        expected = [hc + d for d in HEX_DIRECTIONS]
        assert set(neighbors) == set(expected)

    def test_range_size(self):
        """range(n) 应包含 1 + 3n(n+1) 个六边形"""
        for n in range(0, 5):
            hexes = HexCoord(0, 0).range(n)
            expected_count = 1 + 3 * n * (n + 1)
            assert len(hexes) == expected_count
            for h in hexes:
                assert HexCoord(0, 0).distance_to(h) <= n

    def test_range_centered(self):
        """range 应以原点为中心"""
        center = HexCoord(2, 3)
        hexes = center.range(2)
        assert center in hexes
        for h in hexes:
            assert center.distance_to(h) <= 2

    def test_to_cube_roundtrip(self):
        """to_cube + from_cube 应返回原值"""
        original = HexCoord(3, -2)
        q, r, s = original.to_cube()
        assert q + r + s == 0
        restored = HexCoord.from_cube(q, r, s)
        assert restored == original

    def test_repr(self):
        assert repr(HexCoord(1, 2)) == "Hex(1,2)"


# ============================================================
# 2) HexGrid — 网格生成与坐标转换
# ============================================================
class TestHexGrid:
    def test_grid_size_zero(self):
        grid = HexGrid(size=0)
        assert len(grid.hexes) == 1
        assert grid.hexes[0] == HexCoord(0, 0)

    def test_grid_size_count(self):
        """size N 的网格应有 1 + 3N(N+1) 个六边形"""
        for size in [1, 2, 3, 5]:
            grid = HexGrid(size=size)
            expected = 1 + 3 * size * (size + 1)
            assert len(grid.hexes) == expected

    def test_hex_center_origin(self):
        grid = HexGrid(size=1)
        cx, cy = grid.hex_center(HexCoord(0, 0), 10.0)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_hex_center_q_axis(self):
        """q 轴方向: x = size * 3/2 * q, y 随 q 变化"""
        grid = HexGrid(size=2)
        cx, cy = grid.hex_center(HexCoord(2, 0), 10.0)
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(10.0 * math.sqrt(3) / 2.0 * 2)

    def test_hex_corners_count(self):
        grid = HexGrid(size=1)
        corners = grid.hex_corners(HexCoord(0, 0), 10.0)
        assert len(corners) == 6

    def test_hex_corners_closed_polygon(self):
        """六边形首尾顶点距离应约等于边长"""
        grid = HexGrid(size=1)
        corners = grid.hex_corners(HexCoord(0, 0), 10.0)
        x0, y0 = corners[0]
        x1, y1 = corners[1]
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        assert dist == pytest.approx(10.0, rel=0.01)

    def test_hex_round_exact(self):
        """精确的六边形中心坐标应 round 回自身"""
        grid = HexGrid(size=3)
        for hc in grid.hexes[:20]:
            cx, cy = grid.hex_center(hc, 15.0)
            result = grid.hex_at_pixel(cx, cy, 15.0)
            assert result == hc

    def test_all_hex_centers_length(self):
        grid = HexGrid(size=3)
        centers = grid.all_hex_centers(10.0)
        assert len(centers) == len(grid.hexes)
        for hc, cx, cy in centers:
            assert isinstance(hc, HexCoord)

    def test_get_random_hex(self):
        grid = HexGrid(size=5)
        rng = np.random.default_rng(42)
        for _ in range(20):
            hc = grid.get_random_hex(rng)
            assert hc in grid.hexes

    def test_get_hexes_within_radius(self):
        grid = HexGrid(size=5)
        center = HexCoord(0, 0)
        hexes = grid.get_hexes_within_radius(center, 2)
        assert len(hexes) == 1 + 3 * 2 * 3
        for h in hexes:
            assert center.distance_to(h) <= 2


# ============================================================
# 3) classify_biome — 生物群落分类 (复杂决策逻辑, 边界条件)
# ============================================================
class TestClassifyBiome:
    def setup_method(self):
        self.tg = TerrainGenerator()

    def test_deep_ocean(self):
        """高程 < 0.15 应为深海"""
        assert self.tg.classify_biome(0.0, 0.5, 0.5, False) == BIOME_OCEAN
        assert self.tg.classify_biome(0.1, 0.8, 0.7, False) == BIOME_OCEAN

    def test_shallow_lake(self):
        """0.15 <= 高程 < water_level 应为湖泊"""
        mid = (0.15 + self.tg.water_level) / 2
        assert self.tg.classify_biome(mid, 0.5, 0.5, False) == BIOME_LAKE

    def test_beach_coast_low_elevation(self):
        """海岸且高程略高于水面应为沙滩"""
        beach_elev = self.tg.water_level + 0.05
        assert self.tg.classify_biome(beach_elev, 0.5, 0.5, True) == BIOME_BEACH

    def test_no_beach_without_coast(self):
        """非海岸即使高程合适也不是沙滩"""
        beach_elev = self.tg.water_level + 0.05
        result = self.tg.classify_biome(beach_elev, 0.5, 0.5, False)
        assert result != BIOME_BEACH

    def test_snow_high_elevation_low_temp(self):
        """高海拔 + 低温 = 雪地"""
        assert self.tg.classify_biome(0.9, 0.5, 0.2, False) == BIOME_SNOW

    def test_high_mountains(self):
        """高程 > 0.80 应为高山"""
        result = self.tg.classify_biome(0.85, 0.5, 0.5, False)
        assert result == BIOME_HIGH_MOUNTAINS

    def test_mountains(self):
        """0.65 < 高程 <= 0.80 应为山脉"""
        assert self.tg.classify_biome(0.75, 0.5, 0.5, False) == BIOME_MOUNTAINS

    def test_hills(self):
        """0.55 < 高程 <= 0.65 应为丘陵"""
        assert self.tg.classify_biome(0.60, 0.5, 0.5, False) == BIOME_HILLS

    def test_desert_low_moisture_high_temp(self):
        """低湿度 + 高温 = 沙漠"""
        assert self.tg.classify_biome(0.4, 0.1, 0.8, False) == BIOME_DESERT

    def test_tundra_low_moisture_low_temp(self):
        """低湿度 + 低温 = 苔原"""
        assert self.tg.classify_biome(0.4, 0.1, 0.2, False) == BIOME_TUNDRA

    def test_savanna_medium_low_moisture_high_temp(self):
        """中低湿度 + 高温 = 稀树草原"""
        assert self.tg.classify_biome(0.4, 0.3, 0.8, False) == BIOME_SAVANNA

    def test_plains_medium_moisture_high_temp(self):
        """中湿度 + 高温 = 平原"""
        assert self.tg.classify_biome(0.4, 0.45, 0.6, False) == BIOME_PLAINS

    def test_taiga_medium_moisture_low_temp(self):
        """中湿度 + 低温 = 针叶林"""
        assert self.tg.classify_biome(0.4, 0.45, 0.2, False) == BIOME_TAIGA

    def test_forest_mid_high_moisture(self):
        """中高湿度 + 中温 = 森林"""
        assert self.tg.classify_biome(0.4, 0.65, 0.5, False) == BIOME_FOREST

    def test_rainforest_high_moisture_high_temp(self):
        """高湿度 + 高温 = 热带雨林"""
        assert self.tg.classify_biome(0.4, 0.85, 0.9, False) == BIOME_RAINFOREST

    def test_dense_forest_high_moisture_medium_temp(self):
        """高湿度 + 中温 = 密林"""
        assert self.tg.classify_biome(0.4, 0.85, 0.6, False) == BIOME_DENSE_FOREST

    def test_swamp(self):
        """高湿度 + 低海拔 = 沼泽"""
        assert self.tg.classify_biome(0.38, 0.85, 0.4, False) == BIOME_SWAMP

    def test_boundary_water_level(self):
        """水面边界附近的分类"""
        wl = self.tg.water_level
        assert self.tg.classify_biome(wl - 0.01, 0.5, 0.5, False) == BIOME_LAKE
        land_biome = self.tg.classify_biome(wl + 0.01, 0.5, 0.5, False)
        assert land_biome not in (BIOME_OCEAN, BIOME_LAKE)

    def test_boundary_mountain(self):
        """山脉边界"""
        assert self.tg.classify_biome(0.66, 0.5, 0.5, False) == BIOME_MOUNTAINS
        assert self.tg.classify_biome(0.56, 0.5, 0.5, False) == BIOME_HILLS


# ============================================================
# 4) TerrainGenerator.generate — 海岸检测与火山
# ============================================================
class TestTerrainGeneration:
    def test_coast_detection_water_adjacent_to_land(self):
        """水格相邻有陆地时 is_coast 应为 True"""
        tg = TerrainGenerator()
        grid = HexGrid(size=2)
        hex_list = grid.all_hex_centers(10.0)
        n = len(hex_list)
        coord_map = {hc: i for i, (hc, _, _) in enumerate(hex_list)}
        elev = np.full(n, 0.5)
        center_idx = coord_map[HexCoord(0, 0)]
        elev[center_idx] = 0.1
        moist = np.full(n, 0.5)
        temp = np.full(n, 0.5)
        td = tg.generate(elev, moist, temp, hex_list)
        center_hc = HexCoord(0, 0)
        assert td[center_hc].is_water
        assert td[center_hc].is_coast

    def test_is_water_flag(self):
        """低于水位的格子 is_water 应为 True"""
        tg = TerrainGenerator()
        grid = HexGrid(size=2)
        hex_list = grid.all_hex_centers(10.0)
        n = len(hex_list)
        elev = np.zeros(n)
        moist = np.full(n, 0.5)
        temp = np.full(n, 0.5)
        td = tg.generate(elev, moist, temp, hex_list)
        for hc, data in td.items():
            assert data.is_water
            assert data.biome in (BIOME_OCEAN, BIOME_LAKE)

    def test_terrain_data_attributes(self):
        """TerrainData 应包含所有预期属性"""
        td = TerrainData()
        assert hasattr(td, "elevation")
        assert hasattr(td, "moisture")
        assert hasattr(td, "temperature")
        assert hasattr(td, "biome")
        assert hasattr(td, "is_water")
        assert hasattr(td, "is_coast")
        assert hasattr(td, "river_flow")
        assert hasattr(td, "resource")
        assert hasattr(td, "resource_amount")
        assert hasattr(td, "settlement")
        assert hasattr(td, "settlement_name")
        assert hasattr(td, "settlement_size")
        assert hasattr(td, "road")
        assert hasattr(td, "shipping")
        assert hasattr(td, "volcanic")

    def test_volcano_deterministic_with_seed(self):
        """相同输入应产生相同的火山分布"""
        tg = TerrainGenerator()
        grid = HexGrid(size=5)
        hex_list = grid.all_hex_centers(10.0)
        n = len(hex_list)
        rng = np.random.default_rng(123)
        elev = rng.uniform(0, 1, n)
        moist = rng.uniform(0, 1, n)
        temp = rng.uniform(0, 1, n)
        td1 = tg.generate(elev, moist, temp, hex_list)
        td2 = tg.generate(elev, moist, temp, hex_list)
        volcanic1 = {hc for hc, d in td1.items() if d.volcanic}
        volcanic2 = {hc for hc, d in td2.items() if d.volcanic}
        assert volcanic1 == volcanic2


# ============================================================
# 5) FeatureGenerator — 聚落评分
# ============================================================
class TestSettlementScore:
    def _make_fg_with_terrain(self, biome: str, is_coast: bool = False, river_flow: float = 0.0):
        hc = HexCoord(0, 0)
        td = TerrainData()
        td.biome = biome
        td.is_coast = is_coast
        td.river_flow = river_flow
        td.elevation = 0.3
        terrain_data = {hc: td}
        hex_list = [(hc, 0.0, 0.0)]
        return FeatureGenerator(terrain_data, hex_list), hc, td

    def test_plains_high_score(self):
        fg, hc, td = self._make_fg_with_terrain("plains")
        score = fg._settlement_score(td, hc)
        assert score > 5

    def test_tundra_low_score(self):
        fg, hc, td = self._make_fg_with_terrain("tundra")
        score = fg._settlement_score(td, hc)
        assert score < 5

    def test_coast_bonus(self):
        fg1, hc1, td1 = self._make_fg_with_terrain("plains", is_coast=False)
        fg2, hc2, td2 = self._make_fg_with_terrain("plains", is_coast=True)
        assert fg2._settlement_score(td2, hc2) > fg1._settlement_score(td1, hc1)

    def test_river_bonus(self):
        fg1, hc1, td1 = self._make_fg_with_terrain("plains", river_flow=0.0)
        fg2, hc2, td2 = self._make_fg_with_terrain("plains", river_flow=0.5)
        assert fg2._settlement_score(td2, hc2) > fg1._settlement_score(td1, hc1)

    def test_high_elevation_penalty(self):
        hc = HexCoord(0, 0)
        td_low = TerrainData()
        td_low.biome = "plains"
        td_low.elevation = 0.3
        td_high = TerrainData()
        td_high.biome = "plains"
        td_high.elevation = 0.8
        terrain_data = {hc: td_low}
        fg = FeatureGenerator(terrain_data, [(hc, 0.0, 0.0)])
        score_low = fg._settlement_score(td_low, hc)
        score_high = fg._settlement_score(td_high, hc)
        assert score_high < score_low

    def test_volcanic_excluded_from_settlements(self):
        """火山地形不应被选为聚落地"""
        hc = HexCoord(0, 0)
        td = TerrainData()
        td.biome = "volcano"
        td.volcanic = True
        td.elevation = 0.7
        terrain_data = {hc: td}
        fg = FeatureGenerator(terrain_data, [(hc, 0.0, 0.0)])
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(rng, num_villages=1)
        assert len(settlements) == 0


# ============================================================
# 6) FeatureGenerator — 河流追踪
# ============================================================
class TestRiverTracing:
    def test_river_flows_to_water(self):
        """河流应从高地流向水体"""
        grid = HexGrid(size=4)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            dist = hc.distance_to(HexCoord(0, 0))
            td.elevation = 0.9 - dist * 0.15
            td.is_water = td.elevation < 0.35
            td.biome = BIOME_OCEAN if td.is_water else BIOME_PLAINS
            terrain_data[hc] = td
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=5)
        assert len(rivers) > 0
        for river in rivers:
            assert len(river) > 3
            assert terrain_data[river[-1]].is_water
            for hc in river:
                assert terrain_data[hc].river_flow >= 0.0

    def test_no_rivers_when_no_high_land(self):
        """没有高地时不生成河流"""
        hc = HexCoord(0, 0)
        td = TerrainData()
        td.elevation = 0.3
        td.is_water = False
        td.biome = BIOME_PLAINS
        terrain_data = {hc: td}
        fg = FeatureGenerator(terrain_data, [(hc, 0.0, 0.0)])
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=5)
        assert rivers == []

    def test_trace_river_no_infinite_loop(self):
        """河流追踪不应超过 max_steps"""
        grid = HexGrid(size=2)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            td.elevation = 0.6
            td.is_water = False
            td.biome = BIOME_HILLS
            terrain_data[hc] = td
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        start = HexCoord(0, 0)
        path = fg._trace_river(start, rng)
        assert len(path) <= 200 + 1


# ============================================================
# 7) FeatureGenerator — 道路 A* 寻路
# ============================================================
class TestRoadPathfinding:
    def _make_flat_land_grid(self, size=5):
        grid = HexGrid(size=size)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            td.elevation = 0.5
            td.is_water = False
            td.biome = BIOME_PLAINS
            terrain_data[hc] = td
        return FeatureGenerator(terrain_data, hex_list)

    def test_find_path_adjacent(self):
        """相邻六边形之间的路径长度应为 2"""
        fg = self._make_flat_land_grid(size=2)
        path = fg._find_road_path(HexCoord(0, 0), HexCoord(1, 0))
        assert len(path) == 2
        assert path[0] == HexCoord(0, 0)
        assert path[-1] == HexCoord(1, 0)

    def test_find_path_distance(self):
        """路径长度应约等于六边形距离+1"""
        fg = self._make_flat_land_grid(size=5)
        start = HexCoord(-3, 1)
        end = HexCoord(2, -1)
        path = fg._find_road_path(start, end)
        assert len(path) >= start.distance_to(end) + 1
        assert path[0] == start
        assert path[-1] == end

    def test_find_path_avoids_water(self):
        """道路应避开水域"""
        grid = HexGrid(size=3)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            if abs(hc.q) <= 1 and hc.r == 0:
                td.elevation = 0.1
                td.is_water = True
                td.biome = BIOME_OCEAN
            else:
                td.elevation = 0.5
                td.is_water = False
                td.biome = BIOME_PLAINS
            terrain_data[hc] = td
        fg = FeatureGenerator(terrain_data, hex_list)
        start = HexCoord(-2, 0)
        end = HexCoord(2, 0)
        path = fg._find_road_path(start, end)
        assert len(path) > 0
        for hc in path[1:-1]:
            assert not terrain_data[hc].is_water

    def test_roads_connect_settlements(self):
        """生成道路后聚落间应被连接"""
        grid = HexGrid(size=4)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        land_hexes = []
        for hc, _, _ in hex_list:
            td = TerrainData()
            td.elevation = 0.5
            td.is_water = False
            td.biome = BIOME_PLAINS
            terrain_data[hc] = td
            land_hexes.append(hc)
        terrain_data[land_hexes[0]].settlement = SETTLEMENT_CITY
        terrain_data[land_hexes[0]].settlement_size = 4
        terrain_data[land_hexes[10]].settlement = SETTLEMENT_TOWN
        terrain_data[land_hexes[10]].settlement_size = 3
        fg = FeatureGenerator(terrain_data, hex_list)
        roads = fg.generate_roads()
        assert len(roads) >= 1
        assert any(td.road for td in terrain_data.values())


# ============================================================
# 8) FeatureGenerator — 资源生成
# ============================================================
class TestResourceGeneration:
    def test_resources_only_on_land(self):
        """资源只应生成在陆地上"""
        grid = HexGrid(size=3)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            if hc.q > 0:
                td.elevation = 0.1
                td.is_water = True
                td.biome = BIOME_OCEAN
            else:
                td.elevation = 0.5
                td.is_water = False
                td.biome = BIOME_FOREST
            terrain_data[hc] = td
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_resources(rng, density=1.0)
        for hc, td in terrain_data.items():
            if td.is_water:
                assert td.resource is None

    def test_resource_biome_matching(self):
        """森林应产生木材，山脉应产生铁/石"""
        hc_forest = HexCoord(0, 0)
        td_forest = TerrainData()
        td_forest.biome = "forest"
        td_forest.is_water = False
        td_forest.elevation = 0.4
        hc_mountain = HexCoord(1, 0)
        td_mountain = TerrainData()
        td_mountain.biome = "mountains"
        td_mountain.is_water = False
        td_mountain.elevation = 0.7
        terrain_data = {hc_forest: td_forest, hc_mountain: td_mountain}
        hex_list = [(hc_forest, 0.0, 0.0), (hc_mountain, 10.0, 0.0)]
        wood_count = 0
        iron_stone_count = 0
        for seed in range(50):
            fg = FeatureGenerator({k: v for k, v in terrain_data.items()}, hex_list)
            for hc in fg.terrain:
                fg.terrain[hc].resource = None
            rng = np.random.default_rng(seed)
            fg.generate_resources(rng, density=1.0)
            if fg.terrain[hc_forest].resource == RESOURCE_WOOD:
                wood_count += 1
            if fg.terrain[hc_mountain].resource in (RESOURCE_IRON, RESOURCE_STONE):
                iron_stone_count += 1
        assert wood_count > 0
        assert iron_stone_count > 0

    def test_volcano_has_minerals(self):
        """火山附近有 30% 概率生成矿物资源"""
        hc = HexCoord(0, 0)
        found_mineral = False
        for seed in range(100):
            td = TerrainData()
            td.biome = "volcano"
            td.volcanic = True
            td.is_water = False
            td.elevation = 0.8
            terrain_data = {hc: td}
            fg = FeatureGenerator(terrain_data, [(hc, 0.0, 0.0)])
            rng = np.random.default_rng(seed)
            fg.generate_resources(rng)
            if td.resource in (RESOURCE_IRON, RESOURCE_GOLD, RESOURCE_STONE):
                found_mineral = True
                assert td.resource_amount >= 2
                break
        assert found_mineral, "火山在 100 次尝试中应至少生成一次矿物"


# ============================================================
# 9) FeatureGenerator — 名称生成与航线
# ============================================================
class TestNameGeneration:
    def test_generate_name_returns_string(self):
        fg = FeatureGenerator({}, [])
        rng = np.random.default_rng(42)
        for typ in ("capital", "city", "town", "village"):
            name = fg._generate_name(rng, typ)
            assert isinstance(name, str)
            assert len(name) == 2

    def test_generate_name_deterministic(self):
        fg = FeatureGenerator({}, [])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert fg._generate_name(rng1, "city") == fg._generate_name(rng2, "city")


class TestShippingRoutes:
    def test_no_shipping_without_coastal_settlements(self):
        grid = HexGrid(size=3)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            td.is_water = False
            td.is_coast = False
            td.biome = BIOME_PLAINS
            td.settlement = SETTLEMENT_CITY
            td.elevation = 0.5
            terrain_data[hc] = td
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        routes = fg.generate_shipping_routes(rng)
        assert routes == []

    def test_find_nearest_water(self):
        grid = HexGrid(size=3)
        hex_list = grid.all_hex_centers(10.0)
        terrain_data = {}
        for hc, _, _ in hex_list:
            td = TerrainData()
            td.is_water = hc.q >= 2
            td.biome = BIOME_OCEAN if td.is_water else BIOME_PLAINS
            td.elevation = 0.1 if td.is_water else 0.5
            terrain_data[hc] = td
        fg = FeatureGenerator(terrain_data, hex_list)
        result = fg._find_nearest_water(HexCoord(0, 0))
        assert result is not None
        assert terrain_data[result].is_water


# ============================================================
# 10) NoiseGenerator — set_seed 与效果验证
# ============================================================
class TestNoiseGeneratorSetSeed:
    def test_set_seed_changes_output(self):
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.3, i * 0.7) for i in range(50)]
        e1 = ng.generate_elevation(coords)
        ng.set_seed(99)
        e2 = ng.generate_elevation(coords)
        assert not np.array_equal(e1, e2)

    def test_same_seed_same_output(self):
        ng1 = NoiseGenerator(seed=123)
        ng2 = NoiseGenerator(seed=123)
        coords = [(i * 0.1, i * 0.2) for i in range(10)]
        e1 = ng1.generate_elevation(coords)
        e2 = ng2.generate_elevation(coords)
        np.testing.assert_array_almost_equal(e1, e2)

    def test_perlin_noise_deterministic(self):
        p = PerlinNoise(seed=777)
        v1 = p.noise2d(0.5, 0.5)
        v2 = p.noise2d(0.5, 0.5)
        assert v1 == v2

    def test_smoothstep_bounds(self):
        p = PerlinNoise(seed=1)
        for t in [0.0, 0.5, 1.0]:
            v = p._smoothstep(t)
            assert 0.0 <= v <= 1.0

    def test_smoothstep_monotonic(self):
        p = PerlinNoise(seed=1)
        prev = -1.0
        for i in range(11):
            t = i / 10.0
            v = p._smoothstep(t)
            assert v >= prev
            prev = v
