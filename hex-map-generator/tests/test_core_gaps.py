"""
测试缺口补充套件 — 覆盖核心模块的高风险未测逻辑路径

优先级排序：
1. HexCoord / HexGrid 核心运算（下游使用最广泛）
2. classify_biome 边界条件（复杂分类逻辑 + 多分支）
3. FeatureGenerator 复杂算法（A*寻路、河流追踪、聚落评分）
4. NoiseGenerator / TerrainData 边缘情况
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

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
# 1) HexCoord — 核心运算（下游使用最广泛的共享模块）
# ============================================================
class TestHexCoordCoreOps:
    def test_eq_same_coords(self):
        assert HexCoord(2, 3) == HexCoord(2, 3)

    def test_eq_diff_coords(self):
        assert HexCoord(2, 3) != HexCoord(2, 4)
        assert HexCoord(2, 3) != HexCoord(3, 3)

    def test_eq_non_hexcoord(self):
        assert HexCoord(0, 0) != (0, 0)
        assert HexCoord(0, 0) != "Hex(0,0)"
        assert HexCoord(0, 0) is not None

    def test_hash_consistency(self):
        h1 = HexCoord(5, -3)
        h2 = HexCoord(5, -3)
        assert hash(h1) == hash(h2)

    def test_hash_usable_in_dict(self):
        d = {}
        d[HexCoord(1, 2)] = "a"
        d[HexCoord(1, 2)] = "b"
        assert len(d) == 1
        assert d[HexCoord(1, 2)] == "b"

    def test_hash_usable_in_set(self):
        s = {HexCoord(0, 0), HexCoord(0, 0), HexCoord(1, 0)}
        assert len(s) == 2

    def test_add(self):
        result = HexCoord(1, 2) + HexCoord(3, 4)
        assert result == HexCoord(4, 6)

    def test_sub(self):
        result = HexCoord(5, 3) - HexCoord(2, 1)
        assert result == HexCoord(3, 2)

    def test_mul(self):
        result = HexCoord(2, -1) * 3
        assert result == HexCoord(6, -3)

    def test_s_property(self):
        h = HexCoord(2, -3)
        assert h.s == 1
        assert h.q + h.r + h.s == 0

    def test_s_zero(self):
        h = HexCoord(0, 0)
        assert h.s == 0

    def test_distance_to_same(self):
        h = HexCoord(3, -1)
        assert h.distance_to(h) == 0

    def test_distance_to_neighbor(self):
        h = HexCoord(0, 0)
        for d in HEX_DIRECTIONS:
            assert h.distance_to(h + d) == 1

    def test_distance_to_opposite(self):
        h = HexCoord(0, 0)
        assert h.distance_to(HexCoord(5, -3)) == max(5, 3, 2)

    def test_neighbors_count(self):
        h = HexCoord(0, 0)
        neighbors = h.neighbors()
        assert len(neighbors) == 6

    def test_neighbors_all_distance_one(self):
        h = HexCoord(2, -1)
        for n in h.neighbors():
            assert h.distance_to(n) == 1

    def test_neighbors_unique(self):
        h = HexCoord(0, 0)
        assert len(set(h.neighbors())) == 6

    def test_range_zero(self):
        h = HexCoord(3, 2)
        result = h.range(0)
        assert len(result) == 1
        assert result[0] == h

    def test_range_one(self):
        h = HexCoord(0, 0)
        result = h.range(1)
        assert len(result) == 7
        assert h in result
        for n in h.neighbors():
            assert n in result

    def test_range_size(self):
        h = HexCoord(0, 0)
        for n in range(6):
            result = h.range(n)
            expected = 1 + 3 * n * (n + 1)
            assert len(result) == expected

    def test_range_all_within_distance(self):
        h = HexCoord(2, -3)
        result = h.range(4)
        for r in result:
            assert h.distance_to(r) <= 4

    def test_to_cube(self):
        h = HexCoord(2, -3)
        q, r, s = h.to_cube()
        assert q == 2
        assert r == -3
        assert s == 1
        assert q + r + s == 0

    def test_repr(self):
        h = HexCoord(1, -2)
        assert repr(h) == "Hex(1,-2)"


# ============================================================
# 2) HexGrid — 核心操作（像素转换、网格生成）
# ============================================================
class TestHexGridCore:
    def test_grid_size_zero(self):
        grid = HexGrid(size=0)
        assert len(grid.hexes) == 1
        assert grid.hexes[0] == HexCoord(0, 0)

    def test_grid_size_one(self):
        grid = HexGrid(size=1)
        assert len(grid.hexes) == 7

    def test_grid_size_formula(self):
        for size in range(1, 8):
            grid = HexGrid(size=size)
            expected = 1 + 3 * size * (size + 1)
            assert len(grid.hexes) == expected

    def test_all_hexes_origin_within_size(self):
        grid = HexGrid(size=5)
        origin = HexCoord(0, 0)
        for h in grid.hexes:
            assert origin.distance_to(h) <= 5

    def test_get_random_hex_in_grid(self):
        grid = HexGrid(size=5)
        rng = np.random.default_rng(42)
        for _ in range(20):
            h = grid.get_random_hex(rng)
            assert h in grid.hexes

    def test_hex_center_origin(self):
        grid = HexGrid(size=3)
        cx, cy = grid.hex_center(HexCoord(0, 0), hex_size=10.0)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_hex_center_q_direction(self):
        grid = HexGrid(size=3)
        cx1, _ = grid.hex_center(HexCoord(1, 0), hex_size=10.0)
        cx2, _ = grid.hex_center(HexCoord(2, 0), hex_size=10.0)
        assert cx2 - cx1 == pytest.approx(15.0)

    def test_hex_corners_count(self):
        grid = HexGrid(size=3)
        corners = grid.hex_corners(HexCoord(0, 0), hex_size=10.0)
        assert len(corners) == 6

    def test_hex_corners_all_same_distance_from_center(self):
        grid = HexGrid(size=3)
        cx, cy = grid.hex_center(HexCoord(2, -1), hex_size=15.0)
        corners = grid.hex_corners(HexCoord(2, -1), hex_size=15.0)
        for x, y in corners:
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            assert dist == pytest.approx(15.0, abs=0.01)

    def test_hex_at_pixel_origin(self):
        grid = HexGrid(size=5)
        h = grid.hex_at_pixel(0.0, 0.0, hex_size=10.0)
        assert h == HexCoord(0, 0)

    def test_hex_round_round_trip(self):
        grid = HexGrid(size=5)
        hex_size = 10.0
        for h in grid.hexes[:30]:
            cx, cy = grid.hex_center(h, hex_size)
            result = grid.hex_at_pixel(cx, cy, hex_size)
            assert result == h

    def test_all_hex_centers_count(self):
        grid = HexGrid(size=3)
        centers = grid.all_hex_centers(hex_size=10.0)
        assert len(centers) == len(grid.hexes)

    def test_all_hex_centers_tuples(self):
        grid = HexGrid(size=2)
        centers = grid.all_hex_centers(hex_size=10.0)
        for hc, cx, cy in centers:
            assert isinstance(hc, HexCoord)
            assert isinstance(cx, float)
            assert isinstance(cy, float)

    def test_get_hexes_within_radius(self):
        grid = HexGrid(size=5)
        center = HexCoord(0, 0)
        result = grid.get_hexes_within_radius(center, 2)
        assert len(result) == 1 + 3 * 2 * 3
        for h in result:
            assert center.distance_to(h) <= 2


# ============================================================
# 3) classify_biome — 生物群落分类边界条件（高复杂度）
# ============================================================
class TestClassifyBiomeBoundaries:
    def setup_method(self):
        self.tg = TerrainGenerator()

    def test_ocean_deep(self):
        assert self.tg.classify_biome(0.0, 0.5, 0.5, False) == BIOME_OCEAN
        assert self.tg.classify_biome(0.1, 0.5, 0.5, False) == BIOME_OCEAN

    def test_lake_shallow_water(self):
        assert self.tg.classify_biome(0.25, 0.5, 0.5, False) == BIOME_LAKE

    def test_beach_coast_near_water(self):
        result = self.tg.classify_biome(0.38, 0.5, 0.5, True)
        assert result == BIOME_BEACH

    def test_beach_not_coast(self):
        result = self.tg.classify_biome(0.38, 0.5, 0.5, False)
        assert result != BIOME_BEACH

    def test_snow_high_elevation_cold(self):
        assert self.tg.classify_biome(0.9, 0.5, 0.2, False) == BIOME_SNOW

    def test_snow_warm_not_snow(self):
        result = self.tg.classify_biome(0.9, 0.5, 0.5, False)
        assert result != BIOME_SNOW

    def test_high_mountains(self):
        assert self.tg.classify_biome(0.9, 0.5, 0.5, False) == BIOME_HIGH_MOUNTAINS
        assert self.tg.classify_biome(0.85, 0.5, 0.5, False) == BIOME_HIGH_MOUNTAINS

    def test_mountains(self):
        assert self.tg.classify_biome(0.75, 0.5, 0.5, False) == BIOME_MOUNTAINS

    def test_hills(self):
        assert self.tg.classify_biome(0.60, 0.5, 0.5, False) == BIOME_HILLS

    def test_desert_hot_dry(self):
        assert self.tg.classify_biome(0.5, 0.1, 0.8, False) == BIOME_DESERT

    def test_tundra_cold_dry(self):
        assert self.tg.classify_biome(0.5, 0.1, 0.2, False) == BIOME_TUNDRA

    def test_savanna_hot_moderate_dry(self):
        assert self.tg.classify_biome(0.5, 0.3, 0.8, False) == BIOME_SAVANNA

    def test_plains_moderate_temp_moderate_dry(self):
        assert self.tg.classify_biome(0.5, 0.3, 0.4, False) == BIOME_PLAINS

    def test_plains_mid_moisture_warm(self):
        assert self.tg.classify_biome(0.5, 0.45, 0.6, False) == BIOME_PLAINS

    def test_taiga_cool_mid_moisture(self):
        assert self.tg.classify_biome(0.5, 0.45, 0.4, False) == BIOME_TAIGA

    def test_forest_warm_humid(self):
        assert self.tg.classify_biome(0.5, 0.65, 0.8, False) == BIOME_FOREST

    def test_forest_moderate_temp_humid(self):
        assert self.tg.classify_biome(0.5, 0.65, 0.5, False) == BIOME_FOREST

    def test_taiga_cool_humid(self):
        assert self.tg.classify_biome(0.5, 0.65, 0.2, False) == BIOME_TAIGA

    def test_rainforest_hot_very_humid(self):
        assert self.tg.classify_biome(0.5, 0.85, 0.9, False) == BIOME_RAINFOREST

    def test_dense_forest_warm_very_humid(self):
        assert self.tg.classify_biome(0.5, 0.85, 0.6, False) == BIOME_DENSE_FOREST

    def test_swamp_low_elevation_very_wet(self):
        assert self.tg.classify_biome(0.38, 0.85, 0.4, False) == BIOME_SWAMP

    def test_swamp_high_elevation_not_swamp(self):
        result = self.tg.classify_biome(0.6, 0.9, 0.4, False)
        assert result != BIOME_SWAMP

    def test_boundary_ocean_lake(self):
        r1 = self.tg.classify_biome(0.14, 0.5, 0.5, False)
        r2 = self.tg.classify_biome(0.16, 0.5, 0.5, False)
        assert r1 == BIOME_OCEAN
        assert r2 == BIOME_LAKE

    def test_boundary_lake_land(self):
        wl = self.tg.water_level
        r1 = self.tg.classify_biome(wl - 0.01, 0.5, 0.5, False)
        r2 = self.tg.classify_biome(wl + 0.01, 0.5, 0.5, False)
        assert r1 == BIOME_LAKE
        assert r2 != BIOME_LAKE


# ============================================================
# 4) TerrainData 数据类
# ============================================================
class TestTerrainData:
    def test_default_values(self):
        td = TerrainData()
        assert td.elevation == 0.0
        assert td.moisture == 0.0
        assert td.temperature == 0.5
        assert td.biome == BIOME_OCEAN
        assert td.is_water is True
        assert td.is_coast is False
        assert td.river_flow == 0.0
        assert td.resource is None
        assert td.resource_amount == 0
        assert td.settlement == SETTLEMENT_NONE
        assert td.settlement_name == ""
        assert td.settlement_size == 1
        assert td.road is False
        assert td.shipping is False
        assert td.volcanic is False

    def test_repr(self):
        td = TerrainData()
        td.elevation = 0.42
        td.biome = BIOME_FOREST
        assert "forest" in repr(td)
        assert "0.42" in repr(td)


# ============================================================
# 5) NoiseGenerator — 边缘情况与确定性
# ============================================================
class TestNoiseGeneratorEdgeCases:
    def test_set_seed_changes_output(self):
        ng = NoiseGenerator(seed=42)
        coords = [(0.0, 0.0), (1.0, 1.0)]
        e1 = ng.generate_elevation(coords)
        ng.set_seed(123)
        e2 = ng.generate_elevation(coords)
        assert not np.array_equal(e1, e2)

    def test_set_seed_deterministic(self):
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.1, i * 0.2) for i in range(10)]
        e1 = ng.generate_elevation(coords)
        ng.set_seed(42)
        e2 = ng.generate_elevation(coords)
        np.testing.assert_array_equal(e1, e2)

    def test_generate_elevation_single_coord(self):
        ng = NoiseGenerator(seed=42)
        elev = ng.generate_elevation([(5.0, 5.0)])
        assert elev.shape == (1,)
        assert 0.0 <= elev[0] <= 1.0

    def test_generate_moisture_single_coord(self):
        ng = NoiseGenerator(seed=42)
        elev = np.array([0.5])
        moist = ng.generate_moisture([(0.0, 0.0)], elev)
        assert moist.shape == (1,)
        assert 0.0 <= moist[0] <= 1.0

    def test_generate_temperature_single_coord(self):
        ng = NoiseGenerator(seed=42)
        elev = np.array([0.5])
        temp = ng.generate_temperature(elev, [(0.0, 0.0)])
        assert temp.shape == (1,)
        assert 0.0 <= temp[0] <= 1.0

    def test_elevation_normalized_range(self):
        ng = NoiseGenerator(seed=999)
        coords = [(i * 0.2, j * 0.2) for i in range(10) for j in range(10)]
        elev = ng.generate_elevation(coords)
        assert elev.min() >= 0.0
        assert elev.max() <= 1.0

    def test_moisture_clipped_range(self):
        ng = NoiseGenerator(seed=123)
        coords = [(i * 0.2, j * 0.2) for i in range(10) for j in range(10)]
        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev)
        assert moist.min() >= 0.0
        assert moist.max() <= 1.0

    def test_temperature_clipped_range(self):
        ng = NoiseGenerator(seed=456)
        coords = [(i * 0.2, j * 0.2) for i in range(10) for j in range(10)]
        elev = ng.generate_elevation(coords)
        temp = ng.generate_temperature(elev, coords)
        assert temp.min() >= 0.0
        assert temp.max() <= 1.0

    def test_perlin_noise_deterministic(self):
        p1 = PerlinNoise(seed=42)
        p2 = PerlinNoise(seed=42)
        for x, y in [(0.5, 0.5), (3.2, -1.7), (10.0, 10.0)]:
            assert p1.noise2d(x, y) == p2.noise2d(x, y)

    def test_perlin_smoothstep_range(self):
        p = PerlinNoise(seed=42)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            v = p._smoothstep(t)
            assert 0.0 <= v <= 1.0

    def test_perlin_lerp(self):
        p = PerlinNoise(seed=42)
        assert p._lerp(0.0, 10.0, 0.0) == 0.0
        assert p._lerp(0.0, 10.0, 1.0) == 10.0
        assert p._lerp(0.0, 10.0, 0.5) == 5.0

    def test_monsoon_dir_effect(self):
        ng = NoiseGenerator(seed=42)
        coords = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (0.0, -5.0)]
        elev = np.array([0.5] * len(coords))
        m1 = ng.generate_moisture(coords, elev, monsoon_dir=0.0)
        m2 = ng.generate_moisture(coords, elev, monsoon_dir=180.0)
        assert not np.array_equal(m1, m2)


# ============================================================
# 6) FeatureGenerator — 复杂算法逻辑
# ============================================================
def _build_simple_terrain(grid_size=5):
    """构建一个简单的测试地形：中心高、四周水"""
    grid = HexGrid(size=grid_size)
    terrain_data = {}
    hex_coords_list = []

    for h in grid.hexes:
        td = TerrainData()
        dist = HexCoord(0, 0).distance_to(h)
        td.elevation = max(0.1, 1.0 - dist / (grid_size + 1))
        td.moisture = 0.5
        td.temperature = 0.5
        td.is_water = td.elevation < 0.35
        td.is_coast = False
        td.biome = BIOME_OCEAN if td.is_water else BIOME_PLAINS
        terrain_data[h] = td
        cx, cy = grid.hex_center(h, 10.0)
        hex_coords_list.append((h, cx, cy))

    for h, td in terrain_data.items():
        if not td.is_water:
            for nh in h.neighbors():
                if nh in terrain_data and terrain_data[nh].is_water:
                    td.is_coast = True
                    break

    return grid, terrain_data, hex_coords_list


class TestFeatureGeneratorSettlementScore:
    def test_plains_high_score(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.4
        td.is_coast = False
        td.river_flow = 0.0
        td.volcanic = False
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score > 0

    def test_coast_bonus(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.4
        td.is_coast = False
        td.river_flow = 0.0
        td.volcanic = False
        score_no_coast = fg._settlement_score(td, HexCoord(0, 0))
        td.is_coast = True
        score_coast = fg._settlement_score(td, HexCoord(0, 0))
        assert score_coast > score_no_coast

    def test_river_bonus(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        td = TerrainData()
        td.biome = "plains"
        td.elevation = 0.4
        td.is_coast = False
        td.river_flow = 0.0
        td.volcanic = False
        score_no_river = fg._settlement_score(td, HexCoord(0, 0))
        td.river_flow = 0.5
        score_river = fg._settlement_score(td, HexCoord(0, 0))
        assert score_river > score_no_river

    def test_high_elevation_penalty(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        td_low = TerrainData()
        td_low.biome = "plains"
        td_low.elevation = 0.4
        td_low.is_coast = False
        td_low.river_flow = 0.0
        td_low.volcanic = False
        td_high = TerrainData()
        td_high.biome = "plains"
        td_high.elevation = 0.8
        td_high.is_coast = False
        td_high.river_flow = 0.0
        td_high.volcanic = False
        assert fg._settlement_score(td_low, HexCoord(0, 0)) > fg._settlement_score(
            td_high, HexCoord(0, 0)
        )

    def test_desert_low_score(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        td = TerrainData()
        td.biome = "desert"
        td.elevation = 0.4
        td.is_coast = False
        td.river_flow = 0.0
        td.volcanic = False
        score = fg._settlement_score(td, HexCoord(0, 0))
        td.biome = "plains"
        score_plains = fg._settlement_score(td, HexCoord(0, 0))
        assert score < score_plains


class TestFeatureGeneratorSettlements:
    def test_generate_settlements_basic(self):
        _, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=3, num_towns=2, num_cities=1, has_capital=True
        )
        assert len(settlements) > 0
        for hc in settlements:
            assert terrain_data[hc].settlement != SETTLEMENT_NONE

    def test_generate_settlements_no_land(self):
        grid = HexGrid(size=2)
        terrain_data = {}
        hex_list = []
        for h in grid.hexes:
            td = TerrainData()
            td.elevation = 0.1
            td.is_water = True
            td.biome = BIOME_OCEAN
            terrain_data[h] = td
            cx, cy = grid.hex_center(h, 10.0)
            hex_list.append((h, cx, cy))
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(rng)
        assert settlements == []

    def test_settlement_types_placed(self):
        _, terrain_data, hex_list = _build_simple_terrain(6)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_settlements(
            rng, num_villages=4, num_towns=2, num_cities=1, has_capital=True
        )
        capitals = [h for h, td in terrain_data.items() if td.settlement == SETTLEMENT_CAPITAL]
        cities = [h for h, td in terrain_data.items() if td.settlement == SETTLEMENT_CITY]
        towns = [h for h, td in terrain_data.items() if td.settlement == SETTLEMENT_TOWN]
        villages = [h for h, td in terrain_data.items() if td.settlement == SETTLEMENT_VILLAGE]
        assert len(capitals) == 1
        assert len(cities) >= 1
        assert len(towns) >= 1
        assert len(villages) >= 1

    def test_settlement_names_generated(self):
        _, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(rng, has_capital=True)
        for hc in settlements:
            assert len(terrain_data[hc].settlement_name) >= 2


class TestFeatureGeneratorRoads:
    def test_generate_roads_with_settlements(self):
        _, terrain_data, hex_list = _build_simple_terrain(6)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_settlements(rng, num_villages=5, num_towns=3, num_cities=2, has_capital=True)
        roads = fg.generate_roads()
        assert isinstance(roads, list)
        if roads:
            for hc1, hc2 in roads:
                assert isinstance(hc1, HexCoord)
                assert isinstance(hc2, HexCoord)

    def test_generate_roads_empty_settlements(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        roads = fg.generate_roads()
        assert roads == []

    def test_road_path_avoid_water(self):
        grid, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        land_hexes = [h for h, td in terrain_data.items() if not td.is_water]
        if len(land_hexes) >= 2:
            path = fg._find_road_path(land_hexes[0], land_hexes[-1])
            if path:
                for h in path:
                    assert h in terrain_data
                    if h != land_hexes[0] and h != land_hexes[-1]:
                        assert not terrain_data[h].is_water

    def test_road_path_start_equals_end(self):
        _, terrain_data, hex_list = _build_simple_terrain(4)
        fg = FeatureGenerator(terrain_data, hex_list)
        start = HexCoord(0, 0)
        path = fg._find_road_path(start, start)
        assert len(path) == 1
        assert path[0] == start


class TestFeatureGeneratorRivers:
    def test_generate_rivers_no_high_land(self):
        grid = HexGrid(size=3)
        terrain_data = {}
        hex_list = []
        for h in grid.hexes:
            td = TerrainData()
            td.elevation = 0.3
            td.is_water = td.elevation < 0.35
            td.biome = BIOME_PLAINS if not td.is_water else BIOME_LAKE
            terrain_data[h] = td
            cx, cy = grid.hex_center(h, 10.0)
            hex_list.append((h, cx, cy))
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=5)
        assert rivers == []

    def test_generate_rivers_with_high_land(self):
        grid, terrain_data, hex_list = _build_simple_terrain(6)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=3)
        assert isinstance(rivers, list)
        for river in rivers:
            assert len(river) > 3
            for hc in river:
                assert isinstance(hc, HexCoord)

    def test_river_flows_to_water(self):
        grid, terrain_data, hex_list = _build_simple_terrain(6)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=2)
        for river in rivers:
            last = river[-1]
            assert terrain_data[last].is_water


class TestFeatureGeneratorResources:
    def test_resources_placed(self):
        _, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_resources(rng, density=0.3)
        resources = [h for h, td in terrain_data.items() if td.resource is not None]
        assert len(resources) > 0

    def test_no_resources_on_water(self):
        _, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_resources(rng, density=1.0)
        for h, td in terrain_data.items():
            if td.is_water:
                assert td.resource is None

    def test_no_resources_on_settlements(self):
        _, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_settlements(rng, num_villages=5, has_capital=True)
        fg.generate_resources(rng, density=1.0)
        for h, td in terrain_data.items():
            if td.settlement != SETTLEMENT_NONE:
                assert td.resource is None

    def test_resources_in_correct_biomes(self):
        _, terrain_data, hex_list = _build_simple_terrain(5)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_resources(rng, density=1.0)
        biome_resources = {
            RESOURCE_WOOD: {"forest", "dense_forest", "rainforest", "taiga"},
            RESOURCE_IRON: {"mountains", "hills", "high_mountains"},
            RESOURCE_STONE: {"mountains", "hills", "high_mountains", "volcano"},
            RESOURCE_FOOD: {"plains", "grassland", "savanna"},
        }
        for h, td in terrain_data.items():
            if td.resource and td.resource in biome_resources:
                assert td.biome in biome_resources[td.resource]


class TestFeatureGeneratorShipping:
    def test_shipping_routes_no_coastal_settlements(self):
        _, terrain_data, hex_list = _build_simple_terrain(4)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        routes = fg.generate_shipping_routes(rng)
        assert routes == []

    def test_find_nearest_water_on_water(self):
        _, terrain_data, hex_list = _build_simple_terrain(4)
        fg = FeatureGenerator(terrain_data, hex_list)
        water_hexes = [h for h, td in terrain_data.items() if td.is_water]
        if water_hexes:
            result = fg._find_nearest_water(water_hexes[0])
            assert result == water_hexes[0]

    def test_find_nearest_water_near_water(self):
        _, terrain_data, hex_list = _build_simple_terrain(4)
        fg = FeatureGenerator(terrain_data, hex_list)
        land_hexes = [h for h, td in terrain_data.items() if not td.is_water and td.is_coast]
        if land_hexes:
            result = fg._find_nearest_water(land_hexes[0])
            assert result is not None
            assert terrain_data[result].is_water


class TestFeatureGeneratorNameGen:
    def test_generate_name_length(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        for typ in ["capital", "city", "town", "village"]:
            name = fg._generate_name(rng, typ)
            assert len(name) == 2

    def test_generate_name_unknown_type(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng = np.random.default_rng(42)
        name = fg._generate_name(rng, "unknown_type")
        assert len(name) == 2

    def test_generate_name_deterministic(self):
        _, terrain_data, hex_list = _build_simple_terrain(3)
        fg = FeatureGenerator(terrain_data, hex_list)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        n1 = fg._generate_name(rng1, "city")
        n2 = fg._generate_name(rng2, "city")
        assert n1 == n2


# ============================================================
# 7) TerrainGenerator — 海岸检测与火山
# ============================================================
class TestTerrainGeneratorCoastVolcano:
    def test_is_coast_water_adjacent_to_land(self):
        grid, terrain_data, hex_list = _build_simple_terrain(4)
        tg = TerrainGenerator()
        elev = np.array([terrain_data[h].elevation for h, _, _ in hex_list])
        moist = np.array([0.5] * len(hex_list))
        temp = np.array([0.5] * len(hex_list))
        result = tg.generate(elev, moist, temp, hex_list)
        coast_count = sum(1 for td in result.values() if td.is_coast)
        assert coast_count > 0

    def test_water_tiles_is_water(self):
        grid, terrain_data, hex_list = _build_simple_terrain(3)
        tg = TerrainGenerator()
        elev = np.array([terrain_data[h].elevation for h, _, _ in hex_list])
        moist = np.array([0.5] * len(hex_list))
        temp = np.array([0.5] * len(hex_list))
        result = tg.generate(elev, moist, temp, hex_list)
        for hc, td in result.items():
            if td.elevation < tg.water_level:
                assert td.is_water == True
            else:
                assert td.is_water == False

    def test_volcano_rare(self):
        grid = HexGrid(size=5)
        hex_list = []
        for h in grid.hexes:
            td = TerrainData()
            dist = HexCoord(0, 0).distance_to(h)
            td.elevation = max(0.61, 0.9 - dist * 0.05)
            cx, cy = grid.hex_center(h, 10.0)
            hex_list.append((h, cx, cy))
        tg = TerrainGenerator()
        tg.volcano_chance = 0.0
        elev = np.array([td.elevation for _, td in zip([h for h, _, _ in hex_list], [])])
        elev = np.array([h[0] and 0.7 for h in hex_list])
        elev = np.array([0.7] * len(hex_list))
        moist = np.array([0.2] * len(hex_list))
        temp = np.array([0.5] * len(hex_list))
        result = tg.generate(elev, moist, temp, hex_list)
        volcano_count = sum(1 for td in result.values() if td.volcanic)
        assert volcano_count == 0
