"""
测试套件 — 核心功能覆盖缺口
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.feature_gen import FeatureGenerator
from core.hex_grid import HexCoord, HexGrid, HEX_DIRECTIONS
from core.noise_gen import NoiseGenerator, PerlinNoise
from core.terrain_gen import (
    BIOME_BEACH,
    BIOME_DENSE_FOREST,
    BIOME_DESERT,
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
    TerrainGenerator,
)


class TestHexCoordArithmetic:
    def test_add(self):
        a = HexCoord(2, 3)
        b = HexCoord(-1, 4)
        c = a + b
        assert c.q == 1 and c.r == 7

    def test_add_with_direction(self):
        h = HexCoord(0, 0)
        for d in HEX_DIRECTIONS:
            n = h + d
            assert n.distance_to(h) == 1

    def test_sub(self):
        a = HexCoord(5, -2)
        b = HexCoord(3, 1)
        c = a - b
        assert c.q == 2 and c.r == -3

    def test_sub_inverse_of_add(self):
        a = HexCoord(3, 4)
        b = HexCoord(1, 2)
        assert (a + b) - b == a
        assert (a - b) + b == a

    def test_mul(self):
        h = HexCoord(2, -3)
        assert h * 0 == HexCoord(0, 0)
        assert h * 2 == HexCoord(4, -6)
        assert h * -1 == HexCoord(-2, 3)

    def test_distance_to(self):
        assert HexCoord(0, 0).distance_to(HexCoord(0, 0)) == 0
        assert HexCoord(0, 0).distance_to(HexCoord(1, 0)) == 1
        assert HexCoord(0, 0).distance_to(HexCoord(3, -2)) == 3
        assert HexCoord(5, -3).distance_to(HexCoord(2, 1)) == 4

    def test_neighbors_count(self):
        h = HexCoord(0, 0)
        neighbors = h.neighbors()
        assert len(neighbors) == 6
        for n in neighbors:
            assert h.distance_to(n) == 1

    def test_neighbors_unique(self):
        h = HexCoord(5, -3)
        neighbors = h.neighbors()
        assert len(set(neighbors)) == 6

    def test_range(self):
        h = HexCoord(0, 0)
        assert len(h.range(0)) == 1
        assert len(h.range(1)) == 7
        assert len(h.range(2)) == 19

    def test_range_contains_center(self):
        h = HexCoord(3, -2)
        assert h in h.range(0)
        assert h in h.range(5)

    def test_range_max_distance(self):
        h = HexCoord(0, 0)
        for n in h.range(3):
            assert h.distance_to(n) <= 3


class TestHexGridConversion:
    def test_hex_center_origin(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(0, 0), 10)
        assert abs(cx) < 0.001 and abs(cy) < 0.001

    def test_hex_center_offset(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(2, 3), 10)
        expected_x = 10 * (3.0 / 2.0 * 2)
        expected_y = 10 * (3**0.5 / 2.0 * 2 + 3**0.5 * 3)
        assert abs(cx - expected_x) < 0.001
        assert abs(cy - expected_y) < 0.001

    def test_hex_corners_count(self):
        grid = HexGrid(size=5)
        corners = grid.hex_corners(HexCoord(0, 0), 10)
        assert len(corners) == 6

    def test_hex_corners_equidistant_from_center(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(0, 0), 10)
        corners = grid.hex_corners(HexCoord(0, 0), 10)
        for x, y in corners:
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            assert abs(dist - 10) < 0.001

    def test_hex_round(self):
        grid = HexGrid(size=5)
        h = grid._hex_round(0.1, 0.1)
        assert h == HexCoord(0, 0)
        h = grid._hex_round(0.6, 0.1)
        assert h == HexCoord(1, 0)
        h = grid._hex_round(0.4, 0.4)
        assert h == HexCoord(0, 0) or h == HexCoord(1, 0) or h == HexCoord(0, 1)

    def test_hex_at_pixel(self):
        grid = HexGrid(size=10)
        hex_size = 20
        cx, cy = grid.hex_center(HexCoord(3, 5), hex_size)
        h = grid.hex_at_pixel(cx, cy, hex_size)
        assert h == HexCoord(3, 5)

    def test_hex_at_pixel_near_center(self):
        grid = HexGrid(size=10)
        hex_size = 20
        cx, cy = grid.hex_center(HexCoord(2, 3), hex_size)
        h = grid.hex_at_pixel(cx + 5, cy + 5, hex_size)
        assert h == HexCoord(2, 3)

    def test_all_hex_centers(self):
        grid = HexGrid(size=3)
        centers = grid.all_hex_centers(10)
        assert len(centers) == len(grid.hexes)
        for hc, cx, cy in centers:
            expected_cx, expected_cy = grid.hex_center(hc, 10)
            assert abs(cx - expected_cx) < 0.001
            assert abs(cy - expected_cy) < 0.001

    def test_get_random_hex(self):
        grid = HexGrid(size=5)
        rng = np.random.default_rng(42)
        for _ in range(20):
            h = grid.get_random_hex(rng)
            assert h in grid.hexes

    def test_get_hexes_within_radius(self):
        grid = HexGrid(size=10)
        center = HexCoord(2, 3)
        result = grid.get_hexes_within_radius(center, 2)
        for h in result:
            assert center.distance_to(h) <= 2


class TestPerlinNoise:
    def test_seed_reproducibility(self):
        p1 = PerlinNoise(seed=42)
        p2 = PerlinNoise(seed=42)
        for _ in range(50):
            x, y = np.random.uniform(-100, 100, 2)
            assert abs(p1.noise2d(x, y) - p2.noise2d(x, y)) < 1e-10

    def test_different_seeds_different_output(self):
        p1 = PerlinNoise(seed=42)
        p2 = PerlinNoise(seed=123)
        diffs = []
        for _ in range(50):
            x, y = np.random.uniform(-100, 100, 2)
            diffs.append(abs(p1.noise2d(x, y) - p2.noise2d(x, y)))
        assert np.mean(diffs) > 0.1

    def test_noise2d_range(self):
        p = PerlinNoise(seed=42)
        values = []
        for _ in range(100):
            x, y = np.random.uniform(-10, 10, 2)
            values.append(p.noise2d(x, y))
        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_octave_noise_range(self):
        p = PerlinNoise(seed=42)
        for octaves in [1, 3, 6, 10]:
            for _ in range(20):
                x, y = np.random.uniform(-10, 10, 2)
                v = p.octave_noise(x, y, octaves=octaves)
                assert -1.0 <= v <= 1.0

    def test_octave_noise_persistence(self):
        p = PerlinNoise(seed=42)
        values_low = []
        values_high = []
        for _ in range(20):
            x, y = np.random.uniform(-10, 10, 2)
            values_low.append(p.octave_noise(x, y, octaves=4, persistence=0.3))
            values_high.append(p.octave_noise(x, y, octaves=4, persistence=0.9))
        assert abs(np.mean(values_low) - np.mean(values_high)) > 0.01

    def test_hash_deterministic(self):
        p = PerlinNoise(seed=42)
        assert p._hash(0, 0) == p._hash(0, 0)
        assert p._hash(123, 456) == p._hash(123, 456)

    def test_smoothstep_monotonic(self):
        p = PerlinNoise(seed=42)
        for t in np.linspace(0, 1, 20):
            assert 0 <= p._smoothstep(t) <= 1
        prev = -1
        for t in np.linspace(0, 1, 20):
            val = p._smoothstep(t)
            assert val >= prev
            prev = val


class TestNoiseGenerator:
    def test_empty_coords(self):
        ng = NoiseGenerator(seed=42)
        elev = ng.generate_elevation([])
        assert elev.shape == (0,)
        moist = ng.generate_moisture([], np.array([]))
        assert moist.shape == (0,)

    def test_single_coord(self):
        ng = NoiseGenerator(seed=42)
        elev = ng.generate_elevation([(0.0, 0.0)])
        assert elev.shape == (1,)
        assert 0 <= elev[0] <= 1

    def test_generate_elevation_bounds(self):
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.1, i * 0.2) for i in range(100)]
        elev = ng.generate_elevation(coords)
        assert elev.min() >= 0.0
        assert elev.max() <= 1.0

    def test_generate_moisture_bounds(self):
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.1, i * 0.2) for i in range(50)]
        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev)
        assert moist.min() >= 0.0
        assert moist.max() <= 1.0

    def test_generate_temperature_bounds(self):
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.1, i * 0.2) for i in range(50)]
        elev = ng.generate_elevation(coords)
        temp = ng.generate_temperature(elev, coords)
        assert temp.min() >= 0.0
        assert temp.max() <= 1.0

    def test_monsoon_effect(self):
        ng = NoiseGenerator(seed=42)
        coords = [(x, 0.0) for x in np.linspace(-10, 10, 50)]
        elev = np.full(50, 0.5)
        moist_east = ng.generate_moisture(coords, elev, monsoon_dir=90.0)
        moist_west = ng.generate_moisture(coords, elev, monsoon_dir=270.0)
        assert abs(np.mean(moist_east) - np.mean(moist_west)) > 0.05

    def test_set_seed_updates_perlin(self):
        ng = NoiseGenerator(seed=42)
        coords = [(1.0, 1.0)]
        v1 = ng.generate_elevation(coords)[0]
        ng.set_seed(123)
        v2 = ng.generate_elevation(coords)[0]
        assert abs(v1 - v2) > 0.01


class TestTerrainGeneratorClassifyBiome:
    def test_ocean(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.1, 0.5, 0.5, False) == BIOME_OCEAN

    def test_lake(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.25, 0.5, 0.5, False) == BIOME_LAKE

    def test_beach(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.38, 0.5, 0.5, True) == BIOME_BEACH

    def test_snow_high_elev_low_temp(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.9, 0.5, 0.2, False) == BIOME_SNOW

    def test_high_mountains(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.85, 0.5, 0.5, False) == BIOME_HIGH_MOUNTAINS

    def test_mountains(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.75, 0.5, 0.5, False) == BIOME_MOUNTAINS

    def test_hills(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.6, 0.5, 0.5, False) == BIOME_HILLS

    def test_desert_high_temp_low_moist(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.2, 0.7, False) == BIOME_DESERT

    def test_tundra_low_temp_low_moist(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.2, 0.3, False) == BIOME_TUNDRA

    def test_savanna_high_temp_low_moist(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.3, 0.7, False) == BIOME_SAVANNA

    def test_plains_moderate(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.4, 0.6, False) == BIOME_PLAINS

    def test_taiga_cool_moderate_moist(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.4, 0.3, False) == BIOME_TAIGA

    def test_forest_warm_moderate_moist(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.6, 0.6, False) == BIOME_FOREST

    def test_rainforest_hot_wet(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.85, 0.8, False) == BIOME_RAINFOREST

    def test_dense_forest_moderate_hot_wet(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.4, 0.85, 0.6, False) == BIOME_DENSE_FOREST

    def test_swamp_low_elev_wet(self):
        tg = TerrainGenerator()
        assert tg.classify_biome(0.35, 0.85, 0.4, False) == BIOME_SWAMP


class TestTerrainGeneratorGenerate:
    def test_generate_empty(self):
        tg = TerrainGenerator()
        td = tg.generate(np.array([]), np.array([]), np.array([]), [])
        assert td == {}

    def test_generate_single_hex(self):
        tg = TerrainGenerator()
        elev = np.array([0.5])
        moist = np.array([0.5])
        temp = np.array([0.5])
        hex_list = [(HexCoord(0, 0), 0.0, 0.0)]
        td = tg.generate(elev, moist, temp, hex_list)
        assert len(td) == 1
        assert HexCoord(0, 0) in td

    def test_coast_detection_water_neighbor(self):
        tg = TerrainGenerator()
        elev = np.array([0.4, 0.2])
        moist = np.array([0.5, 0.5])
        temp = np.array([0.5, 0.5])
        hex_list = [
            (HexCoord(0, 0), 0.0, 0.0),
            (HexCoord(1, 0), 1.0, 0.0),
        ]
        td = tg.generate(elev, moist, temp, hex_list)
        assert td[HexCoord(0, 0)].is_coast

    def test_coast_detection_land_neighbor(self):
        tg = TerrainGenerator()
        elev = np.array([0.2, 0.4])
        moist = np.array([0.5, 0.5])
        temp = np.array([0.5, 0.5])
        hex_list = [
            (HexCoord(0, 0), 0.0, 0.0),
            (HexCoord(1, 0), 1.0, 0.0),
        ]
        td = tg.generate(elev, moist, temp, hex_list)
        assert td[HexCoord(0, 0)].is_coast


class TestFeatureGenerator:
    def test_empty_terrain_no_rivers(self):
        fg = FeatureGenerator({}, [])
        rivers = fg.generate_rivers(np.random.default_rng(42))
        assert rivers == []

    def test_empty_terrain_no_settlements(self):
        fg = FeatureGenerator({}, [])
        settlements = fg.generate_settlements(np.random.default_rng(42))
        assert settlements == []

    def test_empty_terrain_no_roads(self):
        fg = FeatureGenerator({}, [])
        roads = fg.generate_roads()
        assert roads == []

    def test_empty_terrain_no_shipping(self):
        fg = FeatureGenerator({}, [])
        routes = fg.generate_shipping_routes(np.random.default_rng(42))
        assert routes == []

    def test_generate_rivers_no_candidates(self):
        td = {}
        hc = HexCoord(0, 0)
        from core.terrain_gen import TerrainData

        td[hc] = TerrainData()
        td[hc].elevation = 0.3
        td[hc].is_water = False
        hex_list = [(hc, 0.0, 0.0)]
        fg = FeatureGenerator(td, hex_list)
        rivers = fg.generate_rivers(np.random.default_rng(42))
        assert rivers == []

    def test_generate_settlements_no_candidates(self):
        td = {}
        hc = HexCoord(0, 0)
        from core.terrain_gen import TerrainData

        td[hc] = TerrainData()
        td[hc].elevation = 0.9
        td[hc].is_water = False
        hex_list = [(hc, 0.0, 0.0)]
        fg = FeatureGenerator(td, hex_list)
        settlements = fg.generate_settlements(np.random.default_rng(42))
        assert settlements == []

    def test_generate_roads_single_settlement(self):
        td = {}
        hc = HexCoord(0, 0)
        from core.terrain_gen import TerrainData, SETTLEMENT_CITY

        td[hc] = TerrainData()
        td[hc].elevation = 0.4
        td[hc].is_water = False
        td[hc].settlement = SETTLEMENT_CITY
        hex_list = [(hc, 0.0, 0.0)]
        fg = FeatureGenerator(td, hex_list)
        roads = fg.generate_roads()
        assert roads == []

    def test_settlement_score_biome_pref(self):
        from core.terrain_gen import TerrainData

        td_plains = TerrainData()
        td_plains.biome = "plains"
        td_desert = TerrainData()
        td_desert.biome = "desert"

        fg = FeatureGenerator({}, [])
        score_plains = fg._settlement_score(td_plains, HexCoord(0, 0))
        score_desert = fg._settlement_score(td_desert, HexCoord(0, 0))
        assert score_plains > score_desert

    def test_settlement_score_coast_bonus(self):
        from core.terrain_gen import TerrainData

        td_coast = TerrainData()
        td_coast.biome = "plains"
        td_coast.is_coast = True
        td_inland = TerrainData()
        td_inland.biome = "plains"
        td_inland.is_coast = False

        fg = FeatureGenerator({}, [])
        score_coast = fg._settlement_score(td_coast, HexCoord(0, 0))
        score_inland = fg._settlement_score(td_inland, HexCoord(0, 0))
        assert score_coast > score_inland

    def test_settlement_score_elevation_penalty(self):
        from core.terrain_gen import TerrainData

        td_low = TerrainData()
        td_low.biome = "plains"
        td_low.elevation = 0.4
        td_high = TerrainData()
        td_high.biome = "plains"
        td_high.elevation = 0.8

        fg = FeatureGenerator({}, [])
        score_low = fg._settlement_score(td_low, HexCoord(0, 0))
        score_high = fg._settlement_score(td_high, HexCoord(0, 0))
        assert score_low > score_high

    def test_generate_resources_no_op(self):
        td = {}
        hc = HexCoord(0, 0)
        from core.terrain_gen import TerrainData

        td[hc] = TerrainData()
        td[hc].elevation = 0.4
        td[hc].is_water = False
        hex_list = [(hc, 0.0, 0.0)]
        fg = FeatureGenerator(td, hex_list)
        fg.generate_resources(np.random.default_rng(42), density=0.0)
        assert td[hc].resource is None

    def test_generate_name(self):
        fg = FeatureGenerator({}, [])
        rng = np.random.default_rng(42)
        for typ in ["capital", "city", "town", "village"]:
            name = fg._generate_name(rng, typ)
            assert len(name) >= 2
            assert len(name) <= 4

    def test_find_nearest_water(self):
        td = {}
        for q in range(-2, 3):
            for r in range(-2, 3):
                hc = HexCoord(q, r)
                from core.terrain_gen import TerrainData

                td[hc] = TerrainData()
                td[hc].elevation = 0.6 if (q == 0 and r == 0) else 0.2
                td[hc].is_water = td[hc].elevation < 0.3

        hex_list = [(hc, float(hc.q), float(hc.r)) for hc in td.keys()]
        fg = FeatureGenerator(td, hex_list)
        near = fg._find_nearest_water(HexCoord(0, 0))
        assert near is not None
        assert td[near].is_water