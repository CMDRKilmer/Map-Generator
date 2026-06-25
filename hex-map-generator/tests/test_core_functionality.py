"""
核心功能行为测试 — 覆盖回归测试安全网缺口
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
    TerrainData,
    TerrainGenerator,
)
from core.feature_gen import FeatureGenerator


class TestHexCoordCore:
    def test_distance_to(self):
        hc1 = HexCoord(0, 0)
        hc2 = HexCoord(3, -2)
        assert hc1.distance_to(hc2) == 3

    def test_neighbors_count(self):
        hc = HexCoord(0, 0)
        neighbors = hc.neighbors()
        assert len(neighbors) == 6
        for n in neighbors:
            assert hc.distance_to(n) == 1

    def test_neighbor_symmetry(self):
        hc = HexCoord(2, 3)
        neighbors = hc.neighbors()
        for n in neighbors:
            assert hc in n.neighbors()

    def test_range_zero(self):
        hc = HexCoord(0, 0)
        assert len(hc.range(0)) == 1
        assert hc.range(0)[0] == hc

    def test_range_one(self):
        hc = HexCoord(0, 0)
        assert len(hc.range(1)) == 7

    def test_to_cube_consistency(self):
        hc = HexCoord(2, -3)
        q, r, s = hc.to_cube()
        assert q + r + s == 0
        assert q == 2 and r == -3 and s == 1

    def test_add_sub_mul(self):
        hc1 = HexCoord(3, -2)
        hc2 = HexCoord(1, 4)
        assert hc1 + hc2 == HexCoord(4, 2)
        assert hc1 - hc2 == HexCoord(2, -6)
        assert hc1 * 2 == HexCoord(6, -4)

    def test_hash_and_eq(self):
        hc1 = HexCoord(2, 3)
        hc2 = HexCoord(2, 3)
        hc3 = HexCoord(3, 2)
        assert hc1 == hc2
        assert hc1 != hc3
        assert hash(hc1) == hash(hc2)


class TestHexGridCore:
    def test_grid_size(self):
        grid = HexGrid(size=2)
        assert len(grid.hexes) == 19

    def test_hex_center_origin(self):
        grid = HexGrid(size=1)
        cx, cy = grid.hex_center(HexCoord(0, 0), 10.0)
        assert abs(cx) < 0.001 and abs(cy) < 0.001

    def test_hex_center_offset(self):
        grid = HexGrid(size=1)
        cx, cy = grid.hex_center(HexCoord(1, 0), 2.0)
        assert abs(cx - 3.0) < 0.001
        assert abs(cy - math.sqrt(3)) < 0.001

    def test_hex_corners_count(self):
        grid = HexGrid(size=1)
        corners = grid.hex_corners(HexCoord(0, 0), 10.0)
        assert len(corners) == 6

    def test_hex_corners_symmetry(self):
        grid = HexGrid(size=1)
        corners = grid.hex_corners(HexCoord(0, 0), 10.0)
        for i, (x, y) in enumerate(corners):
            opp_x, opp_y = corners[(i + 3) % 6]
            assert abs(x + opp_x) < 0.001 and abs(y + opp_y) < 0.001

    def test_hex_at_pixel_origin(self):
        grid = HexGrid(size=5)
        hc = grid.hex_at_pixel(0, 0, 20.0)
        assert hc == HexCoord(0, 0)

    def test_hex_at_pixel_offset(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(2, 1), 20.0)
        hc = grid.hex_at_pixel(cx, cy, 20.0)
        assert hc == HexCoord(2, 1)

    def test_hex_at_pixel_boundary(self):
        grid = HexGrid(size=5)
        cx, cy = grid.hex_center(HexCoord(1, 0), 20.0)
        cx2, cy2 = grid.hex_center(HexCoord(2, 0), 20.0)
        mid_x = (cx + cx2) / 2 + 20.0
        mid_y = (cy + cy2) / 2
        hc = grid.hex_at_pixel(mid_x, mid_y, 20.0)
        assert hc == HexCoord(2, 0)

    def test_all_hex_centers_count(self):
        grid = HexGrid(size=3)
        centers = grid.all_hex_centers(10.0)
        assert len(centers) == len(grid.hexes)
        for hc, _, _ in centers:
            assert hc in grid.hexes

    def test_get_random_hex(self):
        grid = HexGrid(size=5)
        rng = np.random.default_rng(42)
        for _ in range(10):
            hc = grid.get_random_hex(rng)
            assert hc in grid.hexes

    def test_get_hexes_within_radius(self):
        grid = HexGrid(size=10)
        center = HexCoord(0, 0)
        hexes = grid.get_hexes_within_radius(center, 3)
        assert len(hexes) == 37
        for hc in hexes:
            assert center.distance_to(hc) <= 3


class TestTerrainGeneratorClassifyBiome:
    def test_ocean_deep(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.10, 0.5, 0.5, False)
        assert biome == BIOME_OCEAN

    def test_lake_shallow(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.25, 0.5, 0.5, False)
        assert biome == BIOME_LAKE

    def test_beach_coast(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.38, 0.5, 0.5, True)
        assert biome == BIOME_BEACH

    def test_snow_high_cold(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.90, 0.5, 0.2, False)
        assert biome == BIOME_SNOW

    def test_high_mountains(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.85, 0.5, 0.6, False)
        assert biome == BIOME_HIGH_MOUNTAINS

    def test_mountains(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.75, 0.5, 0.5, False)
        assert biome == BIOME_MOUNTAINS

    def test_hills(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.60, 0.5, 0.5, False)
        assert biome == BIOME_HILLS

    def test_desert_hot_dry(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.20, 0.7, False)
        assert biome == BIOME_DESERT

    def test_tundra_cold_dry(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.20, 0.3, False)
        assert biome == BIOME_TUNDRA

    def test_savanna_hot_semi_dry(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.30, 0.7, False)
        assert biome == BIOME_SAVANNA

    def test_plains_mild(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.45, 0.55, False)
        assert biome == BIOME_PLAINS

    def test_taiga_cold_moist(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.60, 0.25, False)
        assert biome == BIOME_TAIGA

    def test_forest_moderate(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.65, 0.6, False)
        assert biome == BIOME_FOREST

    def test_rainforest_hot_wet(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.85, 0.8, False)
        assert biome == BIOME_RAINFOREST

    def test_dense_forest_warm_wet(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.40, 0.85, 0.6, False)
        assert biome == BIOME_DENSE_FOREST

    def test_swamp_low_wet(self):
        tg = TerrainGenerator()
        biome = tg.classify_biome(0.38, 0.85, 0.4, False)
        assert biome == BIOME_SWAMP


class TestFeatureGeneratorCore:
    def _create_minimal_terrain(self, hex_count: int = 20):
        terrain_data = {}
        hex_coords_list = []
        idx = 0
        for q in range(-3, 4):
            for r in range(-3, 4):
                if idx >= hex_count:
                    break
                hc = HexCoord(q, r)
                td = TerrainData()
                td.elevation = 0.4 + (q + r) * 0.02
                td.moisture = 0.5
                td.temperature = 0.5
                td.is_water = td.elevation < 0.35
                td.biome = BIOME_PLAINS if not td.is_water else BIOME_OCEAN
                terrain_data[hc] = td
                hex_coords_list.append((hc, float(q), float(r)))
                idx += 1
            if idx >= hex_count:
                break
        return terrain_data, hex_coords_list

    def test_generate_settlements_basic(self):
        terrain_data = {}
        hex_coords_list = []
        for q in range(-5, 6):
            for r in range(-5, 6):
                if abs(q) + abs(r) + abs(-q - r) <= 8:
                    hc = HexCoord(q, r)
                    td = TerrainData()
                    td.elevation = 0.5 + (q + r) * 0.01
                    td.moisture = 0.5
                    td.temperature = 0.5
                    td.is_water = False
                    td.biome = BIOME_PLAINS
                    terrain_data[hc] = td
                    hex_coords_list.append((hc, float(q), float(r)))
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=3, num_towns=2, num_cities=1, has_capital=True
        )
        assert len(settlements) == 7

    def test_generate_settlements_no_candidates(self):
        terrain_data = {}
        hex_coords_list = []
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(rng, num_villages=5)
        assert settlements == []

    def test_settlement_score_plains(self):
        td = TerrainData()
        td.biome = BIOME_PLAINS
        td.elevation = 0.4
        td.is_coast = False
        td.river_flow = 0
        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score == 10.0

    def test_settlement_score_coast_river_bonus(self):
        td = TerrainData()
        td.biome = BIOME_PLAINS
        td.elevation = 0.4
        td.is_coast = True
        td.river_flow = 0.5
        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score == 19.0

    def test_settlement_score_high_elevation_penalty(self):
        td = TerrainData()
        td.biome = BIOME_PLAINS
        td.elevation = 0.7
        td.is_coast = False
        td.river_flow = 0
        fg = FeatureGenerator({}, [])
        score = fg._settlement_score(td, HexCoord(0, 0))
        assert score == 10.0 - (0.7 - 0.55) * 15

    def test_generate_resources_basic(self):
        terrain_data, hex_coords_list = self._create_minimal_terrain(30)
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        rng = np.random.default_rng(42)
        fg.generate_resources(rng, density=0.5)
        resource_count = sum(1 for td in terrain_data.values() if td.resource is not None)
        assert resource_count > 0

    def test_generate_resources_no_water(self):
        terrain_data, hex_coords_list = self._create_minimal_terrain(20)
        for hc in terrain_data:
            terrain_data[hc].is_water = True
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        rng = np.random.default_rng(42)
        fg.generate_resources(rng, density=1.0)
        resource_count = sum(1 for td in terrain_data.values() if td.resource is not None)
        assert resource_count == 0

    def test_generate_shipping_routes_coastal(self):
        terrain_data = {}
        hex_coords_list = []
        for q in range(-5, 6):
            for r in range(-5, 6):
                hc = HexCoord(q, r)
                td = TerrainData()
                td.elevation = 0.3 if abs(q) == 4 else 0.5
                td.is_water = td.elevation < 0.35
                td.is_coast = td.is_water and abs(q) == 4
                td.biome = BIOME_OCEAN if td.is_water else BIOME_PLAINS
                td.settlement = 1 if td.is_coast and q in [-3, 3] else 0
                terrain_data[hc] = td
                hex_coords_list.append((hc, float(q), float(r)))
        fg = FeatureGenerator(terrain_data, hex_coords_list)
        rng = np.random.default_rng(42)
        routes = fg.generate_shipping_routes(rng)
        assert isinstance(routes, list)


class TestPerlinNoiseCore:
    def test_noise2d_range(self):
        p = PerlinNoise(seed=42)
        values = [p.noise2d(x * 0.5, x * 0.3) for x in range(20)]
        assert min(values) >= -1.0
        assert max(values) <= 1.0

    def test_noise2d_seed_dependency(self):
        p1 = PerlinNoise(seed=42)
        p2 = PerlinNoise(seed=42)
        p3 = PerlinNoise(seed=43)
        assert p1.noise2d(1.5, 2.3) == p2.noise2d(1.5, 2.3)
        assert p1.noise2d(1.5, 2.3) != p3.noise2d(1.5, 2.3)

    def test_octave_noise_range(self):
        p = PerlinNoise(seed=42)
        for _ in range(20):
            v = p.octave_noise(
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                octaves=4,
            )
            assert -1.0 <= v <= 1.0

    def test_octave_noise_zero_scale(self):
        p = PerlinNoise(seed=42)
        v = p.octave_noise(1.0, 1.0, scale=0.001)
        assert -1.0 <= v <= 1.0

    def test_generate_elevation_shape(self):
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.1, i * 0.2) for i in range(50)]
        elev = ng.generate_elevation(coords)
        assert elev.shape == (50,)
        assert elev.min() >= 0.0
        assert elev.max() <= 1.0

    def test_generate_moisture_monsoon_effect(self):
        ng = NoiseGenerator(seed=42)
        coords = [(x, 0) for x in range(20)]
        elev = np.full(20, 0.5)
        moist_east = ng.generate_moisture(coords, elev, monsoon_dir=90.0)
        moist_west = ng.generate_moisture(coords, elev, monsoon_dir=270.0)
        assert not np.array_equal(moist_east, moist_west)

    def test_generate_temperature_latitude_effect(self):
        ng = NoiseGenerator(seed=42)
        coords = [(0, y) for y in range(-10, 11)]
        elev = np.full(21, 0.3)
        temp = ng.generate_temperature(elev, coords)
        assert temp[0] > temp[-1]