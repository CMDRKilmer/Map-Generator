"""地形分类与地形生成的单元测试

核心风险点：
- classify_biome 的各个阈值边界（elevation 0.15/0.35/0.55/0.65/0.80，
  moisture desert/swamp，temperature 雪线）
- TerrainGenerator.generate 的海岸检测逻辑（陆地邻居是否有水 / 水邻居是否有陆）
- resource / settlement / road 等字段初始化状态的正确性
"""
from __future__ import annotations
import numpy as np
import pytest

from core.terrain_gen import (
    TerrainGenerator, TerrainData,
    BIOME_OCEAN, BIOME_LAKE, BIOME_BEACH,
    BIOME_PLAINS, BIOME_FOREST, BIOME_DENSE_FOREST,
    BIOME_RAINFOREST, BIOME_SWAMP,
    BIOME_DESERT, BIOME_TUNDRA, BIOME_SAVANNA, BIOME_TAIGA,
    BIOME_HILLS, BIOME_MOUNTAINS, BIOME_HIGH_MOUNTAINS,
    BIOME_SNOW, BIOME_VOLCANO,
)


# ---------------------------------------------------------------------------
# classify_biome — 分支覆盖
# ---------------------------------------------------------------------------


class TestClassifyBiome:
    def setup_method(self):
        self.tg = TerrainGenerator()

    def _classify(self, elev, moist, temp, coast=False):
        return self.tg.classify_biome(elev, moist, temp, coast)

    # --- 水域 ---
    def test_ocean_below_015(self):
        assert self._classify(0.05, 0.5, 0.5) == BIOME_OCEAN
        assert self._classify(0.14, 0.9, 0.9) == BIOME_OCEAN

    def test_lake_between_015_and_water_level(self):
        # 默认 water_level=0.35
        assert self._classify(0.25, 0.5, 0.5) == BIOME_LAKE
        assert self._classify(0.34, 0.1, 0.9) == BIOME_LAKE

    def test_beach_at_coast_above_water_level(self):
        assert self._classify(0.36, 0.5, 0.5, coast=True) == BIOME_BEACH

    def test_not_beach_without_coast_flag(self):
        # 有海岸标志才会返回 beach，否则按低地分类
        assert self._classify(0.40, 0.5, 0.5, coast=False) != BIOME_BEACH

    # --- 山地 ---
    def test_snow_when_high_elevation_and_cold(self):
        assert self._classify(0.90, 0.5, 0.1) == BIOME_SNOW

    def test_no_snow_when_warm_high_mountain(self):
        assert self._classify(0.90, 0.5, 0.6) != BIOME_SNOW

    def test_high_mountains_band(self):
        # 0.80+ 且温度不低 -> HIGH_MOUNTAINS
        assert self._classify(0.85, 0.1, 0.6) == BIOME_HIGH_MOUNTAINS
        assert self._classify(0.95, 0.9, 0.7) == BIOME_HIGH_MOUNTAINS

    def test_mountains_band(self):
        # 0.65 ~ 0.80
        assert self._classify(0.70, 0.5, 0.5) == BIOME_MOUNTAINS
        assert self._classify(0.79, 0.5, 0.5) == BIOME_MOUNTAINS

    def test_hills_band(self):
        # 0.55 ~ 0.65
        assert self._classify(0.56, 0.5, 0.5) == BIOME_HILLS
        assert self._classify(0.64, 0.5, 0.5) == BIOME_HILLS

    # --- 低地生物群落（按 moisture / temperature） ---
    def test_desert_dry_hot(self):
        # moisture < desert_moisture (0.25) 且温度高 -> DESERT
        assert self._classify(0.45, 0.1, 0.8) == BIOME_DESERT

    def test_tundra_dry_cold(self):
        # moisture < 0.25 且温度低 -> TUNDRA
        assert self._classify(0.45, 0.1, 0.1) == BIOME_TUNDRA

    def test_savanna_moderate_dry_hot(self):
        # 0.25 <= moisture < 0.35 且温度高 -> SAVANNA
        assert self._classify(0.45, 0.30, 0.8) == BIOME_SAVANNA

    def test_taiga_moderate_dry_cold(self):
        # 0.35 <= moisture < 0.55 且温度 <= 0.5 -> TAIGA
        assert self._classify(0.45, 0.45, 0.2) == BIOME_TAIGA

    def test_plains_warm_mid_moisture(self):
        # 0.35 <= moisture < 0.55 且温度 > 0.5 -> PLAINS
        assert self._classify(0.45, 0.45, 0.7) == BIOME_PLAINS

    def test_rainforest_wet_hot(self):
        # moisture >= 0.75, temp > 0.75 -> RAINFOREST
        assert self._classify(0.45, 0.9, 0.9) == BIOME_RAINFOREST

    def test_dense_forest_wet_warm(self):
        # moisture >= 0.75, temp > 0.5 -> DENSE_FOREST
        assert self._classify(0.45, 0.80, 0.6) == BIOME_DENSE_FOREST

    def test_swamp_wet_cool_low_elevation(self):
        # moisture > swamp_moisture(0.80) AND elevation < 0.4 AND temp <= 0.5 -> SWAMP
        # 注意 elevation 必须 >= water_level 否则会被归类为湖/海
        assert self._classify(0.37, 0.82, 0.4) == BIOME_SWAMP

    def test_forest_band(self):
        # moisture 在 0.55 ~ 0.75，温度适中 -> FOREST
        assert self._classify(0.45, 0.70, 0.6) == BIOME_FOREST


# ---------------------------------------------------------------------------
# generate() — 海岸检测 + 数据一致性
# ---------------------------------------------------------------------------


def _build_simple_coords():
    """构造一个 3x3 的扁平“地图”：中间是陆地，周围是水。
    返回 (hex_coords_list, elevation, moisture, temperature, water_level)
    为了让海岸检测真正起作用，我们需要邻居关系。我们手动提供 7 个六边形。
    """
    from core.hex_grid import HexCoord

    # 中心 + 6 个邻居（即半径 1 的范围）共 7 个六边形
    cells = [HexCoord(0, 0)] + [d for d in [
        HexCoord(1, 0), HexCoord(1, -1), HexCoord(0, -1),
        HexCoord(-1, 0), HexCoord(-1, 1), HexCoord(0, 1),
    ]]
    hex_size = 10.0
    hex_coords_list = [(hc, hc.q * hex_size * 1.5, 0.0) for hc in cells]
    return cells, hex_coords_list


class TestGenerate:
    def test_generate_runs_and_returns_dict(self):
        tg = TerrainGenerator()
        cells, hex_coords_list = _build_simple_coords()
        n = len(cells)
        # 把中心保持为陆地（elevation 高），邻居设为水（elevation 低）
        elevation = np.array([0.7 if i == 0 else 0.1 for i in range(n)])
        moisture = np.full(n, 0.5)
        temperature = np.full(n, 0.6)
        result = tg.generate(elevation, moisture, temperature, hex_coords_list)
        assert isinstance(result, dict)
        assert len(result) == n

    def test_generate_coast_flag_for_water_cell_next_to_land(self):
        tg = TerrainGenerator()
        cells, hex_coords_list = _build_simple_coords()
        n = len(cells)
        # 中心为陆地，邻居为水
        elevation = np.array([0.7 if i == 0 else 0.1 for i in range(n)])
        moisture = np.full(n, 0.5)
        temperature = np.full(n, 0.6)
        result = tg.generate(elevation, moisture, temperature, hex_coords_list)

        center = cells[0]
        # 中心是陆地，且邻居有水，所以应当 is_coast=True
        assert bool(result[center].is_water) is False
        assert bool(result[center].is_coast) is True

        # 邻居是水，且邻居与陆地相邻，也是海岸
        for c in cells[1:]:
            assert bool(result[c].is_water) is True
            assert bool(result[c].is_coast) is True

    def test_generate_center_biome_in_land_set(self):
        tg = TerrainGenerator()
        cells, hex_coords_list = _build_simple_coords()
        n = len(cells)
        elevation = np.array([0.7 if i == 0 else 0.1 for i in range(n)])
        moisture = np.full(n, 0.5)
        temperature = np.full(n, 0.6)
        result = tg.generate(elevation, moisture, temperature, hex_coords_list)
        land_biomes = {
            BIOME_BEACH, BIOME_PLAINS, BIOME_FOREST, BIOME_RAINFOREST,
            BIOME_TAIGA, BIOME_DESERT, BIOME_SAVANNA, BIOME_HILLS,
            BIOME_MOUNTAINS, BIOME_HIGH_MOUNTAINS, BIOME_SNOW,
            BIOME_SWAMP, BIOME_VOLCANO, BIOME_TUNDRA,
        }
        center = cells[0]
        assert result[center].biome in land_biomes

    def test_terrain_data_defaults(self):
        td = TerrainData()
        assert td.river_flow == 0.0
        assert td.resource is None
        assert td.resource_amount == 0
        assert td.settlement == 0
        assert td.settlement_name == ""
        assert td.road is False
        assert td.shipping is False
        assert td.volcanic is False


# ---------------------------------------------------------------------------
# get_biome_color_key
# ---------------------------------------------------------------------------


def test_biome_color_key_identity():
    tg = TerrainGenerator()
    assert tg.get_biome_color_key("plains") == "plains"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
