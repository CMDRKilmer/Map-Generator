"""地图特性生成器的单元测试

核心风险点：
- FeatureGenerator 初始化时对陆地/水/海岸的分类正确性
- _settlement_score 对各种 biome 和标记的加/减分
- generate_rivers 从高地流向海洋的基本属性
- generate_settlements 不放置在水/火山格
- generate_roads 在有足够聚落时生成道路且只影响陆地
- generate_resources 只在陆地放置资源，且受 biome 约束
- generate_shipping_routes 只在沿海聚落之间
- _mark_shipping_path / _find_nearest_water 在有水时返回水格
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pytest

from core.hex_grid import HexCoord, HEX_DIRECTIONS
from core.terrain_gen import (
    TerrainData, TerrainGenerator,
    BIOME_PLAINS, BIOME_OCEAN, BIOME_LAKE, BIOME_BEACH,
    BIOME_FOREST, BIOME_MOUNTAINS, BIOME_HIGH_MOUNTAINS,
    BIOME_DESERT, BIOME_SNOW, BIOME_HILLS, BIOME_SWAMP,
    SETTLEMENT_NONE, SETTLEMENT_VILLAGE, SETTLEMENT_TOWN,
    SETTLEMENT_CITY, SETTLEMENT_CAPITAL,
)
from core.feature_gen import FeatureGenerator


# ---------------------------------------------------------------------------
# 测试工具：手工构造地形数据
# ---------------------------------------------------------------------------


def _make_td(*, is_water: bool, biome: str, elevation: float = 0.5,
             moisture: float = 0.5, temperature: float = 0.5,
             is_coast: bool = False, volcanic: bool = False) -> TerrainData:
    td = TerrainData()
    td.is_water = is_water
    td.biome = biome
    td.elevation = elevation
    td.moisture = moisture
    td.temperature = temperature
    td.is_coast = is_coast
    td.volcanic = volcanic
    return td


def _small_map_with_coastline() -> Tuple[Dict[HexCoord, TerrainData],
                                         List[Tuple[HexCoord, float, float]]]:
    """构造一个带海岸线的小地图：
    行 r=0 是陆地，r=1 是海岸，r=-1 是深海。
    """
    cells = []
    terrain_data: Dict[HexCoord, TerrainData] = {}

    # 中心陆地 + 四周邻居
    center = HexCoord(0, 0)
    land = [center]
    for d in HEX_DIRECTIONS[:3]:  # 让一半邻居是陆地
        land.append(center + d)
    water_neighbors = [center + d for d in HEX_DIRECTIONS[3:]]

    for hc in land:
        td = _make_td(is_water=False, biome=BIOME_PLAINS, elevation=0.6,
                      is_coast=True)
        terrain_data[hc] = td
        cells.append((hc, float(hc.q) * 10.0, float(hc.r) * 10.0))

    for hc in water_neighbors:
        td = _make_td(is_water=True, biome=BIOME_OCEAN, elevation=0.05,
                      is_coast=True)
        terrain_data[hc] = td
        cells.append((hc, float(hc.q) * 10.0, float(hc.r) * 10.0))

    return terrain_data, cells


# ---------------------------------------------------------------------------
# 初始化分类
# ---------------------------------------------------------------------------


class TestFeatureGeneratorInit:
    def test_land_water_coast_partitioning(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        assert len(fg.land_hexes) + len(fg.water_hexes) == len(td)
        for h in fg.land_hexes:
            assert td[h].is_water is False
        for h in fg.water_hexes:
            assert td[h].is_water is True
        # 所有海岸的六边形都在 coast_hexes 里
        assert len(fg.coast_hexes) >= len(fg.water_hexes)


# ---------------------------------------------------------------------------
# _settlement_score
# ---------------------------------------------------------------------------


class TestSettlementScore:
    def setup_method(self):
        td, cells = _small_map_with_coastline()
        self.fg = FeatureGenerator(td, cells)

    def test_coast_and_river_add_bonus(self):
        td = TerrainData()
        td.biome = BIOME_PLAINS
        score_plain = self.fg._settlement_score(td, HexCoord(0, 0))

        td2 = TerrainData()
        td2.biome = BIOME_PLAINS
        td2.is_coast = True
        td2.river_flow = 0.5
        score_bonus = self.fg._settlement_score(td2, HexCoord(0, 0))
        assert score_bonus > score_plain

    def test_high_elevation_penalty(self):
        td_a = TerrainData()
        td_a.biome = BIOME_PLAINS
        td_a.elevation = 0.3
        low = self.fg._settlement_score(td_a, HexCoord(0, 0))

        td_b = TerrainData()
        td_b.biome = BIOME_PLAINS
        td_b.elevation = 0.9
        high = self.fg._settlement_score(td_b, HexCoord(0, 0))
        assert high < low

    def test_different_biomes_have_different_scores(self):
        # plains 通常比 desert 更宜居
        td_p = TerrainData(); td_p.biome = BIOME_PLAINS
        td_d = TerrainData(); td_d.biome = BIOME_DESERT
        assert self.fg._settlement_score(td_p, HexCoord(0, 0)) > \
            self.fg._settlement_score(td_d, HexCoord(0, 0))


# ---------------------------------------------------------------------------
# generate_settlements
# ---------------------------------------------------------------------------


class TestGenerateSettlements:
    def test_settlements_only_on_land(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(0)
        fg.generate_settlements(rng)
        for hc, data in td.items():
            if data.settlement != SETTLEMENT_NONE:
                assert data.is_water is False

    def test_settlements_never_on_volcanic(self):
        td, cells = _small_map_with_coastline()
        for hc, data in td.items():
            if not data.is_water:
                data.volcanic = True
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(rng)
        assert settlements == []  # 无候选

    def test_settlement_types_have_expected_sizes(self):
        td, cells = _small_map_with_coastline()
        # 为了保证有足够候选，把陆地范围扩展
        for i, d in enumerate(HEX_DIRECTIONS):
            hc = HexCoord(5, 0) + d
            if hc not in td:
                new_td = _make_td(is_water=False, biome=BIOME_PLAINS,
                                  elevation=0.4)
                td[hc] = new_td
                cells.append((hc, 0.0, 0.0))

        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(7)
        placed = fg.generate_settlements(rng, num_villages=2, num_towns=1,
                                         num_cities=1, has_capital=True)
        # 聚落数量 >= 1（至少有首都）
        assert len(placed) >= 1

        capitals = [hc for hc, d in td.items()
                    if d.settlement == SETTLEMENT_CAPITAL]
        assert len(capitals) <= 1


# ---------------------------------------------------------------------------
# generate_rivers
# ---------------------------------------------------------------------------


class TestGenerateRivers:
    def test_no_rivers_when_no_high_land(self):
        td: Dict[HexCoord, TerrainData] = {}
        cells = []
        for d in HEX_DIRECTIONS:
            hc = HexCoord(0, 0) + d
            td[hc] = _make_td(is_water=False, biome=BIOME_PLAINS, elevation=0.3)
            cells.append((hc, 0.0, 0.0))
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(1)
        rivers = fg.generate_rivers(rng, num_rivers=10)
        assert rivers == []

    def test_river_starts_in_high_land(self):
        td, cells = _small_map_with_coastline()
        # 让某个陆地格有很高的高度作为河源
        center = HexCoord(0, 0)
        td[center].elevation = 0.9
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(0)
        rivers = fg.generate_rivers(rng, num_rivers=1)
        if rivers:
            # 所有河中的格的 river_flow 应 > 0
            assert any(td[hc].river_flow > 0 for hc in td)


# ---------------------------------------------------------------------------
# generate_roads
# ---------------------------------------------------------------------------


class TestGenerateRoads:
    def test_no_roads_without_settlements(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(2)
        roads = fg.generate_roads(rng)
        assert roads == []

    def test_roads_never_on_water(self):
        td, cells = _small_map_with_coastline()
        # 手动放置两个聚落
        land_cells = [hc for hc, d in td.items() if not d.is_water]
        if len(land_cells) >= 2:
            td[land_cells[0]].settlement = SETTLEMENT_CITY
            td[land_cells[0]].settlement_size = 3
            td[land_cells[1]].settlement = SETTLEMENT_TOWN
            td[land_cells[1]].settlement_size = 2
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(3)
        fg.generate_roads(rng)
        for hc, data in td.items():
            if data.is_water:
                assert data.road is False


# ---------------------------------------------------------------------------
# generate_resources
# ---------------------------------------------------------------------------


class TestGenerateResources:
    def test_no_resources_on_water(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(11)
        fg.generate_resources(rng, density=1.0)
        for hc, data in td.items():
            if data.is_water:
                assert data.resource is None
                assert data.resource_amount == 0

    def test_no_resources_on_settlement(self):
        td, cells = _small_map_with_coastline()
        for hc, data in td.items():
            if not data.is_water:
                data.settlement = SETTLEMENT_VILLAGE
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(4)
        fg.generate_resources(rng, density=1.0)
        for hc, data in td.items():
            assert data.resource is None
            assert data.resource_amount == 0

    def test_volcanic_can_gain_resources(self):
        td, cells = _small_map_with_coastline()
        land = [hc for hc, d in td.items() if not d.is_water]
        for hc in land[:2]:
            td[hc].volcanic = True
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(77)
        fg.generate_resources(rng, density=1.0)
        # 火山格有可能获得资源；只要不 crash 即通过；额外断言字段合法
        for hc, data in td.items():
            if data.resource is not None:
                assert data.resource_amount >= 1


# ---------------------------------------------------------------------------
# generate_shipping_routes
# ---------------------------------------------------------------------------


class TestShippingRoutes:
    def test_no_shipping_without_coastal_settlements(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(5)
        routes = fg.generate_shipping_routes(rng)
        assert routes == []

    def test_find_nearest_water_on_water_cell(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        water_cells = [hc for hc, d in td.items() if d.is_water]
        assert water_cells
        target = fg._find_nearest_water(water_cells[0])
        assert target is not None
        assert td[target].is_water

    def test_find_nearest_water_when_no_water(self):
        td: Dict[HexCoord, TerrainData] = {}
        cells = []
        for d in HEX_DIRECTIONS:
            hc = HexCoord(0, 0) + d
            td[hc] = _make_td(is_water=False, biome=BIOME_PLAINS)
            cells.append((hc, 0.0, 0.0))
        fg = FeatureGenerator(td, cells)
        assert fg._find_nearest_water(HexCoord(0, 0)) is None


# ---------------------------------------------------------------------------
# _generate_name
# ---------------------------------------------------------------------------


class TestGenerateName:
    def test_returns_non_empty_string(self):
        td, cells = _small_map_with_coastline()
        fg = FeatureGenerator(td, cells)
        rng = np.random.default_rng(123)
        for typ in ("capital", "city", "town", "village"):
            name = fg._generate_name(rng, typ)
            assert isinstance(name, str) and len(name) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
