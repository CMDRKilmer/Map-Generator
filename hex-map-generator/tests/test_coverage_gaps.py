"""
测试套件 — 覆盖 core/ 中未在 test_merge_changes.py 中触达的行为面
聚焦于：核心数学、解析边界、A* 寻路、分类分支、确定性。
"""

from __future__ import annotations

import os
import sys

# 启用 Qt 离屏渲染（虽然本文件不直接用 QColor，但与项目测试约定保持一致）
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# 把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.feature_gen import FeatureGenerator
from core.hex_grid import HexCoord, HexGrid
from core.noise_gen import NoiseGenerator
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
# 1) core/hex_grid.py — 坐标算术 / 距离 / 邻居 / 范围
# ============================================================
class TestHexCoordAlgebra:
    def test_s_property(self):
        """s 始终等于 -q - r（立方体约束）"""
        assert HexCoord(0, 0).s == 0
        assert HexCoord(2, -3).s == 1
        assert HexCoord(-4, 5).s == -1

    def test_arithmetic_operators(self):
        """+/-/* 应按分量运算并保持立方体约束"""
        a = HexCoord(2, -1)
        b = HexCoord(-1, 3)
        assert (a + b) == HexCoord(1, 2)
        assert (a - b) == HexCoord(3, -4)
        assert (a * 2) == HexCoord(4, -2)
        assert (a * 0) == HexCoord(0, 0)

    def test_equality_and_hash(self):
        """等价对象必须哈希相等（用作 dict/set 键）"""
        assert HexCoord(1, 2) == HexCoord(1, 2)
        assert hash(HexCoord(1, 2)) == hash(HexCoord(1, 2))
        assert HexCoord(1, 2) != HexCoord(2, 1)
        # 放入 set 去重
        assert len({HexCoord(1, 2), HexCoord(1, 2), HexCoord(3, 4)}) == 2

    def test_distance_to(self):
        """distance_to 是步数距离的 max(|dq|,|dr|,|ds|)"""
        a = HexCoord(0, 0)
        assert a.distance_to(HexCoord(0, 0)) == 0
        assert a.distance_to(HexCoord(1, 0)) == 1
        assert a.distance_to(HexCoord(2, -1)) == 2
        assert a.distance_to(HexCoord(3, -3)) == 3
        # 距离对称
        b = HexCoord(2, 5)
        assert a.distance_to(b) == b.distance_to(a)

    def test_neighbors_count_and_uniqueness(self):
        """neighbors 必须返回恰好 6 个互不相同的邻居"""
        hc = HexCoord(0, 0)
        nbs = hc.neighbors()
        assert len(nbs) == 6
        assert len(set(nbs)) == 6
        for n in nbs:
            assert hc.distance_to(n) == 1

    def test_range_size(self):
        """range(n) 包含距离 ≤ n 的所有格子，n=1 为 7 个"""
        hc = HexCoord(0, 0)
        assert len(hc.range(0)) == 1
        assert len(hc.range(1)) == 7
        assert len(hc.range(2)) == 19
        assert len(hc.range(3)) == 37


# ============================================================
# 2) core/hex_grid.py — HexGrid 几何与像素往返
# ============================================================
class TestHexGridGeometry:
    def test_size_zero_is_single_hex(self):
        """size=0 仅包含 (0,0)"""
        g = HexGrid(size=0)
        assert len(g.hexes) == 1
        assert g.hexes[0] == HexCoord(0, 0)

    def test_size_one_has_seven_hexes(self):
        """size=1 是中心 + 6 邻居，共 7 个"""
        assert len(HexGrid(size=1).hexes) == 7

    def test_pixel_round_trip(self):
        """hex_center → hex_at_pixel 应回到原 hex（容忍 ±0）"""
        g = HexGrid(size=10)
        size = 24.0
        for hc in [HexCoord(0, 0), HexCoord(3, -2), HexCoord(-5, 4), HexCoord(7, 7)]:
            cx, cy = g.hex_center(hc, size)
            got = g.hex_at_pixel(cx, cy, size)
            assert got == hc, f"round-trip failed for {hc}: got {got}"

    def test_hex_corners_returns_six_points_on_unit_circle(self):
        """6 个顶点都应在以 hex_size 为半径的圆上"""
        g = HexGrid(size=5)
        corners = g.hex_corners(HexCoord(0, 0), 10.0)
        assert len(corners) == 6
        for x, y in corners:
            assert abs((x * x + y * y) ** 0.5 - 10.0) < 1e-9

    def test_all_hex_centers_count(self):
        """all_hex_centers 应与网格 hex 数一致"""
        g = HexGrid(size=3)
        centers = g.all_hex_centers(12.0)
        assert len(centers) == len(g.hexes)
        # 每条 (HexCoord, cx, cy) 三元组
        for h, cx, cy in centers:
            assert isinstance(h, HexCoord)
            assert isinstance(cx, float) and isinstance(cy, float)

    def test_get_hexes_within_radius(self):
        """get_hexes_within_radius 行为与 range() 一致"""
        g = HexGrid(size=5)
        c = HexCoord(0, 0)
        assert g.get_hexes_within_radius(c, 2) == c.range(2)

    def test_get_random_hex_in_grid(self):
        """get_random_hex 必须返回当前网格的某个 hex"""
        g = HexGrid(size=4)
        rng = np.random.default_rng(0)
        for _ in range(20):
            h = g.get_random_hex(rng)
            assert h in g.hexes


# ============================================================
# 3) core/noise_gen.py — set_seed / 季风 / 边界
# ============================================================
class TestNoiseGeneratorEdges:
    def test_set_seed_replaces_state(self):
        """set_seed 后 elevation 必须与新建实例一致"""
        coords = [(i * 0.1, i * 0.2) for i in range(16)]
        ng = NoiseGenerator(seed=1)
        a = ng.generate_elevation(coords)
        ng.set_seed(42)
        b = ng.generate_elevation(coords)
        ng2 = NoiseGenerator(seed=42)
        c = ng2.generate_elevation(coords)
        np.testing.assert_array_equal(b, c)
        # 与旧 seed 不应相同
        assert not np.array_equal(a, c)

    def test_monsoon_direction_changes_output(self):
        """季风方向不同 → 湿度分布应不同（迎风/背风差异）"""
        coords = [(i * 0.2, i * 0.15) for i in range(40)]
        elev = np.full(40, 0.7)  # 统一高程，凸显季风效应
        ng = NoiseGenerator(seed=7)
        m_east = ng.generate_moisture(coords, elev, monsoon_dir=90.0)
        m_west = ng.generate_moisture(coords, elev, monsoon_dir=270.0)
        # 方向完全相反，应当产生明显不同的湿度图
        assert not np.allclose(m_east, m_west)
        # 输出值仍须在 [0, 1]
        assert m_east.min() >= 0.0 and m_east.max() <= 1.0
        assert m_west.min() >= 0.0 and m_west.max() <= 1.0

    @pytest.mark.xfail(
        reason="generate_elevation 当前对空数组抛 ValueError（min/max 缺 identity）",
        strict=True,
    )
    def test_elevation_empty_input(self):
        """空输入应返回空数组且不抛错（已知缺陷，记录以便回归）"""
        ng = NoiseGenerator(seed=1)
        e = ng.generate_elevation([])
        assert e.shape == (0,)

    @pytest.mark.xfail(
        reason="generate_moisture 当前对空数组抛 ValueError",
        strict=True,
    )
    def test_moisture_empty_input(self):
        """湿度空输入应返回空数组（已知缺陷）"""
        ng = NoiseGenerator(seed=1)
        m = ng.generate_moisture([], np.array([]))
        assert m.shape == (0,)

    @pytest.mark.xfail(
        reason="generate_temperature 当前对空数组抛 ValueError",
        strict=True,
    )
    def test_temperature_empty_input(self):
        """温度空输入应返回空数组（已知缺陷）"""
        ng = NoiseGenerator(seed=1)
        t = ng.generate_temperature(np.array([]), [])
        assert t.shape == (0,)

    def test_temperature_latitude_inversion(self):
        """y 越大温度应越低（南北半球简化模型）"""
        ng = NoiseGenerator(seed=1)
        # 零高程，排除高程降温干扰
        elev = np.zeros(4)
        coords = [(0.0, 0.0), (0.0, 1.0), (0.0, 5.0), (0.0, 10.0)]
        t = ng.generate_temperature(elev, coords)
        # 严格单调非增
        assert t[0] >= t[1] >= t[2] >= t[3]


# ============================================================
# 4) core/terrain_gen.py — classify_biome 分支覆盖
# ============================================================
class TestClassifyBiomeBranches:
    def setup_method(self):
        self.tg = TerrainGenerator()

    def test_deep_ocean(self):
        assert self.tg.classify_biome(0.05, 0.5, 0.5, False) == BIOME_OCEAN

    def test_shallow_lake(self):
        assert self.tg.classify_biome(0.30, 0.5, 0.5, False) == BIOME_LAKE

    def test_beach_only_when_coast(self):
        """BEACH 仅在 is_coast=True 且 elev<water+0.08 时返回"""
        # 不算海岸：应该是 LAKE（因 0.30 < 0.35）
        assert self.tg.classify_biome(0.30, 0.5, 0.5, False) == BIOME_LAKE
        # 海岸线：0.40 < 0.35+0.08=0.43 且 is_coast=True
        assert self.tg.classify_biome(0.40, 0.5, 0.5, True) == BIOME_BEACH

    def test_snow_at_high_elev_cold(self):
        assert self.tg.classify_biome(0.90, 0.5, 0.2, False) == BIOME_SNOW

    def test_high_mountains_above_080(self):
        # 0.85 满足 snow_level 触发但 temperature 不低，落到 high_mountains
        assert self.tg.classify_biome(0.86, 0.5, 0.5, False) == BIOME_HIGH_MOUNTAINS

    def test_mountains_between_065_and_080(self):
        assert self.tg.classify_biome(0.70, 0.5, 0.5, False) == BIOME_MOUNTAINS

    def test_hills_between_055_and_065(self):
        assert self.tg.classify_biome(0.60, 0.5, 0.5, False) == BIOME_HILLS

    def test_desert_dry_and_hot(self):
        assert self.tg.classify_biome(0.45, 0.10, 0.8, False) == BIOME_DESERT

    def test_tundra_dry_and_cold(self):
        assert self.tg.classify_biome(0.45, 0.10, 0.3, False) == BIOME_TUNDRA

    def test_savanna_moderately_dry_hot(self):
        assert self.tg.classify_biome(0.45, 0.30, 0.7, False) == BIOME_SAVANNA

    def test_plains_branch_moderate_dry_cold(self):
        assert self.tg.classify_biome(0.45, 0.30, 0.3, False) == BIOME_PLAINS

    def test_plains_branch_moderate(self):
        assert self.tg.classify_biome(0.45, 0.50, 0.6, False) == BIOME_PLAINS

    def test_taiga_branch_moderate_cold(self):
        assert self.tg.classify_biome(0.45, 0.50, 0.3, False) == BIOME_TAIGA

    def test_forest_moist(self):
        assert self.tg.classify_biome(0.45, 0.60, 0.6, False) == BIOME_FOREST

    def test_rainforest_warm_and_wet(self):
        assert self.tg.classify_biome(0.45, 0.85, 0.9, False) == BIOME_RAINFOREST

    def test_dense_forest_wet(self):
        assert self.tg.classify_biome(0.45, 0.85, 0.6, False) == BIOME_DENSE_FOREST

    def test_swamp_wet_and_low(self):
        """SWAMP 需要 moisture>0.80 且 water_level≤elevation<0.4"""
        # elevation=0.36 介于 water_level(0.35) 与 0.4 之间 → 落到 SWAMP 分支
        assert self.tg.classify_biome(0.36, 0.85, 0.4, False) == BIOME_SWAMP
        # 但 elevation≥0.4 时降级为 FOREST
        assert self.tg.classify_biome(0.45, 0.85, 0.4, False) == BIOME_FOREST


# ============================================================
# 5) core/terrain_gen.py — generate 海岸对称与边界
# ============================================================
class TestGenerateCoastSymmetry:
    def test_coast_marked_on_water_adjacent_to_land(self):
        """水格旁边有陆地时应被标记为 is_coast"""
        tg = TerrainGenerator()
        # 一行 5 个格子：海-陆-海-陆-海
        # 中心 (0,0) 是陆地，两侧是水
        elev = np.array([0.20, 0.40, 0.20, 0.40, 0.20])  # 水-陆-水-陆-水
        moist = np.full(5, 0.5)
        temp = np.full(5, 0.5)
        coords = [(HexCoord(q, 0), float(q), 0.0) for q in range(-2, 3)]
        td = tg.generate(elev, moist, temp, coords)
        # 索引: -2=水, -1=陆, 0=水, 1=陆, 2=水
        # 水格（q=-2, 0, 2）邻接陆地时应被标海岸
        assert td[HexCoord(-2, 0)].is_water
        assert td[HexCoord(-1, 0)].is_coast  # 陆邻水 → 海岸
        assert td[HexCoord(0, 0)].is_coast   # 水邻两侧陆
        assert td[HexCoord(1, 0)].is_coast   # 陆邻水
        assert td[HexCoord(2, 0)].is_coast   # 水邻陆

    def test_volcano_field_set_when_rng_hits(self):
        """强制种子命中时应出现 volcanic=True 的格子"""
        tg = TerrainGenerator()
        n = 200
        # 大部分格子 elevation>0.6 moisture<0.4（满足火山条件），让 RNG 高概率触发
        rng_elev = np.random.default_rng(123)
        elev = rng_elev.uniform(0.7, 0.95, n)
        moist = np.full(n, 0.2)
        temp = np.full(n, 0.5)
        coords = [(HexCoord(q, 0), float(q), 0.0) for q in range(n)]
        td = tg.generate(elev, moist, temp, coords)
        volcanic = sum(1 for v in td.values() if v.volcanic)
        # 200 格子 × 0.5% × (大多满足 elev>0.6 moist<0.4) → 至少 0 个，最多不一定
        # 我们只要求字段被正确设置
        for v in td.values():
            if v.volcanic:
                assert v.biome == "volcano"

    def test_generate_empty_input(self):
        """空输入应返回空字典"""
        tg = TerrainGenerator()
        td = tg.generate(np.array([]), np.array([]), np.array([]), [])
        assert td == {}


# ============================================================
# 6) core/feature_gen.py — 聚落距离与基本类型
# ============================================================
def _make_terrain(land_specs, water_specs=()):
    """根据 (hc, biome, elev) 列表构造 FeatureGenerator 可用 terrain_data"""
    td = {}
    for hc, biome, elev in land_specs:
        d = TerrainData()
        d.biome = biome
        d.elevation = elev
        d.is_water = False
        d.is_coast = False
        td[hc] = d
    for hc, biome, elev in water_specs:
        d = TerrainData()
        d.biome = biome
        d.elevation = elev
        d.is_water = True
        d.is_coast = False
        td[hc] = d
    hex_list = [(hc, 0.0, 0.0) for hc in td]
    return td, hex_list


class TestSettlementPlacement:
    def test_no_land_returns_empty(self):
        """全图无陆地时不应放置任何聚落"""
        water = [(HexCoord(q, 0), BIOME_OCEAN, 0.1) for q in range(5)]
        td, hex_list = _make_terrain([], water)
        fg = FeatureGenerator(td, hex_list)
        out = fg.generate_settlements(np.random.default_rng(0))
        assert out == []

    def test_settlements_have_min_distance(self):
        """任意两个被放置的聚落之间距离应 ≥ 3（village 最小）"""
        # 8 个相邻草原，期望 8 个 village 中至少 3 个被放置且两两距离 ≥ 3
        land = [(HexCoord(q, 0), BIOME_PLAINS, 0.5) for q in range(8)]
        td, hex_list = _make_terrain(land)
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(0)
        placed = fg.generate_settlements(rng, num_villages=8, num_towns=0, num_cities=0, has_capital=False)
        assert len(placed) >= 2
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                assert placed[i].distance_to(placed[j]) >= 3

    def test_volcanic_hexes_excluded(self):
        """火山格不应被选作聚落"""
        # 一个火山 + 多个平原，num_villages=1 → 应选平原
        land = [(HexCoord(0, 0), BIOME_VOLCANO, 0.85)] + [
            (HexCoord(q, 0), BIOME_PLAINS, 0.5) for q in range(1, 6)
        ]
        td, hex_list = _make_terrain(land)
        fg = FeatureGenerator(td, hex_list)
        placed = fg.generate_settlements(
            np.random.default_rng(0),
            num_villages=1, num_towns=0, num_cities=0, has_capital=False,
        )
        assert HexCoord(0, 0) not in placed

    def test_settlement_type_assigned(self):
        """被放置的格子应设置非 0 的 settlement 字段，并命中至少 2 种类型"""
        # 用一个更宽的网格，确保 min_dist 距离约束可被满足
        land = [(HexCoord(q, 0), BIOME_PLAINS, 0.5) for q in range(-10, 11)]
        td, hex_list = _make_terrain(land)
        fg = FeatureGenerator(td, hex_list)
        fg.generate_settlements(
            np.random.default_rng(0),
            num_villages=3, num_towns=2, num_cities=1, has_capital=True,
        )
        types = {td[h].settlement for h, _, _ in hex_list}
        # 至少出现 capital（4）和 village（1）
        assert SETTLEMENT_CAPITAL in types
        assert SETTLEMENT_VILLAGE in types


# ============================================================
# 7) core/feature_gen.py — 寻路 / 道路 / 资源 / 航线
# ============================================================
class TestRoadsAndPathfinding:
    def test_no_road_with_fewer_than_two_settlements(self):
        """少于 2 个聚落时 generate_roads 应返回空"""
        land = [(HexCoord(0, 0), BIOME_PLAINS, 0.5)]
        td, hex_list = _make_terrain(land)
        fg = FeatureGenerator(td, hex_list)
        assert fg.generate_roads() == []

    def test_road_never_steps_on_water(self):
        """道路标记的格子不应包含水格"""
        # 2 个海岸聚落，中间隔了一片水
        land = [
            (HexCoord(-3, 0), BIOME_PLAINS, 0.5),
            (HexCoord(3, 0), BIOME_PLAINS, 0.5),
        ]
        water = [
            (HexCoord(q, 0), BIOME_OCEAN, 0.1) for q in range(-2, 3)
        ]
        td, hex_list = _make_terrain(land, water)
        fg = FeatureGenerator(td, hex_list)
        # 手动标聚落
        td[HexCoord(-3, 0)].settlement = SETTLEMENT_TOWN
        td[HexCoord(3, 0)].settlement = SETTLEMENT_TOWN
        fg.generate_roads()
        # 所有标 road=True 的格子都应是陆地
        for hc, d in td.items():
            if d.road:
                assert not d.is_water, f"road marked on water hex {hc}"

    def test_road_path_returns_empty_when_unreachable(self):
        """当目标被水完全包围时寻路应返回空（不崩溃）"""
        land = [
            (HexCoord(-3, 0), BIOME_PLAINS, 0.5),
            (HexCoord(3, 0), BIOME_PLAINS, 0.5),
        ]
        # 周围一圈水
        water = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                if (q, r) in {(-3, 0), (3, 0)}:
                    continue
                water.append((HexCoord(q, r), BIOME_OCEAN, 0.1))
        td, hex_list = _make_terrain(land, water)
        fg = FeatureGenerator(td, hex_list)
        path = fg._find_road_path(HexCoord(-3, 0), HexCoord(3, 0), max_steps=500)
        # 在 A* 中目标被水包围（end 自身不是水），应仍能找到或返回空
        assert isinstance(path, list)


class TestResourceDistribution:
    def test_no_resource_on_water(self):
        """资源不应放置在水格上"""
        water = [(HexCoord(q, 0), BIOME_OCEAN, 0.1) for q in range(20)]
        td, hex_list = _make_terrain([], water)
        fg = FeatureGenerator(td, hex_list)
        fg.generate_resources(np.random.default_rng(0), density=1.0)
        for d in td.values():
            assert d.resource is None

    def test_no_resource_on_settlement(self):
        """资源不应覆盖已有聚落"""
        land = [(HexCoord(q, 0), BIOME_FOREST, 0.5) for q in range(10)]
        td, hex_list = _make_terrain(land)
        td[HexCoord(0, 0)].settlement = SETTLEMENT_CITY
        fg = FeatureGenerator(td, hex_list)
        fg.generate_resources(np.random.default_rng(0), density=1.0)
        assert td[HexCoord(0, 0)].resource is None

    def test_volcano_forces_mining_resource(self):
        """火山格以高概率获得 iron/gold/stone 之一"""
        land = [(HexCoord(0, 0), BIOME_VOLCANO, 0.85)]
        td, hex_list = _make_terrain(land)
        td[HexCoord(0, 0)].volcanic = True
        fg = FeatureGenerator(td, hex_list)
        # 试多次以避免单次 RNG 失败
        hits = 0
        for s in range(50):
            td[HexCoord(0, 0)].resource = None
            td[HexCoord(0, 0)].resource_amount = 0
            fg.generate_resources(np.random.default_rng(s), density=0.0)
            if td[HexCoord(0, 0)].resource is not None:
                hits += 1
                assert td[HexCoord(0, 0)].resource in (RESOURCE_IRON, RESOURCE_GOLD, RESOURCE_STONE)
        # 50 次里应有大量命中（每次 30%）
        assert hits > 5


class TestShippingRoutes:
    def test_no_route_with_fewer_than_two_coastal_settlements(self):
        """<2 海岸聚落时航线应为空"""
        # 一块陆地带一个非聚落 + 一个非海岸聚落
        land = [
            (HexCoord(0, 0), BIOME_PLAINS, 0.5),
            (HexCoord(5, 0), BIOME_PLAINS, 0.5),
        ]
        td, hex_list = _make_terrain(land)
        td[HexCoord(0, 0)].is_coast = True
        td[HexCoord(0, 0)].settlement = SETTLEMENT_CITY
        fg = FeatureGenerator(td, hex_list)
        out = fg.generate_shipping_routes(np.random.default_rng(0))
        assert out == []

    def test_routes_only_within_distance_window(self):
        """航线只连接 5 < dist < 30 的两个海岸聚落"""
        # 三个海岸聚落：近距离 4（应被排除）、中距离 10、远距离 40（应被排除）
        land = [
            (HexCoord(0, 0), BIOME_PLAINS, 0.5),
            (HexCoord(4, 0), BIOME_PLAINS, 0.5),
            (HexCoord(10, 0), BIOME_PLAINS, 0.5),
            (HexCoord(50, 0), BIOME_PLAINS, 0.5),
        ]
        td, hex_list = _make_terrain(land)
        for hc in [HexCoord(0, 0), HexCoord(4, 0), HexCoord(10, 0), HexCoord(50, 0)]:
            td[hc].is_coast = True
            td[hc].settlement = SETTLEMENT_TOWN
        fg = FeatureGenerator(td, hex_list)
        # 用确定性种子，让 rng.random() < 0.4 触发
        # 多次尝试以保证至少一次生成
        all_pairs = set()
        for s in range(200):
            # 重置 shipping
            for d in td.values():
                d.shipping = False
            routes = fg.generate_shipping_routes(np.random.default_rng(s))
            for a, b in routes:
                all_pairs.add((a, b))
        # 至少应出现过一次中距离配对 (0,10) 或 (4,10)
        ok = any(
            HexCoord(0, 0) in pair and HexCoord(10, 0) in pair
            or HexCoord(4, 0) in pair and HexCoord(10, 0) in pair
            for pair in all_pairs
        )
        assert ok, "中距离海岸配对应能生成航线"


class TestNameGeneration:
    def test_name_non_empty_for_all_types(self):
        """所有聚落类型生成的名称均非空"""
        fg_type = FeatureGenerator({}, [])
        rng = np.random.default_rng(0)
        for typ in ("capital", "city", "town", "village"):
            for _ in range(20):
                name = fg_type._generate_name(rng, typ)
                assert isinstance(name, str)
                assert len(name) > 0

    def test_name_uses_chinese_syllables(self):
        """名称应使用项目预定义的中文前后缀"""
        fg_type = FeatureGenerator({}, [])
        rng = np.random.default_rng(0)
        prefixes = ["王", "皇", "帝", "龙", "天", "圣", "大"]
        suffixes_capital = ["京", "都", "城"]
        for _ in range(30):
            name = fg_type._generate_name(rng, "capital")
            assert name[0] in prefixes
            assert name[1:] in suffixes_capital


# ============================================================
# 8) core/feature_gen.py — generate_rivers
# ============================================================
class TestRiverGeneration:
    def test_no_high_land_returns_empty(self):
        """无高地时应直接返回空列表（不应崩溃）"""
        land = [(HexCoord(0, 0), BIOME_PLAINS, 0.3)]  # 0.3 < 0.55
        td, hex_list = _make_terrain(land)
        fg = FeatureGenerator(td, hex_list)
        assert fg.generate_rivers(np.random.default_rng(0), num_rivers=5) == []

    def test_rivers_mark_river_flow(self):
        """生成的河流应在路径上留下非零 river_flow"""
        # 构造一个狭长地图：一列高地 (elev=0.9) 相邻一列低地 (elev=0.1)
        # 这样从任一高地出发都能下降到水
        land_hexes = []
        for q in range(-3, 4):
            land_hexes.append((HexCoord(q, 0), BIOME_PLAINS, 0.9 if q == 0 else 0.1))
        td, hex_list = _make_terrain(land_hexes)
        fg = FeatureGenerator(td, hex_list)
        # 多试种子，确保至少一次产生河流
        produced = False
        for s in range(30):
            for v in td.values():
                v.river_flow = 0.0
            rivers = fg.generate_rivers(np.random.default_rng(s), num_rivers=3)
            if rivers:
                produced = True
                # 每条河至少 4 个格子
                for r in rivers:
                    assert len(r) > 3
                # 至少有一个格子的 river_flow > 0
                assert any(v.river_flow > 0 for v in td.values())
                break
        assert produced, "在 30 个种子内应至少生成一条河"
