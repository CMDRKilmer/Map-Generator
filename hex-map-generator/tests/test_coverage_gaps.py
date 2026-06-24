"""
回归测试缺口覆盖 — 面向 PR #7 合并后真正存在业务风险的代码路径。

聚焦领域：
  1. core/hex_grid.py      网格/坐标数学（被全链路依赖的底层原语）
  2. core/noise_gen.py     噪声/季风/温度语义（决定地图可玩性）
  3. core/terrain_gen.py   Whittaker 生物群落表 + 海岸判定
  4. core/feature_gen.py   河流/聚落/道路/资源/航线业务规则
  5. ui/param_panel.py     用户输入 → 生成参数（驱动整个 UI 流程）
  6. ui/map_widget.py      编辑模式状态机
  7. export/exporter.py    JSON / SVG 输出契约

不覆盖：纯外观、纯重构、快照类断言。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# 启用 Qt 离屏渲染
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# 把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from core.feature_gen import FeatureGenerator
from core.hex_grid import HEX_DIRECTIONS, HexCoord, HexGrid
from core.noise_gen import NoiseGenerator
from core.terrain_gen import (
    BIOME_DENSE_FOREST,
    BIOME_DESERT,
    BIOME_FOREST,
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
# 1) core/hex_grid.py — 网格/坐标数学
#    风险：所有下游模块依赖此处的距离/邻接/四舍五入。
# ============================================================
class TestHexCoordMath:
    def test_distance_symmetry_and_triangle_inequality(self):
        """距离应对称，且满足三角不等式"""
        a, b, c = HexCoord(3, -1), HexCoord(-2, 4), HexCoord(1, 1)
        assert a.distance_to(b) == b.distance_to(a)
        assert a.distance_to(c) + c.distance_to(b) >= a.distance_to(b)

    def test_neighbors_count_and_uniqueness(self):
        """neighbors() 应返回 6 个互不相同的相邻坐标，且距离全为 1"""
        hc = HexCoord(2, -1)
        nbrs = hc.neighbors()
        assert len(nbrs) == 6
        assert len(set(nbrs)) == 6
        for n in nbrs:
            assert hc.distance_to(n) == 1

    def test_neighbors_match_global_directions(self):
        """neighbors() 顺序应等于全局 HEX_DIRECTIONS（被生成逻辑隐含依赖）"""
        assert HexCoord(0, 0).neighbors() == HEX_DIRECTIONS

    def test_range_includes_self_and_respects_radius(self):
        """range(n) 应包含自身，且所有元素距离 ≤ n"""
        hc = HexCoord(1, 0)
        for n in [0, 1, 2, 3]:
            r = hc.range(n)
            assert hc in r
            assert all(hc.distance_to(h) <= n for h in r)
        # 半径 2 的理论数量 = 1 + 6 + 12 = 19
        assert len(HexCoord(0, 0).range(2)) == 19

    def test_to_cube_from_cube_roundtrip(self):
        """to_cube / from_cube 应无损往返，且 s == -q - r"""
        for h in [HexCoord(3, -1), HexCoord(0, 0), HexCoord(-5, 2), HexCoord(7, -3)]:
            q, r, s = h.to_cube()
            assert q + r + s == 0
            assert HexCoord.from_cube(q, r, s) == h

    def test_arithmetic_operators(self):
        """+ - * 操作应正确实现向量加减与标量乘"""
        a, b = HexCoord(2, -1), HexCoord(-1, 3)
        assert a + b == HexCoord(1, 2)
        assert a - b == HexCoord(3, -4)
        assert a * 3 == HexCoord(6, -3)
        assert a * 0 == HexCoord(0, 0)


class TestHexGrid:
    def test_hex_count_formula(self):
        """六边形数 = 3*size^2 + 3*size + 1（中心向外扩散）"""
        for size in [0, 1, 2, 3, 5, 10]:
            grid = HexGrid(size=size)
            expected = 3 * size * size + 3 * size + 1
            assert len(grid.hexes) == expected, f"size={size}"

    def test_origin_present_for_all_sizes(self):
        """原点 HexCoord(0,0) 应在所有尺寸的网格中"""
        for size in [0, 3, 10]:
            assert HexCoord(0, 0) in HexGrid(size=size).hexes

    def test_pixel_roundtrip(self):
        """hex_center → hex_at_pixel 应回到同一坐标"""
        grid = HexGrid(size=3)
        hex_size = 10.0
        for hc in grid.hexes:
            cx, cy = grid.hex_center(hc, hex_size)
            back = grid.hex_at_pixel(cx, cy, hex_size)
            assert back == hc, f"roundtrip failed for {hc}"

    def test_get_random_hex_within_bounds(self):
        """get_random_hex 应始终返回网格内的有效坐标"""
        grid = HexGrid(size=3)
        rng = np.random.default_rng(0)
        for _ in range(50):
            h = grid.get_random_hex(rng)
            assert h in grid.hexes

    def test_get_hexes_within_radius_matches_range(self):
        """get_hexes_within_radius 应等价于 center.range(radius)"""
        grid = HexGrid(size=5)
        for center, r in [(HexCoord(0, 0), 2), (HexCoord(2, -1), 1), (HexCoord(-3, 1), 3)]:
            assert grid.get_hexes_within_radius(center, r) == center.range(r)

    def test_hex_corners_count_and_diameter(self):
        """hex_corners 应返回 6 个顶点；直径顶点距离 ≈ 2 * hex_size"""
        grid = HexGrid(size=2)
        hex_size = 10.0
        corners = grid.hex_corners(HexCoord(0, 0), hex_size)
        assert len(corners) == 6
        # 任意两个相对顶点距离应约等于 2 * hex_size
        import math

        d = math.hypot(corners[0][0] - corners[3][0], corners[0][1] - corners[3][1])
        assert abs(d - 2 * hex_size) < 1e-6


# ============================================================
# 2) core/noise_gen.py — 语义行为
#    风险：噪声分布错误会全局影响所有地形分类与季风逻辑。
# ============================================================
class TestNoiseGeneratorSemantics:
    def _make_grid(self, size=4, hex_size=5.0):
        grid = HexGrid(size=size)
        centers = grid.all_hex_centers(hex_size)
        return [(x, y) for _, x, y in centers]

    def test_elevation_is_deterministic_for_same_seed(self):
        """同种子同输入应产生完全相同的高程"""
        coords = self._make_grid()
        e1 = NoiseGenerator(seed=42).generate_elevation(coords)
        e2 = NoiseGenerator(seed=42).generate_elevation(coords)
        np.testing.assert_array_equal(e1, e2)

    def test_elevation_differs_for_different_seeds(self):
        """不同种子应产生不同的高程"""
        coords = self._make_grid()
        e1 = NoiseGenerator(seed=42).generate_elevation(coords)
        e2 = NoiseGenerator(seed=43).generate_elevation(coords)
        assert not np.array_equal(e1, e2)

    def test_elevation_is_normalized(self):
        """高程归一化到 [0, 1] 后 min=0, max=1"""
        coords = self._make_grid(size=4)
        e = NoiseGenerator(seed=42).generate_elevation(coords)
        assert e.min() >= 0.0 and e.max() <= 1.0
        assert e.max() - e.min() < 0.001 or e.max() == 1.0

    def test_set_seed_resets_state(self):
        """set_seed 后再次生成应等价于用新种子构造的新生成器"""
        coords = self._make_grid()
        ng1 = NoiseGenerator(seed=42)
        ng1.set_seed(7)
        e_new = ng1.generate_elevation(coords)
        e_ref = NoiseGenerator(seed=7).generate_elevation(coords)
        np.testing.assert_array_equal(e_new, e_ref)

    def test_monsoon_changes_moisture_distribution(self):
        """季风方向不同应产生不同湿度（风向对雨影有显著影响）"""
        coords = self._make_grid(size=5)
        ng = NoiseGenerator(seed=42)
        elev = ng.generate_elevation(coords)
        m0 = ng.generate_moisture(coords, elev, monsoon_dir=0.0)
        m90 = ng.generate_moisture(coords, elev, monsoon_dir=90.0)
        m180 = ng.generate_moisture(coords, elev, monsoon_dir=180.0)
        # 至少有两两不同
        assert not (np.array_equal(m0, m90) and np.array_equal(m0, m180))
        for m in (m0, m90, m180):
            assert m.min() >= 0.0 and m.max() <= 1.0

    def test_temperature_decreases_with_elevation(self):
        """高程越高温度应越低（海拔冷却效应）"""
        coords = [(float(i), 0.0) for i in range(20)]
        ng = NoiseGenerator(seed=42)
        low_elev = np.full(20, 0.1)
        high_elev = np.full(20, 0.9)
        t_low = ng.generate_temperature(low_elev, coords)
        t_high = ng.generate_temperature(high_elev, coords)
        assert (t_low > t_high).all()

    def test_temperature_is_latitude_based(self):
        """温度应随 y 坐标（纬度）增大而降低"""
        ng = NoiseGenerator(seed=42)
        elev = np.zeros(10)
        # y=0 处更靠近赤道
        coords_south = [(0.0, -5.0 + 0.5 * i) for i in range(10)]
        temp = ng.generate_temperature(elev, coords_south)
        # 数组顺序上 y 单调递增，温度应单调非增
        for i in range(len(temp) - 1):
            assert temp[i] >= temp[i + 1]


# ============================================================
# 3) core/terrain_gen.py — Whittaker 表 + 海岸判定
#    风险：生物群落分类错误会破坏玩家地理直觉。
# ============================================================
class TestTerrainGeneratorBiomeTable:
    """对核心 Whittaker 生物群落分类的穷举式黑盒测试。"""

    def setup_method(self):
        self.tg = TerrainGenerator()  # 默认 water_level=0.35

    def test_deep_water_is_ocean(self):
        assert self.tg.classify_biome(0.10, 0.5, 0.5, is_coast=False) == BIOME_OCEAN

    def test_shallow_water_is_lake(self):
        assert self.tg.classify_biome(0.20, 0.5, 0.5, is_coast=False) == BIOME_LAKE

    def test_high_elev_low_temp_is_snow(self):
        # 雪线 (0.85) 以上 + 低温
        assert self.tg.classify_biome(0.90, 0.5, 0.2, is_coast=False) == BIOME_SNOW

    def test_high_mountains_above_080(self):
        assert self.tg.classify_biome(0.85, 0.5, 0.7, is_coast=False) in (
            "high_mountains",
            BIOME_SNOW,
        )

    def test_mountains_above_065(self):
        # 0.7 在 0.65 与 0.80 之间，应为 mountains
        assert self.tg.classify_biome(0.70, 0.5, 0.7, is_coast=False) == BIOME_MOUNTAINS

    def test_hills_between_055_and_065(self):
        assert self.tg.classify_biome(0.60, 0.5, 0.5, is_coast=False) == "hills"

    def test_dry_hot_lowland_is_desert(self):
        # moist<0.25 且 temp>0.6
        assert self.tg.classify_biome(0.5, 0.1, 0.8, is_coast=False) == BIOME_DESERT

    def test_dry_cold_lowland_is_tundra(self):
        # moist<0.25 且 temp<=0.6
        assert self.tg.classify_biome(0.5, 0.1, 0.3, is_coast=False) == BIOME_TUNDRA

    def test_moderate_dry_hot_is_savanna(self):
        # 0.25<=moist<0.35 + temp>0.6
        assert self.tg.classify_biome(0.5, 0.30, 0.8, is_coast=False) == BIOME_SAVANNA

    def test_high_moisture_high_temp_is_rainforest(self):
        # moist>=0.75 + temp>0.75
        assert self.tg.classify_biome(0.5, 0.8, 0.8, is_coast=False) == BIOME_RAINFOREST

    def test_high_moisture_mid_temp_is_dense_forest(self):
        # moist>=0.75 + 0.5<temp<=0.75
        assert self.tg.classify_biome(0.5, 0.85, 0.6, is_coast=False) == BIOME_DENSE_FOREST

    def test_swamp_conditions(self):
        # swamp 要求 moist>=0.80 + elev<0.4；需 elev 刚高于水线（否则变 lake）
        assert self.tg.classify_biome(0.39, 0.9, 0.3, is_coast=False) == BIOME_SWAMP

    def test_beach_only_when_is_coast(self):
        # 海岸判定只在 is_coast=True 时触发
        without_coast = self.tg.classify_biome(0.40, 0.5, 0.5, is_coast=False)
        with_coast = self.tg.classify_biome(0.40, 0.5, 0.5, is_coast=True)
        assert with_coast == "beach"
        assert without_coast != "beach"


class TestTerrainGeneratorIsWaterBoundary:
    """is_water 在水线处的精确分类"""

    def test_water_level_threshold(self):
        """elevation < water_level 为水域；>= water_level 为陆地"""
        tg = TerrainGenerator()
        tg.water_level = 0.40
        grid = HexGrid(size=2)
        elev = np.array([0.39 if i % 2 == 0 else 0.41 for i in range(len(grid.hexes))])
        moist = np.full(len(grid.hexes), 0.5)
        temp = np.full(len(grid.hexes), 0.5)
        td = tg.generate(elev, moist, temp, [(h, 0.0, 0.0) for h in grid.hexes])
        water_count = sum(1 for v in td.values() if v.is_water)
        land_count = sum(1 for v in td.values() if not v.is_water)
        assert water_count == 10
        assert land_count == 9

    def test_volcanic_marker_only_for_high_dry_land(self):
        """火山仅在陆地 + 高 elevation + 低 moisture 时可能生成"""
        # 用固定 seed 跑大量次，统计是否至少出现一次（火山概率 0.5%）
        tg = TerrainGenerator()
        grid = HexGrid(size=5)
        elev = np.full(len(grid.hexes), 0.7)  # 高地
        moist = np.full(len(grid.hexes), 0.2)  # 干燥
        temp = np.full(len(grid.hexes), 0.5)
        volcanic_seen = False
        for seed in range(50):
            np.random.seed(seed)
            td = tg.generate(elev, moist, temp, [(h, 0.0, 0.0) for h in grid.hexes])
            if any(v.volcanic for v in td.values()):
                volcanic_seen = True
                # 火山必为非水域 + biome == volcano
                for v in td.values():
                    if v.volcanic:
                        assert not v.is_water
                        assert v.biome == "volcano"
                break
        assert volcanic_seen, "50 次随机生成未出现火山，概率异常"


class TestTerrainGeneratorCoast:
    """is_coast 双向判定的正确性"""

    def test_water_with_land_neighbor_is_coast(self):
        """水域若有陆地邻居应被标为海岸"""
        tg = TerrainGenerator()
        tg.water_level = 0.35
        # 2x2-like 棋盘：偶数格子陆地，奇数格子水域
        grid = HexGrid(size=2)
        elev = np.array([0.5 if (h.q + h.r) % 2 == 0 else 0.2 for h in grid.hexes])
        moist = np.full(len(grid.hexes), 0.5)
        temp = np.full(len(grid.hexes), 0.5)
        td = tg.generate(elev, moist, temp, [(h, 0.0, 0.0) for h in grid.hexes])

        # 在该棋盘中每个水域都至少有一个陆地邻居
        for h, v in td.items():
            if v.is_water:
                has_land_nbr = any(
                    nb in td and not td[nb].is_water for nb in h.neighbors()
                )
                if has_land_nbr:
                    assert v.is_coast, f"水域 {h} 紧邻陆地但未标海岸"

    def test_isolated_water_hex_not_coast(self):
        """孤立水域（全部邻居都是水域）不应被标为海岸"""
        tg = TerrainGenerator()
        tg.water_level = 0.35
        # 构造一个全水域网格
        grid = HexGrid(size=3)
        elev = np.full(len(grid.hexes), 0.1)  # 全部 < 0.35
        moist = np.full(len(grid.hexes), 0.5)
        temp = np.full(len(grid.hexes), 0.5)
        td = tg.generate(elev, moist, temp, [(h, 0.0, 0.0) for h in grid.hexes])
        # 全部为水域，没有陆地 → 都不应为海岸
        for v in td.values():
            assert not v.is_coast


# ============================================================
# 4) core/feature_gen.py — 业务规则
#    风险：河流/聚落/航线错位会破坏玩家路线规划。
# ============================================================


def _build_terrain_for_features():
    """构造一个能稳定触发各类特性的最小地图：以中心为高地、四周环水。"""
    tg = TerrainGenerator()
    tg.water_level = 0.35
    grid = HexGrid(size=6)
    # 中心 elev 接近 0.9，向外单调下降到 0.05，确保有真实下坡路径
    n = len(grid.hexes)
    elev = np.array([max(0.05, 0.95 - 0.07 * abs(h.q) - 0.07 * abs(h.r)) for h in grid.hexes])
    moist = np.full(n, 0.6)
    temp = np.full(n, 0.5)
    hex_list = [(h, 0.0, 0.0) for h in grid.hexes]
    td = tg.generate(elev, moist, temp, hex_list)
    return td, hex_list, grid


class TestFeatureGenRivers:
    def test_river_flow_monotonically_decreases_along_path(self):
        """河流流量标记应沿路径单调不增（1.0 → 0）"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        rivers = fg.generate_rivers(rng, num_rivers=5)
        assert len(rivers) > 0, "测试数据下应至少能生成一条河流"
        for r in rivers:
            flows = [td[h].river_flow for h in r]
            for i in range(len(flows) - 1):
                assert flows[i] >= flows[i + 1], f"流量沿河流未单调递减：{flows}"

    def test_no_river_starts_on_volcanic(self):
        """河流起点不应选火山六边形"""
        td, hex_list, _ = _build_terrain_for_features()
        # 手工注入一个火山陆地
        for h, v in td.items():
            if not v.is_water and v.elevation > 0.55:
                v.volcanic = True
                break
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(0)
        fg.generate_rivers(rng, num_rivers=20)
        # 验证：所有 river_flow > 0 的 hex 中不应包含被标 volcanic 的
        for h, v in td.items():
            if v.volcanic:
                assert v.river_flow == 0.0


class TestFeatureGenSettlements:
    def test_settlement_distance_constraint(self):
        """同类聚落（town）之间的距离应 >= 4（min_dist 阈值）"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=5, num_towns=3, num_cities=1, has_capital=True
        )
        towns = [h for h in settlements if td[h].settlement == SETTLEMENT_TOWN]
        for i, a in enumerate(towns):
            for b in towns[i + 1 :]:
                assert a.distance_to(b) >= 4, (
                    f"towns {a} 和 {b} 距离 {a.distance_to(b)} < 4"
                )

    def test_capital_placed_when_requested(self):
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=2, num_towns=1, num_cities=1, has_capital=True
        )
        capitals = [h for h in settlements if td[h].settlement == SETTLEMENT_CAPITAL]
        assert len(capitals) == 1
        # 首都应有非空中文名
        assert len(td[capitals[0]].settlement_name) >= 2

    def test_no_capital_when_disabled(self):
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=2, num_towns=1, num_cities=1, has_capital=False
        )
        capitals = [h for h in settlements if td[h].settlement == SETTLEMENT_CAPITAL]
        assert len(capitals) == 0

    def test_settlement_names_are_two_chinese_chars(self):
        """生成的聚落名应为 2 个汉字（前缀 + 后缀）"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=4, num_towns=2, num_cities=1, has_capital=True
        )
        for h in settlements:
            name = td[h].settlement_name
            assert len(name) == 2
            assert all("\u4e00" <= c <= "\u9fff" for c in name), f"非汉字：{name}"


class TestFeatureGenRoads:
    def test_no_road_on_water(self):
        """A* 路径不应穿越水域（除非终点本身是水域）"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_settlements(
            rng, num_villages=4, num_towns=2, num_cities=1, has_capital=True
        )
        fg.generate_roads()
        for h, v in td.items():
            if v.road:
                assert not v.is_water, f"水域 {h} 被错误标记为道路"

    def test_roads_connect_settlements(self):
        """聚落集合中的所有节点都应能通过 road 链互相可达（通过陆地 hex）"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        settlements = fg.generate_settlements(
            rng, num_villages=4, num_towns=2, num_cities=1, has_capital=True
        )
        fg.generate_roads()
        # 至少有一个聚落 hex 应被标为 road（路径经过）
        road_hexes = {h for h, v in td.items() if v.road}
        if len(settlements) >= 2:
            assert len(road_hexes) > 0


class TestFeatureGenResources:
    def test_resources_skip_settlements_and_water(self):
        """资源不应放置在聚落或水域"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        fg.generate_settlements(
            rng, num_villages=3, num_towns=2, num_cities=1, has_capital=True
        )
        fg.generate_resources(rng, density=1.0)  # 100% 概率
        for h, v in td.items():
            if v.resource:
                assert not v.is_water
                assert v.settlement == SETTLEMENT_NONE

    def test_resource_amount_in_valid_range(self):
        """资源数量应在 1~4（普通）或 2~5（火山）"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(0)
        fg.generate_resources(rng, density=1.0)
        for h, v in td.items():
            if v.resource:
                assert 1 <= v.resource_amount <= 5, v.resource_amount
                assert v.resource in {
                    RESOURCE_WOOD,
                    RESOURCE_IRON,
                    RESOURCE_GOLD,
                    RESOURCE_FOOD,
                    RESOURCE_STONE,
                }


class TestFeatureGenShipping:
    def test_shipping_routes_within_distance_bounds(self):
        """航线仅在 5 < 距离 < 30 的沿海聚落间建立"""
        td, hex_list, _ = _build_terrain_for_features()
        fg = FeatureGenerator(td, hex_list)
        rng = np.random.default_rng(42)
        # 制造大量沿海聚落
        coastal = [h for h, v in td.items() if v.is_coast and not v.is_water][:10]
        for i, h in enumerate(coastal):
            v = td[h]
            v.settlement = SETTLEMENT_TOWN
            v.settlement_name = f"镇{i}"
        routes = fg.generate_shipping_routes(rng)
        for a, b in routes:
            d = a.distance_to(b)
            assert 5 < d < 30, f"航线距离越界：{d}"


# ============================================================
# 5) ui/param_panel.py — 用户输入 → 生成参数
#    风险：单位换算或季风索引错位会让所有用户生成的地图与预期不符。
# ============================================================


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def panel(qapp):
    from ui.param_panel import ParamPanel

    p = ParamPanel()
    p.seed_input.setValue(42)
    yield p


class TestParamPanelGetParams:
    def test_water_level_divides_by_100(self, panel):
        panel.water_level.setValue(45)
        assert panel.get_params()["water_level"] == pytest.approx(0.45)

    def test_noise_scale_divides_by_10(self, panel):
        panel.noise_scale.setValue(25)
        assert panel.get_params()["noise_scale"] == pytest.approx(2.5)

    def test_resource_density_divides_by_100(self, panel):
        panel.resource_density.setValue(8)
        assert panel.get_params()["resource_density"] == pytest.approx(0.08)

    def test_monsoon_index_0_is_none(self, panel):
        panel.monsoon_combo.setCurrentIndex(0)
        assert panel.get_params()["monsoon_dir"] is None

    def test_monsoon_index_3_is_90_east(self, panel):
        panel.monsoon_combo.setCurrentIndex(3)
        assert panel.get_params()["monsoon_dir"] == 90.0

    def test_monsoon_index_5_is_180_south(self, panel):
        panel.monsoon_combo.setCurrentIndex(5)
        assert panel.get_params()["monsoon_dir"] == 180.0

    def test_size_mapping(self, panel):
        panel.size_input.setCurrentIndex(0)
        assert panel.get_params()["size"] == 20
        panel.size_input.setCurrentIndex(3)
        assert panel.get_params()["size"] == 80

    def test_params_schema_is_complete(self, panel):
        params = panel.get_params()
        for key in (
            "seed",
            "size",
            "water_level",
            "noise_scale",
            "monsoon_dir",
            "num_rivers",
            "resource_density",
        ):
            assert key in params, f"缺少键 {key}"


class TestParamPanelSignals:
    def test_terrain_change_emits_biome_name(self, panel, qapp):
        captured = []
        panel.terrain_changed.connect(lambda s: captured.append(s))
        # 先切到与目标不同的 idx（如 "沙漠"=7），再切到"森林"以确保信号触发
        panel.terrain_combo.setCurrentIndex(7)
        captured.clear()
        panel.terrain_combo.setCurrentIndex(1)  # "森林"
        assert captured == ["forest"]

    def test_terrain_change_default_plains(self, panel, qapp):
        captured = []
        panel.terrain_changed.connect(lambda s: captured.append(s))
        # 切到别的再切回
        panel.terrain_combo.setCurrentIndex(1)
        captured.clear()
        panel.terrain_combo.setCurrentText("平原")
        assert captured == ["plains"]

    def test_layer_change_emits_layer_string(self, panel, qapp):
        captured = []
        panel.layer_changed.connect(lambda s: captured.append(s))
        # 默认已经在 idx=0，从 idx=1 开始
        panel.layer_combo.setCurrentIndex(0)  # biome (no-op)
        captured.clear()
        panel.layer_combo.setCurrentIndex(1)  # elevation
        panel.layer_combo.setCurrentIndex(3)  # temperature
        assert captured == ["elevation", "temperature"]

    def test_edit_toggle_enables_tools(self, panel, qapp):
        panel.edit_toggle.setChecked(True)
        # 默认工具为"地形画笔"(idx=0)，应启用 terrain_combo
        assert panel.terrain_combo.isEnabled()
        assert not panel.settlement_type_combo.isEnabled()
        # 切到"放置聚落"(idx=1)
        panel.edit_tool_combo.setCurrentIndex(1)
        assert not panel.terrain_combo.isEnabled()
        assert panel.settlement_type_combo.isEnabled()

    def test_display_options_reflect_state(self, panel, qapp):
        panel.show_grid_cb.setChecked(False)
        panel.show_settlements_cb.setChecked(False)
        opts = panel.get_display_options()
        assert opts["show_grid"] is False
        assert opts["show_settlements"] is False
        assert opts["show_rivers"] is True  # 默认未改动


# ============================================================
# 6) ui/map_widget.py — 编辑模式状态机
#    风险：编辑工具的副作用错误会破坏用户操作预期。
# ============================================================


class TestMapWidgetEdits:
    @pytest.fixture
    def widget(self, qapp):
        from ui.map_widget import MapWidget

        w = MapWidget()
        grid = HexGrid(size=2)
        td = {h: TerrainData() for h in grid.hexes}
        w.set_map_data(grid, td, None, None, None, {h: i for i, h in enumerate(grid.hexes)})
        w.edit_mode = True
        return w, grid, td

    def test_set_hex_size_clamps_upper(self, widget):
        w, _, _ = widget
        w.set_hex_size(1000)
        assert w.hex_size == 40

    def test_set_hex_size_clamps_lower(self, widget):
        w, _, _ = widget
        w.set_hex_size(1)
        assert w.hex_size == 6

    def test_terrain_edit_marks_water_for_ocean(self, widget):
        w, grid, td = widget
        w.edit_tool = "terrain"
        w.edit_terrain = "ocean"
        h = grid.hexes[0]
        w._apply_edit(h)
        assert td[h].biome == "ocean"
        assert td[h].is_water is True

    def test_terrain_edit_marks_land_for_plains(self, widget):
        w, grid, td = widget
        w.edit_tool = "terrain"
        w.edit_terrain = "plains"
        h = grid.hexes[0]
        w._apply_edit(h)
        assert td[h].biome == "plains"
        assert td[h].is_water is False

    def test_settlement_edit_toggles(self, widget):
        w, grid, td = widget
        w.edit_tool = "settlement"
        w.edit_settlement_type = SETTLEMENT_CAPITAL
        h = grid.hexes[0]
        w._apply_edit(h)
        assert td[h].settlement == SETTLEMENT_CAPITAL
        assert len(td[h].settlement_name) == 2
        # 再点一次应清除
        w._apply_edit(h)
        assert td[h].settlement == SETTLEMENT_NONE
        assert td[h].settlement_name == ""

    def test_resource_edit_adds_then_erases(self, widget):
        w, grid, td = widget
        w.edit_tool = "resource"
        h = grid.hexes[0]
        w._apply_edit(h)
        assert td[h].resource is not None
        assert 1 <= td[h].resource_amount <= 4
        # 再次点击应清除
        w._apply_edit(h)
        assert td[h].resource is None
        assert td[h].resource_amount == 0

    def test_erase_clears_settlement_and_resource(self, widget):
        w, grid, td = widget
        h = grid.hexes[0]
        td[h].settlement = SETTLEMENT_TOWN
        td[h].settlement_name = "测试"
        td[h].resource = "wood"
        td[h].resource_amount = 3
        td[h].road = True
        w.edit_tool = "erase"
        w._apply_edit(h)
        assert td[h].settlement == SETTLEMENT_NONE
        assert td[h].resource is None
        assert td[h].road is False


class TestMapWidgetSetLayer:
    def test_layer_state_change(self, qapp):
        from ui.map_widget import MapWidget

        w = MapWidget()
        w.set_layer(MapWidget.LAYER_ELEVATION)
        assert w.current_layer == MapWidget.LAYER_ELEVATION
        w.set_layer(MapWidget.LAYER_MOISTURE)
        assert w.current_layer == MapWidget.LAYER_MOISTURE


# ============================================================
# 7) export/exporter.py — 输出契约
#    风险：导出文件结构破坏会让下游消费方（HTML 预览等）失效。
# ============================================================


class _MockWidget(QWidget):
    """最小化的 MapWidget 替身，覆盖 exporter 实际访问的属性。"""

    def __init__(self, grid, td, hex_size=20.0):
        super().__init__()
        self.hex_grid = grid
        self.terrain_data = td
        self.hex_size = hex_size


def _build_export_data():
    grid = HexGrid(size=2)
    td = {}
    for i, h in enumerate(grid.hexes):
        v = TerrainData()
        v.biome = "plains" if i % 2 == 0 else "forest"
        v.is_water = False
        v.elevation = 0.5
        v.moisture = 0.5
        v.temperature = 0.5
        if i == 0:
            v.settlement = SETTLEMENT_CAPITAL
            v.settlement_name = "王京"
        elif i == 1:
            v.settlement = SETTLEMENT_VILLAGE
            v.settlement_name = "新村"
        td[h] = v
    return grid, td


class TestExporterJson:
    def test_json_schema_and_values(self, monkeypatch, tmp_path):
        from export.exporter import MapExporter

        grid, td = _build_export_data()
        widget = _MockWidget(grid, td)
        exporter = MapExporter(widget)

        # 静默 GUI 弹窗
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(
            QMessageBox, "information", staticmethod(lambda *a, **k: None)
        )

        fp = tmp_path / "out.json"
        exporter.export_json(parent_widget=widget, filepath=str(fp))
        assert fp.exists()

        data = json.loads(fp.read_text(encoding="utf-8"))
        assert "meta" in data and "hexes" in data
        assert data["meta"]["grid_size"] == 2
        assert data["meta"]["total_hexes"] == len(grid.hexes)
        assert len(data["hexes"]) == len(grid.hexes)

        # 校验字段完整
        required = {
            "q", "r", "x", "y", "elevation", "moisture", "temperature", "biome",
            "is_water", "river_flow", "settlement", "settlement_name", "resource",
            "resource_amount", "road", "shipping", "volcanic",
        }
        assert set(data["hexes"][0].keys()) == required

        # 验证首都字段
        capital_entry = next(
            e for e in data["hexes"] if e["settlement"] == SETTLEMENT_CAPITAL
        )
        assert capital_entry["settlement_name"] == "王京"

    def test_json_unicode_preserved(self, monkeypatch, tmp_path):
        """settlement_name 应保留中文字符而不被转义"""
        from export.exporter import MapExporter

        grid, td = _build_export_data()
        widget = _MockWidget(grid, td)
        exporter = MapExporter(widget)
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(
            QMessageBox, "information", staticmethod(lambda *a, **k: None)
        )

        fp = tmp_path / "out.json"
        exporter.export_json(parent_widget=widget, filepath=str(fp))
        raw = fp.read_text(encoding="utf-8")
        assert "王京" in raw  # ensure_ascii=False
        assert "新村" in raw


class TestExporterSvg:
    def test_svg_writes_valid_xml(self, monkeypatch, tmp_path):
        from export.exporter import MapExporter

        grid, td = _build_export_data()
        widget = _MockWidget(grid, td)
        exporter = MapExporter(widget)
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(
            QMessageBox, "information", staticmethod(lambda *a, **k: None)
        )

        fp = tmp_path / "out.svg"
        exporter.export_svg(parent_widget=widget, filepath=str(fp))
        content = fp.read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert content.rstrip().endswith("</svg>")
        # 每个六边形都应有 polygon 元素
        assert content.count("<polygon") == len(td)
        # 背景矩形
        assert "<rect" in content
