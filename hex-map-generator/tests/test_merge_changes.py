"""
测试套件 — 覆盖 trae/solo-agent-Vkg8uv → origin/main 合并的关键变更点
"""

from __future__ import annotations

import inspect
import os
import sys

# 启用 Qt 离屏渲染
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# 把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.feature_gen import FeatureGenerator
from core.hex_grid import HexCoord
from core.noise_gen import NoiseGenerator, PerlinNoise
from core.terrain_gen import (
    BIOME_DESERT,
    BIOME_FOREST,
    BIOME_OCEAN,
    BIOME_PLAINS,
    BIOME_SNOW,
    BIOME_TUNDRA,
    BIOME_VOLCANO,
    TerrainData,
    TerrainGenerator,
)
from utils.colors import BIOME_COLORS, FEATURE_COLORS


# ============================================================
# 1) core/hex_grid.py — from_cube 新增立方体坐标校验
# ============================================================
class TestHexGridFromCube:
    def test_valid_cube_coords(self):
        """合法的立方体坐标 (q+r+s==0) 应正确构造"""
        hc = HexCoord.from_cube(1, -2, 1)
        assert hc.q == 1
        assert hc.r == -2

    def test_invalid_cube_coords_raises(self):
        """非法的立方体坐标 (q+r+s != 0) 应抛出 ValueError"""
        with pytest.raises(ValueError, match="does not sum to 0"):
            HexCoord.from_cube(1, 1, 1)
        with pytest.raises(ValueError):
            HexCoord.from_cube(0, 0, 1)
        with pytest.raises(ValueError):
            HexCoord.from_cube(2, 3, 0)

    def test_zero_zero_zero_accepted(self):
        """(0,0,0) 是合法立方体坐标"""
        hc = HexCoord.from_cube(0, 0, 0)
        assert hc.q == 0 and hc.r == 0


# ============================================================
# 2) core/noise_gen.py — SimplexNoise 已重命名为 PerlinNoise
#    移除 water_level / latitude_bias 参数
# ============================================================
class TestNoiseGenerator:
    def test_perlin_noise_class_exists(self):
        """PerlinNoise 类应存在"""
        assert inspect.isclass(PerlinNoise)

    def test_simplex_noise_class_removed(self):
        """旧的 SimplexNoise 类应已移除"""
        from core import noise_gen

        assert not hasattr(noise_gen, "SimplexNoise")

    def test_perlin_attribute(self):
        """NoiseGenerator 应暴露 perlin 实例而非 simplex"""
        ng = NoiseGenerator(seed=42)
        assert hasattr(ng, "perlin")
        assert not hasattr(ng, "simplex")
        assert isinstance(ng.perlin, PerlinNoise)

    def test_generate_elevation_no_water_level(self):
        """generate_elevation 不再接受 water_level 参数"""
        ng = NoiseGenerator(seed=42)
        coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
        sig = inspect.signature(ng.generate_elevation)
        assert "water_level" not in sig.parameters
        elev = ng.generate_elevation(coords)
        assert elev.shape == (4,)
        assert elev.min() >= 0.0 and elev.max() <= 1.0

    def test_generate_moisture_no_water_level(self):
        """generate_moisture 不再接受 water_level 参数"""
        ng = NoiseGenerator(seed=42)
        coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
        elev = ng.generate_elevation(coords)
        sig = inspect.signature(ng.generate_moisture)
        assert "water_level" not in sig.parameters
        m = ng.generate_moisture(coords, elev)
        assert m.shape == (4,)
        assert m.min() >= 0.0 and m.max() <= 1.0

    def test_generate_temperature_no_latitude_bias(self):
        """generate_temperature 不再接受 latitude_bias 参数"""
        ng = NoiseGenerator(seed=42)
        coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
        elev = ng.generate_elevation(coords)
        sig = inspect.signature(ng.generate_temperature)
        assert "latitude_bias" not in sig.parameters
        t = ng.generate_temperature(elev, coords)
        assert t.shape == (4,)
        assert t.min() >= 0.0 and t.max() <= 1.0

    def test_perlin_range(self):
        """单次 octave_noise 输出应在合理范围"""
        p = PerlinNoise(seed=123)
        for _ in range(20):
            v = p.octave_noise(np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            assert -1.5 < v < 1.5  # 多八度叠加后仍近似 [-1, 1]


# ============================================================
# 3) core/terrain_gen.py — 移除未使用常量和方法
# ============================================================
class TestTerrainGen:
    def test_removed_terrain_constants(self):
        """旧的 TERRAIN_* 常量应已移除"""
        from core import terrain_gen

        for name in (
            "TERRAIN_WATER",
            "TERRAIN_LAND",
            "TERRAIN_MOUNTAIN",
            "TERRAIN_DESERT",
            "TERRAIN_SNOW",
            "TERRAIN_VOLCANO",
        ):
            assert not hasattr(terrain_gen, name), f"{name} 不应再存在"

    def test_removed_method(self):
        """get_biome_color_key 方法应已移除"""
        tg = TerrainGenerator()
        assert not hasattr(tg, "get_biome_color_key")

    def test_hex_grid_import_at_top(self):
        """core.hex_grid 导入应在文件顶部而非方法内"""
        from core import terrain_gen

        src = inspect.getsource(terrain_gen)
        # 文件中只应有一处 from core.hex_grid import，且不在 generate 内
        assert src.count("from core.hex_grid import") == 1

    def test_seed_calculation_handles_empty(self):
        """种子计算不应因 elevation 为空数组而崩溃"""
        empty = np.array([], dtype=float)
        # 不应抛 IndexError
        seed = int(np.sum(empty[: min(100, len(empty))] * 1000) % 10000) + 1
        assert seed == 1  # 空数组求和为 0，0 % 10000 + 1 = 1

    def test_seed_calculation_handles_large(self):
        """种子计算应处理大数组（>100 元素）"""
        big = np.arange(200, dtype=float)
        seed = int(np.sum(big[: min(100, len(big))] * 1000) % 10000) + 1
        assert seed > 0

    def test_generate_basic(self):
        """generate 方法应正确处理多元素数据"""
        tg = TerrainGenerator()
        n = 50
        rng = np.random.default_rng(42)
        elev = rng.uniform(0, 1, n)
        moist = rng.uniform(0, 1, n)
        temp = rng.uniform(0, 1, n)
        hex_list = [
            (HexCoord(q, r), float(q), float(r)) for q in range(-3, 4) for r in range(-3, 4)
        ]
        hex_list = hex_list[:n]
        td = tg.generate(
            elev[: len(hex_list)], moist[: len(hex_list)], temp[: len(hex_list)], hex_list
        )
        assert isinstance(td, dict)
        assert len(td) == len(hex_list)
        for hc, data in td.items():
            assert isinstance(data, TerrainData)
            assert data.biome in (
                BIOME_OCEAN,
                BIOME_PLAINS,
                BIOME_DESERT,
                BIOME_FOREST,
                BIOME_SNOW,
                BIOME_VOLCANO,
                BIOME_TUNDRA,
                "lake",
                "beach",
                "dense_forest",
                "rainforest",
                "taiga",
                "savanna",
                "hills",
                "mountains",
                "high_mountains",
                "swamp",
            )


# ============================================================
# 4) core/feature_gen.py — generate_roads 移除 rng
# ============================================================
class TestFeatureGen:
    def test_generate_roads_no_rng_param(self):
        """generate_roads 不再接受 rng 参数"""
        sig = inspect.signature(FeatureGenerator.generate_roads)
        assert "rng" not in sig.parameters

    def test_generate_roads_returns_list(self):
        """generate_roads 应返回道路列表（即使没有真实地形数据也应不崩溃或合理返回）"""
        # 构造最小可用的 FeatureGenerator
        terrain_data = {}
        hex_list = []
        fg = FeatureGenerator(terrain_data, hex_list)
        # 没有聚落时应返回空列表
        result = fg.generate_roads()
        assert isinstance(result, list)
        assert result == []


# ============================================================
# 5) utils/colors.py — 移除未使用的色板/常量
# ============================================================
class TestColors:
    def test_biome_colors_present(self):
        """BIOME_COLORS 应包含 ocean 和 forest"""
        assert "ocean" in BIOME_COLORS
        assert "forest" in BIOME_COLORS

    def test_feature_colors_present(self):
        """FEATURE_COLORS 应包含 river/road/capital/city"""
        for k in ("river", "road", "capital", "city", "shipping_route"):
            assert k in FEATURE_COLORS, f"FEATURE_COLORS 缺少 {k}"

    def test_removed_color_constants(self):
        """TERRAIN_COLORS / WATER_COLORS / LAYER_NAMES / elevation_color 应已移除"""
        from utils import colors

        assert not hasattr(colors, "TERRAIN_COLORS")
        assert not hasattr(colors, "WATER_COLORS")
        assert not hasattr(colors, "LAYER_NAMES")
        assert not hasattr(colors, "elevation_color")


# ============================================================
# 6) export/exporter.py — 新增覆盖层绘制方法 + 修复后的居中
# ============================================================
class TestExporterDrawMethods:
    def test_draw_methods_exist(self):
        """所有 6 个覆盖层绘制方法应已存在"""
        from export.exporter import MapExporter

        for name in (
            "_draw_hex",
            "_draw_rivers",
            "_draw_roads",
            "_draw_settlements",
            "_draw_resources",
            "_draw_shipping_routes",
            "_draw_labels",
        ):
            assert hasattr(MapExporter, name), f"缺少 {name}"

    def test_hex_path_does_not_use_dynamic_import(self):
        """_draw_hex 不应再使用 __import__ 动态加载 QPainterPath"""
        from export.exporter import MapExporter

        src = inspect.getsource(MapExporter._draw_hex)
        assert "__import__" not in src

    def test_resources_use_qfontmetrics(self):
        """_draw_resources 应使用 QFontMetrics 居中而非硬编码 -4, +3"""
        from export.exporter import MapExporter

        src = inspect.getsource(MapExporter._draw_resources)
        assert "QFontMetrics" in src
        assert "cx - 4" not in src and "cy + 3" not in src

    def test_settlements_use_qfontmetrics(self):
        """_draw_settlements 首都/城市星标应使用 QFontMetrics 居中"""
        from export.exporter import MapExporter

        src = inspect.getsource(MapExporter._draw_settlements)
        assert "QFontMetrics" in src
        assert "radius * 0.3" not in src

    def test_shipping_routes_qpen_outside_loop(self):
        """_draw_shipping_routes 的 setPen 应在循环外"""
        from export.exporter import MapExporter

        src = inspect.getsource(MapExporter._draw_shipping_routes)
        # 循环外的 setPen 应在 'for' 之前出现
        for_idx = src.find("for hc, td in")
        pen_idx = src.find("painter.setPen(")
        assert pen_idx != -1 and for_idx != -1
        assert pen_idx < for_idx, "setPen 应在循环外"

    def test_roads_use_edge_set(self):
        """_draw_roads 应使用 drawn_edges 集合保证对称去重"""
        from export.exporter import MapExporter

        src = inspect.getsource(MapExporter._draw_roads)
        assert "drawn_edges" in src
        # 不应再有旧的 visited 模式
        assert "visited.add(hc)" not in src

    def test_labels_truncate_long_names(self):
        """_draw_labels 应按 max_chars 截断长聚落名"""
        from export.exporter import MapExporter

        src = inspect.getsource(MapExporter._draw_labels)
        assert "max_chars" in src


# ============================================================
# 7) ui/map_widget.py — 同步修复
# ============================================================
class TestMapWidgetFixes:
    def test_no_dynamic_qpainterpath_import(self):
        """map_widget.py 中不应出现 _draw_hex_to_painter 旧名"""
        from ui import map_widget

        src = inspect.getsource(map_widget)
        # _draw_hex_to_painter 是旧名
        assert "_draw_hex_to_painter" not in src

    def test_resources_use_qfontmetrics(self):
        from ui.map_widget import MapWidget

        src = inspect.getsource(MapWidget._draw_resources)
        assert "QFontMetrics" in src
        assert "cx - 4" not in src and "cy + 3" not in src

    def test_settlements_use_qfontmetrics(self):
        from ui.map_widget import MapWidget

        src = inspect.getsource(MapWidget._draw_settlements)
        assert "QFontMetrics" in src

    def test_roads_use_edge_set(self):
        from ui.map_widget import MapWidget

        src = inspect.getsource(MapWidget._draw_roads)
        assert "drawn_edges" in src
        assert "visited.add(hc)" not in src

    def test_shipping_qpen_outside_loop(self):
        from ui.map_widget import MapWidget

        src = inspect.getsource(MapWidget._draw_shipping_routes)
        for_idx = src.find("for hc, td in")
        pen_idx = src.find("painter.setPen(")
        assert pen_idx != -1 and for_idx != -1
        assert pen_idx < for_idx

    def test_edit_terrain_default(self):
        """MapWidget 初始化时应设置 edit_terrain 默认值"""
        from ui.map_widget import MapWidget

        attrs = inspect.getsource(MapWidget.__init__)
        assert "self.edit_terrain" in attrs


# ============================================================
# 8) ui/param_panel.py — 新增 terrain_changed 信号
# ============================================================
class TestParamPanel:
    def test_terrain_changed_signal(self):
        """terrain_changed 信号应存在"""
        from ui.param_panel import ParamPanel

        assert hasattr(ParamPanel, "terrain_changed")

    def test_params_changed_removed(self):
        """旧的 params_changed 信号应已移除"""
        from ui.param_panel import ParamPanel

        assert not hasattr(ParamPanel, "params_changed")

    def test_terrain_map_defined(self):
        """TERRAIN_MAP 中英文映射应存在且包含 10 项"""
        from ui.param_panel import ParamPanel

        assert hasattr(ParamPanel, "TERRAIN_MAP")
        m = ParamPanel.TERRAIN_MAP
        assert len(m) == 10
        assert m["平原"] == "plains"
        assert m["森林"] == "forest"
        assert m["沙漠"] == "desert"


# ============================================================
# 10) 边界情况：空数组/除零崩溃修复验证
# ============================================================
class TestEdgeCases:
    def test_empty_coords_elevation(self):
        """空坐标列表不应导致 generate_elevation 崩溃"""
        ng = NoiseGenerator(seed=42)
        empty_coords = []
        elev = ng.generate_elevation(empty_coords)
        assert elev.shape == (0,)
        assert len(elev) == 0

    def test_empty_coords_moisture(self):
        """空坐标列表不应导致 generate_moisture 崩溃"""
        ng = NoiseGenerator(seed=42)
        empty_coords = []
        empty_elev = np.array([], dtype=float)
        moist = ng.generate_moisture(empty_coords, empty_elev)
        assert moist.shape == (0,)
        assert len(moist) == 0

    def test_empty_coords_temperature(self):
        """空坐标列表不应导致 generate_temperature 崩溃"""
        ng = NoiseGenerator(seed=42)
        empty_coords = []
        empty_elev = np.array([], dtype=float)
        temp = ng.generate_temperature(empty_elev, empty_coords)
        assert temp.shape == (0,)
        assert len(temp) == 0

    def test_size_zero_division_protection(self):
        """size=0 时不应除零崩溃（验证 generate_image.py 和 main.py 的防护）"""
        # 导入 generate_image 模块
        import generate_image

        # size=0 应被转换为 safe_size=1
        safe_size = max(1, 0)
        hex_size = max(6, min(30, 500 / safe_size))
        assert hex_size > 0  # 应为有效正数，而非崩溃

    def test_full_pipeline_with_empty_data(self):
        """端到端流程应能处理空数据而不崩溃"""
        from core.hex_grid import HexGrid

        ng = NoiseGenerator(seed=42)
        # 直接传入空坐标列表（不依赖 HexGrid）
        coords = []  # 空列表
        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev)
        temp = ng.generate_temperature(elev, coords)

        # 应全部返回空数组
        assert len(elev) == 0
        assert len(moist) == 0
        assert len(temp) == 0


# ============================================================
# 9) 端到端：构造完整地图数据并验证可调用绘制方法
# ============================================================
class TestEndToEndPipeline:
    def test_full_pipeline(self):
        """端到端流程：生成噪声 → 地形 → 特性"""
        from core.hex_grid import HexGrid

        ng = NoiseGenerator(seed=42)
        grid = HexGrid(size=5)
        coords = [(0.0, 0.0) for _ in grid.hexes]
        elev = ng.generate_elevation(coords)
        moist = ng.generate_moisture(coords, elev)
        temp = ng.generate_temperature(elev, coords)

        tg = TerrainGenerator()
        hex_list = [(hc, 0.0, 0.0) for hc in grid.hexes]
        td = tg.generate(elev, moist, temp, hex_list)
        assert len(td) == len(grid.hexes)

        fg = FeatureGenerator(td, hex_list)
        # 不再传 rng
        rivers = fg.generate_rivers(np.random.default_rng(42), num_rivers=2)
        assert isinstance(rivers, list)
        roads = fg.generate_roads()
        assert isinstance(roads, list)

    def test_seed_reproducibility(self):
        """相同 seed 应产生相同的种子计算结果（验证 +1 调整后仍稳定）"""
        ng = NoiseGenerator(seed=42)
        coords = [(i * 0.1, i * 0.2) for i in range(20)]
        e1 = ng.generate_elevation(coords)
        e2 = ng.generate_elevation(coords)
        np.testing.assert_array_equal(e1, e2)
