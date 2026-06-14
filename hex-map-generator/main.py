"""
主窗口 — 整合地图生成器所有组件
"""
from __future__ import annotations
import sys
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter,
    QStatusBar, QLabel, QMessageBox,
)

import numpy as np

from core.hex_grid import HexCoord, HexGrid
from core.noise_gen import NoiseGenerator
from core.terrain_gen import TerrainGenerator, TerrainData
from core.feature_gen import FeatureGenerator
from ui.map_widget import MapWidget
from ui.param_panel import ParamPanel
from export.exporter import MapExporter


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("六边形地图生成器")
        self.setMinimumSize(1200, 800)

        # 核心组件
        self.hex_grid: Optional[HexGrid] = None
        self.terrain_data: Dict[HexCoord, TerrainData] = {}
        self.elevation: Optional[np.ndarray] = None
        self.moisture: Optional[np.ndarray] = None
        self.temperature: Optional[np.ndarray] = None
        self.coord_to_idx: Dict[HexCoord, int] = {}
        self.hex_coords_list: List[Tuple[HexCoord, float, float]] = []

        self._setup_ui()
        self._connect_signals()

        # 启动时自动生成一张地图
        QTimer.singleShot(100, self.generate_map)

    def _setup_ui(self):
        """构建 UI"""
        # 地图组件
        self.map_widget = MapWidget()
        self.map_widget.setMinimumWidth(600)

        # 参数面板
        self.param_panel = ParamPanel()

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.param_panel)
        splitter.addWidget(self.map_widget)
        splitter.setStretchFactor(0, 0)  # 参数面板不拉伸
        splitter.setStretchFactor(1, 1)  # 地图拉伸
        splitter.setSizes([280, 920])

        self.setCentralWidget(splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.coord_label = QLabel("坐标: -")
        self.terrain_label = QLabel("地形: -")
        self.hex_count_label = QLabel("六边形: -")
        self.status_bar.addWidget(self.coord_label)
        self.status_bar.addPermanentWidget(self.terrain_label)
        self.status_bar.addPermanentWidget(self.hex_count_label)

        # 菜单栏
        self._setup_menu()

    def _setup_menu(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")
        export_png_action = QAction("导出 PNG...", self)
        export_png_action.setShortcut(QKeySequence("Ctrl+P"))
        export_png_action.triggered.connect(self._export_png)
        file_menu.addAction(export_png_action)

        export_svg_action = QAction("导出 SVG...", self)
        export_svg_action.triggered.connect(self._export_svg)
        file_menu.addAction(export_svg_action)

        export_json_action = QAction("导出 JSON...", self)
        export_json_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_json_action.triggered.connect(self._export_json)
        file_menu.addAction(export_json_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图")
        fit_action = QAction("适配视图", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.map_widget.fit_to_view)
        view_menu.addAction(fit_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """连接信号"""
        # 参数面板 -> 生成
        self.param_panel.generate_clicked.connect(self.generate_map)

        # 显示选项 -> 地图
        panel = self.param_panel
        panel.show_grid_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_grid', v) or self.map_widget.update()
        )
        panel.show_rivers_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_rivers', v) or self.map_widget.update()
        )
        panel.show_roads_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_roads', v) or self.map_widget.update()
        )
        panel.show_settlements_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_settlements', v) or self.map_widget.update()
        )
        panel.show_resources_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_resources', v) or self.map_widget.update()
        )
        panel.show_shipping_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_shipping', v) or self.map_widget.update()
        )
        panel.show_labels_cb.toggled.connect(
            lambda v: setattr(self.map_widget, 'show_labels', v) or self.map_widget.update()
        )

        # 图层切换
        panel.layer_changed.connect(self.map_widget.set_layer)

        # 编辑模式
        panel.edit_mode_toggled.connect(self.map_widget.toggle_edit_mode)

        # 编辑工具
        panel.edit_tool_combo.currentIndexChanged.connect(
            lambda: self.map_widget.set_edit_tool(
                ["terrain", "settlement", "resource", "erase"][
                    panel.edit_tool_combo.currentIndex()
                ]
            )
        )
        panel.terrain_changed.connect(
            lambda v: setattr(self.map_widget, 'edit_terrain', v)
        )
        panel.settlement_type_combo.currentIndexChanged.connect(
            lambda: setattr(
                self.map_widget, 'edit_settlement_type',
                [1, 2, 3, 4][panel.settlement_type_combo.currentIndex()]
            )
        )

        # 导出按钮
        panel.export_png_btn.clicked.connect(self._export_png)
        panel.export_svg_btn.clicked.connect(self._export_svg)
        panel.export_json_btn.clicked.connect(self._export_json)

        # 地图悬停
        self.map_widget.hex_hovered.connect(self._on_hex_hovered)

        # 编辑
        self.map_widget.hex_edited.connect(self._on_hex_edited)

    def generate_map(self):
        """生成地图"""
        params = self.param_panel.get_params()

        seed = params["seed"]
        size = params["size"]
        water_level = params["water_level"]
        noise_scale = params["noise_scale"]
        monsoon_dir = params["monsoon_dir"]
        num_rivers = params["num_rivers"]
        resource_density = params["resource_density"]

        # 1. 生成六边形网格
        self.hex_grid = HexGrid(size=size)

        # 2. 计算六边形中心像素坐标
        hex_size = max(6, min(30, 500 / size))
        self.hex_coords_list = self.hex_grid.all_hex_centers(hex_size)
        self.coord_to_idx = {}
        for i, (hc, _, _) in enumerate(self.hex_coords_list):
            self.coord_to_idx[hc] = i

        coords_xy = [(x, y) for _, x, y in self.hex_coords_list]

        # 3. 生成噪声数据
        noise_gen = NoiseGenerator(seed=seed)
        self.elevation = noise_gen.generate_elevation(
            coords_xy, scale=noise_scale
        )
        monsoon_angle = monsoon_dir if monsoon_dir is not None else 90.0
        self.moisture = noise_gen.generate_moisture(
            coords_xy, self.elevation,
            scale=noise_scale * 1.2,
            monsoon_dir=monsoon_angle
        )
        self.temperature = noise_gen.generate_temperature(
            self.elevation, coords_xy
        )

        # 4. 生成地形数据
        terrain_gen = TerrainGenerator()
        terrain_gen.water_level = water_level
        self.terrain_data = terrain_gen.generate(
            self.elevation, self.moisture, self.temperature,
            self.hex_coords_list
        )

        # 5. 生成特性
        rng = np.random.Generator(np.random.PCG64(seed + 42))
        feature_gen = FeatureGenerator(self.terrain_data, self.hex_coords_list)

        # 河流
        feature_gen.generate_rivers(rng, num_rivers=num_rivers)

        # 聚落
        feature_gen.generate_settlements(
            rng, num_villages=10, num_towns=5, num_cities=2, has_capital=True
        )

        # 道路
        feature_gen.generate_roads()

        # 资源
        feature_gen.generate_resources(rng, density=resource_density)

        # 航线
        feature_gen.generate_shipping_routes(rng)

        # 6. 将数据传给地图组件
        self.map_widget.set_hex_size(hex_size)
        self.map_widget.set_map_data(
            self.hex_grid, self.terrain_data,
            self.elevation, self.moisture, self.temperature,
            self.coord_to_idx,
        )

        # 更新状态栏
        self.hex_count_label.setText(f"六边形: {len(self.hex_grid.hexes)}")

        # 适配视图
        self.map_widget.fit_to_view()

    def _on_hex_hovered(self, hc: HexCoord, td: TerrainData):
        """鼠标悬停时更新状态栏"""
        self.coord_label.setText(f"坐标: ({hc.q}, {hc.r})")
        self.terrain_label.setText(f"地形: {td.biome} | 高程: {td.elevation:.2f}")

    def _on_hex_edited(self, hc: HexCoord):
        """六边形被编辑后更新状态"""
        self.status_bar.showMessage(f"已编辑六边形 ({hc.q}, {hc.r})", 3000)

    def _export_png(self):
        exporter = MapExporter(self.map_widget)
        exporter.export_png(self)

    def _export_svg(self):
        exporter = MapExporter(self.map_widget)
        exporter.export_svg(self)

    def _export_json(self):
        exporter = MapExporter(self.map_widget)
        exporter.export_json(self)

    def _show_about(self):
        QMessageBox.about(
            self, "关于 六边形地图生成器",
            "六边形策略游戏地图生成器 v1.0\n\n"
            "功能:\n"
            "• 六边形策略地图生成\n"
            "• 多种生物群落\n"
            "• 河流、道路、聚落、资源系统\n"
            "• 季风与航线模拟\n"
            "• 多层数据展示\n"
            "• 手动编辑模式\n"
            "• 导出 PNG/SVG/JSON"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()