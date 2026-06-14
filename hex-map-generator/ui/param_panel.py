"""
地图生成参数调节面板 — 暗色游戏主题
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.styles import ThemeColors, ThemeFonts


class ParamPanel(QWidget):
    """参数控制面板"""

    # 信号：生成地图
    generate_clicked = Signal()
    # 信号：编辑模式切换
    edit_mode_toggled = Signal(bool)
    # 信号：图层切换
    layer_changed = Signal(str)
    # 信号：地形画笔类型切换
    terrain_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(340)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ========== 标题 ==========
        title_label = QLabel("⚔️ 地图生成器")
        title_label.setFont(ThemeFonts.title())
        title_label.setStyleSheet("color: #78b4ff; font-size: 14px;")
        layout.addWidget(title_label)

        # ========== 地图生成参数 ==========
        gen_group = QGroupBox("🗺️ 地图生成")
        gen_group.setFont(ThemeFonts.subtitle())
        gen_layout = QFormLayout(gen_group)
        gen_layout.setSpacing(10)
        gen_layout.setLabelAlignment(Qt.AlignLeft)
        gen_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # 种子
        seed_layout = QHBoxLayout()
        seed_layout.setSpacing(8)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(42)
        self.seed_input.setToolTip("相同的种子生成相同的地图")
        self.seed_input.setMinimumWidth(100)
        self.random_seed_btn = QPushButton("🎲")
        self.random_seed_btn.setFixedWidth(36)
        self.random_seed_btn.setToolTip("随机种子")
        self.random_seed_btn.clicked.connect(self._random_seed)
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.random_seed_btn)
        gen_layout.addRow(self._label("种子:"), seed_layout)

        # 地图大小
        self.size_input = QComboBox()
        self.size_input.addItems(["20 (小)", "40 (中)", "60 (大)", "80 (超大)"])
        self.size_input.setCurrentIndex(2)
        self.size_input.setToolTip("地图半径（六边形步数）")
        gen_layout.addRow(self._label("地图尺寸:"), self.size_input)

        # 海平面
        self.water_level = QSlider(Qt.Horizontal)
        self.water_level.setRange(15, 60)
        self.water_level.setValue(35)
        self.water_level.setToolTip("海平面高度 (15% ~ 60%)")
        self.water_label = QLabel("35%")
        self.water_label.setObjectName("valueLabel")
        self.water_level.valueChanged.connect(lambda v: self.water_label.setText(f"{v}%"))
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(self.water_level)
        wl_layout.addWidget(self.water_label)
        gen_layout.addRow(self._label("海平面:"), wl_layout)

        # 噪声缩放
        self.noise_scale = QSlider(Qt.Horizontal)
        self.noise_scale.setRange(10, 80)
        self.noise_scale.setValue(30)
        self.noise_scale.setToolTip("地形起伏频率 (10=粗糙, 80=平滑)")
        self.scale_label = QLabel("3.0")
        self.scale_label.setObjectName("valueLabel")
        self.noise_scale.valueChanged.connect(lambda v: self.scale_label.setText(f"{v / 10:.1f}"))
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(self.noise_scale)
        scale_layout.addWidget(self.scale_label)
        gen_layout.addRow(self._label("地形细节:"), scale_layout)

        # 季风方向
        self.monsoon_combo = QComboBox()
        self.monsoon_combo.addItems(
            ["无季风", "北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]
        )
        self.monsoon_combo.setCurrentIndex(3)
        gen_layout.addRow(self._label("季风方向:"), self.monsoon_combo)

        # 河流数量
        self.river_count = QSpinBox()
        self.river_count.setRange(0, 50)
        self.river_count.setValue(12)
        gen_layout.addRow(self._label("河流数量:"), self.river_count)

        # 资源密度
        self.resource_density = QSlider(Qt.Horizontal)
        self.resource_density.setRange(2, 20)
        self.resource_density.setValue(8)
        self.resource_density.setToolTip("资源分布密度")
        self.resource_label = QLabel("8%")
        self.resource_label.setObjectName("valueLabel")
        self.resource_density.valueChanged.connect(lambda v: self.resource_label.setText(f"{v}%"))
        rd_layout = QHBoxLayout()
        rd_layout.addWidget(self.resource_density)
        rd_layout.addWidget(self.resource_label)
        gen_layout.addRow(self._label("资源密度:"), rd_layout)

        # 生成按钮
        self.generate_btn = QPushButton("🎲 生成地图")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.setMinimumHeight(42)
        self.generate_btn.setFont(ThemeFonts.subtitle())
        self.generate_btn.clicked.connect(self.generate_clicked.emit)

        layout.addWidget(gen_group)
        layout.addWidget(self.generate_btn)

        # ========== 显示控制 ==========
        display_group = QGroupBox("👁️ 显示控制")
        display_group.setFont(ThemeFonts.subtitle())
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(8)

        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["生物群落", "高程图", "湿度图", "温度图"])
        self.layer_combo.currentIndexChanged.connect(self._on_layer_change)
        display_layout.addWidget(self._label("当前图层:"))
        display_layout.addWidget(self.layer_combo)

        # 显示选项网格
        checks_layout = QHBoxLayout()
        left_checks = QVBoxLayout()
        right_checks = QVBoxLayout()

        self.show_grid_cb = QCheckBox("网格线")
        self.show_grid_cb.setChecked(True)
        self.show_rivers_cb = QCheckBox("河流")
        self.show_rivers_cb.setChecked(True)
        self.show_roads_cb = QCheckBox("道路")
        self.show_roads_cb.setChecked(True)
        self.show_settlements_cb = QCheckBox("聚落")
        self.show_settlements_cb.setChecked(True)

        left_checks.addWidget(self.show_grid_cb)
        left_checks.addWidget(self.show_rivers_cb)
        left_checks.addWidget(self.show_roads_cb)
        left_checks.addWidget(self.show_settlements_cb)

        self.show_resources_cb = QCheckBox("资源")
        self.show_resources_cb.setChecked(True)
        self.show_shipping_cb = QCheckBox("航线")
        self.show_shipping_cb.setChecked(True)
        self.show_labels_cb = QCheckBox("名称")
        self.show_labels_cb.setChecked(True)

        right_checks.addWidget(self.show_resources_cb)
        right_checks.addWidget(self.show_shipping_cb)
        right_checks.addWidget(self.show_labels_cb)

        checks_layout.addLayout(left_checks)
        checks_layout.addLayout(right_checks)
        display_layout.addLayout(checks_layout)

        layout.addWidget(display_group)

        # ========== 编辑模式 ==========
        edit_group = QGroupBox("✏️ 编辑模式")
        edit_group.setFont(ThemeFonts.subtitle())
        edit_layout = QVBoxLayout(edit_group)
        edit_layout.setSpacing(10)

        self.edit_toggle = QPushButton("🖊️ 进入编辑")
        self.edit_toggle.setCheckable(True)
        self.edit_toggle.setMinimumHeight(36)
        self.edit_toggle.toggled.connect(self._on_edit_toggle)
        edit_layout.addWidget(self.edit_toggle)

        tool_layout = QHBoxLayout()
        self.edit_tool_combo = QComboBox()
        self.edit_tool_combo.addItems(["地形画笔", "放置聚落", "放置资源", "擦除"])
        self.edit_tool_combo.currentIndexChanged.connect(self._on_edit_tool_change)
        tool_layout.addWidget(QLabel("工具:"))
        tool_layout.addWidget(self.edit_tool_combo)
        edit_layout.addLayout(tool_layout)

        self.terrain_combo = QComboBox()
        self.terrain_combo.addItems(
            ["平原", "森林", "密林", "雨林", "针叶林", "丘陵", "山地", "沙漠", "稀树草原", "冻土"]
        )
        self.terrain_combo.setCurrentIndex(0)
        self.terrain_combo.setEnabled(False)
        self.terrain_combo.currentIndexChanged.connect(self._on_terrain_change)
        edit_layout.addWidget(QLabel("地形类型:"))
        edit_layout.addWidget(self.terrain_combo)

        self.settlement_type_combo = QComboBox()
        self.settlement_type_combo.addItems(["村庄", "城镇", "城市", "首都"])
        self.settlement_type_combo.setCurrentIndex(1)
        self.settlement_type_combo.setEnabled(False)
        edit_layout.addWidget(QLabel("聚落类型:"))
        edit_layout.addWidget(self.settlement_type_combo)

        layout.addWidget(edit_group)

        # ========== 导出 ==========
        export_group = QGroupBox("💾 导出")
        export_group.setFont(ThemeFonts.subtitle())
        export_layout = QHBoxLayout(export_group)
        export_layout.setSpacing(8)

        self.export_png_btn = QPushButton("📷 PNG")
        self.export_svg_btn = QPushButton("✏️ SVG")
        self.export_json_btn = QPushButton("📄 JSON")
        export_layout.addWidget(self.export_png_btn)
        export_layout.addWidget(self.export_svg_btn)
        export_layout.addWidget(self.export_json_btn)

        layout.addWidget(export_group)

        # 弹性空间
        layout.addStretch()

    def _label(self, text: str) -> QLabel:
        """创建样式化的标签"""
        label = QLabel(text)
        label.setFont(ThemeFonts.body())
        return label

    def _random_seed(self):
        import random
        self.seed_input.setValue(random.randint(0, 999999))

    def _on_layer_change(self, idx: int):
        layers = ["biome", "elevation", "moisture", "temperature"]
        if idx < len(layers):
            self.layer_changed.emit(layers[idx])

    def _on_edit_toggle(self, checked: bool):
        self.edit_toggle.setText("🔧 退出编辑" if checked else "🖊️ 进入编辑")
        self.edit_tool_combo.setEnabled(checked)
        self.terrain_combo.setEnabled(checked and self.edit_tool_combo.currentIndex() == 0)
        self.settlement_type_combo.setEnabled(checked and self.edit_tool_combo.currentIndex() == 1)
        self.edit_mode_toggled.emit(checked)

    def _on_edit_tool_change(self, idx: int):
        self.settlement_type_combo.setEnabled(idx == 1)
        self.terrain_combo.setEnabled(idx == 0)

    TERRAIN_MAP = {
        "平原": "plains",
        "森林": "forest",
        "密林": "dense_forest",
        "雨林": "rainforest",
        "针叶林": "taiga",
        "丘陵": "hills",
        "山地": "mountains",
        "沙漠": "desert",
        "稀树草原": "savanna",
        "冻土": "tundra",
    }

    def _on_terrain_change(self, idx: int):
        name = self.terrain_combo.currentText()
        biome = self.TERRAIN_MAP.get(name, "plains")
        self.terrain_changed.emit(biome)

    def get_params(self) -> dict:
        """获取所有参数"""
        size_idx = self.size_input.currentIndex()
        sizes = [20, 40, 60, 80]
        monsoon_map = {
            0: None,
            1: 0,
            2: 45,
            3: 90,
            4: 135,
            5: 180,
            6: 225,
            7: 270,
            8: 315,
        }
        return {
            "seed": self.seed_input.value(),
            "size": sizes[size_idx],
            "water_level": self.water_level.value() / 100.0,
            "noise_scale": self.noise_scale.value() / 10.0,
            "monsoon_dir": monsoon_map.get(self.monsoon_combo.currentIndex(), None),
            "num_rivers": self.river_count.value(),
            "resource_density": self.resource_density.value() / 100.0,
        }

    def get_display_options(self) -> dict:
        return {
            "show_grid": self.show_grid_cb.isChecked(),
            "show_rivers": self.show_rivers_cb.isChecked(),
            "show_roads": self.show_roads_cb.isChecked(),
            "show_settlements": self.show_settlements_cb.isChecked(),
            "show_resources": self.show_resources_cb.isChecked(),
            "show_shipping": self.show_shipping_cb.isChecked(),
            "show_labels": self.show_labels_cb.isChecked(),
        }

    def get_edit_options(self) -> dict:
        return {
            "edit_mode": self.edit_toggle.isChecked(),
            "edit_tool": ["terrain", "settlement", "resource", "erase"][
                self.edit_tool_combo.currentIndex()
            ],
            "settlement_type": [1, 2, 3, 4][self.settlement_type_combo.currentIndex()],
        }
