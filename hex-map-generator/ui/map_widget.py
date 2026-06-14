"""
地图渲染引擎 — 六边形地图绘制、图层切换、交互编辑
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush,
    QFont, QFontMetrics, QPolygonF,
)
from PySide6.QtWidgets import QWidget

import numpy as np

from core.hex_grid import HexCoord, HexGrid
from core.terrain_gen import (
    TerrainData, SETTLEMENT_NONE, SETTLEMENT_VILLAGE,
    SETTLEMENT_TOWN, SETTLEMENT_CITY, SETTLEMENT_CAPITAL,
)
from utils.colors import BIOME_COLORS, FEATURE_COLORS


class MapWidget(QWidget):
    """六边形地图渲染组件"""

    # 信号：鼠标悬停时显示坐标和地形信息
    hex_hovered = Signal(HexCoord, object)
    # 信号：六边形被编辑
    hex_edited = Signal(HexCoord)

    LAYER_ELEVATION = "elevation"
    LAYER_MOISTURE = "moisture"
    LAYER_TEMPERATURE = "temperature"
    LAYER_BIOME = "biome"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)

        # 地图数据
        self.hex_grid: Optional[HexGrid] = None
        self.terrain_data: Dict[HexCoord, TerrainData] = {}
        self.coord_to_idx: Dict[HexCoord, int] = {}
        self.elevation: Optional[np.ndarray] = None
        self.moisture: Optional[np.ndarray] = None
        self.temperature: Optional[np.ndarray] = None

        # 渲染参数
        self.hex_size = 20
        self.current_layer = self.LAYER_BIOME
        self.show_grid = True
        self.show_rivers = True
        self.show_roads = True
        self.show_settlements = True
        self.show_resources = True
        self.show_shipping = True
        self.show_labels = True

        # 交互状态
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPointF(0, 0)
        self.hovered_hex: Optional[HexCoord] = None
        self.selected_hex: Optional[HexCoord] = None

        # 编辑模式
        self.edit_mode = False
        self.edit_tool = "terrain"  # terrain, settlement, resource, erase
        self.edit_terrain = "plains"
        self.edit_settlement_type = SETTLEMENT_TOWN

        # 地图尺寸缓存
        self.map_bounds = QRectF()

    def set_map_data(self, hex_grid: HexGrid, terrain_data: Dict[HexCoord, TerrainData],
                     elevation: np.ndarray, moisture: np.ndarray,
                     temperature: np.ndarray, coord_to_idx: Dict[HexCoord, int]):
        """设置地图数据"""
        self.hex_grid = hex_grid
        self.terrain_data = terrain_data
        self.elevation = elevation
        self.moisture = moisture
        self.temperature = temperature
        self.coord_to_idx = coord_to_idx
        self._calc_bounds()
        self.update()

    def _calc_bounds(self):
        """计算地图边界"""
        if not self.hex_grid:
            return
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for hc in self.hex_grid.hexes:
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)
            min_x = min(min_x, cx - self.hex_size)
            max_x = max(max_x, cx + self.hex_size)
            min_y = min(min_y, cy - self.hex_size)
            max_y = max(max_y, cy + self.hex_size)
        self.map_bounds = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def set_layer(self, layer: str):
        self.current_layer = layer
        self.update()

    def set_hex_size(self, size: float):
        self.hex_size = max(6, min(40, size))
        self._calc_bounds()
        self.update()

    def toggle_edit_mode(self, enabled: bool):
        self.edit_mode = enabled

    def set_edit_tool(self, tool: str):
        self.edit_tool = tool

    def paintEvent(self, event):
        if not self.hex_grid:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "请生成地图")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor(30, 30, 40))

        # 应用平移
        painter.save()
        painter.translate(
            self.width() / 2 + self.pan_offset.x(),
            self.height() / 2 + self.pan_offset.y(),
        )

        # 视锥裁剪 - 只绘制可见区域的六边形
        vis_center_x = self.width() / 2 + self.pan_offset.x()
        vis_center_y = self.height() / 2 + self.pan_offset.y()
        vis_radius = math.sqrt(
            (self.width() / 2) ** 2 + (self.height() / 2) ** 2
        ) + self.hex_size * 2

        for hc in self.hex_grid.hexes:
            td = self.terrain_data.get(hc)
            if td is None:
                continue

            # 快速可见性测试
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)
            dx = cx - (self.width() / 2 - vis_center_x)
            dy = cy - (self.height() / 2 - vis_center_y)
            if dx * dx + dy * dy > vis_radius * vis_radius:
                # 粗略跳过 — 只对大地图有效
                if self.hex_grid.size > 30:
                    continue

            self._draw_hex(painter, hc, td)

        # 绘制特性覆盖层
        if self.show_shipping:
            self._draw_shipping_routes(painter)
        if self.show_rivers:
            self._draw_rivers(painter)
        if self.show_roads:
            self._draw_roads(painter)
        if self.show_settlements:
            self._draw_settlements(painter)
        if self.show_resources:
            self._draw_resources(painter)
        if self.show_labels:
            self._draw_labels(painter)

        # 绘制选中高亮
        if self.selected_hex and self.selected_hex in self.terrain_data:
            self._draw_hex_highlight(painter, self.selected_hex, QColor(255, 255, 0, 120))

        # 绘制悬停高亮
        if self.hovered_hex and self.hovered_hex in self.terrain_data:
            self._draw_hex_highlight(painter, self.hovered_hex, QColor(255, 255, 255, 60))

        painter.restore()

    def _get_hex_color(self, hc: HexCoord, td: TerrainData) -> QColor:
        """根据当前图层返回六边形颜色"""
        if self.current_layer == self.LAYER_ELEVATION:
            v = int(255 * td.elevation)
            return QColor(v, v, max(50, v))
        elif self.current_layer == self.LAYER_MOISTURE:
            v = int(255 * td.moisture)
            return QColor(max(30, 200 - v), max(30, 200 - v // 2), max(200, v))
        elif self.current_layer == self.LAYER_TEMPERATURE:
            v = int(255 * td.temperature)
            return QColor(255, max(50, 255 - v), max(50, 255 - v))
        else:
            return BIOME_COLORS.get(td.biome, QColor(100, 100, 100))

    def _draw_hex(self, painter: QPainter, hc: HexCoord, td: TerrainData):
        """绘制单个六边形"""
        corners = self.hex_grid.hex_corners(hc, self.hex_size)
        polygon = QPolygonF([QPointF(x, y) for x, y in corners])

        # 填充
        color = self._get_hex_color(hc, td)
        painter.setBrush(QBrush(color))

        # 描边
        if self.show_grid:
            if td.is_water:
                painter.setPen(QPen(QColor(20, 70, 160), 0.5))
            else:
                painter.setPen(QPen(QColor(60, 60, 70), 0.5))
        else:
            painter.setPen(Qt.NoPen)

        painter.drawPolygon(polygon)

    def _draw_hex_highlight(self, painter: QPainter, hc: HexCoord, color: QColor):
        corners = self.hex_grid.hex_corners(hc, self.hex_size + 2)
        polygon = QPolygonF([QPointF(x, y) for x, y in corners])
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1.5))
        painter.drawPolygon(polygon)

    def _draw_rivers(self, painter: QPainter):
        painter.setPen(QPen(FEATURE_COLORS.get("river", QColor(70, 140, 230)), 2.0))
        for hc, td in self.terrain_data.items():
            if td.river_flow > 0.3:
                cx, cy = self.hex_grid.hex_center(hc, self.hex_size)
                # 向最低邻居画线
                min_elev = td.elevation
                best_nh = None
                for nh in hc.neighbors():
                    if nh in self.terrain_data:
                        ntd = self.terrain_data[nh]
                        if ntd.elevation < min_elev:
                            min_elev = ntd.elevation
                            best_nh = nh
                if best_nh:
                    nx, ny = self.hex_grid.hex_center(best_nh, self.hex_size)
                    painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    def _draw_roads(self, painter: QPainter):
        painter.setPen(QPen(FEATURE_COLORS.get("road", QColor(160, 120, 80)), 1.5))
        visited = set()
        for hc, td in self.terrain_data.items():
            if not td.road:
                continue
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)
            for nh in hc.neighbors():
                if nh in visited:
                    continue
                if nh in self.terrain_data and self.terrain_data[nh].road:
                    nx, ny = self.hex_grid.hex_center(nh, self.hex_size)
                    painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))
            visited.add(hc)

    def _draw_settlements(self, painter: QPainter):
        for hc, td in self.terrain_data.items():
            if td.settlement == SETTLEMENT_NONE:
                continue
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)

            # 根据聚落类型选择颜色和大小
            if td.settlement == SETTLEMENT_CAPITAL:
                color = FEATURE_COLORS.get("capital", QColor(220, 180, 40))
                radius = self.hex_size * 0.45
                pen_width = 2.5
            elif td.settlement == SETTLEMENT_CITY:
                color = FEATURE_COLORS.get("city", QColor(180, 40, 40))
                radius = self.hex_size * 0.38
                pen_width = 2.0
            elif td.settlement == SETTLEMENT_TOWN:
                color = FEATURE_COLORS.get("town", QColor(200, 60, 60))
                radius = self.hex_size * 0.30
                pen_width = 1.5
            else:
                color = FEATURE_COLORS.get("village", QColor(220, 140, 60))
                radius = self.hex_size * 0.22
                pen_width = 1.0

            # 画圆形标记
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.white, pen_width))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            # 首都/城市加星
            if td.settlement in (SETTLEMENT_CAPITAL, SETTLEMENT_CITY):
                painter.setPen(QPen(Qt.white, 1.0))
                font = QFont("sans-serif", max(8, int(self.hex_size * 0.5)))
                painter.setFont(font)
                painter.drawText(QPointF(cx - radius * 0.3, cy + radius * 0.4), "★")

    def _draw_resources(self, painter: QPainter):
        for hc, td in self.terrain_data.items():
            if not td.resource:
                continue
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)

            # 用字母符号表示资源类型（跨平台兼容）
            text_symbols = {
                "wood": "W", "iron": "I", "gold": "G",
                "food": "F", "stone": "S",
            }
            sym = text_symbols.get(td.resource, "?")
            color = FEATURE_COLORS.get(f"resource_{td.resource}", QColor(200, 200, 200))

            painter.setPen(QPen(color, 1.0))
            font = QFont("sans-serif", max(7, int(self.hex_size * 0.4)), QFont.Bold)
            painter.setFont(font)
            painter.drawText(QPointF(cx - 4, cy + 3), sym)

    def _draw_shipping_routes(self, painter: QPainter):
        for hc, td in self.terrain_data.items():
            if not td.shipping:
                continue
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)
            painter.setPen(QPen(FEATURE_COLORS.get("shipping_route", QColor(40, 80, 180)), 1.0, Qt.DashLine))
            for nh in hc.neighbors():
                if nh in self.terrain_data and self.terrain_data[nh].shipping:
                    nx, ny = self.hex_grid.hex_center(nh, self.hex_size)
                    painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))
                    break  # 只画一条避免重复

    def _draw_labels(self, painter: QPainter):
        """绘制聚落名称标签"""
        for hc, td in self.terrain_data.items():
            if td.settlement == SETTLEMENT_NONE or not td.settlement_name:
                continue
            cx, cy = self.hex_grid.hex_center(hc, self.hex_size)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
            font = QFont("sans-serif", max(7, int(self.hex_size * 0.45)))
            painter.setFont(font)

            # 名称位于六边形下方
            fm = QFontMetrics(font)
            text = td.settlement_name
            tw = fm.horizontalAdvance(text)
            painter.drawText(
                QPointF(cx - tw / 2, cy + self.hex_size * 0.7),
                text
            )

    def mousePressEvent(self, event):
        if not self.hex_grid:
            return super().mousePressEvent(event)

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier
        ):
            self.is_panning = True
            self.last_mouse_pos = QPointF(event.position())
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            hc = self._hex_at_pos(event.position())
            if hc and hc in self.terrain_data:
                if self.edit_mode:
                    self._apply_edit(hc)
                else:
                    self.selected_hex = hc
                    self.hex_edited.emit(hc)
                    self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = QPointF(event.position()) - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = QPointF(event.position())
            self.update()
            return

        if self.hex_grid:
            hc = self._hex_at_pos(event.position())
            if hc != self.hovered_hex:
                self.hovered_hex = hc
                if hc and hc in self.terrain_data:
                    self.hex_hovered.emit(hc, self.terrain_data[hc])
                self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.set_hex_size(self.hex_size * factor)

    def _hex_at_pos(self, pos: QPointF) -> Optional[HexCoord]:
        """将鼠标位置转换为六边形坐标"""
        if not self.hex_grid:
            return None
        # 反算平移
        wx = pos.x() - self.width() / 2 - self.pan_offset.x()
        wy = pos.y() - self.height() / 2 - self.pan_offset.y()
        return self.hex_grid.hex_at_pixel(wx, wy, self.hex_size)

    def _apply_edit(self, hc: HexCoord):
        """在编辑模式下应用修改"""
        td = self.terrain_data.get(hc)
        if td is None:
            return

        if self.edit_tool == "terrain":
            # 切换地形
            td.biome = self.edit_terrain
            td.is_water = self.edit_terrain in ("ocean", "lake")
        elif self.edit_tool == "settlement":
            # 手动放置/移除聚落
            if td.settlement != SETTLEMENT_NONE:
                td.settlement = SETTLEMENT_NONE
                td.settlement_name = ""
            else:
                td.settlement = self.edit_settlement_type
                td.settlement_name = self._gen_name()
        elif self.edit_tool == "resource":
            if td.resource:
                td.resource = None
                td.resource_amount = 0
            else:
                resources = ["wood", "iron", "gold", "food", "stone"]
                import random as _random
                td.resource = _random.choice(resources)
                td.resource_amount = _random.randint(1, 4)
        elif self.edit_tool == "erase":
            td.settlement = SETTLEMENT_NONE
            td.settlement_name = ""
            td.resource = None
            td.resource_amount = 0
            td.road = False

        self.hex_edited.emit(hc)
        self.update()

    def _gen_name(self) -> str:
        prefixes = ["河", "山", "林", "湖", "溪", "谷", "岚", "雾", "霜", "风", "云", "星"]
        suffixes = ["村", "庄", "屯", "寨", "店", "镇", "集", "堡", "城"]
        import random as _random
        return f"{_random.choice(prefixes)}{_random.choice(suffixes)}"

    def fit_to_view(self):
        """适配地图到视图"""
        if not self.hex_grid or self.map_bounds.isEmpty():
            return
        margin = self.hex_size * 2
        available_w = self.width() - margin * 2
        available_h = self.height() - margin * 2
        if available_w <= 0 or available_h <= 0:
            return
        scale_x = available_w / self.map_bounds.width()
        scale_y = available_h / self.map_bounds.height()
        self.hex_size = min(scale_x, scale_y) * self.hex_size
        self.pan_offset = QPointF(0, 0)
        self._calc_bounds()
        self.update()