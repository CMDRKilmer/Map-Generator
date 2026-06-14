"""
导出功能 — PNG / SVG / JSON 格式导出
"""

from __future__ import annotations

import json
from typing import Dict, Set, Tuple

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QFileDialog, QMessageBox

from core.hex_grid import HexCoord
from core.terrain_gen import (
    SETTLEMENT_CAPITAL,
    SETTLEMENT_CITY,
    SETTLEMENT_NONE,
    SETTLEMENT_TOWN,
    TerrainData,
)
from utils.colors import BIOME_COLORS, FEATURE_COLORS


class MapExporter:
    """地图导出器"""

    def __init__(self, widget):
        self.widget = widget

    def export_png(self, parent_widget, filepath: str = ""):
        """导出为 PNG 图片（含所有特性覆盖层）"""
        if not self.widget.hex_grid:
            QMessageBox.warning(parent_widget, "提示", "请先生成地图")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                parent_widget, "导出 PNG", "map.png", "PNG Images (*.png)"
            )
            if not filepath:
                return

        margin = 20
        hex_size = self.widget.hex_size
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for hc in self.widget.hex_grid.hexes:
            cx, cy = self.widget.hex_grid.hex_center(hc, hex_size)
            min_x = min(min_x, cx - hex_size)
            max_x = max(max_x, cx + hex_size)
            min_y = min(min_y, cy - hex_size)
            max_y = max(max_y, cy + hex_size)

        width = int(max_x - min_x + margin * 2)
        height = int(max_y - min_y + margin * 2)

        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(30, 30, 40))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(margin - min_x, margin - min_y)

        # 绘制所有六边形
        for hc in self.widget.hex_grid.hexes:
            td = self.widget.terrain_data.get(hc)
            if td is None:
                continue
            self._draw_hex(painter, hc, td, hex_size)

        # 绘制特性覆盖层
        terrain_data = self.widget.terrain_data
        hex_grid = self.widget.hex_grid

        self._draw_shipping_routes(painter, terrain_data, hex_grid, hex_size)
        self._draw_rivers(painter, terrain_data, hex_grid, hex_size)
        self._draw_roads(painter, terrain_data, hex_grid, hex_size)
        self._draw_settlements(painter, terrain_data, hex_grid, hex_size)
        self._draw_resources(painter, terrain_data, hex_grid, hex_size)
        self._draw_labels(painter, terrain_data, hex_grid, hex_size)

        painter.end()
        image.save(filepath)
        QMessageBox.information(parent_widget, "完成", f"地图已导出到:\n{filepath}")

    def export_svg(self, parent_widget, filepath: str = ""):
        """导出为 SVG 矢量图（简易文本版）"""
        if not self.widget.hex_grid:
            QMessageBox.warning(parent_widget, "提示", "请先生成地图")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                parent_widget, "导出 SVG", "map.svg", "SVG Images (*.svg)"
            )
            if not filepath:
                return

        hex_size = self.widget.hex_size
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for hc in self.widget.hex_grid.hexes:
            cx, cy = self.widget.hex_grid.hex_center(hc, hex_size)
            min_x = min(min_x, cx - hex_size)
            max_x = max(max_x, cx + hex_size)
            min_y = min(min_y, cy - hex_size)
            max_y = max(max_y, cy + hex_size)

        w = max_x - min_x + 40
        h = max_y - min_y + 40

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"'
            f' width="{w}" height="{h}">',
            f'<rect width="{w}" height="{h}" fill="#1e1e28"/>',
        ]

        for hc in self.widget.hex_grid.hexes:
            td = self.widget.terrain_data.get(hc)
            if td is None:
                continue
            corners = self.widget.hex_grid.hex_corners(hc, hex_size)
            color = BIOME_COLORS.get(td.biome, QColor(100, 100, 100))
            pts = " ".join(f"{x - min_x + 20},{y - min_y + 20}" for x, y in corners)
            svg_lines.append(
                f'<polygon points="{pts}" fill="{color.name()}" stroke="#444" stroke-width="0.5"/>'
            )

        svg_lines.append("</svg>")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        QMessageBox.information(parent_widget, "完成", f"SVG已导出到:\n{filepath}")

    def export_json(self, parent_widget, filepath: str = ""):
        """导出为 JSON 数据文件"""
        if not self.widget.hex_grid:
            QMessageBox.warning(parent_widget, "提示", "请先生成地图")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                parent_widget, "导出 JSON", "map_data.json", "JSON Files (*.json)"
            )
            if not filepath:
                return

        hex_size = self.widget.hex_size
        data = {
            "meta": {
                "grid_size": self.widget.hex_grid.size,
                "hex_size": hex_size,
                "total_hexes": len(self.widget.hex_grid.hexes),
            },
            "hexes": [],
        }

        for hc in self.widget.hex_grid.hexes:
            td = self.widget.terrain_data.get(hc)
            if td is None:
                continue
            cx, cy = self.widget.hex_grid.hex_center(hc, hex_size)
            settlement_name = ""
            if td.settlement != SETTLEMENT_NONE:
                settlement_name = td.settlement_name

            data["hexes"].append(
                {
                    "q": hc.q,
                    "r": hc.r,
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                    "elevation": round(td.elevation, 3),
                    "moisture": round(td.moisture, 3),
                    "temperature": round(td.temperature, 3),
                    "biome": td.biome,
                    "is_water": td.is_water,
                    "river_flow": round(td.river_flow, 3),
                    "settlement": td.settlement,
                    "settlement_name": settlement_name,
                    "resource": td.resource or "",
                    "resource_amount": td.resource_amount,
                    "road": td.road,
                    "shipping": td.shipping,
                    "volcanic": td.volcanic,
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        QMessageBox.information(parent_widget, "完成", f"JSON数据已导出到:\n{filepath}")

    # ─── 绘制方法 ─────────────────────────────────────────────

    def _draw_hex(self, painter: QPainter, hc: HexCoord, td: TerrainData, hex_size: float):
        """在 Painter 上绘制单个六边形"""
        corners = self.widget.hex_grid.hex_corners(hc, hex_size)
        path = QPainterPath()
        path.moveTo(QPointF(*corners[0]))
        for x, y in corners[1:]:
            path.lineTo(QPointF(x, y))
        path.closeSubpath()

        color = BIOME_COLORS.get(td.biome, QColor(100, 100, 100))
        painter.fillPath(path, color)
        painter.setPen(QPen(QColor(60, 60, 70), 0.5))
        painter.drawPath(path)

    def _draw_rivers(self, painter: QPainter, terrain_data: Dict, hex_grid, hex_size: float):
        painter.setPen(QPen(FEATURE_COLORS.get("river", QColor(70, 140, 230)), 2.0))
        for hc, td in terrain_data.items():
            if td.river_flow > 0.3:
                cx, cy = hex_grid.hex_center(hc, hex_size)
                min_elev = td.elevation
                best_nh = None
                for nh in hc.neighbors():
                    if nh in terrain_data:
                        ntd = terrain_data[nh]
                        if ntd.elevation < min_elev:
                            min_elev = ntd.elevation
                            best_nh = nh
                if best_nh:
                    nx, ny = hex_grid.hex_center(best_nh, hex_size)
                    painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    def _draw_roads(self, painter: QPainter, terrain_data: Dict, hex_grid, hex_size: float):
        painter.setPen(QPen(FEATURE_COLORS.get("road", QColor(160, 120, 80)), 1.5))
        # 用无向边集合保证对称画线且不重复
        drawn_edges: Set[Tuple[int, int]] = set()
        for hc, td in terrain_data.items():
            if not td.road:
                continue
            cx, cy = hex_grid.hex_center(hc, hex_size)
            for nh in hc.neighbors():
                if nh not in terrain_data or not terrain_data[nh].road:
                    continue
                edge = (
                    hc.q * 100000 + hc.r,
                    nh.q * 100000 + nh.r,
                )
                key = (min(edge), max(edge))
                if key in drawn_edges:
                    continue
                drawn_edges.add(key)
                nx, ny = hex_grid.hex_center(nh, hex_size)
                painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    def _draw_settlements(self, painter: QPainter, terrain_data: Dict, hex_grid, hex_size: float):
        for hc, td in terrain_data.items():
            if td.settlement == SETTLEMENT_NONE:
                continue
            cx, cy = hex_grid.hex_center(hc, hex_size)

            if td.settlement == SETTLEMENT_CAPITAL:
                color = FEATURE_COLORS.get("capital", QColor(220, 180, 40))
                radius = hex_size * 0.45
                pen_width = 2.5
            elif td.settlement == SETTLEMENT_CITY:
                color = FEATURE_COLORS.get("city", QColor(180, 40, 40))
                radius = hex_size * 0.38
                pen_width = 2.0
            elif td.settlement == SETTLEMENT_TOWN:
                color = FEATURE_COLORS.get("town", QColor(200, 60, 60))
                radius = hex_size * 0.30
                pen_width = 1.5
            else:
                color = FEATURE_COLORS.get("village", QColor(220, 140, 60))
                radius = hex_size * 0.22
                pen_width = 1.0

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.white, pen_width))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            if td.settlement in (SETTLEMENT_CAPITAL, SETTLEMENT_CITY):
                font = QFont("sans-serif", max(8, int(hex_size * 0.5)))
                painter.setFont(font)
                fm = QFontMetrics(font)
                star = "★"
                tw = fm.horizontalAdvance(star)
                th = fm.ascent()
                # 度量居中而非硬编码偏移
                painter.setPen(QPen(Qt.white, 1.0))
                painter.drawText(
                    QPointF(cx - tw / 2, cy + th / 2 - fm.descent() / 2),
                    star,
                )

    def _draw_resources(self, painter: QPainter, terrain_data: Dict, hex_grid, hex_size: float):
        text_symbols = {
            "wood": "W",
            "iron": "I",
            "gold": "G",
            "food": "F",
            "stone": "S",
        }
        font = QFont("sans-serif", max(7, int(hex_size * 0.4)), QFont.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        for hc, td in terrain_data.items():
            if not td.resource:
                continue
            cx, cy = hex_grid.hex_center(hc, hex_size)
            sym = text_symbols.get(td.resource, "?")
            color = FEATURE_COLORS.get(f"resource_{td.resource}", QColor(200, 200, 200))
            painter.setPen(QPen(color, 1.0))
            tw = fm.horizontalAdvance(sym)
            th = fm.ascent()
            # 用 QFontMetrics 居中而非硬编码 -4, +3
            painter.drawText(
                QPointF(cx - tw / 2, cy + th / 2 - fm.descent() / 2),
                sym,
            )

    def _draw_shipping_routes(
        self, painter: QPainter, terrain_data: Dict, hex_grid, hex_size: float
    ):
        # QPen 移到循环外避免重复构造
        painter.setPen(
            QPen(
                FEATURE_COLORS.get("shipping_route", QColor(40, 80, 180)),
                1.0,
                Qt.DashLine,
            )
        )
        drawn_edges: Set[Tuple[int, int]] = set()
        for hc, td in terrain_data.items():
            if not td.shipping:
                continue
            cx, cy = hex_grid.hex_center(hc, hex_size)
            for nh in hc.neighbors():
                if nh not in terrain_data or not terrain_data[nh].shipping:
                    continue
                edge = (
                    hc.q * 100000 + hc.r,
                    nh.q * 100000 + nh.r,
                )
                key = (min(edge), max(edge))
                if key in drawn_edges:
                    continue
                drawn_edges.add(key)
                nx, ny = hex_grid.hex_center(nh, hex_size)
                painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    def _draw_labels(self, painter: QPainter, terrain_data: Dict, hex_grid, hex_size: float):
        font = QFont("sans-serif", max(7, int(hex_size * 0.45)))
        painter.setFont(font)
        fm = QFontMetrics(font)
        # 名称过长时按字符截断，避免与邻接标签重叠
        max_chars = 6
        for hc, td in terrain_data.items():
            if td.settlement == SETTLEMENT_NONE or not td.settlement_name:
                continue
            cx, cy = hex_grid.hex_center(hc, hex_size)
            text = td.settlement_name
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            tw = fm.horizontalAdvance(text)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
            painter.drawText(QPointF(cx - tw / 2, cy + hex_size * 0.7), text)
