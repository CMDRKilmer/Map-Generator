"""
导出功能 — PNG / SVG / JSON 格式导出
"""
from __future__ import annotations
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from PySide6.QtGui import QPainter, QColor, QImage, QPen, QBrush, QFont
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtWidgets import QFileDialog, QMessageBox

from core.hex_grid import HexCoord, HexGrid
from core.terrain_gen import TerrainData, SETTLEMENT_NONE, SETTLEMENT_CAPITAL
from utils.colors import BIOME_COLORS


class MapExporter:
    """地图导出器"""

    def __init__(self, widget):
        self.widget = widget

    def export_png(self, parent_widget, filepath: str = ""):
        """导出为 PNG 图片"""
        if not self.widget.hex_grid:
            QMessageBox.warning(parent_widget, "提示", "请先生成地图")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                parent_widget, "导出 PNG", "map.png", "PNG Images (*.png)"
            )
            if not filepath:
                return

        # 计算地图实际大小
        margin = 20
        hex_size = self.widget.hex_size
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

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
            self._draw_hex_to_painter(painter, hc, td, hex_size)

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
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

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
            cx, cy = self.widget.hex_grid.hex_center(hc, hex_size)
            color = BIOME_COLORS.get(td.biome, QColor(100, 100, 100))
            pts = " ".join(
                f"{x - min_x + 20},{y - min_y + 20}" for x, y in corners
            )
            svg_lines.append(
                f'<polygon points="{pts}" fill="{color.name()}" '
                f'stroke="#444" stroke-width="0.5"/>'
            )

        svg_lines.append('</svg>')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg_lines))

        QMessageBox.information(parent_widget, "完成", f"SVG已导出到:\n{filepath}")

    def export_json(self, parent_widget, filepath: str = ""):
        """导出为 JSON 数据文件"""
        if not self.widget.hex_grid:
            QMessageBox.warning(parent_widget, "提示", "请先生成地图")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                parent_widget, "导出 JSON", "map_data.json",
                "JSON Files (*.json)"
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

            data["hexes"].append({
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
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        QMessageBox.information(parent_widget, "完成", f"JSON数据已导出到:\n{filepath}")

    def _draw_hex_to_painter(self, painter: QPainter, hc: HexCoord,
                              td: TerrainData, hex_size: float):
        """在 Painter 上绘制单个六边形"""
        corners = self.widget.hex_grid.hex_corners(hc, hex_size)
        poly = [QPointF(x, y) for x, y in corners]

        # 由于 QPolygonF 构建问题，逐点绘制
        path = __import__('PySide6.QtGui', fromlist=['QPainterPath']).QPainterPath()
        path.moveTo(poly[0])
        for p in poly[1:]:
            path.lineTo(p)
        path.closeSubpath()

        color = BIOME_COLORS.get(td.biome, QColor(100, 100, 100))
        painter.fillPath(path, color)
        painter.setPen(QPen(QColor(60, 60, 70), 0.5))
        painter.drawPath(path)