"""
离屏生成地图图片 — 直接使用 QPainter 绘制
"""

import math
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QApplication

from core.feature_gen import FeatureGenerator
from core.hex_grid import HexGrid
from core.noise_gen import NoiseGenerator
from core.terrain_gen import (
    SETTLEMENT_CAPITAL,
    SETTLEMENT_CITY,
    SETTLEMENT_NONE,
    SETTLEMENT_TOWN,
    TerrainGenerator,
)
from utils.colors import BIOME_COLORS, FEATURE_COLORS

_app = None


def draw_map(seed=42, size=40, output_path="generated_map.png", img_w=1600, img_h=1200):
    """直接生成地图图片"""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)

    # 1. 生成网格
    hex_grid = HexGrid(size=size)

    # 2. 计算六边形坐标
    # 防止 size=0 时除零崩溃
    safe_size = max(1, size)
    hex_size = max(6, min(30, 500 / safe_size))
    hex_coords_list = hex_grid.all_hex_centers(hex_size)
    coord_to_idx = {}
    for i, (hc, _, _) in enumerate(hex_coords_list):
        coord_to_idx[hc] = i
    coords_xy = [(x, y) for _, x, y in hex_coords_list]

    # 3. 生成噪声
    noise_gen = NoiseGenerator(seed=seed)
    elevation = noise_gen.generate_elevation(coords_xy, scale=3.0)
    moisture = noise_gen.generate_moisture(coords_xy, elevation, scale=3.6, monsoon_dir=90.0)
    temperature = noise_gen.generate_temperature(elevation, coords_xy)

    # 4. 生成地形
    terrain_gen = TerrainGenerator()
    terrain_gen.water_level = 0.35
    terrain_data = terrain_gen.generate(elevation, moisture, temperature, hex_coords_list)

    # 5. 生成特性
    rng = np.random.Generator(np.random.PCG64(seed + 42))
    feature_gen = FeatureGenerator(terrain_data, hex_coords_list)
    feature_gen.generate_rivers(rng, num_rivers=12)
    feature_gen.generate_settlements(
        rng, num_villages=10, num_towns=5, num_cities=2, has_capital=True
    )
    feature_gen.generate_roads()
    feature_gen.generate_resources(rng, density=0.08)
    feature_gen.generate_shipping_routes(rng)

    # 6. 计算地图实际边界
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    hex_width = hex_size * 1.5
    hex_height = hex_size * math.sqrt(3) / 2
    for hc in hex_grid.hexes:
        cx, cy = hex_grid.hex_center(hc, hex_size)
        min_x = min(min_x, cx - hex_width)
        max_x = max(max_x, cx + hex_width)
        min_y = min(min_y, cy - hex_height)
        max_y = max(max_y, cy + hex_height)

    map_w = max_x - min_x
    map_h = max_y - min_y

    # 7. 创建图片
    margin = 30
    image = QImage(img_w, img_h, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(22, 22, 30))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)

    # 8. 计算缩放和平移
    scale = min((img_w - margin * 2) / map_w, (img_h - margin * 2) / map_h)

    painter.translate(img_w / 2, img_h / 2)
    painter.scale(scale, scale)
    painter.translate(-(min_x + map_w / 2), -(min_y + map_h / 2))

    # 9. 绘制所有六边形
    for hc in hex_grid.hexes:
        td = terrain_data.get(hc)
        if td is None:
            continue
        corners = hex_grid.hex_corners(hc, hex_size)
        polygon = QPolygonF([QPointF(x, y) for x, y in corners])
        color = BIOME_COLORS.get(td.biome, QColor(100, 100, 100))
        painter.setBrush(QBrush(color))
        if td.is_water:
            painter.setPen(QPen(QColor(20, 70, 160), 0.5 / scale))
        else:
            painter.setPen(QPen(QColor(60, 60, 70), 0.5 / scale))
        painter.drawPolygon(polygon)

    # 10. 绘制河流
    painter.setPen(QPen(FEATURE_COLORS.get("river", QColor(70, 140, 230)), 2.0 / scale))
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

    # 11. 绘制道路
    painter.setPen(QPen(FEATURE_COLORS.get("road", QColor(160, 120, 80)), 1.5 / scale))
    drawn_edges = set()
    for hc, td in terrain_data.items():
        if not td.road:
            continue
        cx, cy = hex_grid.hex_center(hc, hex_size)
        for nh in hc.neighbors():
            if nh not in terrain_data or not terrain_data[nh].road:
                continue
            edge = (hc.q * 100000 + hc.r, nh.q * 100000 + nh.r)
            key = (min(edge), max(edge))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            nx, ny = hex_grid.hex_center(nh, hex_size)
            painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    # 12. 绘制聚落
    font = QFont("Microsoft YaHei", int(hex_size * 0.5))
    painter.setFont(font)
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
        painter.setPen(QPen(Qt.white, pen_width / scale))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        if td.settlement in (SETTLEMENT_CAPITAL, SETTLEMENT_CITY):
            star = "★"
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(star)
            th = fm.ascent()
            painter.setPen(QPen(Qt.white, 1.0 / scale))
            painter.drawText(
                QPointF(cx - tw / 2, cy + th / 2 - fm.descent() / 2),
                star,
            )

    # 13. 绘制资源
    text_symbols = {"wood": "W", "iron": "I", "gold": "G", "food": "F", "stone": "S"}
    resource_font = QFont("Microsoft YaHei", int(hex_size * 0.4), QFont.Bold)
    painter.setFont(resource_font)
    for hc, td in terrain_data.items():
        if not td.resource:
            continue
        cx, cy = hex_grid.hex_center(hc, hex_size)
        sym = text_symbols.get(td.resource, "?")
        color = FEATURE_COLORS.get(f"resource_{td.resource}", QColor(200, 200, 200))
        painter.setPen(QPen(color, 1.0 / scale))
        fm = QFontMetrics(resource_font)
        tw = fm.horizontalAdvance(sym)
        th = fm.ascent()
        painter.drawText(
            QPointF(cx - tw / 2, cy + th / 2 - fm.descent() / 2),
            sym,
        )

    # 14. 绘制航线
    painter.setPen(
        QPen(
            FEATURE_COLORS.get("shipping_route", QColor(40, 80, 180)),
            1.0 / scale,
            Qt.DashLine,
        )
    )
    drawn_edges = set()
    for hc, td in terrain_data.items():
        if not td.shipping:
            continue
        cx, cy = hex_grid.hex_center(hc, hex_size)
        for nh in hc.neighbors():
            if nh not in terrain_data or not terrain_data[nh].shipping:
                continue
            edge = (hc.q * 100000 + hc.r, nh.q * 100000 + nh.r)
            key = (min(edge), max(edge))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            nx, ny = hex_grid.hex_center(nh, hex_size)
            painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

    # 15. 绘制聚落名称
    label_font = QFont("Microsoft YaHei", int(hex_size * 0.45))
    painter.setFont(label_font)
    fm = QFontMetrics(label_font)
    for hc, td in terrain_data.items():
        if td.settlement == SETTLEMENT_NONE or not td.settlement_name:
            continue
        cx, cy = hex_grid.hex_center(hc, hex_size)
        text = td.settlement_name
        if len(text) > 6:
            text = text[:6] + "…"
        tw = fm.horizontalAdvance(text)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.0 / scale))
        painter.drawText(QPointF(cx - tw / 2, cy + hex_size * 0.7), text)

    painter.end()

    # 16. 保存
    if image.save(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ 地图已保存到: {output_path}")
        print(f"   图片尺寸: {image.width()} x {image.height()}")
        print(f"   文件大小: {file_size / 1024:.1f} KB")
        print(f"   种子: {seed}, 地图大小: {size}")
        return output_path
    else:
        print("❌ 保存图片失败")
        return None


if __name__ == "__main__":
    output = draw_map(
        seed=42, size=40, output_path="/workspace/hex-map-generator/generated_map.png"
    )
    if output:
        print("\n🎉 成功生成地图图片!")
        print(f"   路径: {output}")
