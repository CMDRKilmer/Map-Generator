"""
地图配色方案
"""
from PySide6.QtGui import QColor

# 特征颜色
FEATURE_COLORS = {
    "town": QColor(200, 60, 60),           # 城镇 (红色)
    "city": QColor(180, 40, 40),           # 城市 (深红)
    "village": QColor(220, 140, 60),       # 村庄 (橙色)
    "capital": QColor(220, 180, 40),       # 首都 (金色)
    "road": QColor(160, 120, 80),          # 道路 (棕色)
    "river": QColor(70, 140, 230),         # 河流 (蓝色)
    "shipping_route": QColor(40, 80, 180), # 航线 (蓝色)
    "resource_wood": QColor(40, 130, 40),   # 木材
    "resource_iron": QColor(140, 130, 140), # 铁矿
    "resource_gold": QColor(210, 180, 40),  # 金矿
    "resource_food": QColor(200, 180, 80),  # 粮食
    "resource_stone": QColor(150, 140, 130),# 石材
}

# 生物群落背景色
BIOME_COLORS = {
    "ocean": QColor(25, 90, 190),
    "lake": QColor(50, 130, 210),
    "beach": QColor(238, 214, 150),
    "plains": QColor(140, 200, 90),
    "forest": QColor(60, 160, 60),
    "dense_forest": QColor(30, 130, 40),
    "rainforest": QColor(20, 120, 40),
    "taiga": QColor(80, 140, 100),
    "tundra": QColor(180, 190, 170),
    "snow": QColor(240, 240, 255),
    "desert": QColor(240, 210, 120),
    "savanna": QColor(200, 190, 80),
    "hills": QColor(180, 170, 80),
    "mountains": QColor(160, 140, 110),
    "high_mountains": QColor(200, 190, 180),
    "swamp": QColor(100, 150, 120),
    "volcano": QColor(180, 60, 30),
}