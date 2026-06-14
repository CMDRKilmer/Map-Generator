"""
明亮风格的地图配色方案
"""
from PySide6.QtGui import QColor

# 地形颜色 (明亮风格)
TERRAIN_COLORS = {
    "deep_water": QColor(30, 100, 200),      # 深水
    "shallow_water": QColor(60, 140, 220),    # 浅水
    "sand": QColor(238, 214, 150),             # 沙滩
    "plains": QColor(140, 200, 90),            # 平原
    "grassland": QColor(120, 190, 80),         # 草地
    "forest": QColor(60, 160, 60),             # 森林
    "dense_forest": QColor(30, 130, 40),       # 密林
    "hills": QColor(180, 170, 80),             # 丘陵
    "mountains": QColor(160, 140, 110),        # 山地
    "high_mountains": QColor(200, 190, 180),   # 高山
    "snow": QColor(240, 240, 255),             # 雪地
    "desert": QColor(240, 210, 120),           # 沙漠
    "tundra": QColor(180, 190, 170),           # 冻土
    "swamp": QColor(100, 150, 120),            # 沼泽
    "volcano": QColor(180, 60, 30),            # 火山
    "taiga": QColor(80, 140, 100),             # 针叶林
    "savanna": QColor(200, 190, 80),           # 稀树草原
}

# 水体颜色
WATER_COLORS = {
    "ocean": QColor(25, 90, 190),
    "lake": QColor(50, 130, 210),
    "river": QColor(70, 140, 230),
}

# 特征颜色
FEATURE_COLORS = {
    "town": QColor(200, 60, 60),           # 城镇 (红色)
    "city": QColor(180, 40, 40),           # 城市 (深红)
    "village": QColor(220, 140, 60),       # 村庄 (橙色)
    "capital": QColor(220, 180, 40),       # 首都 (金色)
    "road": QColor(160, 120, 80),          # 道路 (棕色)
    "shipping_route": QColor(40, 80, 180), # 航线 (蓝色)
    "resource_wood": QColor(40, 130, 40),   # 木材
    "resource_iron": QColor(140, 130, 140), # 铁矿
    "resource_gold": QColor(210, 180, 40),  # 金矿
    "resource_food": QColor(200, 180, 80),  # 粮食
    "resource_stone": QColor(150, 140, 130),# 石材
}

# 图层名称映射
LAYER_NAMES = {
    "elevation": "高程图",
    "moisture": "湿度图",
    "temperature": "温度图",
    "biome": "生物群落",
    "political": "政治地图",
}

# 高程渲染 (灰度)
def elevation_color(value: float) -> QColor:
    """value: 0.0 ~ 1.0"""
    g = int(200 * value + 55)
    return QColor(g, g, g)


# 生物群落背景色 (更鲜艳版本)
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