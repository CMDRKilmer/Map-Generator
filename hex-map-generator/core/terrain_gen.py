"""
地形与生物群落生成系统
根据高程、湿度、温度数据为每个六边形分配地形和生物群落
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.hex_grid import HexCoord

# 生物群落枚举
BIOME_OCEAN = "ocean"
BIOME_LAKE = "lake"
BIOME_BEACH = "beach"
BIOME_PLAINS = "plains"
BIOME_FOREST = "forest"
BIOME_DENSE_FOREST = "dense_forest"
BIOME_RAINFOREST = "rainforest"
BIOME_TAIGA = "taiga"
BIOME_TUNDRA = "tundra"
BIOME_SNOW = "snow"
BIOME_DESERT = "desert"
BIOME_SAVANNA = "savanna"
BIOME_HILLS = "hills"
BIOME_MOUNTAINS = "mountains"
BIOME_HIGH_MOUNTAINS = "high_mountains"
BIOME_SWAMP = "swamp"
BIOME_VOLCANO = "volcano"

# 资源类型
RESOURCE_WOOD = "wood"
RESOURCE_IRON = "iron"
RESOURCE_GOLD = "gold"
RESOURCE_FOOD = "food"
RESOURCE_STONE = "stone"

# 聚落类型
SETTLEMENT_NONE = 0
SETTLEMENT_VILLAGE = 1
SETTLEMENT_TOWN = 2
SETTLEMENT_CITY = 3
SETTLEMENT_CAPITAL = 4


class TerrainData:
    """单个六边形的地形数据"""

    def __init__(self):
        self.elevation: float = 0.0  # 0~1
        self.moisture: float = 0.0  # 0~1
        self.temperature: float = 0.5  # 0~1
        self.biome: str = BIOME_OCEAN
        self.is_water: bool = True
        self.is_coast: bool = False
        self.river_flow: float = 0.0  # 河流流量
        self.resource: Optional[str] = None
        self.resource_amount: int = 0
        self.settlement: int = SETTLEMENT_NONE
        self.settlement_name: str = ""
        self.settlement_size: int = 1  # 1-5等级
        self.road: bool = False
        self.shipping: bool = False  # 是否航线经过
        self.volcanic: bool = False  # 是否火山

    def __repr__(self):
        return f"Terrain({self.biome}, elev={self.elevation:.2f})"


class TerrainGenerator:
    """地形与生物群落生成器"""

    def __init__(self):
        self.water_level = 0.35
        self.mountain_level = 0.70
        self.snow_level = 0.85
        self.volcano_chance = 0.005
        self.swamp_moisture = 0.80
        self.desert_moisture = 0.25

        # 生物群落图 (Whittaker 分类风格)
        # 基于 elevation(0~1) 和 moisture(0~1)
        self.biome_scheme = "whittaker"

    def classify_biome(
        self, elevation: float, moisture: float, temperature: float, is_coast: bool
    ) -> str:
        """根据高程、湿度、温度分类生物群落"""
        # 深海
        if elevation < 0.15:
            return BIOME_OCEAN
        # 浅水
        if elevation < self.water_level:
            return BIOME_LAKE
        # 沙滩 (海岸线)
        if is_coast and elevation < self.water_level + 0.08:
            return BIOME_BEACH

        # 高山雪地
        if elevation > self.snow_level and temperature < 0.3:
            return BIOME_SNOW

        # 高山
        if elevation > 0.80:
            return BIOME_HIGH_MOUNTAINS
        if elevation > 0.65:
            return BIOME_MOUNTAINS
        if elevation > 0.55:
            return BIOME_HILLS

        # 低地 — 基于湿度和温度
        if moisture < self.desert_moisture:
            if temperature > 0.6:
                return BIOME_DESERT
            else:
                return BIOME_TUNDRA

        if moisture < 0.35:
            if temperature > 0.6:
                return BIOME_SAVANNA
            else:
                return BIOME_PLAINS

        if moisture < 0.55:
            if temperature > 0.5:
                return BIOME_PLAINS
            else:
                return BIOME_TAIGA

        if moisture < 0.75:
            if temperature > 0.7:
                return BIOME_FOREST
            elif temperature > 0.3:
                return BIOME_FOREST
            else:
                return BIOME_TAIGA

        # 高湿度
        if temperature > 0.75:
            return BIOME_RAINFOREST
        if temperature > 0.5:
            return BIOME_DENSE_FOREST
        if moisture > self.swamp_moisture and elevation < 0.4:
            return BIOME_SWAMP
        return BIOME_FOREST

    def generate(
        self,
        elevation: np.ndarray,
        moisture: np.ndarray,
        temperature: np.ndarray,
        hex_coords_list: List,
    ) -> Dict:
        """
        生成所有六边形的完整地形数据

        Args:
            elevation: (N,) 0~1
            moisture: (N,) 0~1
            temperature: (N,) 0~1
            hex_coords_list: [(HexCoord, x, y), ...]

        Returns:
            {HexCoord: TerrainData}
        """
        terrain_data = {}
        rng = np.random.Generator(
            np.random.PCG64(int(np.sum(elevation[: min(100, len(elevation))] * 1000) % 10000) + 1)
        )

        # 构建快速查找
        coord_map = {}
        for i, (hc, _, _) in enumerate(hex_coords_list):
            coord_map[(hc.q, hc.r)] = i

        def _is_coast(hc: HexCoord, idx: int) -> bool:
            if elevation[idx] >= self.water_level:
                return False
            for nh in hc.neighbors():
                nk = (nh.q, nh.r)
                if nk in coord_map:
                    nidx = coord_map[nk]
                    if elevation[nidx] >= self.water_level:
                        return True
            return False

        for i, (hc, x, y) in enumerate(hex_coords_list):
            td = TerrainData()
            td.elevation = float(elevation[i])
            td.moisture = float(moisture[i])
            td.temperature = float(temperature[i])
            td.is_water = elevation[i] < self.water_level

            # 判断是否海岸
            if td.is_water:
                td.is_coast = _is_coast(hc, i)
            else:
                # 检查邻居是否有水
                for nh in hc.neighbors():
                    nk = (nh.q, nh.r)
                    if nk in coord_map:
                        nidx = coord_map[nk]
                        if elevation[nidx] < self.water_level:
                            td.is_coast = True
                            break

            # 火山（稀有）
            if (
                not td.is_water
                and elevation[i] > 0.6
                and moisture[i] < 0.4
                and rng.random() < self.volcano_chance
            ):
                td.volcanic = True
                td.biome = BIOME_VOLCANO
            else:
                td.biome = self.classify_biome(
                    elevation[i], moisture[i], temperature[i], td.is_coast
                )

            terrain_data[hc] = td

        return terrain_data
