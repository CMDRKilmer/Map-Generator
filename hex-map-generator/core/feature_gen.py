"""
特性生成系统 — 河流、季风、聚落、道路、资源、航线
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from core.hex_grid import HexCoord, HEX_DIRECTIONS
from core.terrain_gen import (
    TerrainData, SETTLEMENT_NONE, SETTLEMENT_VILLAGE,
    SETTLEMENT_TOWN, SETTLEMENT_CITY, SETTLEMENT_CAPITAL,
    RESOURCE_WOOD, RESOURCE_IRON, RESOURCE_GOLD,
    RESOURCE_FOOD, RESOURCE_STONE, BIOME_VOLCANO,
)


class FeatureGenerator:
    """生成河流、聚落、道路、资源、航线等地图特性"""

    def __init__(self, terrain_data: Dict[HexCoord, TerrainData],
                 hex_coords_list: List[Tuple[HexCoord, float, float]]):
        self.terrain = terrain_data
        self.hex_coords_list = hex_coords_list

        # 构建快速索引
        self.coord_to_idx: Dict[HexCoord, int] = {}
        for i, (hc, _, _) in enumerate(hex_coords_list):
            self.coord_to_idx[hc] = i

        # 构建地形查找
        self.land_hexes: List[HexCoord] = []
        self.water_hexes: List[HexCoord] = []
        self.coast_hexes: List[HexCoord] = []
        for hc, td in terrain_data.items():
            if td.is_water:
                self.water_hexes.append(hc)
                if td.is_coast:
                    self.coast_hexes.append(hc)
            else:
                self.land_hexes.append(hc)
                if td.is_coast:
                    self.coast_hexes.append(hc)

    def generate_rivers(self, rng: np.random.Generator,
                        num_rivers: int = 12) -> List[List[HexCoord]]:
        """
        河流生成 — 从高地流向低地到海洋
        使用流水模拟：从随机高地起点开始，沿最陡下降方向流动
        """
        rivers = []
        candidates = [h for h in self.land_hexes
                      if self.terrain[h].elevation > 0.55
                      and not self.terrain[h].volcanic]

        if not candidates:
            return rivers

        river_starts = rng.choice(len(candidates),
                                  size=min(num_rivers, len(candidates)),
                                  replace=False)

        for idx in river_starts:
            river = self._trace_river(candidates[idx], rng)
            if river and len(river) > 3:
                rivers.append(river)
                # 标记河流流量
                for i, hc in enumerate(river):
                    flow = 1.0 - (i / max(len(river), 1))
                    self.terrain[hc].river_flow = max(
                        self.terrain[hc].river_flow, flow
                    )

        return rivers

    def _trace_river(self, start: HexCoord, rng: np.random.Generator) -> List[HexCoord]:
        """从起点追踪河流路径到海洋"""
        path = [start]
        current = start
        visited = {start}
        max_steps = 200

        for _ in range(max_steps):
            td = self.terrain[current]
            if td.is_water:
                break

            # 找最低邻居（最陡下降）
            best_neighbor = None
            best_elev = td.elevation

            for nh in current.neighbors():
                if nh in visited:
                    continue
                if nh not in self.terrain:
                    continue
                ntd = self.terrain[nh]
                if ntd.elevation < best_elev:
                    best_elev = ntd.elevation
                    best_neighbor = nh

            if best_neighbor is None:
                # 随机选一个未访问的
                unvisited = [nh for nh in current.neighbors()
                             if nh in self.terrain and nh not in visited]
                if not unvisited:
                    break
                best_neighbor = rng.choice(unvisited)

            current = best_neighbor
            visited.add(current)
            path.append(current)

            if self.terrain[current].is_water:
                break

        return path

    def generate_settlements(self, rng: np.random.Generator,
                             num_villages: int = 8,
                             num_towns: int = 4,
                             num_cities: int = 2,
                             has_capital: bool = True) -> List[HexCoord]:
        """
        聚落生成 — 在地形合适的陆地上放置聚落
        偏好：平原 > 丘陵 > 森林 > 海岸
        """
        settlements = []

        # 根据地形评分排序
        scored_land = []
        for hc in self.land_hexes:
            td = self.terrain[hc]
            if td.volcanic:
                continue
            score = self._settlement_score(td, hc)
            scored_land.append((hc, score))

        scored_land.sort(key=lambda x: -x[1])
        candidates = [h for h, s in scored_land[:50]]

        if not candidates:
            return []

        used = set()

        def _place(typ: str, num: int, min_dist: int) -> List[HexCoord]:
            placed = []
            for hc in candidates:
                if len(placed) >= num:
                    break
                if hc in used:
                    continue
                # 确保与已有聚落保持距离
                too_close = False
                for uh in used:
                    if hc.distance_to(uh) < min_dist:
                        too_close = True
                        break
                if too_close:
                    continue
                used.add(hc)
                placed.append(hc)

                # 设置聚落类型
                td = self.terrain[hc]
                if typ == "capital":
                    td.settlement = SETTLEMENT_CAPITAL
                    td.settlement_name = self._generate_name(rng, "capital")
                    td.settlement_size = 5
                elif typ == "city":
                    td.settlement = SETTLEMENT_CITY
                    td.settlement_name = self._generate_name(rng, "city")
                    td.settlement_size = 4
                elif typ == "town":
                    td.settlement = SETTLEMENT_TOWN
                    td.settlement_name = self._generate_name(rng, "town")
                    td.settlement_size = 3
                else:
                    td.settlement = SETTLEMENT_VILLAGE
                    td.settlement_name = self._generate_name(rng, "village")
                    td.settlement_size = 2

            return placed

        # 首都（最好位置）
        if has_capital and candidates:
            capital = _place("capital", 1, 8)
            settlements.extend(capital)

        # 城市
        cities = _place("city", num_cities, 6)
        settlements.extend(cities)

        # 城镇
        towns = _place("town", num_towns, 4)
        settlements.extend(towns)

        # 村庄
        villages = _place("village", num_villages, 3)
        settlements.extend(villages)

        return settlements

    def _settlement_score(self, td: TerrainData, hc: HexCoord) -> float:
        """评估六边形作为聚落位置的适宜度"""
        score = 0.0

        # 地形偏好
        biome_scores = {
            "plains": 10, "grassland": 9, "savanna": 8,
            "forest": 7, "dense_forest": 5,
            "hills": 6, "taiga": 4,
            "beach": 7, "desert": 2, "tundra": 1,
        }
        score += biome_scores.get(td.biome, 0)

        # 海岸加成
        if td.is_coast:
            score += 5

        # 河流加成
        if td.river_flow > 0:
            score += 4

        # 高程惩罚（太高不好）
        if td.elevation > 0.55:
            score -= (td.elevation - 0.55) * 15

        return score

    def generate_roads(self, rng: np.random.Generator) -> List[Tuple[HexCoord, HexCoord]]:
        """道路生成 — 连接聚落之间的路径"""
        # 收集所有有聚落的六边形
        settlements = [(hc, td) for hc, td in self.terrain.items()
                       if td.settlement != SETTLEMENT_NONE]
        settlements.sort(key=lambda x: -x[1].settlement_size)

        roads = []
        connected: Set[HexCoord] = set()

        if len(settlements) < 2:
            return roads

        # 连接主要聚落形成路网
        for i, (hc, td) in enumerate(settlements):
            if len(connected) == 0:
                connected.add(hc)
                continue

            # 找到到已连接网络的最短路径
            best_target = None
            best_dist = 999
            for ch in connected:
                d = hc.distance_to(ch)
                if d < best_dist and d > 1:
                    best_dist = d
                    best_target = ch

            if best_target:
                path = self._find_road_path(hc, best_target)
                if path:
                    for ph in path:
                        if ph in self.terrain and not self.terrain[ph].is_water:
                            self.terrain[ph].road = True
                    roads.append((hc, best_target))
                connected.add(hc)

        return roads

    def _find_road_path(self, start: HexCoord, end: HexCoord,
                        max_steps: int = 100) -> List[HexCoord]:
        """A* 寻路找到两个聚落之间的道路路径"""
        from heapq import heappush, heappop

        open_set = [(0, 0, start)]
        came_from: Dict[HexCoord, Optional[HexCoord]] = {start: None}
        g_score: Dict[HexCoord, float] = {start: 0.0}

        while open_set:
            _, _, current = heappop(open_set)

            if current == end:
                # 回溯路径
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                    if current is None:
                        break
                path.reverse()
                return path

            if len(g_score) > max_steps:
                break

            for nh in current.neighbors():
                if nh not in self.terrain:
                    continue
                td = self.terrain[nh]
                # 避免水路
                if td.is_water and nh != end:
                    continue
                # 避免高山和火山
                if td.biome in ("high_mountains", "volcano") and nh != end:
                    continue

                # 移动成本
                move_cost = 1.0
                if td.biome in ("mountains", "hills"):
                    move_cost = 2.0
                if td.biome in ("dense_forest", "rainforest", "swamp"):
                    move_cost = 1.5
                if td.road:
                    move_cost = 0.5

                tentative = g_score[current] + move_cost
                if nh not in g_score or tentative < g_score[nh]:
                    g_score[nh] = tentative
                    priority = tentative + nh.distance_to(end) * 1.2
                    heappush(open_set, (priority, id(nh), nh))
                    came_from[nh] = current

        return []

    def generate_resources(self, rng: np.random.Generator,
                           density: float = 0.08) -> None:
        """在地图上分布资源"""
        resource_biomes = {
            RESOURCE_WOOD: ["forest", "dense_forest", "rainforest", "taiga"],
            RESOURCE_IRON: ["mountains", "hills", "high_mountains"],
            RESOURCE_GOLD: ["mountains", "hills", "volcano"],
            RESOURCE_FOOD: ["plains", "grassland", "savanna"],
            RESOURCE_STONE: ["mountains", "hills", "high_mountains", "volcano"],
        }

        for hc, td in self.terrain.items():
            if td.is_water or td.settlement != SETTLEMENT_NONE:
                continue
            if td.volcanic:
                # 火山附近有矿
                if rng.random() < 0.3:
                    td.resource = rng.choice([RESOURCE_IRON, RESOURCE_GOLD, RESOURCE_STONE])
                    td.resource_amount = rng.integers(2, 5)
                continue

            if rng.random() > density:
                continue

            # 根据生物群落分配资源
            for res, biomes in resource_biomes.items():
                if td.biome in biomes and rng.random() < 0.25:
                    td.resource = res
                    td.resource_amount = rng.integers(1, 4)
                    break

    def generate_shipping_routes(self, rng: np.random.Generator) -> List[Tuple[HexCoord, HexCoord]]:
        """航线生成 — 连接沿海聚落"""
        coastal_settlements = [
            hc for hc in self.coast_hexes
            if hc in self.terrain and self.terrain[hc].settlement != SETTLEMENT_NONE
        ]

        if len(coastal_settlements) < 2:
            return []

        routes = []
        used_pairs = set()

        for i, hc1 in enumerate(coastal_settlements):
            for j, hc2 in enumerate(coastal_settlements):
                if i >= j:
                    continue
                pair = (hc1, hc2) if hc1.q < hc2.q or (hc1.q == hc2.q and hc1.r < hc2.r) else (hc2, hc1)
                if pair in used_pairs:
                    continue

                # 距离适中才建航线
                dist = hc1.distance_to(hc2)
                if 5 < dist < 30 and rng.random() < 0.4:
                    used_pairs.add(pair)
                    routes.append((hc1, hc2))

                    # 标记航线经过的六边形
                    self._mark_shipping_path(hc1, hc2)

        return routes

    def _mark_shipping_path(self, start: HexCoord, end: HexCoord):
        """标记航线路径上的水格"""
        # 使用直线插值标记沿途水格
        steps = max(abs(end.q - start.q), abs(end.r - start.r))
        if steps == 0:
            return

        for t in range(1, steps):
            q = start.q + (end.q - start.q) * t / steps
            r = start.r + (end.r - start.r) * t / steps
            # 找最近的水六边形
            near = self._find_nearest_water(HexCoord(round(q), round(r)))
            if near and near in self.terrain:
                self.terrain[near].shipping = True

    def _find_nearest_water(self, hc: HexCoord) -> Optional[HexCoord]:
        """找到最近的六边形（偏好水域）"""
        if hc in self.terrain and self.terrain[hc].is_water:
            return hc
        for d in range(1, 4):
            for nh in hc.range(d):
                if nh in self.terrain and self.terrain[nh].is_water:
                    return nh
        return None

    def _generate_name(self, rng: np.random.Generator, typ: str) -> str:
        """生成随机聚落名称"""
        prefixes = {
            "capital": ["王", "皇", "帝", "龙", "天", "圣", "大"],
            "city": ["风", "云", "星", "月", "铁", "金", "石", "海", "白", "红"],
            "town": ["河", "山", "林", "湖", "溪", "谷", "岚", "雾", "霜"],
            "village": ["小", "新", "老", "北", "南", "东", "西", "上", "下"],
        }
        suffixes = {
            "capital": ["京", "都", "城"],
            "city": ["城", "关", "堡", "镇"],
            "town": ["镇", "集", "市", "渡"],
            "village": ["村", "庄", "屯", "寨", "店"],
        }

        p = rng.choice(prefixes.get(typ, prefixes["village"]))
        s = rng.choice(suffixes.get(typ, suffixes["village"]))
        return f"{p}{s}"