"""
六边形网格核心库 — 平顶(flat-top)六边形，轴向坐标 (q, r)
"""
from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np


class HexCoord:
    """六边形轴向坐标 (q, r)。s = -q - r"""

    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r

    @property
    def s(self) -> int:
        return -self.q - self.r

    def __eq__(self, other):
        if not isinstance(other, HexCoord):
            return False
        return self.q == other.q and self.r == other.r

    def __hash__(self):
        return hash((self.q, self.r))

    def __repr__(self):
        return f"Hex({self.q},{self.r})"

    def __add__(self, other):
        return HexCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other):
        return HexCoord(self.q - other.q, self.r - other.r)

    def __mul__(self, k: int):
        return HexCoord(self.q * k, self.r * k)

    def distance_to(self, other: HexCoord) -> int:
        """两个六边形之间的步数距离"""
        dq = abs(self.q - other.q)
        dr = abs(self.r - other.r)
        ds = abs(self.s - other.s)
        return max(dq, dr, ds)

    def neighbors(self) -> List[HexCoord]:
        """返回6个邻居"""
        return [self + d for d in HEX_DIRECTIONS]

    def range(self, n: int) -> List[HexCoord]:
        """返回距离 ≤ n 的所有六边形"""
        results = []
        for dq in range(-n, n + 1):
            for dr in range(max(-n, -dq - n), min(n, -dq + n) + 1):
                results.append(HexCoord(self.q + dq, self.r + dr))
        return results

    def to_cube(self) -> Tuple[int, int, int]:
        return (self.q, self.r, self.s)

    @staticmethod
    def from_cube(q: int, r: int, s: int) -> HexCoord:
        if q + r + s != 0:
            raise ValueError(f"Invalid cube coordinates: ({q}, {r}, {s}) does not sum to 0")
        return HexCoord(q, r)


# 平顶六边形的6个方向 (轴向坐标)
HEX_DIRECTIONS = [
    HexCoord(1, 0), HexCoord(1, -1), HexCoord(0, -1),
    HexCoord(-1, 0), HexCoord(-1, 1), HexCoord(0, 1),
]


class HexGrid:
    """平顶六边形网格系统"""

    def __init__(self, size: int = 60):
        """
        Args:
            size: 六边形网格的半径（从中心到边缘的步数）
        """
        self.size = size
        self.hexes: List[HexCoord] = []
        self._generate_grid()

    def _generate_grid(self):
        """生成以原点为中心的六边形网格"""
        self.hexes = []
        for q in range(-self.size, self.size + 1):
            r1 = max(-self.size, -q - self.size)
            r2 = min(self.size, -q + self.size)
            for r in range(r1, r2 + 1):
                self.hexes.append(HexCoord(q, r))

    def get_random_hex(self, rng: np.random.Generator) -> HexCoord:
        """获取网格内的随机六边形"""
        idx = rng.integers(0, len(self.hexes))
        return self.hexes[idx]

    def hex_at_pixel(self, px: float, py: float, hex_size: float) -> HexCoord:
        """将像素坐标转换为六边形轴向坐标（平顶）"""
        q = (2.0 / 3.0 * px) / hex_size
        r = (-1.0 / 3.0 * px + math.sqrt(3.0) / 3.0 * py) / hex_size
        return self._hex_round(q, r)

    def _hex_round(self, q_frac: float, r_frac: float) -> HexCoord:
        """将浮点坐标四舍五入到最近的六边形"""
        s_frac = -q_frac - r_frac
        q_int = round(q_frac)
        r_int = round(r_frac)
        s_int = round(s_frac)

        q_diff = abs(q_int - q_frac)
        r_diff = abs(r_int - r_frac)
        s_diff = abs(s_int - s_frac)

        if q_diff > r_diff and q_diff > s_diff:
            q_int = -r_int - s_int
        elif r_diff > s_diff:
            r_int = -q_int - s_int

        return HexCoord(q_int, r_int)

    def hex_center(self, h: HexCoord, hex_size: float) -> Tuple[float, float]:
        """返回六边形中心像素坐标（平顶）"""
        x = hex_size * (3.0 / 2.0 * h.q)
        y = hex_size * (math.sqrt(3.0) / 2.0 * h.q + math.sqrt(3.0) * h.r)
        return (x, y)

    def hex_corners(self, h: HexCoord, hex_size: float) -> List[Tuple[float, float]]:
        """返回六边形的6个顶点坐标（平顶）"""
        cx, cy = self.hex_center(h, hex_size)
        corners = []
        for i in range(6):
            angle_deg = 60.0 * i
            angle_rad = math.radians(angle_deg)
            corners.append((
                cx + hex_size * math.cos(angle_rad),
                cy + hex_size * math.sin(angle_rad),
            ))
        return corners

    def all_hex_centers(self, hex_size: float) -> List[Tuple[HexCoord, float, float]]:
        """返回所有六边形的中心坐标"""
        result = []
        for h in self.hexes:
            cx, cy = self.hex_center(h, hex_size)
            result.append((h, cx, cy))
        return result

    def get_hexes_within_radius(self, center: HexCoord, radius: int) -> List[HexCoord]:
        return center.range(radius)