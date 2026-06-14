"""六边形网格核心库的单元测试

覆盖：
- HexCoord 的代数性质（加法、减法、标量乘法、距离、邻居、range、cube 转换）
- HexGrid 生成的六边形数量（半径公式）
- hex_center / hex_corners 的几何不变量
- _hex_round / hex_at_pixel 的往返一致性（pixel -> hex -> pixel 接近原点）
- get_random_hex 返回网格内的六边形
"""
from __future__ import annotations
import math

import numpy as np
import pytest

from core.hex_grid import HexCoord, HexGrid, HEX_DIRECTIONS


# ---------------------------------------------------------------------------
# HexCoord 代数
# ---------------------------------------------------------------------------


class TestHexCoord:
    def test_eq_and_hash(self):
        a = HexCoord(3, -2)
        b = HexCoord(3, -2)
        c = HexCoord(3, -1)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        # 与非 HexCoord 比较
        assert (a == "(3,-2)") is False

    def test_add_sub_mul(self):
        a = HexCoord(2, 1)
        b = HexCoord(-1, 3)
        assert a + b == HexCoord(1, 4)
        assert a - b == HexCoord(3, -2)
        assert a * 3 == HexCoord(6, 3)

    def test_s_coord_is_minus_q_minus_r(self):
        for q, r in [(-5, 3), (0, 0), (7, -2), (10, 10)]:
            h = HexCoord(q, r)
            assert h.q + h.r + h.s == 0

    def test_distance_to_self_is_zero(self):
        h = HexCoord(5, -3)
        assert h.distance_to(h) == 0

    def test_distance_to_adjacent_is_one(self):
        origin = HexCoord(0, 0)
        for d in HEX_DIRECTIONS:
            assert origin.distance_to(origin + d) == 1

    def test_distance_is_symmetric(self):
        a = HexCoord(7, -2)
        b = HexCoord(-3, 4)
        assert a.distance_to(b) == b.distance_to(a)

    def test_distance_triangle_inequality(self):
        a = HexCoord(1, 0)
        b = HexCoord(4, -2)
        c = HexCoord(-2, 3)
        assert a.distance_to(c) <= a.distance_to(b) + b.distance_to(c)

    def test_neighbors_has_six_distinct_coords(self):
        origin = HexCoord(2, 3)
        ns = origin.neighbors()
        assert len(ns) == 6
        assert len({(n.q, n.r) for n in ns}) == 6
        for n in ns:
            assert origin.distance_to(n) == 1

    def test_range_size_matches_formula(self):
        for n in [0, 1, 3, 7]:
            cells = HexCoord(0, 0).range(n)
            # 半径 n 的六边形格数 = 1 + 3n(n+1)
            assert len(cells) == 1 + 3 * n * (n + 1)
            for c in cells:
                assert HexCoord(0, 0).distance_to(c) <= n

    def test_range_is_symmetric_under_translation(self):
        base = HexCoord(-4, 2)
        cells_a = base.range(2)
        cells_b = [HexCoord(-4, 2) + d for d in HexCoord(0, 0).range(2)]
        assert {(c.q, c.r) for c in cells_a} == {(c.q, c.r) for c in cells_b}

    def test_cube_roundtrip(self):
        h = HexCoord(-3, 5)
        q, r, s = h.to_cube()
        assert q == h.q and r == h.r and s == h.s
        assert HexCoord.from_cube(q, r, s) == h


# ---------------------------------------------------------------------------
# HexGrid
# ---------------------------------------------------------------------------


class TestHexGrid:
    def test_grid_size_matches_formula(self):
        for size in [1, 5, 10]:
            grid = HexGrid(size=size)
            # 半径 size 的六边形总数 = 1 + 3*size*(size+1)
            assert len(grid.hexes) == 1 + 3 * size * (size + 1)

    def test_all_cells_within_radius(self):
        grid = HexGrid(size=8)
        origin = HexCoord(0, 0)
        for h in grid.hexes:
            assert origin.distance_to(h) <= 8

    def test_no_duplicate_cells(self):
        grid = HexGrid(size=6)
        assert len({(h.q, h.r) for h in grid.hexes}) == len(grid.hexes)

    def test_get_random_hex_is_in_grid(self):
        grid = HexGrid(size=5)
        rng = np.random.default_rng(0)
        for _ in range(50):
            h = grid.get_random_hex(rng)
            assert any(h == g for g in grid.hexes)

    def test_hex_center_and_corners_geometry(self):
        grid = HexGrid(size=2)
        hex_size = 12.0
        for h in grid.hexes[:20]:
            cx, cy = grid.hex_center(h, hex_size)
            corners = grid.hex_corners(h, hex_size)
            assert len(corners) == 6
            # 所有角到中心距离应近似等于 hex_size
            for (px, py) in corners:
                dist = math.hypot(px - cx, py - cy)
                assert abs(dist - hex_size) < 1e-9

    def test_hex_round_returns_known_coord(self):
        grid = HexGrid(size=5)
        hex_size = 10.0
        # 直接给出中心像素坐标 -> 应该精确返回自身
        for h in grid.hexes[:30]:
            cx, cy = grid.hex_center(h, hex_size)
            got = grid.hex_at_pixel(cx, cy, hex_size)
            assert got == h

    def test_all_hex_centers_list(self):
        grid = HexGrid(size=4)
        centers = grid.all_hex_centers(hex_size=8.0)
        assert len(centers) == len(grid.hexes)

    def test_get_hexes_within_radius_matches_range(self):
        grid = HexGrid(size=5)
        center = HexCoord(1, -1)
        cells = grid.get_hexes_within_radius(center, 2)
        assert len(cells) == 1 + 3 * 2 * 3
        for c in cells:
            assert center.distance_to(c) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
