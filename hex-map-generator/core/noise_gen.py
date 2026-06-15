"""
噪声生成器 — 使用 Perlin 噪声算法生成高程、湿度、温度等地图数据
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


class PerlinNoise:
    """
    简易 Perlin 噪声实现（梯度噪声）
    不需要额外依赖，用 n 个八度的噪声叠加
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))
        # 预生成梯度表
        self._grad_table = self._build_grad_table()

    def _build_grad_table(self) -> np.ndarray:
        """生成随机梯度向量表"""
        rng = self.rng
        # 生成 256 个随机梯度向量
        angles = rng.uniform(0, 2 * math.pi, 256)
        grads = np.column_stack([np.cos(angles), np.sin(angles)])
        return grads

    def _hash(self, x: int, y: int) -> int:
        """简单的哈希函数"""
        h = (x * 374761393 + y * 668265263 + self.seed) & 0x7FFFFFFF
        h = (h ^ (h >> 13)) * 1274126177
        h = h ^ (h >> 16)
        return h % 256

    def _dot_grad(self, hash_val: int, dx: float, dy: float) -> float:
        """梯度点积"""
        grad = self._grad_table[hash_val]
        return grad[0] * dx + grad[1] * dy

    def _smoothstep(self, t: float) -> float:
        """平滑插值 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def noise2d(self, x: float, y: float) -> float:
        """返回 [-1, 1] 范围的 2D 噪声值"""
        # 整数格点
        xi = math.floor(x)
        yi = math.floor(y)
        # 局部坐标
        xf = x - xi
        yf = y - yi

        # 四个角落的哈希
        h00 = self._hash(xi, yi)
        h10 = self._hash(xi + 1, yi)
        h01 = self._hash(xi, yi + 1)
        h11 = self._hash(xi + 1, yi + 1)

        # 平滑插值
        sx = self._smoothstep(xf)
        sy = self._smoothstep(yf)

        # 插值
        n0 = self._dot_grad(h00, xf, yf)
        n1 = self._dot_grad(h10, xf - 1.0, yf)
        nx0 = self._lerp(n0, n1, sx)

        n0 = self._dot_grad(h01, xf, yf - 1.0)
        n1 = self._dot_grad(h11, xf - 1.0, yf - 1.0)
        nx1 = self._lerp(n0, n1, sx)

        return self._lerp(nx0, nx1, sy)

    def octave_noise(
        self,
        x: float,
        y: float,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 1.0,
    ) -> float:
        """多八度分形噪声"""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            value += amplitude * self.noise2d(x * frequency / scale, y * frequency / scale)
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return value / max_value


class NoiseGenerator:
    """地图噪声数据生成器"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.perlin = PerlinNoise(seed)

    def set_seed(self, seed: int):
        self.seed = seed
        self.perlin = PerlinNoise(seed)

    def generate_elevation(
        self, hex_coords: List[Tuple[float, float]], scale: float = 3.0, octaves: int = 6
    ) -> np.ndarray:
        """
        生成高程数据

        Args:
            hex_coords: [(x, y), ...] 六边形坐标列表
            scale: 噪声缩放
            octaves: 八度数

        Returns:
            elevation: (N,) 0~1 的高程值
        """
        n = len(hex_coords)
        elevation = np.zeros(n)

        for i, (x, y) in enumerate(hex_coords):
            val = self.perlin.octave_noise(x, y, octaves=octaves, scale=scale)
            # 映射到 0~1
            elevation[i] = (val + 1.0) * 0.5

        # 标准化
        e_min, e_max = elevation.min(), elevation.max()
        if e_max - e_min > 0.001:
            elevation = (elevation - e_min) / (e_max - e_min)

        return elevation

    def generate_moisture(
        self,
        hex_coords: List[Tuple[float, float]],
        elevation: np.ndarray,
        scale: float = 4.0,
        octaves: int = 4,
        monsoon_dir: Optional[float] = 90.0,
    ) -> np.ndarray:
        """
        生成湿度数据，受高程和季风影响

        Args:
            monsoon_dir: 季风方向角度 (度)，0=北, 90=东，None=无季风

        Returns:
            moisture: (N,) 0~1 的湿度值
        """
        n = len(hex_coords)
        moisture = np.zeros(n)

        if monsoon_dir is not None:
            monsoon_rad = math.radians(monsoon_dir)
            monsoon_dx = math.sin(monsoon_rad)
            monsoon_dy = -math.cos(monsoon_rad)

        for i, (x, y) in enumerate(hex_coords):
            val = self.perlin.octave_noise(x, y, octaves=octaves, scale=scale)
            m = (val + 1.0) * 0.5

            # 季风效应：迎风坡湿润，背风坡干燥
            if monsoon_dir is not None:
                wind_effect = (x * monsoon_dx + y * monsoon_dy) * 0.03
                m += wind_effect

            # 高程效应：山地阻挡形成雨影
            if elevation[i] > 0.5:
                m -= (elevation[i] - 0.5) * 0.3

            moisture[i] = np.clip(m, 0.0, 1.0)

        # 再次标准化
        m_min, m_max = moisture.min(), moisture.max()
        if m_max - m_min > 0.001:
            moisture = (moisture - m_min) / (m_max - m_min)

        return moisture

    def generate_temperature(
        self, elevation: np.ndarray, hex_coords: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        生成温度数据
        受纬度（y坐标）和高程影响
        """
        n = len(hex_coords)
        temperature = np.zeros(n)

        # 找出y范围
        ys = [y for _, y in hex_coords]
        y_min, y_max = min(ys), max(ys)

        for i, (x, y) in enumerate(hex_coords):
            # 纬度效应：y越大越冷
            lat_factor = (y - y_min) / (y_max - y_min + 0.001)
            temp = 1.0 - lat_factor

            # 高程效应：每升高0.1降温
            temp -= elevation[i] * 0.3

            temperature[i] = np.clip(temp, 0.0, 1.0)

        return temperature
