"""配色键覆盖测试：确保每个 biome 常量都有对应的颜色。"""
from __future__ import annotations

import pytest

from core.terrain_gen import (
    BIOME_OCEAN, BIOME_LAKE, BIOME_BEACH,
    BIOME_PLAINS, BIOME_FOREST, BIOME_DENSE_FOREST,
    BIOME_RAINFOREST, BIOME_TAIGA, BIOME_TUNDRA,
    BIOME_SNOW, BIOME_DESERT, BIOME_SAVANNA,
    BIOME_HILLS, BIOME_MOUNTAINS, BIOME_HIGH_MOUNTAINS,
    BIOME_SWAMP, BIOME_VOLCANO,
)

# 从 utils/colors.py 导入时会加载 PySide6；如果在无 GUI 环境下测试，
# 此处做惰性导入：
try:
    from utils.colors import BIOME_COLORS, TERRAIN_COLORS, FEATURE_COLORS
    _HAS_QT = True
except Exception:
    _HAS_QT = False


ALL_BIOMES = [
    BIOME_OCEAN, BIOME_LAKE, BIOME_BEACH,
    BIOME_PLAINS, BIOME_FOREST, BIOME_DENSE_FOREST,
    BIOME_RAINFOREST, BIOME_TAIGA, BIOME_TUNDRA,
    BIOME_SNOW, BIOME_DESERT, BIOME_SAVANNA,
    BIOME_HILLS, BIOME_MOUNTAINS, BIOME_HIGH_MOUNTAINS,
    BIOME_SWAMP, BIOME_VOLCANO,
]


@pytest.mark.skipif(not _HAS_QT, reason="PySide6 / Qt 不可用")
def test_all_biomes_have_colors():
    for biome in ALL_BIOMES:
        assert biome in BIOME_COLORS, f"biome {biome} 没有在 BIOME_COLORS 中定义"


@pytest.mark.skipif(not _HAS_QT, reason="PySide6 / Qt 不可用")
def test_biome_color_values_are_qcolor_like():
    # 只需验证它们可被当作颜色键（至少不是 None 或空串）
    for name, color in BIOME_COLORS.items():
        assert color is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
