"""Pokemon game version enum"""

from enum import IntEnum


class Game(IntEnum):
    """Pokemon game version enum"""

    NONE = 0
    RUBY = 1 << 0
    SAPPHIRE = 1 << 1
    EMERALD = 1 << 2
    FIRE_RED = 1 << 3
    LEAF_GREEN = 1 << 4
    GALES = 1 << 5
    COLOSSEUM = 1 << 6
    DIAMOND = 1 << 9
    PEARL = 1 << 8
    PLATINUM = 1 << 9
    HEART_GOLD = 1 << 10
    SOUL_SILVER = 1 << 11
    BLACK = 1 << 12
    WHITE = 1 << 13
    BLACK2 = 1 << 14
    WHITE2 = 1 << 15
    X = 1 << 16
    Y = 1 << 17
    OMEGA_RUBY = 1 << 18
    ALPHA_SAPPHIRE = 1 << 19
    SUN = 1 << 20
    MOON = 1 << 21
    ULTRA_SUN = 1 << 22
    ULTRA_MOON = 1 << 23
    SWORD = 1 << 24
    SHIELD = 1 << 25
    BRILLIANT_DIAMOND = 1 << 26
    SHINING_PEARL = 1 << 27
    LEGENDS_ARCEUS = 1 << 28
