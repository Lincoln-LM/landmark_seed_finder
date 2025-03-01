"""32-bit Linear Congruential Pseudo Random Number Generator"""

from __future__ import annotations
from typing import Type, Callable
import enum
import numpy as np
from ..compilation import optional_jitclass, optional_njit
from ..compatability import mod_inv


class LCRNG32RandomDistribution(enum.IntEnum):
    """
    Various LCRNG32 random distribution types

    MODULO -> next_u16() % maximum

    RECIPROCAL_DIVISION -> next_u16() // ((0xFFFF // maximum) + 1)
    """

    NONE = 0
    MODULO = 1
    RECIPROCAL_DIVISION = 2


# TODO: reverse next
# TODO: float rands
class LCRNG32:
    """32-bit LCRNG parent class"""

    seed: np.uint32

    def __init__(self, seed: np.uint32) -> None:
        self.seed: np.uint32 = np.uint32(seed)

    def re_init(self, seed: np.uint32) -> None:
        """Reinitialize without creating a new object"""
        self.seed = np.uint32(seed)

    def next(self) -> np.uint32:
        """
        Generate and return the next full 32-bit random uint

        () -> seed = seed * mult + add
        """
        raise NotImplementedError()

    def jump(self, adv: np.uint32) -> np.uint32:
        """Jump ahead the LCRNG sequence by adv"""
        raise NotImplementedError()

    def advance(self, adv: np.uint32) -> np.uint32:
        """Advance the LCRNG sequence by adv"""
        adv = np.uint32(adv)
        for _ in range(adv):
            self.next()
        return np.uint32(self.seed)

    def next_u16(self) -> np.uint16:
        """Generate and return the next 16-bit random uint"""
        return np.uint16(np.uint32(self.next()) >> np.uint32(16))

    def next_rand(self, maximum: np.uint16) -> np.uint16:
        """
        Generate and return the next [0, maximum) random uint

        () -> distribution(self.next_u16(), maximum)
        """
        raise NotImplementedError()

    @staticmethod
    def const_jump(adv: np.uint32, **kwargs) -> Callable[[LCRNG32]]:
        """Compile @njit jump function with a const adv"""
        raise NotImplementedError()

    @staticmethod
    def const_rand(maximum: np.uint16, **kwargs) -> Callable[[LCRNG32], np.uint16]:
        """Compile @njit rand function with a const maximum"""
        raise NotImplementedError()


def lcrng32_init(
    *,
    add: np.uint32,
    mult: np.uint32,
    distribution: LCRNG32RandomDistribution = LCRNG32RandomDistribution.NONE,
    reverse: bool = False,
) -> Callable[[Type[LCRNG32]], Type[LCRNG32]]:
    """Initialize a LCRNG32 class with constants and random distribution"""
    if reverse:
        mult = np.uint32(mod_inv(mult, 0x100000000))
        add = np.uint32(-np.uint32(add) * mult)
    else:
        mult = np.uint32(mult)
        add = np.uint32(add)

    jump_table = [(np.uint32(add), np.uint32(mult))]
    for i in range(31):
        jump_table.append(
            (
                np.uint32(
                    np.uint32(jump_table[i][0])
                    * np.uint32(np.uint32(jump_table[i][1]) + np.uint32(1))
                ),
                np.uint32(np.uint32(jump_table[i][1]) * np.uint32(jump_table[i][1])),
            )
        )
    jump_table = tuple(jump_table)

    def wrap(lcrng_class: Type[LCRNG32]) -> Type[LCRNG32]:
        def next_(self: LCRNG32) -> np.uint32:
            self.seed = np.uint32(
                np.uint32(self.seed) * np.uint32(mult) + np.uint32(add)
            )
            return np.uint32(self.seed)

        def jump(self: LCRNG32, adv: np.uint32):
            i = 0
            while adv:
                if adv & 1:
                    add, mult = jump_table[i]
                    self.seed = np.uint32(
                        np.uint32(self.seed) * np.uint32(mult) + np.uint32(add)
                    )
                adv >>= 1
                i += 1
            return np.uint32(self.seed)

        def const_jump(adv: np.uint32, **kwargs) -> Callable[[LCRNG32]]:
            i = 0
            mult = np.uint32(1)
            add = np.uint32(0)
            while adv:
                add_val, mult_val = jump_table[i]
                if adv & 1:
                    mult *= mult_val
                    add = add * mult_val + add_val
                adv >>= 1
                i += 1

            @optional_njit(**kwargs)
            def jump_func(self: LCRNG32):
                self.seed = np.uint32(
                    np.uint32(self.seed) * np.uint32(mult) + np.uint32(add)
                )
                return np.uint32(self.seed)

            return jump_func

        if distribution == LCRNG32RandomDistribution.MODULO:

            def next_rand(self: LCRNG32, maximum: np.uint16) -> np.uint16:
                return np.uint16(self.next_u16()) % np.uint16(maximum)

            def const_rand(
                maximum: np.uint16, **kwargs
            ) -> Callable[[LCRNG32], np.uint16]:
                @optional_njit(**kwargs)
                def rand_func(self: LCRNG32) -> np.uint16:
                    return np.uint16(self.next_u16()) % np.uint16(maximum)

                return rand_func

        elif distribution == LCRNG32RandomDistribution.RECIPROCAL_DIVISION:

            def next_rand(self: LCRNG32, maximum: np.uint16) -> np.uint16:
                return np.uint16(self.next_u16()) // np.uint16(
                    (np.uint16(0xFFFF) // np.uint16(maximum)) + np.uint16(1)
                )

            def const_rand(
                maximum: np.uint16, **kwargs
            ) -> Callable[[LCRNG32], np.uint16]:
                @optional_njit(**kwargs)
                def rand_func(self: LCRNG32) -> np.uint16:
                    return np.uint16(self.next_u16()) // np.uint16(
                        (np.uint16(0xFFFF) // np.uint16(maximum)) + np.uint16(1)
                    )

                return rand_func

        else:

            def next_rand(self: LCRNG32, maximum: np.uint16) -> np.uint16:
                raise NotImplementedError()

            def const_rand(
                maximum: np.uint16, **kwargs
            ) -> Callable[[LCRNG32], np.uint16]:
                raise NotImplementedError()

        lcrng_class.next = next_
        lcrng_class.jump = jump
        lcrng_class.next_rand = next_rand

        lcrng_class = optional_jitclass(lcrng_class)

        lcrng_class.const_jump = const_jump
        lcrng_class.const_rand = const_rand

        return lcrng_class

    return wrap
