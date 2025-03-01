"""Mersenne Twister 19937 Pseudo Random Number Generator"""

from __future__ import annotations
import numpy as np
from ..compilation import optional_jitclass, array_type

MAG02 = (np.uint32(0), np.uint32(0x9908B0DF))


# TODO: jump tables
# TODO: staticmethod const functions
# TODO: init_by_array
# TODO: reverse next
@optional_jitclass
class MersenneTwister:
    """Mersenne Twister Pseudo Random Number Generator"""

    state: array_type(np.uint32)  # contiguous array
    index: np.uint16

    def __init__(self, seed: np.uint32) -> None:
        seed = np.uint32(seed)
        self.state = np.empty(624, dtype=np.uint32)
        self.index = np.uint16(624)  # ensures shuffle after initialization
        self.re_init(seed)

    def re_init(self, seed: np.uint32) -> None:
        """Reinitialize without creating a new object"""
        seed = np.uint32(seed)
        self.index = np.uint16(624)  # ensures shuffle after initialization
        self.state[0] = seed

        for i in range(1, 624):
            seed = np.uint32(
                np.uint32(0x6C078965) * (seed ^ (seed >> np.uint32(30))) + np.uint32(i)
            )
            self.state[i] = seed

    def advance(self, adv: np.uint32) -> None:
        """Advance Mersenne Twister sequence by adv"""
        adv = np.uint32(adv)
        adv += np.uint32(self.index)
        while adv >= np.uint32(624):
            self.shuffle()
            adv -= np.uint32(624)
        self.index = np.uint16(adv)

    def next(self) -> np.uint32:
        """Access and return the next 32-bit tempered rand"""
        if self.index == np.uint16(624):
            self.shuffle()

        y_rand = self.state[self.index]
        self.index += np.uint16(1)

        y_rand ^= y_rand >> np.uint32(11)
        y_rand ^= (y_rand << np.uint32(7)) & np.uint32(0x9D2C5680)
        y_rand ^= (y_rand << np.uint32(15)) & np.uint32(0xEFC60000)
        y_rand ^= y_rand >> np.uint32(18)

        return y_rand

    def shuffle(self) -> None:
        """Advance and shuffle the entire state (624 advances)"""
        state = self.state  # accessing self.state directly fails to vectorize
        upper_mask = np.uint32(0x80000000)
        lower_mask = np.uint32(0x7FFFFFFF)
        one = np.uint32(1)

        for i in range(227):
            y_val = (state[i] & upper_mask) | (state[i + 1] & lower_mask)
            state[i] = (y_val >> one) ^ MAG02[y_val & one] ^ state[i + 397]

        for i in range(227, 623):
            y_val = (state[i] & upper_mask) | (state[i + 1] & lower_mask)
            state[i] = (y_val >> one) ^ MAG02[y_val & one] ^ state[i - 227]

        y_val = (state[623] & upper_mask) | (state[0] & lower_mask)
        state[623] = (y_val >> one) ^ MAG02[y_val & one] ^ state[396]

        self.index = np.uint16(0)

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint
        via multiplication-shift distribution"""
        return np.uint32((np.uint64(self.next()) * np.uint64(maximum)) >> np.uint64(32))

    def next_rand_mod(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint via modulo distribution"""
        return self.next() % np.uint32(maximum)
