"""Xoroshiro128+ Pseudo Random Number Generator"""

from __future__ import annotations
import numpy as np
from ..util import rotate_left_u64, rotate_right_u64
from ..compilation import optional_jitclass, array_type


# TODO: jump tables
# TODO: staticmethod const functions
class Xoroshiro128Plus:
    """Xoroshiro128+ Pseudo Random Number Generator Parent Class"""

    state: array_type(np.uint64)  # contiguous array

    def __init__(self) -> None:
        raise NotImplementedError()

    def re_init(self) -> None:
        """Reinitialize without creating a new object"""
        raise NotImplementedError()

    def advance(self, adv: np.uint64) -> None:
        """Advance Xoroshiro128+ sequence by adv"""
        for _ in range(adv):
            self.next()

    def next(self) -> np.uint64 | np.uint32:
        """Advance and return the next random uint"""
        raise NotImplementedError()

    def previous(self) -> np.uint64 | np.uint32:
        """Advance backwards and return the previos random uint"""
        raise NotImplementedError()

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        """Generate and return the next [0, maximum) random uint"""
        raise NotImplementedError()


@optional_jitclass
class Xoroshiro128PlusRejection(Xoroshiro128Plus):
    """Xoroshiro128+ Pseudo Random Number Generator w/rejection sampling"""

    # pylint: disable=arguments-differ, super-init-not-called
    def __init__(
        self, seed_0: np.uint64, seed_1: np.uint64 = np.uint64(0x82A2B175229D6A5B)
    ) -> None:
        self.state = np.empty(2, np.uint64)
        self.re_init(np.uint64(seed_0), np.uint64(seed_1))

    def re_init(
        self, seed_0: np.uint64, seed_1: np.uint64 = np.uint64(0x82A2B175229D6A5B)
    ) -> None:
        self.state[0] = np.uint64(seed_0)
        self.state[1] = np.uint64(seed_1)

    # pylint: enable=arguments-differ, super-init-not-called

    def next(self) -> np.uint64:
        seed_0 = self.state[0]
        seed_1 = self.state[1]
        result = np.uint64(seed_0 + seed_1)

        seed_1 ^= seed_0
        self.state[0] = rotate_left_u64(seed_0, 24) ^ seed_1 ^ (seed_1 << np.uint64(16))
        self.state[1] = rotate_left_u64(seed_1, 37)

        return np.uint64(result)

    def previous(self) -> np.uint64:
        seed_1 = rotate_right_u64(self.state[1], 37)
        seed_0 = rotate_right_u64(
            self.state[0] ^ seed_1 ^ (seed_1 << np.uint64(16)), 24
        )
        seed_1 ^= seed_0

        self.state[0] = seed_0
        self.state[1] = seed_1

        return np.uint64(seed_0 + seed_1)

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        mask = self.bit_mask(maximum)
        result = self.next() & mask
        while result >= maximum:
            result = self.next() & mask
        return result

    @staticmethod
    def bit_mask(val: np.uint32) -> np.uint32:
        """Create a bitmask that includes only up to the MSB of val"""
        val = np.uint32(val)
        val -= np.uint32(1)
        val |= val >> np.uint32(1)
        val |= val >> np.uint32(2)
        val |= val >> np.uint32(4)
        val |= val >> np.uint32(8)
        val |= val >> np.uint32(16)
        return val


@optional_jitclass
class SplitMixXoroshiro128Plus(Xoroshiro128Plus):
    """Xoroshiro128+ Pseudo Random Number Generator w/splitmix initialization"""

    # pylint: disable=arguments-differ, super-init-not-called
    def __init__(self, seed: np.uint64) -> None:
        self.state = np.empty(2, np.uint64)
        self.re_init(np.uint64(seed))

    def re_init(self, seed: np.uint64) -> None:
        seed = np.uint64(seed)
        self.state[0] = self.splitmix(seed, np.uint64(0x9E3779B97F4A7C15))
        # 0x9E3779B97F4A7C15 + 0x9E3779B97F4A7C15 = 0x3C6EF372FE94F82A
        self.state[1] = self.splitmix(seed, np.uint64(0x3C6EF372FE94F82A))

    # pylint: enable=arguments-differ, super-init-not-called

    @staticmethod
    def splitmix(seed: np.uint64, state: np.uint64) -> np.uint64:
        """Splitmix initialization function"""
        seed = np.uint64(seed)
        state = np.uint64(state)

        seed += state
        seed = np.uint64(0xBF58476D1CE4E5B9) * (seed ^ (seed >> np.uint64(30)))
        seed = np.uint64(0x94D049BB133111EB) * (seed ^ (seed >> np.uint64(27)))

        return np.uint64(seed ^ (seed >> np.uint64(31)))

    def next(self) -> np.uint32:
        seed_0 = self.state[0]
        seed_1 = self.state[1]
        result = np.uint64(seed_0 + seed_1)

        seed_1 ^= seed_0
        self.state[0] = rotate_left_u64(seed_0, 24) ^ seed_1 ^ (seed_1 << np.uint64(16))
        self.state[1] = rotate_left_u64(seed_1, 37)

        return np.uint32(result >> np.uint64(32))

    def previous(self) -> np.uint32:
        seed_1 = rotate_right_u64(self.state[1], 37)
        seed_0 = rotate_right_u64(
            self.state[0] ^ seed_1 ^ (seed_1 << np.uint64(16)), 24
        )
        seed_1 ^= seed_0

        self.state[0] = seed_0
        self.state[1] = seed_1

        return np.uint32(np.uint64(seed_0 + seed_1) >> np.uint64(32))

    def next_rand(self, maximum: np.uint32) -> np.uint32:
        return self.next() % maximum
