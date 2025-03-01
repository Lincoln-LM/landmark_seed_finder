"""SIMD-oriented Fast Mersenne Twister 19937 Pseudo Random Number Generator"""

from __future__ import annotations
import numpy as np
from ..compilation import (
    optional_jitclass,
    optional_ir_function,
    array_type,
)
from ..options import USE_NUMBA


def sfmt_shuffle(state: np.ndarray):
    """SFMT shuffling function"""
    # the contents of this function are only ever used when USE_NUMBA is set to False
    # otherwise, it is overloaded with the ir version

    # compute the first two states manually

    x_0 = state[0] << 8
    x_1 = (state[1] << 8) | (state[0] >> 24)
    x_2 = (state[2] << 8) | (state[1] >> 24)
    x_3 = (state[3] << 8) | (state[2] >> 24)

    y_0 = (state[617] << 24) | (state[616] >> 8)
    y_1 = (state[618] << 24) | (state[617] >> 8)
    y_2 = (state[619] << 24) | (state[618] >> 8)
    y_3 = state[619] >> 8

    b_0 = (state[488] >> 11) & 0xDFFFFFEF
    b_1 = (state[489] >> 11) & 0xDDFECB7F
    b_2 = (state[490] >> 11) & 0xBFFAFFFF
    b_3 = (state[491] >> 11) & 0xBFFFFFF6

    d_0 = state[620] << 18
    d_1 = state[621] << 18
    d_2 = state[622] << 18
    d_3 = state[623] << 18

    state[0] ^= x_0 ^ y_0 ^ b_0 ^ d_0
    state[1] ^= x_1 ^ y_1 ^ b_1 ^ d_1
    state[2] ^= x_2 ^ y_2 ^ b_2 ^ d_2
    state[3] ^= x_3 ^ y_3 ^ b_3 ^ d_3

    x_0 = state[4] << 8
    x_1 = (state[5] << 8) | (state[4] >> 24)
    x_2 = (state[6] << 8) | (state[5] >> 24)
    x_3 = (state[7] << 8) | (state[6] >> 24)

    y_0 = (state[621] << 24) | (state[620] >> 8)
    y_1 = (state[622] << 24) | (state[621] >> 8)
    y_2 = (state[623] << 24) | (state[622] >> 8)
    y_3 = state[623] >> 8

    b_0 = (state[492] >> 11) & 0xDFFFFFEF
    b_1 = (state[493] >> 11) & 0xDDFECB7F
    b_2 = (state[494] >> 11) & 0xBFFAFFFF
    b_3 = (state[495] >> 11) & 0xBFFFFFF6

    d_0 = state[0] << 18
    d_1 = state[1] << 18
    d_2 = state[2] << 18
    d_3 = state[3] << 18

    state[4] ^= x_0 ^ y_0 ^ b_0 ^ d_0
    state[5] ^= x_1 ^ y_1 ^ b_1 ^ d_1
    state[6] ^= x_2 ^ y_2 ^ b_2 ^ d_2
    state[7] ^= x_3 ^ y_3 ^ b_3 ^ d_3

    for i in range(8, 136, 4):
        x_0 = state[i] << 8
        x_1 = (state[i + 1] << 8) | (state[i] >> 24)
        x_2 = (state[i + 2] << 8) | (state[i + 1] >> 24)
        x_3 = (state[i + 3] << 8) | (state[i + 2] >> 24)

        b_0 = (state[i + 488] >> 11) & 0xDFFFFFEF
        b_1 = (state[i + 489] >> 11) & 0xDDFECB7F
        b_2 = (state[i + 490] >> 11) & 0xBFFAFFFF
        b_3 = (state[i + 491] >> 11) & 0xBFFFFFF6

        d_0 = state[i - 4] << 18
        d_1 = state[i - 3] << 18
        d_2 = state[i - 2] << 18
        d_3 = state[i - 1] << 18

        y_0 = (state[i - 7] << 24) | (state[i - 8] >> 8)
        y_1 = (state[i - 6] << 24) | (state[i - 7] >> 8)
        y_2 = (state[i - 5] << 24) | (state[i - 6] >> 8)
        y_3 = state[i - 5] >> 8

        state[i] ^= x_0 ^ b_0 ^ d_0 ^ y_0
        state[i + 1] ^= x_1 ^ b_1 ^ d_1 ^ y_1
        state[i + 2] ^= x_2 ^ b_2 ^ d_2 ^ y_2
        state[i + 3] ^= x_3 ^ b_3 ^ d_3 ^ y_3

    for i in range(136, 624, 4):
        x_0 = state[i] << 8
        x_1 = (state[i + 1] << 8) | (state[i] >> 24)
        x_2 = (state[i + 2] << 8) | (state[i + 1] >> 24)
        x_3 = (state[i + 3] << 8) | (state[i + 2] >> 24)

        b_0 = (state[i - 136] >> 11) & 0xDFFFFFEF
        b_1 = (state[i - 135] >> 11) & 0xDDFECB7F
        b_2 = (state[i - 134] >> 11) & 0xBFFAFFFF
        b_3 = (state[i - 133] >> 11) & 0xBFFFFFF6

        d_0 = state[i - 4] << 18
        d_1 = state[i - 3] << 18
        d_2 = state[i - 2] << 18
        d_3 = state[i - 1] << 18

        y_0 = (state[i - 7] << 24) | (state[i - 8] >> 8)
        y_1 = (state[i - 6] << 24) | (state[i - 7] >> 8)
        y_2 = (state[i - 5] << 24) | (state[i - 6] >> 8)
        y_3 = state[i - 5] >> 8

        state[i] ^= x_0 ^ b_0 ^ d_0 ^ y_0
        state[i + 1] ^= x_1 ^ b_1 ^ d_1 ^ y_1
        state[i + 2] ^= x_2 ^ b_2 ^ d_2 ^ y_2
        state[i + 3] ^= x_3 ^ b_3 ^ d_3 ^ y_3


if USE_NUMBA:
    import numba
    from numba.core.imputils import impl_ret_untracked
    from llvmlite import ir

    I8 = ir.IntType(8)
    I32 = ir.IntType(32)
    I64 = ir.IntType(64)
    I64_ZERO = ir.Constant(I64, 0)
    I64_4 = ir.Constant(I64, 4)
    VI32X4 = ir.VectorType(I32, 4)
    VI32X4_18 = ir.Constant(VI32X4, 18)
    VI32X4_11 = ir.Constant(VI32X4, 11)
    VI32X16 = ir.VectorType(I32, 16)
    VI8X16 = ir.VectorType(I8, 16)
    VI8X16_ZERO = ir.Constant(VI8X16, None)
    MASK = ir.Constant(VI32X4, (0xDFFFFFEF, 0xDDFECB7F, 0xBFFAFFFF, 0xBFFFFFF6))
    SHIFT_LEFT_MASK = ir.Constant(
        VI32X16, [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    )
    SHIFT_RIGHT_MASK = ir.Constant(
        VI32X16, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    )

    @optional_ir_function(sfmt_shuffle, numba.void(numba.uint32[::1]))
    def sfmt_shuffle_ir(context, builder: ir.IRBuilder, signature, arguments):
        """LLVM IR implementation of sfmt_shuffle"""
        # sfmt_shuffle_ir assumes that arguments[0] is a (624,) np.array of dtype np.uint32
        # it does not do any bounds checking or verifying the shape/size of the array or integers

        # state_obj is a reference to an np.array object
        # np.array objects are stored as follows:
        # {
        #     meminfo: i8*,
        #     parent: i8*,
        #     nitems: i64,
        #     itemsize: i64,
        #     data: dtype*,
        #     shape: [ndim x i64],
        #     strides: [ndim x i64]
        # }
        (state_obj,) = arguments
        # the actual pointer to the data of an np.array is stored at index 4
        state_array_pointer = builder.extract_value(state_obj, 4)

        # store index in a pointer to avoid needing to rewrite variable in loop
        # TODO: overwriting variable is probably faster than accessing memory over and over
        # but it wasn't obvious how to do this
        index_ptr = builder.alloca(I64)
        builder.store(I64_ZERO, index_ptr)

        # allocated memory for temporarily storing a <i32 x 4>/<i8 x 16> vector
        temp_vi32x4 = builder.alloca(VI32X4)
        temp_vi8x16 = builder.bitcast(temp_vi32x4, VI8X16.as_pointer())

        # equivalent to _mm_slli_si128(vec, 1)/_mm_srli_si128(vec, 1) depending on mask
        def shift_bytes(vec_ptr, mask):
            vec_ptr_vi8x16 = builder.bitcast(vec_ptr, VI8X16.as_pointer())
            vec_vi8x16 = builder.shuffle_vector(
                builder.load(vec_ptr_vi8x16), VI8X16_ZERO, mask
            )
            builder.store(vec_vi8x16, temp_vi8x16)
            return builder.load(temp_vi32x4)

        def mm_recursion(ofs):
            index = builder.load(index_ptr)
            index_b = builder.add(index, ir.Constant(I64, ofs[0]))
            index_c = builder.add(index, ir.Constant(I64, ofs[1]))
            index_d = builder.add(index, ir.Constant(I64, ofs[2]))

            vec_a_ptr = builder.bitcast(
                builder.gep(state_array_pointer, (index,)), VI32X4.as_pointer()
            )
            vec_b_ptr = builder.bitcast(
                builder.gep(state_array_pointer, (index_b,)), VI32X4.as_pointer()
            )
            vec_c_ptr = builder.bitcast(
                builder.gep(state_array_pointer, (index_c,)), VI32X4.as_pointer()
            )
            vec_d_ptr = builder.bitcast(
                builder.gep(state_array_pointer, (index_d,)), VI32X4.as_pointer()
            )

            vec_a = builder.load(vec_a_ptr)
            vec_b = builder.load(vec_b_ptr)
            vec_d = builder.load(vec_d_ptr)

            vec_x = shift_bytes(vec_a_ptr, SHIFT_LEFT_MASK)
            vec_y = shift_bytes(vec_c_ptr, SHIFT_RIGHT_MASK)

            vec_b1 = builder.and_(builder.lshr(vec_b, VI32X4_11), MASK)
            vec_d1 = builder.shl(vec_d, VI32X4_18)

            vec_a = builder.xor(
                builder.xor(builder.xor(builder.xor(vec_a, vec_x), vec_b1), vec_y),
                vec_d1,
            )

            builder.store(vec_a, vec_a_ptr)

            index = builder.add(index, I64_4)
            builder.store(index, index_ptr)

            return index

        def loop_mm_recursion(ofs, last_index, block=None):
            # first block needs to be created and branched to
            if block is None:
                block = builder.append_basic_block()
                builder.branch(block)
            # create next block to allow jumping to it
            next_block = builder.append_basic_block()
            with builder.goto_block(block):
                # do mm_recursion and check current index
                index = mm_recursion(ofs)
                # if index == last_index: jump to next_block else: jump back to the start of loop
                builder.cbranch(
                    builder.icmp_unsigned("==", index, ir.Constant(I64, last_index)),
                    next_block,
                    block,
                )
            # return the next block
            return next_block

        # compute first two states manually
        mm_recursion((488, 616, 620))
        mm_recursion((488, 616, -4))
        # split the loop to avoid modulo
        next_block = loop_mm_recursion((488, -8, -4), 136)
        final_block = loop_mm_recursion((-136, -8, -4), 624, block=next_block)

        builder.position_at_end(final_block)

        # return void as the array is changed in-place
        return impl_ret_untracked(context, builder, signature.return_type, None)


# TODO: jump tables
# TODO: staticmethod const functions
# TODO: reverse next
@optional_jitclass
class SIMDFastMersenneTwister:
    """SIMD-oriented Fast Mersenne Twister Pseudo Random Number Generator"""

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
        self.state[0] = seed
        self.index = np.uint16(624)  # ensures shuffle after initialization
        inner = np.uint32(seed & np.uint32(1))

        for i in range(1, 624):
            seed = np.uint32(
                np.uint32(0x6C078965) * (seed ^ (seed >> np.uint32(30))) + np.uint32(i)
            )
            self.state[i] = seed

        inner ^= self.state[3] & np.uint32(0x13C9E684)
        inner ^= inner >> 16
        inner ^= inner >> 8
        inner ^= inner >> 4
        inner ^= inner >> 2
        inner ^= inner >> 1

        self.state[0] ^= ~inner & np.uint32(1)

    def advance(self, adv: np.uint32) -> None:
        """Advance SIMD-oriented Fast Mersenne Twister sequence by adv"""
        adv = np.uint32(adv)
        adv = (adv * np.uint32(2)) + np.uint32(self.index)
        while adv >= np.uint32(624):
            self.shuffle()
            adv -= np.uint32(624)
        self.index = np.uint16(adv)

    def next(self) -> np.uint64:
        """Access and return the next 64-bit rand"""
        if self.index == np.uint16(624):
            self.shuffle()

        low = self.state[self.index]
        self.index += 1
        high = self.state[self.index]
        self.index += 1
        return np.uint64(low) | (np.uint64(high) << np.uint64(32))

    def shuffle(self) -> None:
        """Advance and shuffle the entire state (624 advances)"""
        state = self.state

        sfmt_shuffle(state)

        self.index = 0

    def next_rand(self, maximum: np.uint64) -> np.uint64:
        """Generate and return the next [0, maximum) random uint via modulo distribution"""
        return self.next() % np.uint64(maximum)
