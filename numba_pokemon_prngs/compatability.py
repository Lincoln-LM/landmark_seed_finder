"""Compatability functions to support various python versions"""

import sys

# 3.8 added usage of pow for this
if sys.version_info.minor < 8:

    def mod_inv(value, modulus):
        """Compute the modular multiplicative inverse of value under mod modulus"""

        def extended_gcd(a_val, b_val):
            if a_val == 0:
                return (b_val, 0, 1)
            g_val, y_val, x_val = extended_gcd(b_val % a_val, a_val)
            return (g_val, x_val - (b_val // a_val) * y_val, y_val)

        gcd, inv, _ = extended_gcd(value, modulus)
        if gcd != 1:
            raise ValueError("base is not invertible for the given modulus")
        return inv % modulus

else:

    def mod_inv(value, modulus):
        """Compute the modular multiplicative inverse of value under mod modulus"""
        return pow(value, -1, modulus)
