"""Exact conversion of weights and quotas given as arbitrary numbers.

Every finite number Python can hold is rational, so a weighted game with
real weights is scaled to an equivalent game with integer weights:
multiplying all weights and the quota by one positive factor keeps every
coalition's outcome.

Floats are read as written by default ("decimal"): ``0.1`` means 1/10, the
number its shortest representation shows, not the nearby binary fraction
3602879701896397/2^55 the float actually stores.  This keeps sums such as
0.7 + 0.1 equal to a quota of 0.8, as intended.  ``floats="binary"`` uses
the stored binary value instead.
"""
from __future__ import annotations

import numbers
from decimal import Decimal
from fractions import Fraction
from math import gcd, isfinite
from typing import List, Sequence, Tuple

FLOAT_READINGS = ("decimal", "binary")


def to_fraction(value, floats: str = "decimal") -> Fraction:
    """Exact rational value of an int, float, Fraction, Decimal, str or numpy number."""
    if floats not in FLOAT_READINGS:
        raise ValueError(f"floats must be one of {FLOAT_READINGS}, got {floats!r}")
    if isinstance(value, bool):
        raise TypeError("booleans are not valid weights")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"weights and quotas must be finite, got {value}")
        return Fraction(value)
    if isinstance(value, numbers.Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    if isinstance(value, str):
        return Fraction(value.strip())
    if isinstance(value, numbers.Real):
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"weights and quotas must be finite, got {value}")
        return Fraction(repr(number)) if floats == "decimal" else Fraction(number)
    raise TypeError(f"not a real number: {value!r}")


def integer_game(weights: Sequence, quota, floats: str = "decimal") -> Tuple[List[int], int, Fraction]:
    """Equivalent integer game ``(weights, quota, scale)``.

    ``weights[i] == original[i] * scale`` exactly and ``quota`` is the
    smallest integer at least ``original_quota * scale``; the game is the
    same because coalition weights are integers.  ``scale`` is the smallest
    such factor (weights divided by their common divisor).  Integer weights
    are kept unchanged (scale 1).
    """
    values = [to_fraction(w, floats) for w in weights]
    if any(v < 0 for v in values):
        raise ValueError("weights must be nonnegative")
    q = to_fraction(quota, floats)
    if all(v.denominator == 1 for v in values):
        # integer weights stay as given; only a fractional quota is rounded up
        return [int(v) for v in values], -(-q.numerator // q.denominator), Fraction(1)
    denominator = 1
    for v in values:
        denominator = denominator * v.denominator // gcd(denominator, v.denominator)
    integers = [int(v * denominator) for v in values]
    common = 0
    for x in integers:
        common = gcd(common, x)
    common = common or 1
    scale = Fraction(denominator, common)
    scaled_quota = q * scale
    int_quota = -(-scaled_quota.numerator // scaled_quota.denominator)
    return [x // common for x in integers], int_quota, scale


def is_plain_integers(weights: Sequence, quota) -> bool:
    return (all(isinstance(w, numbers.Integral) and not isinstance(w, bool) for w in weights)
            and isinstance(quota, numbers.Integral) and not isinstance(quota, bool))
