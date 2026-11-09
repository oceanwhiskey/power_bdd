"""Quota-indexed methods for single weighted games.

* :func:`johnston_dp` - the aggregated Johnston dynamic program:
  O(n^2 q) operations and O(nq) stored values.
* :func:`kakimura_banzhaf`, :func:`kakimura_shapley` - the truncated formal
  power series of Kakimura and Terai (2026), multiplied exactly; uses
  python-flint (0.9 or later) when installed and exact integer packing
  otherwise.

All functions take the quota and the weights in player order and return
exact values in player order.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial, gcd
from typing import List, Sequence

import numpy as np

try:  # optional fast polynomial arithmetic
    from flint import fmpz_poly
except ImportError:  # pragma: no cover - exercised without python-flint
    fmpz_poly = None
if fmpz_poly is not None and not all(hasattr(fmpz_poly, m) for m in ("left_shift", "truncate")):
    fmpz_poly = None    # python-flint before 0.9 lacks truncated arithmetic: use the fallback

#: True if python-flint is installed and recent enough to be used
HAS_FLINT = fmpz_poly is not None


# ---------------------------------------------------------------------------
# aggregated Johnston DP
# ---------------------------------------------------------------------------

def _size_pairs(lo, hi):
    length = np.maximum(hi - lo + 1, 0)
    row = np.repeat(np.arange(len(lo), dtype=np.int64), length)
    first = np.cumsum(length) - length
    k = np.arange(int(length.sum()), dtype=np.int64) - first[row] + lo[row]
    return row, k


def johnston_dp(quota: int, weights: Sequence[int]) -> List[Fraction]:
    """Raw Johnston values J_i by the aggregated O(n^2 q) dynamic program.

    Weights are sorted decreasingly; every winning coalition with a critical
    player is assigned to its last critical player j.  Prefix tables count
    the weight and size of the critical prefix; suffix counts record the
    admissible noncritical continuations; one shared backward reward
    recurrence yields every player's share.  Prefix tables are recovered
    in place (Uno's space-saving technique), so storage is O(nq).
    """
    q = int(quota)
    weights = [int(w) for w in weights]
    result = [Fraction(0)] * len(weights)
    if q <= 0 or sum(weights) < q:
        return result
    order = sorted((i for i, w in enumerate(weights) if w), key=lambda i: -weights[i])
    w = [min(q, weights[i]) for i in order]
    common = 0
    for wi in w:
        common = gcd(common, wi)
    q = (q + common - 1) // common
    w = [wi // common for wi in w]
    n = len(w)
    distinct = sorted(set(w))
    prefix_sums = [0]
    tail_limits = {}
    for wi in w:
        tail_limits[wi] = min(q - 1, prefix_sums[-1])
        prefix_sums.append(prefix_sums[-1] + wi)
    sums = np.array(prefix_sums, dtype=np.int64)
    minimum_size = np.searchsorted(sums, np.arange(q, dtype=np.int64), side="left")

    def maximum_sizes(size, upper):
        targets = sums[size] - np.arange(upper + 1, dtype=np.int64)
        return size - np.searchsorted(sums[:size], targets, side="left")

    tail = {wi: np.zeros(tail_limits[wi] + 1, dtype=object) for wi in distinct}
    counts = np.zeros(q, dtype=object)
    counts[0] = 1
    multiplicities = {wi: 0 for wi in distinct}
    for wi in w:
        multiplicities[wi] += 1
    low = 0
    suffix_upper = 0
    for boundary in distinct:
        prefix = np.zeros(suffix_upper + 2, dtype=object)
        prefix[1:] = np.cumsum(counts[:suffix_upper + 1])
        cap = suffix_upper + 1
        for wi in distinct:
            if wi < boundary:
                continue
            first = max(0, low + q - wi - suffix_upper)
            end = min(tail_limits[wi] + 1, boundary + q - wi)
            if end <= first:
                continue
            offsets = np.arange(end - first, dtype=np.int64)
            start = low + q - wi - first - offsets
            stop = boundary + q - wi - first - offsets
            tail[wi][first:end] += (prefix[np.minimum(stop, cap)]
                                    - prefix[np.maximum(start, 0)])
        for _ in range(multiplicities[boundary]):
            suffix_upper = min(q - 1, suffix_upper + boundary)
            if suffix_upper >= boundary:
                counts[boundary:suffix_upper + 1] += counts[:suffix_upper + 1 - boundary].copy()
        low = boundary

    width = n + 1
    table = np.zeros(q * width, dtype=object)
    table[0] = 1
    for i, wi in enumerate(w):
        top = min(q - 1, prefix_sums[i + 1])
        if top < wi:
            continue
        sources = top - wi + 1
        bounds = maximum_sizes(i, min(q - 1, prefix_sums[i]))
        row, k = _size_pairs(minimum_size[:sources], bounds[:sources])
        source = row * width + k
        table[source + wi * width + 1] += table[source]

    L = 1
    for k in range(1, n + 1):
        L = L * k // gcd(L, k)
    reciprocal = np.array([L // (k + 1) for k in range(n)] + [0], dtype=object)
    rewards = np.zeros(q * width, dtype=object)
    numerators = [0] * n
    for i in range(n - 1, -1, -1):
        wi = w[i]
        upper = min(q - 1, prefix_sums[i])
        bounds = maximum_sizes(i, upper)
        top = min(q - 1, prefix_sums[i + 1])
        if top >= wi:
            sources = top - wi + 1
            row, k = _size_pairs(minimum_size[:sources], bounds[:sources])
            source = row * width + k
            cuts = np.searchsorted(row, np.arange(0, sources + wi, wi), side="left")
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b > a:
                    table[source[a:b] + wi * width + 1] -= table[source[a:b]]
        row, k = _size_pairs(minimum_size[:upper + 1], bounds)
        cells = row * width + k
        contribution = tail[wi][row] * reciprocal[k]
        downstream = row + wi < q
        contribution[downstream] += rewards[cells[downstream] + wi * width + 1]
        reuse = i + 1 < n and wi == w[i + 1]
        numerators[i] = numerators[i + 1] if reuse else int(np.dot(table[cells], contribution))
        rewards[cells] += contribution
    dummies = len(weights) - n          # players of weight 0 double every count
    for i, player in enumerate(order):
        result[player] = Fraction(numerators[i] << dummies, L)
    return result


# ---------------------------------------------------------------------------
# Kakimura--Terai formal power series
# ---------------------------------------------------------------------------

def _encode_univariate(coefficients, block_bytes):
    data = bytearray(len(coefficients) * block_bytes)
    for index, coefficient in enumerate(coefficients):
        if coefficient:
            start = index * block_bytes
            data[start:start + block_bytes] = coefficient.to_bytes(block_bytes, "little")
    return int.from_bytes(data, "little")


def _multiply_univariate(left, right, left_players, right_players, quota):
    players = left_players + right_players
    block_bytes = max(1, (players + 1 + 7) // 8)
    product = _encode_univariate(left, block_bytes) * _encode_univariate(right, block_bytes)
    data = product.to_bytes(max(1, (product.bit_length() + 7) // 8), "little")
    mask = (1 << (8 * block_bytes)) - 1
    result = [0] * quota
    for exponent in range(quota):
        start = exponent * block_bytes
        if start >= len(data):
            break
        result[exponent] = int.from_bytes(data[start:start + block_bytes], "little") & mask
    return result, players


def _univariate_product(weights, quota):
    """prod_p (1 + x^w_p) mod x^quota."""
    if fmpz_poly is not None:
        one = fmpz_poly([1])
        factors = [one + one.left_shift(w) for w in weights if w < quota]
        if not factors:
            return [1] + [0] * (quota - 1)
        while len(factors) > 1:
            factors = [(factors[i] * factors[i + 1]).truncate(quota) if i + 1 < len(factors)
                       else factors[i] for i in range(0, len(factors), 2)]
        coefficients = [int(v) for v in factors[0].coeffs()]
        return coefficients + [0] * (quota - len(coefficients))
    factors = []
    for weight in weights:
        row = [0] * min(quota, weight + 1)
        row[0] = 1
        if weight < quota:
            row[weight] += 1
        factors.append((row, 1))
    while len(factors) > 1:
        nxt = []
        for i in range(0, len(factors), 2):
            if i + 1 == len(factors):
                nxt.append(factors[i])
            else:
                (left, nl), (right, nr) = factors[i], factors[i + 1]
                nxt.append(_multiply_univariate(left, right, nl, nr, quota))
        factors = nxt
    row = factors[0][0] if factors else [1]
    return row + [0] * (quota - len(row))


def _encode_bivariate(rows, block_bytes, stride):
    if not rows:
        return 0
    last = (len(rows) - 1) * stride + max((len(row) for row in rows), default=1)
    data = bytearray(last * block_bytes)
    for cardinality, row in enumerate(rows):
        base = cardinality * stride
        for weight, coefficient in enumerate(row):
            if coefficient:
                start = (base + weight) * block_bytes
                data[start:start + block_bytes] = coefficient.to_bytes(block_bytes, "little")
    return int.from_bytes(data, "little")


def _multiply_bivariate(left, right, left_players, right_players, quota, rows_limit):
    players = left_players + right_players
    block_bytes = max(1, (players + 1 + 7) // 8)
    stride = 2 * quota - 1
    product = (_encode_bivariate(left, block_bytes, stride)
               * _encode_bivariate(right, block_bytes, stride))
    data = product.to_bytes(max(1, (product.bit_length() + 7) // 8), "little")
    row_count = min(rows_limit, len(left) + len(right) - 1)
    result = []
    for cardinality in range(row_count):
        row = [0] * quota
        base = cardinality * stride
        for weight in range(quota):
            start = (base + weight) * block_bytes
            if start >= len(data):
                break
            row[weight] = int.from_bytes(data[start:start + block_bytes], "little")
        result.append(row)
    return result, players


def _bivariate_product(weights, quota, rows_limit):
    """prod_p (1 + y x^w_p) mod x^quota, y^rows_limit."""
    if fmpz_poly is not None:
        rows = [fmpz_poly([1])] + [fmpz_poly([]) for _ in range(rows_limit - 1)]
        for weight in weights:
            if weight < quota:
                for cardinality in range(rows_limit - 1, 0, -1):
                    rows[cardinality] = rows[cardinality] + rows[cardinality - 1].left_shift(weight).truncate(quota)
        result = []
        for row in rows:
            coefficients = [int(v) for v in row.coeffs()]
            result.append(coefficients + [0] * (quota - len(coefficients)))
        return result
    factors = []
    for weight in weights:
        rows = [[1]]
        if rows_limit > 1:
            selected = [0] * min(quota, weight + 1)
            if weight < quota:
                selected[weight] = 1
            rows.append(selected)
        factors.append((rows, 1))
    while len(factors) > 1:
        nxt = []
        for i in range(0, len(factors), 2):
            if i + 1 == len(factors):
                nxt.append(factors[i])
            else:
                (left, nl), (right, nr) = factors[i], factors[i + 1]
                nxt.append(_multiply_bivariate(left, right, nl, nr, quota, rows_limit))
        factors = nxt
    rows = factors[0][0] if factors else [[1]]
    rows = list(rows) + [[0] * quota for _ in range(rows_limit - len(rows))]
    return [row + [0] * (quota - len(row)) for row in rows]


def _prefix_sums(row):
    total, result = 0, []
    for value in row:
        total += value
        result.append(total)
    return result


def _quotient_coefficient(g, weight, target):
    """[x^target] g / (1 + x^weight)."""
    if target < 0:
        return 0
    total, sign = 0, 1
    for exponent in range(target, -1, -weight):
        total += sign * g[exponent]
        sign = -sign
    return total


def _bivariate_quotient_coefficient(g, weight, cardinality, target):
    """[y^cardinality x^target] g / (1 + y x^weight)."""
    if target < 0:
        return 0
    total, sign = 0, 1
    for step in range(min(cardinality, target // weight) + 1):
        total += sign * g[cardinality - step][target - step * weight]
        sign = -sign
    return total


def _check_kakimura(quota, weights):
    if any(w <= 0 for w in weights):
        raise ValueError("Kakimura-Terai assumes positive integer weights")


def kakimura_banzhaf(quota: int, weights: Sequence[int]) -> List[int]:
    """Swing counts B_i from prod (1 + x^w) (Kakimura-Terai, Lemma 3)."""
    weights = [int(w) for w in weights]
    _check_kakimura(quota, weights)
    n = len(weights)
    q = int(quota)
    if q <= 0 or sum(weights) < q:
        return [0] * n
    g = _prefix_sums(_univariate_product(weights, q))
    by_weight = {}
    for weight in set(weights):
        by_weight[weight] = (_quotient_coefficient(g, weight, q - 1)
                             - _quotient_coefficient(g, weight, q - weight - 1))
    return [by_weight[w] for w in weights]


def kakimura_shapley(quota: int, weights: Sequence[int]) -> List[Fraction]:
    """Shapley-Shubik index from prod (1 + y x^w) (Kakimura-Terai)."""
    weights = [int(w) for w in weights]
    _check_kakimura(quota, weights)
    n = len(weights)
    q = int(quota)
    if q <= 0 or sum(weights) < q:
        return [Fraction(0)] * n
    rows_limit = min(n, q) + 1
    g = [_prefix_sums(row) for row in _bivariate_product(weights, q, rows_limit)]
    denominator = factorial(n)
    factorials = [factorial(k) for k in range(n + 1)]
    by_weight = {}
    for weight in set(weights):
        numerator = 0
        for cardinality in range(min(n - 1, rows_limit - 1) + 1):
            upper = _bivariate_quotient_coefficient(g, weight, cardinality, q - 1)
            lower = _bivariate_quotient_coefficient(g, weight, cardinality, q - weight - 1)
            numerator += factorials[cardinality] * factorials[n - cardinality - 1] * (upper - lower)
        by_weight[weight] = Fraction(numerator, denominator)
    return [by_weight[w] for w in weights]
