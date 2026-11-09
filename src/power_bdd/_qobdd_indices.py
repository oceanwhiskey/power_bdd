"""Power indices read off a complete QOBDD (:class:`LevelDiagram`).

All readings are exact and vectorized level by level.

* :func:`banzhaf_swings` - swing counts ``sum_v paths(v)(cnt(T v) - cnt(E v))``
  in O(M) word operations per prime, recovered by Chinese remaindering.
* :func:`row_readings` - forward prefix-size rows ``f(v, k)`` and backward
  rows ``h(v, k)`` stored only on the reachable sizes of each node:

  - ``"johnston"`` / ``"deegan_packel"``: the last-critical-player
    evaluation; valid for complete games whose
    players are on the levels in desirability order;
  - ``"shapley"``: the weighted residual rows with ``k!(n-1-k)!``
    valid for every monotone simple game, any order;
  - ``"member"``: ``h(T v, k)`` with ``L/(k+1)`` on the QOBDD of the
    minimal winning coalitions; every monotone game.

Return values are in level order; callers map them to players.
"""
from __future__ import annotations

from fractions import Fraction
from math import exp, factorial, gcd, lgamma
from typing import List

import numpy as np

from ._diagram import LevelDiagram


def _is_prime(m: int) -> bool:
    """Deterministic Miller-Rabin below 3.2e9 (bases 2, 3, 5, 7)."""
    if m < 2 or m % 2 == 0:
        return m == 2
    d, s = m - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7):
        if a % m == 0:
            continue
        x = pow(a, d, m)
        if x in (1, m - 1):
            continue
        for _ in range(s - 1):
            x = x * x % m
            if x == m - 1:
                break
        else:
            return False
    return True


def _large_primes(count: int) -> List[int]:
    primes, candidate = [], (1 << 31) - 1
    while len(primes) < count:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate -= 2
    return primes


def banzhaf_swings(d: LevelDiagram) -> List[int]:
    """Exact swing counts B_i of the players on levels 0..n-1."""
    n = d.n
    if n == 0:
        return []
    primes = _large_primes((n - 1) // 30 + 1)
    modulus = np.array(primes, dtype=np.int64)[:, None]
    paths = [np.ones((len(primes), 1), dtype=np.int64)]
    for level in range(1, n):
        else_idx, then_idx = d.else_children[level - 1], d.then_children[level - 1]
        width = len(d.else_children[level])
        current = paths[-1]
        if 2 * len(else_idx) < 1 << 22:
            # float64 bin counts are exact: below 2^22 summands of size < 2^31
            following = np.empty((len(primes), width), dtype=np.int64)
            for row in range(len(primes)):
                weights_row = current[row].astype(np.float64)
                following[row] = (np.bincount(else_idx, weights_row, width)
                                  + np.bincount(then_idx, weights_row, width))
        else:
            following = np.zeros((len(primes), width), dtype=np.int64)
            np.add.at(following, (slice(None), else_idx), current)
            np.add.at(following, (slice(None), then_idx), current)
        paths.append(following % modulus)
    residues = np.zeros((len(primes), n), dtype=np.int64)
    count = np.repeat(d.accepting.astype(np.int64)[None, :], len(primes), axis=0)
    for level in range(n, 0, -1):
        else_count = np.take(count, d.else_children[level - 1], axis=1)
        then_count = np.take(count, d.then_children[level - 1], axis=1)
        difference = (then_count - else_count) % modulus
        weighted = (paths[level - 1] * difference) % modulus
        residues[:, level - 1] = weighted.sum(axis=1) % modulus[:, 0]
        count = (else_count + then_count) % modulus
    product = 1
    for prime in primes:
        product *= prime
    factors = [(product // p) * pow(product // p, -1, p) for p in primes]
    return [sum(int(r) * f for r, f in zip(residues[:, i].tolist(), factors)) % product
            for i in range(n)]


def winning_coalitions(d: LevelDiagram) -> int:
    """Number of winning coalitions (exact)."""
    count = d.accepting.astype(np.int64).astype(object)
    for level in range(d.n, 0, -1):
        count = count[d.else_children[level - 1]] + count[d.then_children[level - 1]]
    return int(count[0])


def _row_layout(lo: np.ndarray, hi: np.ndarray):
    length = hi - lo + 1
    start = np.zeros(len(lo) + 1, dtype=np.int64)
    np.cumsum(length, out=start[1:])
    node = np.repeat(np.arange(len(lo), dtype=np.int64), length)
    k = np.arange(start[-1], dtype=np.int64) - start[node] + lo[node]
    return start, node, k


def _lcm_upto(n: int) -> int:
    value = 1
    for k in range(1, n + 1):
        value = value * k // gcd(value, k)
    return value


#: Largest number of players for float64 row readings (counts up to 2^n).
FLOAT_PLAYER_LIMIT = 1000


def row_readings(d: LevelDiagram, mode: str, floating: bool = False):
    """Numerators per level, see the module docstring.

    Exact (default): integers; for ``"shapley"`` the values are
    ``numerators[i] / n!``, for the other modes raw shares scaled by
    ``L = lcm(1..n)``.  With ``floating=True`` all rows are float64 and the
    numerators are the values themselves (Shapley-Shubik) or the raw shares
    (other modes).  The local Johnston rewards are differences of counts;
    they are computed exactly per node and converted afterwards, so the
    rows only add and multiply nonnegative numbers.  Shapley-Shubik reads
    differences of rows and therefore has a small absolute, not relative,
    rounding error.
    """
    if mode not in ("johnston", "deegan_packel", "shapley", "member"):
        raise ValueError(mode)
    n = d.n
    if floating and n > FLOAT_PLAYER_LIMIT:
        raise ValueError(f"float evaluation supports at most {FLOAT_PLAYER_LIMIT} players "
                         "(counts up to 2^n); use exact arithmetic")
    dtype = np.float64 if floating else object
    else_children, then_children, accepting = d.else_children, d.then_children, d.accepting
    weighted = mode in ("shapley", "member")

    lo, hi = [np.zeros(1, dtype=np.int64)], [np.zeros(1, dtype=np.int64)]
    for i in range(n - 1):
        width = len(else_children[i + 1])
        child_lo = np.full(width, np.iinfo(np.int64).max, dtype=np.int64)
        child_hi = np.full(width, -1, dtype=np.int64)
        np.minimum.at(child_lo, else_children[i], lo[i])
        np.minimum.at(child_lo, then_children[i], lo[i] + 1)
        np.maximum.at(child_hi, else_children[i], hi[i])
        np.maximum.at(child_hi, then_children[i], hi[i] + 1)
        lo.append(child_lo)
        hi.append(child_hi)
    layouts = [_row_layout(lo[i], hi[i]) for i in range(n)]

    if floating and mode == "shapley":
        scale = [exp(lgamma(m + 1) + lgamma(n - m) - lgamma(n + 1)) for m in range(n)]
    elif floating:
        scale = [1.0 / (k + 1) for k in range(n)]
    elif mode == "shapley":
        scale = [factorial(m) * factorial(n - 1 - m) for m in range(n)]
    else:
        L = _lcm_upto(n)
        scale = [L // (k + 1) for k in range(n)]
    if weighted:
        h_lo = [np.maximum(lo[i] - 1, 0) for i in range(n)]
        h_starts = [_row_layout(h_lo[i], hi[i])[0] for i in range(n)]
    else:
        h_lo, h_starts = lo, [layout[0] for layout in layouts]
    factors = np.array(scale + [0], dtype=dtype)

    forward = [np.array([1], dtype=dtype)]
    for i in range(n - 1):
        _, node, k = layouts[i]
        else_node, then_node = else_children[i][node], then_children[i][node]
        start, child_lo = layouts[i + 1][0], lo[i + 1]
        targets = np.concatenate((start[else_node] + k - child_lo[else_node],
                                  start[then_node] + k + 1 - child_lo[then_node]))
        order = np.argsort(targets, kind="stable")
        ordered = targets[order]
        cuts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        following = np.zeros(int(start[-1]), dtype=dtype)
        following[ordered[cuts]] = np.add.reduceat(
            np.concatenate((forward[i], forward[i]))[order], cuts)
        forward.append(following)

    terminal_width = n + 1
    rows = np.zeros((len(accepting), terminal_width), dtype=dtype)
    if weighted:
        rows[accepting, :] = factors
    rows = rows.reshape(-1)
    row_start = np.arange(len(accepting) + 1, dtype=np.int64) * terminal_width
    row_lo = np.zeros(len(accepting), dtype=np.int64)
    count = accepting.astype(np.int64).astype(object)
    upper = np.zeros(len(accepting), dtype=object)
    zeta = accepting.astype(bool).copy()
    numerators = [0] * n
    for i in range(n - 1, -1, -1):
        E, T = else_children[i], then_children[i]
        _, node, k = layouts[i]
        E_node, T_node = E[node], T[node]
        if mode == "shapley":
            selected = (rows[row_start[T_node] + k - row_lo[T_node]]
                        - rows[row_start[E_node] + k - row_lo[E_node]])
        elif mode == "member":
            selected = rows[row_start[T_node] + k - row_lo[T_node]]
        else:
            selected = rows[row_start[T_node] + k + 1 - row_lo[T_node]]
            if mode == "johnston":
                difference = count[T] - count[E]
                local = difference - upper[T]           # eta(v)
                upper = difference + upper[E]
                count = count[E] + count[T]
                reward = local[node].astype(np.float64) if floating else local[node]
                selected = selected + reward * factors[k]
            else:
                alpha = zeta[T] & ~zeta[E]               # alpha(v)
                zeta = zeta[E]
                rewarded = np.flatnonzero(alpha[node])
                selected = selected.copy()
                selected[rewarded] += factors[k[rewarded]]
        if len(selected):
            dot = np.dot(forward[i], selected)
            numerators[i] = float(dot) if floating else int(dot)
        else:
            numerators[i] = 0.0 if floating else 0
        if i == 0:
            break
        if weighted:
            _, h_node, h_k = _row_layout(h_lo[i], hi[i])
            E_h, T_h = E[h_node], T[h_node]
            rows = (rows[row_start[E_h] + h_k - row_lo[E_h]]
                    + rows[row_start[T_h] + h_k + 1 - row_lo[T_h]])
        else:
            rows = selected + rows[row_start[E_node] + k - row_lo[E_node]]
        row_start, row_lo = h_starts[i], h_lo[i]
    return numerators


def banzhaf_float(d: LevelDiagram) -> List[float]:
    """Absolute Banzhaf index in float64 from reach and win probabilities.

    p(v) is the probability that a uniformly random set of the earlier
    players reaches v, c(v) the probability that a random set of the later
    players wins from v; beta_i = sum_v p(v) (c(T v) - c(E v)).  No counts,
    so no overflow for any n.
    """
    n = d.n
    reach = [np.ones(1)]
    for level in range(1, n):
        width = len(d.else_children[level])
        half = reach[-1] * 0.5
        reach.append(np.bincount(d.else_children[level - 1], half, width)
                     + np.bincount(d.then_children[level - 1], half, width))
    win = d.accepting.astype(np.float64)
    beta = [0.0] * n
    for level in range(n, 0, -1):
        e_win, t_win = win[d.else_children[level - 1]], win[d.then_children[level - 1]]
        beta[level - 1] = float(np.dot(reach[level - 1], t_win - e_win))
        win = 0.5 * (e_win + t_win)
    return beta


def shapley_values(d: LevelDiagram) -> List[Fraction]:
    denominator = factorial(d.n)
    return [Fraction(x, denominator) for x in row_readings(d, "shapley")]


def normalized_shares(numerators: List[int]) -> List[Fraction]:
    total = sum(numerators)
    return [Fraction(x, total) if total else Fraction(0) for x in numerators]
