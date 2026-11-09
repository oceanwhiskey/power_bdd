"""Exact minimum quota of a weighted game, without an LP solver.

The minimum quota ``q_min`` is the smallest quota of any nonnegative integer
representation ``[q'; w']`` of the same game.  The search works with one
weight per class of equal original weights (averaging within a class keeps
any representation valid) on the polyhedron

    c(S).w - q >= 0  (S winning),    q - c(S).w >= 1  (S losing),    w >= 0.

Its coalition rows are never listed; the winning QOBDD supplies them:

1. *Direct certificate.*  Coalitions that are tight at the given weights
   are harvested from the diagram; a least-squares guess of nonnegative
   multipliers is turned into an exact certificate.  This closes every
   game whose given weights are already minimal.
2. *Active set.*  An exact primal simplex on the implicit polyhedron
   starts at the given weights.  The inverse of the working rows is kept
   as an integer adjugate (Bareiss updates); the ratio test is an exact
   lexicographic dynamic program on the diagram.  At the LP optimum the
   multipliers prove ``q_min >= ceil(q*)``; rounded optimal weights give
   an upper bound after an exact integer check.
3. *Integer gap.*  If the bounds still differ, the active set runs on the
   labelled game (one weight per player, at most 32 players); a basis with
   strictly positive multipliers allows an exact enumeration of all cheaper
   integer slack combinations.

Floating point only proposes rows or multipliers; every bound is checked
with exact integer arithmetic before it is reported.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._diagram import LevelDiagram, weighted_diagram

INT64_SAFE = 1 << 60


# ---------------------------------------------------------------------------
# weight classes and the winning-QOBDD oracle
# ---------------------------------------------------------------------------

class _Classes:
    """Classes of equal positive weight, heaviest first."""

    def __init__(self, quota: int, weights: Sequence[int], labelled: bool = False):
        self.quota = int(quota)
        self.weights = [int(w) for w in weights]
        positive = sorted((p for p, w in enumerate(self.weights) if w > 0),
                          key=lambda p: (-self.weights[p], p))
        if labelled:
            self.players = [[p] for p in positive]
        else:
            groups = {}
            for p in positive:
                groups.setdefault(self.weights[p], []).append(p)
            self.players = [groups[w] for w in sorted(groups, reverse=True)]
        self.values = [self.weights[m[0]] for m in self.players]
        self.multiplicities = [len(m) for m in self.players]
        self.k = len(self.values)
        self.class_of = {p: h for h, members in enumerate(self.players) for p in members}

    def classify(self, counts) -> bool:
        if len(counts) != self.k or any(c > m for c, m in zip(counts, self.multiplicities)):
            raise ValueError("invalid coalition count vector")
        return sum(c * v for c, v in zip(counts, self.values)) >= self.quota

    def expand(self, class_weights) -> List[int]:
        result = [0] * len(self.weights)
        for weight, members in zip(class_weights, self.players):
            for p in members:
                result[p] = int(weight)
        return result


class _Oracle:
    """Exact extremes of linear functions over winning/losing coalitions."""

    def __init__(self, classes: _Classes, diagram: LevelDiagram):
        self.classes = classes
        self.d = diagram
        self.level_class = [classes.class_of.get(p, -1) for p in diagram.order]
        self.n = diagram.n

    # -- lexicographic sweep -------------------------------------------------
    def extremes(self, P, S):
        """(lexmin over winning, lexmax over losing) of (sum P, sum S).

        Each result is ``(primary, secondary, counts)`` or None.  Keys
        ``P*M + S`` in one int64 array when the bounds allow it, otherwise
        two Python-integer arrays; both are exact.
        """
        mult = self.classes.multiplicities
        Sb = sum(abs(int(s)) * m for s, m in zip(S, mult))
        Pb = sum(abs(int(p)) * m for p, m in zip(P, mult))
        M = 2 * Sb + 1
        if Pb * M + Sb < INT64_SAFE:
            return self._packed(P, S, Sb, M)
        return self._pairs(P, S)

    def _trace(self, takes) -> bytes:
        counts = bytearray(self.classes.k)
        node = 0
        for i in range(self.n):
            if takes[i][node]:
                h = self.level_class[i]
                if h >= 0:
                    counts[h] += 1
                node = int(self.d.then_children[i][node])
            else:
                node = int(self.d.else_children[i][node])
        return bytes(counts)

    def _packed(self, P, S, Sb, M):
        key = [int(p) * M + int(s) for p, s in zip(P, S)]
        big = 1 << 62
        acc = self.d.accepting.astype(bool)
        aw = np.where(acc, 0, big).astype(np.int64)
        al = np.where(acc, big, 0).astype(np.int64)
        take_w, take_l = [None] * self.n, [None] * self.n
        for i in range(self.n - 1, -1, -1):
            h = self.level_class[i]
            kv = key[h] if h >= 0 else 0
            e, t = self.d.else_children[i], self.d.then_children[i]
            x, y = aw[e], aw[t] + kv
            take_w[i] = y < x
            aw = np.minimum(x, y)
            x, y = al[e], al[t] - kv
            take_l[i] = y < x
            al = np.minimum(x, y)

        def decode(value, takes, sign):
            if value >= (1 << 61):
                return None
            value *= sign
            p = (value + Sb) // M
            return p, value - p * M, self._trace(takes)

        return decode(int(aw[0]), take_w, 1), decode(int(al[0]), take_l, -1)

    def _pairs(self, P, S):
        acc = self.d.accepting.astype(bool)
        results = []
        for target, minimize in ((True, True), (False, False)):
            a = np.zeros(len(acc), dtype=object)
            b = np.zeros(len(acc), dtype=object)
            valid = acc.copy() if target else ~acc
            takes = [None] * self.n
            for i in range(self.n - 1, -1, -1):
                h = self.level_class[i]
                va, vb = (int(P[h]), int(S[h])) if h >= 0 else (0, 0)
                e, t = self.d.else_children[i], self.d.then_children[i]
                ea, eb, ev = a[e], b[e], valid[e]
                ta, tb, tv = a[t] + va, b[t] + vb, valid[t]
                if minimize:
                    better = (ta < ea) | ((ta == ea) & (tb < eb))
                else:
                    better = (ta > ea) | ((ta == ea) & (tb > eb))
                sel = tv & (~ev | better.astype(bool))
                a, b = np.where(sel, ta, ea), np.where(sel, tb, eb)
                valid = ev | tv
                takes[i] = sel
            results.append(None if not valid[0] else (int(a[0]), int(b[0]), self._trace(takes)))
        return results[0], results[1]

    def integer_extrema(self, class_weights):
        """(maximum losing weight, minimum winning weight) for class weights."""
        win, lose = self.extremes(list(class_weights), [0] * self.classes.k)
        return (-1 if lose is None else lose[0]), (float("inf") if win is None else win[0])

    # -- harvesting --------------------------------------------------------------
    def harvest(self, class_weights) -> set:
        """For every level and choice, the extremal winning and losing coalitions.

        A numerical prefix/suffix sweep proposes about 4n coalitions; callers
        keep only rows that pass an exact test.
        """
        d, n = self.d, self.n
        w = [float(class_weights[h]) if h >= 0 else 0.0 for h in self.level_class]
        acc = d.accepting.astype(bool)
        smin = [None] * (n + 1)
        smax = [None] * (n + 1)
        tmin, tmax = [None] * n, [None] * n
        smin[n] = np.where(acc, 0.0, np.inf)
        smax[n] = np.where(acc, -np.inf, 0.0)
        for i in range(n - 1, -1, -1):
            e, t = d.else_children[i], d.then_children[i]
            sel = w[i] + smin[i + 1][t]
            tmin[i] = sel < smin[i + 1][e]
            smin[i] = np.minimum(smin[i + 1][e], sel)
            sel = w[i] + smax[i + 1][t]
            tmax[i] = sel > smax[i + 1][e]
            smax[i] = np.maximum(smax[i + 1][e], sel)
        pmin = [np.zeros(1)]
        pmax = [np.zeros(1)]
        pred_min, pred_max = [None], [None]
        for i in range(n - 1):
            width = len(d.else_children[i + 1])
            e, t = d.else_children[i], d.then_children[i]
            children = np.concatenate((e, t))
            codes = np.concatenate((np.arange(len(e)) * 2, np.arange(len(e)) * 2 + 1))
            for prefix, store, pred, op in ((pmin, None, pred_min, np.minimum),
                                            (pmax, None, pred_max, np.maximum)):
                incoming = np.concatenate((prefix[i], prefix[i] + w[i]))
                fill = np.inf if op is np.minimum else -np.inf
                nxt = np.full(width, fill)
                op.at(nxt, children, incoming)
                matches = incoming == nxt[children]
                predecessor = np.zeros(width, dtype=np.int64)
                predecessor[children[matches]] = codes[matches]
                prefix.append(nxt)
                pred.append(predecessor)
        rows = set()
        for i in range(n):
            e, t = d.else_children[i], d.then_children[i]
            for bit, targets in ((0, e), (1, t)):
                reward = w[i] if bit else 0.0
                for winning in (True, False):
                    prefix = pmin[i] if winning else pmax[i]
                    suffix = smin[i + 1] if winning else smax[i + 1]
                    scores = prefix + reward + suffix[targets]
                    pos = int(np.argmin(scores) if winning else np.argmax(scores))
                    if not np.isfinite(scores[pos]):
                        continue
                    counts = bytearray(self.classes.k)
                    node = pos
                    for level in range(i - 1, -1, -1):
                        code = int((pred_min if winning else pred_max)[level + 1][node])
                        parent, selected = divmod(code, 2)
                        if selected:
                            h = self.level_class[level]
                            if h >= 0:
                                counts[h] += 1
                        node = parent
                    if bit and self.level_class[i] >= 0:
                        counts[self.level_class[i]] += 1
                    node = int(targets[pos])
                    takes = tmin if winning else tmax
                    for level in range(i + 1, n):
                        if takes[level][node]:
                            h = self.level_class[level]
                            if h >= 0:
                                counts[h] += 1
                            node = int(d.then_children[level][node])
                        else:
                            node = int(d.else_children[level][node])
                    counts = bytes(counts)
                    if self.classes.classify(counts) == winning:
                        rows.add((counts, winning))
        return rows


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------

def _certificate(classes: _Classes, rows, multipliers) -> Optional[Tuple[Fraction, list]]:
    """Exact lower bound sum(z) / (c + sum(r+)) from nonnegative integer multipliers.

    Rows are ``(counts, winning)``; the bound is the one implied by adding
    the coalition inequalities, using ``0 <= w_h <= q`` for the residual.
    Returns (rational bound, terms) after checking every row's outcome.
    """
    residual = [0] * classes.k
    c = rhs = 0
    terms = []
    for (counts, winning), m in zip(rows, multipliers):
        m = int(m)
        if m <= 0:
            continue
        if classes.classify(counts) != winning:
            raise AssertionError("misclassified coalition in a certificate")
        sign = 1 if winning else -1
        for h, cnt in enumerate(counts):
            residual[h] += sign * m * cnt
        c -= sign * m
        if not winning:
            rhs += m
        terms.append((counts, winning, m))
    divisor = c + sum(v for v in residual if v > 0)
    if divisor <= 0:
        return None
    return Fraction(rhs, divisor), terms


def _ceil(x: Fraction) -> int:
    return -(-x.numerator // x.denominator)


def _float_certificate(classes, rows, values):
    """Turn numerical multipliers into exact integers and certify."""
    values = [Fraction.from_float(max(0.0, float(x))) for x in values]
    if not values:
        return None
    denominator = max(v.denominator for v in values)
    numbers = [v.numerator * (denominator // v.denominator) for v in values]
    common = reduce(gcd, numbers, 0) or 1
    return _certificate(classes, rows, [x // common for x in numbers])


def _direct_certificate(classes, rows, incumbent, upper, max_steps=12):
    """Exact lower proof from rows tight at a verified incumbent, if any.

    An SVD whitens the tight-row system for multipliers with q-coefficient
    one; negative trial multipliers are removed greedily.  Heuristic search,
    exact acceptance.
    """
    tight = [(z, b) for z, b in rows
             if sum(c * w for c, w in zip(z, incumbent)) == upper - (not b)]
    if len(tight) < classes.k + 1:
        return None
    matrix = np.empty((classes.k + 1, len(tight)))
    for j, (counts, winning) in enumerate(tight):
        sign = 1 if winning else -1
        matrix[:-1, j] = sign * np.frombuffer(counts, dtype=np.uint8).astype(float)
        matrix[-1, j] = -sign
    try:
        u, singular, whitened = np.linalg.svd(matrix, full_matrices=False)
        if singular[-1] <= np.finfo(float).eps * max(matrix.shape) * singular[0]:
            return None
        rhs = u[-1] / singular
        gram = np.eye(len(singular))
        keep = np.arange(len(tight))
        for _ in range(max_steps):
            prices = whitened[:, keep].T @ np.linalg.solve(gram, rhs)
            negative = prices < -1e-9
            if not np.any(negative):
                return _float_certificate(classes, [tight[int(i)] for i in keep], prices)
            removed = keep[negative]
            keep = keep[~negative]
            if len(keep) < classes.k + 1:
                return None
            block = whitened[:, removed]
            gram -= block @ block.T
    except np.linalg.LinAlgError:
        return None
    return None


# ---------------------------------------------------------------------------
# exact active set
# ---------------------------------------------------------------------------

class _ActiveSet:
    def __init__(self, classes: _Classes, oracle: _Oracle, start, start_quota):
        self.classes, self.oracle = classes, oracle
        k = self.k = classes.k
        K = self.K = k + 1
        self.rows = [("F", j) for j in range(K)]          # phantom rows
        self.b = [int(v) for v in start] + [int(start_quota)]
        self.adj = np.eye(K, dtype=np.int64)
        self.delta = 1
        self.pool_keys, self.pool_index, self._pending = [], {}, []
        self.pool_mat = np.zeros((0, K), dtype=np.int64)
        self.pool_b = np.zeros(0, dtype=np.int64)
        for h in range(k):
            self._add(("N", h))
        self._add(("L", bytes(k)))
        self.X = self._point()
        self.pivots = 0

    def row(self, key):
        kind, data = key
        a = [0] * self.K
        if kind == "W":
            a[:self.k] = list(data)
            a[self.k] = -1
            return a, 0
        if kind == "L":
            a[:self.k] = [-c for c in data]
            a[self.k] = 1
            return a, 1
        a[data] = 1
        return a, 0

    def _add(self, key):
        if key not in self.pool_index:
            self.pool_index[key] = len(self.pool_keys)
            self.pool_keys.append(key)
            self._pending.append(key)

    def _flush(self):
        if self._pending:
            rows = [self.row(key) for key in self._pending]
            self.pool_mat = np.vstack([self.pool_mat, np.array([a for a, _ in rows], dtype=np.int64)])
            self.pool_b = np.concatenate([self.pool_b, np.array([b for _, b in rows], dtype=np.int64)])
            self._pending = []

    def _point(self):
        return [int(v) for v in self.adj.astype(object) @ np.array(self.b, dtype=object)]

    def _reduced(self):
        gx = reduce(gcd, self.X, self.delta)
        return [x // gx for x in self.X], self.delta // gx

    def price(self):
        lam = self.adj[self.K - 1, :]
        candidates = []
        for j, key in enumerate(self.rows):
            value = int(lam[j])
            if key[0] == "F":
                if value:
                    candidates.append((j, -1 if value > 0 else 1, True))
            elif value < 0:
                candidates.append((j, 1, False))
        if not candidates:
            return None
        if self.bland:
            phantoms = [c for c in candidates if c[2]]
            j, sigma, _ = min(phantoms or candidates, key=lambda c: repr(self.rows[c[0]]))
            return j, sigma
        best = max(range(len(candidates)),
                   key=lambda i: (candidates[i][2], abs(float(lam[candidates[i][0]]))))
        j, sigma, _ = candidates[best]
        return j, sigma

    @staticmethod
    def _matvec(mat, vec):
        big = max((abs(v) for v in vec), default=0)
        if big * 256 * max(1, mat.shape[1]) < (1 << 62):
            return mat @ np.array(vec, dtype=np.int64)
        return mat.astype(object) @ np.array(vec, dtype=object)

    def ratio(self, D):
        k = self.k
        Xr, dx = self._reduced()
        gd = reduce(gcd, D, 0)
        Dp = [dx * (v // gd) for v in D]
        self._flush()
        best_t = best_key = best_slope = None
        slopes = self._matvec(self.pool_mat, Dp)
        neg = np.flatnonzero(slopes < 0)
        if len(neg):
            slacks = self._matvec(self.pool_mat[neg], Xr) - dx * self.pool_b[neg].astype(object)
            for idx, slack, slope in zip(neg, slacks, slopes[neg]):
                t = Fraction(int(slack), -int(slope))
                if best_t is None or t < best_t or (t == best_t and int(slope) < best_slope):
                    best_t, best_key, best_slope = t, self.pool_keys[int(idx)], int(slope)
        if best_t == 0:
            return best_t, best_key
        Xw, Dw, Xq, Dq = Xr[:k], Dp[:k], Xr[k], Dp[k]
        win, lose = self.oracle.extremes(Xw, Dw)
        zero = None
        for kind, res in (("W", win), ("L", lose)):
            if res is None:
                continue
            if kind == "W":
                value, slope = res[0] - Xq, res[1] - Dq
            else:
                value, slope = Xq - dx - res[0], Dq - res[1]
            if value < 0:
                raise AssertionError("active set left the feasible region")
            if value == 0 and slope < 0 and (zero is None or slope < zero[0]):
                zero = (slope, (kind, res[2]))
        if zero is not None:
            self._add(zero[1])
            return Fraction(0), zero[1]
        if best_t is None:
            win, lose = self.oracle.extremes(Dw, Xw)
            for kind, res in (("W", win), ("L", lose)):
                if res is None:
                    continue
                if kind == "W":
                    slope, slack = res[0] - Dq, res[1] - Xq
                else:
                    slope, slack = Dq - res[0], Xq - dx - res[1]
                if slope < 0:
                    key = (kind, res[2])
                    self._add(key)
                    t = Fraction(slack, -slope)
                    if best_t is None or t < best_t:
                        best_t, best_key = t, key
            if best_t is None:
                raise AssertionError("unbounded direction")
        while True:  # Dinkelbach from above
            p, s = best_t.numerator, best_t.denominator
            win, lose = self.oracle.extremes([s * x + p * dd for x, dd in zip(Xw, Dw)], Dw)
            improved = False
            for kind, res in (("W", win), ("L", lose)):
                if res is None:
                    continue
                if kind == "W":
                    value = res[0] - (s * Xq + p * Dq)
                else:
                    value = (s * Xq + p * Dq) - s * dx - res[0]
                if value < 0:
                    key = (kind, res[2])
                    a, b = self.row(key)
                    slack = sum(u * v for u, v in zip(a, Xr) if u) - dx * b
                    slope = sum(u * v for u, v in zip(a, Dp) if u)
                    if slope >= 0:
                        raise AssertionError("violated row with nonnegative slope")
                    self._add(key)
                    t = Fraction(slack, -slope)
                    if t < best_t:
                        best_t, best_key, improved = t, key, True
            if not improved:
                return best_t, best_key

    def pivot(self, r, key, moved):
        a, b = self.row(key)
        adj = self.adj
        if adj.dtype != object:
            amax = int(np.abs(adj).max())
            if sum(abs(v) for v in a) * amax < INT64_SAFE:
                U = np.array(a, dtype=np.int64) @ adj
                umax = int(np.abs(U).max())
                if umax * amax < (INT64_SAFE >> 1):
                    Ur = int(U[r])
                    col = adj[:, r].copy()
                    num = Ur * adj - np.outer(col, U)
                    new = num // self.delta
                    if np.any(new * self.delta != num):
                        raise AssertionError("inexact Bareiss division")
                    return self._finish(new, col, Ur, r, key, b, moved)
            adj = adj.astype(object)
        U = np.array(a, dtype=object) @ adj
        Ur = int(U[r])
        col = adj[:, r].copy()
        new = (Ur * adj - np.outer(col, U)) // self.delta
        try:
            small = new.astype(np.int64)
        except OverflowError:
            small = None
        if small is not None and int(np.abs(small).max()) < (1 << 40):
            new, col = small, col.astype(np.int64)
        return self._finish(new, col, Ur, r, key, b, moved)

    def _finish(self, new, col, Ur, r, key, b, moved):
        if Ur == 0:
            raise AssertionError("singular pivot")
        new[:, r] = col
        if Ur < 0:
            new, Ur = -new, -Ur
        old = self.delta
        self.adj, self.delta = new, Ur
        self.rows[r] = key
        self.b[r] = b
        self.X = self._point() if moved else [x * Ur // old for x in self.X]
        self.pivots += 1

    def harvest(self):
        Xr, dx = self._reduced()
        for counts, winning in sorted(self.oracle.harvest(Xr[:self.k])):
            key = ("W" if winning else "L", counts)
            if key in self.pool_index:
                continue
            a, b = self.row(key)
            if sum(u * v for u, v in zip(a, Xr) if u) == dx * b:
                self._add(key)

    def run(self, max_pivots=200_000):
        self.bland = False
        seen, last = set(), None
        self.harvest()
        while True:
            choice = self.price()
            if choice is None:
                return True
            if self.pivots >= max_pivots:
                return False
            r, sigma = choice
            D = [sigma * int(v) for v in self.adj[:, r]]
            t, key = self.ratio(D)
            self.pivot(r, key, moved=t != 0)
            if t != 0:
                self.harvest()
            value = Fraction(self.X[self.k], self.delta)
            if value != last:
                seen.clear()
                last = value
            signature = frozenset(self.rows)
            if signature in seen:
                self.bland = True       # anti-cycling fallback
            seen.add(signature)

    def value(self) -> Fraction:
        return Fraction(self.X[self.k], self.delta)

    def certificate(self):
        lam = self.adj[self.K - 1, :]
        rows, mults = [], []
        for j, key in enumerate(self.rows):
            if key[0] in ("W", "L") and int(lam[j]) > 0:
                rows.append((key[1], key[0] == "W"))
                mults.append(int(lam[j]))
        return _certificate(self.classes, rows, mults)


def _round_candidates(classes, oracle, X, delta, qstar):
    lower = _ceil(qstar)
    xs = [Fraction(int(x), delta) for x in X[:classes.k]]
    scales = [Fraction(1)] if qstar.denominator == 1 else [Fraction(1), Fraction(lower) / qstar]
    best = None
    seen = set()
    for s in scales:
        for mode in ("round", "floor", "ceil"):
            w = []
            for x in xs:
                y = x * s
                if mode == "floor":
                    w.append(y.numerator // y.denominator)
                elif mode == "ceil":
                    w.append(_ceil(y))
                else:
                    w.append((2 * y.numerator + y.denominator) // (2 * y.denominator))
            w = tuple(max(0, v) for v in w)
            if w in seen:
                continue
            seen.add(w)
            losing, winning = oracle.integer_extrema(list(w))
            if losing < winning and (best is None or losing + 1 < best[0]):
                best = (losing + 1, list(w))
    return best


# ---------------------------------------------------------------------------
# integer gap: positive labelled basis and slack enumeration
# ---------------------------------------------------------------------------

def _integer_inverse(matrix):
    """(N, D) with A N = D I, D > 0, by fraction-free Gauss-Jordan."""
    n = len(matrix)
    a = [list(row) + [int(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    previous = 1
    for k in range(n):
        candidates = [i for i in range(k, n) if a[i][k]]
        if not candidates:
            raise ValueError("singular basis")
        pivot_row = min(candidates, key=lambda i: abs(a[i][k]))
        a[k], a[pivot_row] = a[pivot_row], a[k]
        pivot = a[k][k]
        for i in range(n):
            if i == k:
                continue
            factor = a[i][k]
            for j in range(2 * n):
                if j != k:
                    value, remainder = divmod(pivot * a[i][j] - factor * a[k][j], previous)
                    if remainder:
                        raise ArithmeticError("fraction-free division was not exact")
                    a[i][j] = value
            a[i][k] = 0
        previous = pivot
    denominator = a[0][0]
    numerator = [row[n:] for row in a]
    if denominator < 0:
        denominator, numerator = -denominator, [[-x for x in row] for row in numerator]
    for i in range(n):
        for j in range(n):
            if sum(matrix[i][k] * numerator[k][j] for k in range(n)) != denominator * (i == j):
                raise ArithmeticError("integer inverse identity failed")
    return numerator, denominator


def _positive_basis_enumeration(oracle, rows, quota_upper, max_candidates=100_000):
    """Search every cheaper integer slack combination of a positive basis.

    ``rows`` are n+1 labelled coalitions (counts over levels, winning).
    Returns (quota, level weights) or None.
    """
    n = oracle.classes.k
    matrix, right = [], []
    for counts, winning in rows:
        sign = 1 if winning else -1
        matrix.append([sign * c for c in counts] + [-sign])
        right.append(int(not winning))
    inverse, denominator = _integer_inverse(matrix)
    costs = inverse[-1]
    if any(x <= 0 for x in costs):
        return None
    origin = [sum(x * b for x, b in zip(row, right)) for row in inverse]
    budget = quota_upper * denominator - origin[-1]
    if budget < 0:
        return None
    active = [i for i, cost in enumerate(costs) if cost <= budget]
    columns = [[row[i] for row in inverse] for i in active]
    steps = [costs[i] for i in active]
    zero = (0,) * len(active)
    queue, seen = [(0, zero)], {zero}
    examined = 0
    while queue:
        cost, slacks = heapq.heappop(queue)
        examined += 1
        if examined > max_candidates:
            return None
        values = [x + sum(col[i] * s for col, s in zip(columns, slacks)) for i, x in enumerate(origin)]
        if all(x % denominator == 0 for x in values):
            point = [x // denominator for x in values]
            weights, quota = point[:-1], point[-1]
            if quota >= 1 and min(weights) >= 0:
                losing, winning = oracle.integer_extrema(weights)
                if losing < quota <= winning:
                    return quota, weights
        for i, step in enumerate(steps):
            if cost + step <= budget:
                following = list(slacks)
                following[i] += 1
                following = tuple(following)
                if following not in seen:
                    seen.add(following)
                    heapq.heappush(queue, (cost + step, following))
    return None


def _labelled_gap(quota, weights, upper):
    """Close an integer gap of a game without dummies and with n <= 32."""
    classes = _Classes(quota, weights, labelled=True)
    order = [m[0] for m in classes.players]
    diagram = weighted_diagram(quota, weights, order)
    oracle = _Oracle(classes, diagram)
    clipped = [min(quota, v) for v in classes.values]
    common = reduce(gcd, clipped, 0)
    solver = _ActiveSet(classes, oracle, [v // common for v in clipped], _ceil(Fraction(quota, common)))
    if not solver.run():
        return None
    lam = solver.adj[solver.K - 1, :]
    rows = [(key[1], key[0] == "W") for j, key in enumerate(solver.rows)
            if key[0] in ("W", "L") and int(lam[j]) > 0]
    if len(rows) != classes.k + 1:
        return None
    try:
        found = _positive_basis_enumeration(oracle, rows, upper)
    except (ValueError, ArithmeticError):
        return None
    if found is None:
        return None
    q, level_weights = found
    return q, classes.expand(level_weights)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinimumQuotaResult:
    """Exact bounds on the minimum quota; ``optimal`` when they coincide.

    ``weights`` is an integer representation ``[upper_bound; weights]`` of
    the game, ``lp_bound`` the optimal value of the LP relaxation (None if
    not reached), ``stage`` the step that closed the bounds.
    """

    quota: Optional[int]
    weights: Tuple[int, ...]
    lower_bound: int
    upper_bound: int
    lp_bound: Optional[Fraction]
    stage: str

    @property
    def optimal(self) -> bool:
        return self.lower_bound == self.upper_bound


def minimum_quota(quota: int, weights: Sequence[int], *, max_pivots: int = 200_000,
                  gap_players: int = 32) -> MinimumQuotaResult:
    quota = int(quota)
    weights = [int(w) for w in weights]
    if any(w < 0 for w in weights):
        raise ValueError("weights must be nonnegative")
    if quota < 1 or quota > sum(weights):
        raise ValueError("the minimum quota needs a game with winning and losing coalitions")
    order = sorted(range(len(weights)), key=lambda p: (-weights[p], p))
    diagram = weighted_diagram(quota, weights, order)
    essential = sorted(order[i] for i in range(diagram.n)
                       if np.any(diagram.else_children[i] != diagram.then_children[i]))
    if len(essential) < len(weights):
        inner = minimum_quota(quota, [weights[p] for p in essential],
                              max_pivots=max_pivots, gap_players=gap_players)
        lifted = [0] * len(weights)
        for p, w in zip(essential, inner.weights):
            lifted[p] = w
        return MinimumQuotaResult(inner.quota, tuple(lifted), inner.lower_bound,
                                  inner.upper_bound, inner.lp_bound,
                                  inner.stage + " (dummies removed)")
    classes = _Classes(quota, weights)
    oracle = _Oracle(classes, diagram)
    clipped = [min(quota, v) for v in classes.values]
    common = reduce(gcd, clipped, 0)
    incumbent = [v // common for v in clipped]
    upper = _ceil(Fraction(quota, common))
    lower, lp_bound, stage = 1, None, "trivial"
    if lower < upper:
        rows = set()
        rng = np.random.default_rng(len(weights))
        for g in range(4):
            guide = np.array(classes.values, dtype=float)
            if g:
                guide += rng.uniform(-1e-3, 1e-3, len(guide))
            rows |= oracle.harvest(guide)
        rows.add((bytes(classes.k), False))
        proof = _direct_certificate(classes, sorted(rows), incumbent, upper)
        if proof is not None and _ceil(proof[0]) > lower:
            lower, stage = _ceil(proof[0]), "direct certificate"
            if lower > upper:
                raise AssertionError("direct proof contradicts the given representation")
    if lower < upper:
        solver = _ActiveSet(classes, oracle, incumbent, upper)
        optimal = solver.run(max_pivots)
        if optimal:
            qstar = solver.value()
            proof = solver.certificate()
            if proof is None or proof[0] != qstar:
                raise AssertionError("active-set certificate differs from the LP value")
            lp_bound = qstar
            if _ceil(qstar) > lower:
                lower = _ceil(qstar)
            best = _round_candidates(classes, oracle, solver.X, solver.delta, qstar)
            if best is not None and best[0] < upper:
                upper, incumbent = best[0], best[1]
        stage = "active set"
        if lower > upper:
            raise AssertionError("active-set proof contradicts an integer representation")
    final_weights = classes.expand(incumbent)
    if lower < upper and len(weights) <= gap_players and all(w > 0 for w in weights):
        gap = _labelled_gap(quota, weights, upper)
        if gap is not None:
            upper, final_weights = gap
            lower, stage = upper, "integer gap enumeration"
    return MinimumQuotaResult(upper if lower == upper else None, tuple(final_weights),
                              lower, upper, lp_bound, stage)
