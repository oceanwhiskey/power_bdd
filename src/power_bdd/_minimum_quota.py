"""Exact minimum quota of a weighted game, without an LP solver.

The minimum quota ``q_min`` is the smallest quota of any nonnegative integer
representation ``[q'; w']`` of the same game.  The search works with one
weight per class of equal original weights (averaging within a class keeps
any representation valid) on the polyhedron

    c(S).w - q >= 0  (S winning),    q - c(S).w >= 1  (S losing),    w >= 0.

Its coalition rows are never listed; the winning QOBDD supplies them:

1. *Active set.*  An exact primal simplex on the implicit polyhedron
   starts at the given weights.  The inverse of the working rows is kept
   as an integer adjugate (Bareiss updates); the ratio test is an exact
   lexicographic dynamic program on the diagram.  At every new vertex,
   the start included, nonnegative least squares on the tight rows, with
   column generation from the diagram, proposes a dual solution, and its
   support is exchanged into the basis without moving the point; this
   skips the long runs of degenerate pivots at a degenerate optimum.  At
   the LP optimum the multipliers prove ``q_min >= ceil(q*)``; rounded
   optimal weights give an upper bound after an exact integer check.  If
   the optimum is the start point, the given weights are already minimal.
2. *Integer gap.*  If the bounds still differ, the active set runs on the
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
        if Pb < INT64_SAFE and Sb < INT64_SAFE:
            return self._pairs(P, S, np.int64)       # two int64 arrays, still exact
        return self._pairs(P, S, object)

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

    def _pairs(self, P, S, dtype=object):
        acc = self.d.accepting.astype(bool)
        results = []
        for target, minimize in ((True, True), (False, False)):
            a = np.zeros(len(acc), dtype=dtype)
            b = np.zeros(len(acc), dtype=dtype)
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
    def _start_levels(self):
        """Levels where coalitions are harvested: the first and last level of
        every run of one class (players of a class are interchangeable)."""
        lc = self.level_class
        return [i for i in range(self.n)
                if i == 0 or i == self.n - 1 or lc[i] != lc[i - 1] or lc[i] != lc[i + 1]]

    def harvest(self, class_weights) -> set:
        """For every start level and choice, the extremal winning and losing coalitions.

        A numerical prefix/suffix sweep proposes up to four coalitions per
        start level (:meth:`_start_levels`); callers keep only rows that
        pass an exact test.  Values are float32 when every sum is an
        integer below 2^24 (exact), otherwise float64.
        """
        d, n = self.d, self.n
        w = [float(class_weights[h]) if h >= 0 else 0.0 for h in self.level_class]
        # float32 halves the memory traffic and is exact for integer sums < 2^24
        exact32 = all(v == int(v) for v in w) and sum(abs(v) for v in w) < (1 << 24)
        real = np.float32 if exact32 else np.float64
        w = [real(v) for v in w]
        acc = d.accepting.astype(bool)
        smin = [None] * (n + 1)
        smax = [None] * (n + 1)
        tmin, tmax = [None] * n, [None] * n
        smin[n] = np.where(acc, 0.0, np.inf).astype(real)
        smax[n] = np.where(acc, -np.inf, 0.0).astype(real)
        for i in range(n - 1, -1, -1):
            e, t = d.else_children[i], d.then_children[i]
            sel = w[i] + smin[i + 1][t]
            tmin[i] = sel < smin[i + 1][e]
            smin[i] = np.minimum(smin[i + 1][e], sel)
            sel = w[i] + smax[i + 1][t]
            tmax[i] = sel > smax[i + 1][e]
            smax[i] = np.maximum(smax[i + 1][e], sel)
        pmin = [np.zeros(1, dtype=real)]
        pmax = [np.zeros(1, dtype=real)]
        pred_min, pred_max = [None], [None]
        for i in range(n - 1):
            width = len(d.else_children[i + 1])
            e, t = d.else_children[i], d.then_children[i]
            children = np.concatenate((e, t))
            codes = np.concatenate((np.arange(len(e), dtype=np.int32) * 2,
                                    np.arange(len(e), dtype=np.int32) * 2 + 1))
            for prefix, pred, op in ((pmin, pred_min, np.minimum), (pmax, pred_max, np.maximum)):
                incoming = np.concatenate((prefix[i], prefix[i] + w[i]))
                nxt = np.full(width, np.inf if op is np.minimum else -np.inf, dtype=real)
                op.at(nxt, children, incoming)
                matches = incoming == nxt[children]
                predecessor = np.zeros(width, dtype=np.int32)
                predecessor[children[matches]] = codes[matches]
                prefix.append(nxt)
                pred.append(predecessor)
        # one extremal coalition per (level, choice, outcome): its node on the
        # level, then all of them traced back and forward together
        starts_level, starts_bit, starts_win, starts_node, starts_child = [], [], [], [], []
        for i in self._start_levels():
            e, t = d.else_children[i], d.then_children[i]
            for bit, targets in ((0, e), (1, t)):
                for winning in (True, False):
                    prefix = pmin[i] if winning else pmax[i]
                    suffix = smin[i + 1] if winning else smax[i + 1]
                    scores = prefix + suffix[targets]     # + reward: same argmin
                    pos = int(np.argmin(scores) if winning else np.argmax(scores))
                    if not np.isfinite(scores[pos]):
                        continue
                    starts_level.append(i)
                    starts_bit.append(bit)
                    starts_win.append(winning)
                    starts_node.append(pos)
                    starts_child.append(int(targets[pos]))
        if not starts_level:
            return set()
        level_of = np.array(starts_level)
        win = np.array(starts_win)
        lc = np.array(self.level_class)
        counts = np.zeros((len(level_of), self.classes.k), dtype=np.int64)
        node = np.array(starts_node, dtype=np.int64)
        for level in range(n - 2, -1, -1):          # prefixes: from level+1 back to level
            for flag, pred in ((True, pred_min), (False, pred_max)):
                sub = np.flatnonzero((level_of > level) & (win == flag))
                if len(sub) == 0:
                    continue
                code = pred[level + 1][node[sub]]
                if lc[level] >= 0:
                    np.add.at(counts[:, lc[level]], sub[code % 2 == 1], 1)
                node[sub] = code // 2
        own = np.flatnonzero((np.array(starts_bit) == 1) & (lc[level_of] >= 0))
        np.add.at(counts, (own, lc[level_of[own]]), 1)
        node = np.array(starts_child, dtype=np.int64)
        for level in range(1, n):                   # suffixes: follow the best choices
            for flag, takes in ((True, tmin), (False, tmax)):
                sub = np.flatnonzero((level_of < level) & (win == flag))
                if len(sub) == 0:
                    continue
                here = node[sub]
                take = takes[level][here]
                if lc[level] >= 0:
                    np.add.at(counts[:, lc[level]], sub[take], 1)
                node[sub] = np.where(take, d.then_children[level][here], d.else_children[level][here])
        rows = set()
        for row, winning in zip(counts.astype(np.uint8), starts_win):
            key = row.tobytes()
            if self.classes.classify(key) == winning:
                rows.add((key, winning))
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
        self.amax = 1                                      # max |adj| (int64 adjugate)
        self.delta = 1
        self.pool_keys, self.pool_index, self._pending = [], {}, []
        self.pool_mat = np.zeros((0, K), dtype=np.int64)
        self.pool_b = np.zeros(0, dtype=np.int64)
        for h in range(k):
            self._add(("N", h))
        self._add(("L", bytes(k)))
        self.X = self._point()
        self.pivots = 0
        self.moved = False

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
            down = -slopes[neg]
            zero = np.flatnonzero(slacks == 0)
            if len(zero):
                # degenerate step: t = 0, steepest blocking row first
                j = zero[int(np.argmax(down[zero]))]
                best_t, best_key, best_slope = Fraction(0), self.pool_keys[int(neg[j])], -int(down[j])
            else:
                # smallest slack/down: floats preselect, exact comparison decides
                approx = np.array([float(s) for s in slacks]) / np.array([float(v) for v in down])
                cut = float(approx.min())
                for j in np.flatnonzero(approx <= cut * (1 + 1e-9) + 1e-300):
                    t = Fraction(int(slacks[j]), int(down[j]))
                    slope = -int(down[j])
                    if best_t is None or t < best_t or (t == best_t and slope < best_slope):
                        best_t, best_key, best_slope = t, self.pool_keys[int(neg[j])], slope
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

    def pivot(self, r, key, moved, U=None):
        a, b = self.row(key)
        adj = self.adj
        if adj.dtype != object:
            amax = self.amax
            if sum(abs(v) for v in a) * amax < INT64_SAFE:
                if U is None or U.dtype == object:
                    U = np.array(a, dtype=np.int64) @ adj
                umax = max(int(U.max()), -int(U.min()))
                if umax * amax < (INT64_SAFE >> 1):
                    Ur = int(U[r])
                    col = adj[:, r].copy()
                    new = adj * Ur
                    new -= np.outer(col, U)
                    if self.delta != 1:
                        new, remainder = np.divmod(new, self.delta)
                        if remainder.any():
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
        if new.dtype != object:
            self.amax = max(int(new.max()), -int(new.min()))
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

    # -- guided basis switch ---------------------------------------------------
    def _tight(self):
        """Indices of pool rows that are tight at the current point (exact)."""
        self._flush()
        Xr, dx = self._reduced()
        values = self._matvec(self.pool_mat, Xr)
        if values.dtype == object or dx >= (1 << 60):
            rhs = self.pool_b.astype(object) * dx
        else:
            rhs = self.pool_b * dx
        return np.flatnonzero(values == rhs)

    def guided_switch(self, rounds: int = 60) -> bool:
        """Exchange the basis for one that a numerical dual solution predicts optimal.

        At a degenerate vertex many rows are tight, and the pricing rule can
        take hundreds of degenerate pivots, finding the missing rows of the
        certificate one dynamic program at a time.  Here nonnegative least
        squares on the tight rows proposes multipliers lambda >= 0 with
        sum(lambda_j a_j) = e_Q.  While the residual r is not zero, one pass
        over the diagram finds the tight winning and losing coalitions with
        the largest a.r (column generation) and adds them.  The support is
        then made part of the basis without moving the point, at once when
        the determinant is small (:meth:`_install`), otherwise by exact
        Bareiss pivots; if the search stalls, the support found so far is
        exchanged all the same.  The pricing step checks optimality exactly,
        so a wrong proposal only costs time.
        """
        k = self.k
        target = np.zeros(self.K)
        target[self.K - 1] = 1.0
        passive = mult = None
        history = []
        solved = False
        for _ in range(rounds):
            tight = self._tight()
            if len(tight) == 0:
                return False
            matrix = self.pool_mat[tight].astype(float).T        # columns a_j
            start = None
            if passive is not None:
                start = np.isin(tight, passive)
            mult, residual_vec = _nnls(matrix, target, start)
            scale = 1.0 + float(np.abs(mult).max(initial=0.0))
            norm = float(np.linalg.norm(residual_vec))
            if norm <= 1e-7 * scale:
                solved = True
                break
            history.append(norm)
            if len(history) >= 3 and history[-3] - norm <= 1e-6 * history[-3]:
                break                 # no progress
            passive = tight[mult > 0]
            # column generation: tight coalitions with the largest a.r
            Xr, dx = self._reduced()
            r_x = residual_vec[:k]
            top = float(np.abs(r_x).max(initial=0.0))
            if top == 0.0:
                break
            secondary = [-int(round(v / top * (1 << 20))) for v in r_x]
            win, lose = self.oracle.extremes(Xr[:k], secondary)
            added = False
            for kind, res in (("W", win), ("L", lose)):
                if res is None:
                    continue
                tight_value = Xr[k] if kind == "W" else Xr[k] - dx
                if res[0] != tight_value:
                    continue
                key = (kind, res[2])
                a, _ = self.row(key)
                gain = float(np.dot(np.array(a, dtype=float), residual_vec))
                if gain > 1e-9 * scale and key not in self.pool_index:
                    self._add(key)
                    added = True
            if not added:
                break
        if mult is None or not np.any(mult > 0):
            return False
        # exchange the support into the basis; when the search stopped early
        # it is still a set of independent tight rows close to a certificate
        support = [int(tight[j]) for j in np.flatnonzero(mult > 1e-9 * max(1.0, mult.max()))]
        if len(support) > self.K:
            return False
        wanted = [self.pool_keys[i] for i in support]
        if self._install(wanted):
            return solved
        wanted_set = set(wanted)
        for key in wanted:
            if key in self.rows:
                continue
            a, _ = self.row(key)
            adj = self.adj
            if adj.dtype == object or max(abs(v) for v in a) * self.amax * self.K >= INT64_SAFE:
                U = np.array(a, dtype=object) @ adj.astype(object)
            else:
                U = np.array(a, dtype=np.int64) @ adj
            options = [r for r in range(self.K)
                       if self.rows[r] not in wanted_set and int(U[r]) != 0]
            if not options:
                return False
            r = min(options, key=lambda j: (self.rows[j][0] != "F", abs(int(U[j]))))
            self.pivot(r, key, moved=False, U=U)
        return solved

    def _install(self, keys) -> bool:
        """Make the given tight rows part of the basis at once, without moving.

        Missing rows are taken from the current basis (all tight at the
        point), non-phantom rows first, so that the matrix is regular: a
        candidate c is usable while c N, N a null-space basis of the given
        rows, adds rank.  For a small determinant the adjugate is |det|
        times a floating point inverse, rounded and accepted only if
        A adj = |det| I holds exactly; otherwise the caller exchanges the
        rows one by one.
        """
        K = self.K
        if not keys or len(keys) > K:
            return False
        wanted = set(keys)
        rows = [self.row(key) for key in keys]
        keys = list(keys)
        missing = K - len(keys)
        if missing:
            W = np.array([a for a, _ in rows], dtype=float)
            _, singular, vt = np.linalg.svd(W)
            if singular[-1] <= 1e-9 * singular[0]:
                return False
            null = vt[len(keys):].T
            current = [j for j in range(K) if self.rows[j] not in wanted]
            current.sort(key=lambda j: self.rows[j][0] == "F")
            basis = np.zeros((missing, missing))
            chosen = []
            for j in current:
                a, _ = self.row(self.rows[j])
                v = np.array(a, dtype=float) @ null
                residual = v - basis[:len(chosen)].T @ (basis[:len(chosen)] @ v) if chosen else v
                norm = float(np.linalg.norm(residual))
                if norm > 1e-8 * max(1.0, float(np.linalg.norm(v))):
                    basis[len(chosen)] = residual / norm
                    chosen.append(j)
                    if len(chosen) == missing:
                        break
            if len(chosen) < missing:
                return False
            for j in chosen:
                keys.append(self.rows[j])
                rows.append((self.row(self.rows[j])[0], self.b[j]))   # keep its right side
        A = np.array([a for a, _ in rows], dtype=np.int64)
        Af = A.astype(float)
        sign, logdet = np.linalg.slogdet(Af)
        if sign == 0 or not np.isfinite(logdet) or logdet > 40 * np.log(2.0):
            return False
        delta = int(round(float(np.exp(logdet))))
        if delta <= 0:
            return False
        try:
            adj = np.rint(np.linalg.inv(Af) * delta)
        except np.linalg.LinAlgError:
            return False
        amax = float(np.abs(adj).max())
        bound = K * float(np.abs(Af).max()) * amax
        if bound >= 2.0 ** 52:
            return False
        if not np.array_equal(Af @ adj, delta * np.eye(K)):   # exact: all values < 2^52
            return False
        adj = adj.astype(np.int64)
        # the point stays: every row is tight at it, so A X = delta_old * b
        b = [bb for _, bb in rows]
        old = self.delta
        X = [x * delta for x in self.X]
        if any(x % old for x in X):
            return False
        X = [x // old for x in X]
        if self._matvec(A, X).tolist() != [delta * bb for bb in b]:
            return False
        self.adj, self.delta, self.amax = adj, delta, int(amax)
        self.rows, self.b, self.X = list(keys), b, X
        self.pivots += 1
        return True

    def run(self, max_pivots=200_000, guided=True):
        self.bland = False
        seen, last = set(), None
        self.harvest()
        if guided:
            self.guided_switch()
        stalled, retries, pool_seen = 0, 0, len(self.pool_keys)
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
                self.moved = True
                self.harvest()
                if guided:
                    self.guided_switch()      # new vertex: try to certify it at once
                stalled, retries, pool_seen = 0, 0, len(self.pool_keys)
            elif guided:
                stalled += 1
                # degenerate pivots have brought new tight rows: try again
                if (stalled >= max(16, self.K // 8) and retries < 8
                        and len(self.pool_keys) > pool_seen):
                    self.guided_switch()
                    stalled, pool_seen = 0, len(self.pool_keys)
                    retries += 1
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


def _nnls(E, f, start=None, tol=1e-10, max_iterations=None):
    """Lawson-Hanson nonnegative least squares: min ||E x - f||, x >= 0.

    The passive columns E_P are kept with a whitening factor F (E_P F has
    orthonormal columns, so (E_P^T E_P)^-1 = F F^T).  A column enters by
    one Gram-Schmidt step and leaves by a Householder reflection that moves
    its row of F onto the last coordinate; both cost O(p^2), and the
    gradient O(mp), given the Gram matrix of E, which is formed once.  F
    and the passive columns of the Gram matrix live in preallocated arrays.
    ``start`` (a boolean mask) warm-starts the passive set.  The result is
    only a proposal that the caller checks exactly.  Returns x and the
    residual vector f - E x.
    """
    rows, m = E.shape
    gram = E.T @ E
    squares = np.diag(gram).copy()
    rhs = E.T @ f
    limit = max_iterations if max_iterations is not None else 6 * rows + 50
    scale = max(1.0, float(np.abs(E).max(initial=0.0)))
    cap = min(rows, m) + 1
    F = np.zeros((cap, cap))            # whitening factor, top-left p x p block
    G = np.zeros((m, cap))              # gram[:, passive]
    passive: List[int] = []

    def load(columns):
        """Factor of the given columns from scratch; False if dependent."""
        p = len(columns)
        if p == 0:
            return True
        try:
            lower = np.linalg.cholesky(gram[np.ix_(columns, columns)])
            F[:p, :p] = np.linalg.inv(lower.T)
        except np.linalg.LinAlgError:
            return False
        G[:, :p] = gram[:, columns]
        return True

    def solve():
        p = len(passive)
        Fp = F[:p, :p]
        return Fp @ (Fp.T @ rhs[passive])

    def remove(i):
        """Drop passive position i: reflect row i of F onto the last axis."""
        p = len(passive)
        Fp = F[:p, :p]
        v = Fp[i].copy()
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return False
        v[-1] += norm if v[-1] >= 0 else -norm
        Fp -= np.outer(Fp @ v, v * (2.0 / float(v @ v)))
        F[i:p - 1, :p] = F[i + 1:p, :p]
        F[:, p - 1] = 0.0
        F[p - 1, :] = 0.0
        G[:, i:p - 1] = G[:, i + 1:p]
        del passive[i]
        return True

    if start is not None:
        passive = [int(j) for j in np.flatnonzero(start)][:cap - 1]
        if not load(passive):
            passive = []
    x = np.zeros(m)
    if passive:
        z = solve()
        keep = [c for c, v in zip(passive, z) if v > tol]
        if len(keep) < len(passive):
            passive = keep
            if not load(passive):
                passive = []
            z = solve() if passive else np.zeros(0)
            if (z <= tol).any():
                passive, z = [], np.zeros(0)
        x[passive] = z
    excluded = np.zeros(m, dtype=bool)
    iterations = updates = 0
    while iterations < limit:
        p = len(passive)
        gradient = rhs - G[:, :p] @ x[passive] if p else rhs.copy()
        gradient[passive] = -np.inf
        gradient[excluded] = -np.inf
        j = int(np.argmax(gradient))
        if gradient[j] <= tol * scale or p + 1 >= cap:
            break
        # add column j: one Gram-Schmidt step in the whitened basis
        r = F[:p, :p].T @ gram[passive, j] if p else np.zeros(0)
        d2 = float(squares[j] - r @ r)
        if d2 <= 1e-10 * float(squares[j]):
            excluded[j] = True          # dependent on the passive columns
            continue
        d = d2 ** 0.5
        F[:p, p] = -(F[:p, :p] @ r) / d
        F[p, p] = 1.0 / d
        G[:, p] = gram[:, j]
        passive.append(j)
        while iterations < limit:
            iterations += 1
            z = solve()
            if (z > tol).all():
                x = np.zeros(m)
                x[passive] = z
                break
            current = x[passive]
            bad = z <= tol
            denominator = current[bad] - z[bad]
            ratios = np.where(denominator > 0,
                              current[bad] / np.where(denominator > 0, denominator, 1.0), 0.0)
            step = current + float(ratios.min()) * (z - current)
            keep = step > tol
            if keep.all():
                keep = ~bad
            dropped = [c for c, k in zip(passive, keep) if not k]
            if dropped == [j] and float(ratios.min()) == 0.0:
                excluded[j] = True      # numerically unusable new column
            values = step[keep]
            ok = True
            for i in sorted(np.flatnonzero(~keep), reverse=True):
                ok = ok and remove(int(i))
            updates += len(dropped)
            if not ok:
                passive = [c for c, k in zip(passive, keep) if k] if len(passive) > len(values) else passive
            if not ok or updates >= 64:
                updates = 0
                if not load(passive):       # refresh against drift
                    x = np.zeros(m)
                    x[passive] = values
                    return x, f - E @ x
            x = np.zeros(m)
            x[passive] = values
    return x, f - E @ x


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


def _satisfies(known, point) -> bool:
    """A point >= b for the known valid rows (a cheap necessary test)."""
    matrix, rhs = known
    values = _ActiveSet._matvec(matrix, point)
    return bool(np.all(values >= rhs))


def _positive_basis_enumeration(oracle, rows, quota_upper, max_candidates=100_000, known=None):
    """Search every cheaper integer slack combination of a positive basis.

    ``rows`` are n+1 labelled coalitions (counts over levels, winning).
    ``known`` optionally gives valid rows (A, b) with A [w; q] >= b for
    every representation, used to reject candidates before the exact
    check on the diagram.  Returns (quota, level weights) or None.
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
    # positions sorted by cost, so that siblings come in nondecreasing cost
    active = sorted((i for i, cost in enumerate(costs) if cost <= budget),
                    key=lambda i: (costs[i], i))
    steps = [costs[i] for i in active]
    # values of a slack combination s: origin + columns @ s, all exact
    reach = max(abs(x) for x in origin) + sum(
        max(abs(row[i]) for row in inverse) * (budget // costs[i]) for i in active)
    dtype = np.int64 if reach < INT64_SAFE and denominator < INT64_SAFE else object
    columns = np.array([[row[i] for i in active] for row in inverse], dtype=dtype)
    base = np.array(origin, dtype=dtype)
    # best-first in order of cost over the tree in which a combination's
    # parent has one unit less in its last used position; children are
    # generated lazily (first child, then the next sibling), so every
    # examined combination adds at most two entries to the queue
    count = len(active)
    queue = [(0, 0, (0,) * count, -1)]
    pushed = examined = 0
    while queue:
        cost, _, slacks, j = heapq.heappop(queue)
        examined += 1
        if examined > max_candidates:
            return None
        # the quota coordinate is origin[-1] + cost: test it before the rest
        if (origin[-1] + cost) % denominator == 0:
            values = base + columns @ np.array(slacks, dtype=dtype) if count else base
            if not np.any(values % denominator):
                point = [int(x) // denominator for x in values]
                weights, quota = point[:-1], point[-1]
                if quota >= 1 and min(weights) >= 0 and (
                        known is None or _satisfies(known, point)):
                    losing, winning = oracle.integer_extrema(weights)
                    if losing < quota <= winning:
                        return quota, weights
        first = max(j, 0)
        if first < count and cost + steps[first] <= budget:      # first child
            child = list(slacks)
            child[first] += 1
            pushed += 1
            heapq.heappush(queue, (cost + steps[first], pushed, tuple(child), first))
        if 0 <= j < count - 1:                                  # next sibling
            sibling_cost = cost - steps[j] + steps[j + 1]
            if sibling_cost <= budget:
                sibling = list(slacks)
                sibling[j] -= 1
                sibling[j + 1] += 1
                pushed += 1
                heapq.heappush(queue, (sibling_cost, pushed, tuple(sibling), j + 1))
    return None


def _labelled_gap(quota, weights, upper):
    """Close an integer gap of a game without dummies and with n <= 32."""
    classes = _Classes(quota, weights, labelled=True)
    order = [m[0] for m in classes.players]
    diagram = weighted_diagram(quota, weights, order)
    oracle = _Oracle(classes, diagram)
    clipped = [min(quota, v) for v in classes.values]
    common = reduce(gcd, clipped, 0)
    rows = None
    for guided in (False, True):     # the plain pivot path tends to end in a positive basis
        start = [v // common for v in clipped]
        solver = _ActiveSet(classes, oracle, start, _ceil(Fraction(quota, common)))
        if not solver.run(guided=guided):
            return None
        lam = solver.adj[solver.K - 1, :]
        rows = [(key[1], key[0] == "W") for j, key in enumerate(solver.rows)
                if key[0] in ("W", "L") and int(lam[j]) > 0]
        if len(rows) == classes.k + 1:
            break                # a basis with strictly positive multipliers
    if len(rows) != classes.k + 1:
        return None
    solver._flush()
    try:
        found = _positive_basis_enumeration(oracle, rows, upper,
                                            known=(solver.pool_mat, solver.pool_b))
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
            if lower < upper:
                best = _round_candidates(classes, oracle, solver.X, solver.delta, qstar)
                if best is not None and best[0] < upper:
                    upper, incumbent = best[0], best[1]
        stage = "certificate at the given weights" if optimal and not solver.moved else "active set"
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
