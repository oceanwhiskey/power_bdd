"""Exact minimum quota of a weighted game, without an LP solver.

The minimum quota ``q_min`` is the smallest quota of any nonnegative integer
representation ``[q'; w']`` of the same game.  The search works with one
weight per class of equal original weights (averaging within a class keeps
any representation valid) on the polyhedron

    c(S).w - q >= 0  (S winning),    q - c(S).w >= 1  (S losing),    w >= 0.

Its coalition rows are never listed; the winning QOBDD supplies them:

0. *Start without a diagram.*  At the given weights (v, Q) a coalition is
   tight exactly when it weighs Q or Q - 1, a subset-sum condition.  When
   the diagram would be wide, subset-sum bitsets over the classes harvest
   tight coalitions and an exact-sum knapsack generates columns, so that
   already minimal weights are certified before any diagram is built.
1. *Active set.*  An exact primal simplex on the implicit polyhedron
   starts at the given weights.  The inverse of the working rows is kept
   as an integer adjugate (Bareiss updates); the ratio test is an exact
   lexicographic dynamic program on the diagram.  At every new vertex,
   the start included, nonnegative least squares on the tight rows, with
   column generation from the diagram, proposes a dual solution, and its
   support is exchanged into the basis without moving the point; this
   skips the long runs of degenerate pivots at a degenerate optimum.  At
   the LP optimum the multipliers prove ``q_min >= ceil(q*)``.  The optimum
   times the common denominator of its coordinates is a representation,
   and rounded multiples of it give further upper bounds after an exact
   integer check.
2. *Integer gap.*  At an optimal basis of the labelled LP (one weight per
   player), integer representations are the coset points of a lattice
   whose quotient group has order delta (Gomory's group relaxation): a
   shortest path in that group raises the lower bound.  The coset points
   are then searched by cost in planes spanned by the (at most two)
   cheapest generators: there the known rows cut out a polygon, whose
   cheapest lattice point follows from Fourier-Motzkin elimination, and
   every point that fails on the diagram adds the violated coalitions as
   cutting planes.

Floating point only proposes rows or multipliers; every bound is checked
with exact integer arithmetic before it is reported.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._diagram import LevelDiagram, _meet_in_the_middle, weighted_diagram

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

    def extrema_many(self, weight_vectors) -> List[Tuple[int, float]]:
        """:meth:`integer_extrema` for many class weight vectors in one pass.

        int64 when every weight sum is below 2^60, else one exact pass each.
        """
        if not weight_vectors:
            return []
        mult = self.classes.multiplicities
        if max(sum(abs(int(v)) * m for v, m in zip(w, mult)) for w in weight_vectors) >= INT64_SAFE:
            return [self.integer_extrema(w) for w in weight_vectors]
        width = max(len(e) for e in self.d.else_children)
        chunk = max(1, (1 << 23) // width)
        if len(weight_vectors) > chunk:
            return [r for i in range(0, len(weight_vectors), chunk)
                    for r in self.extrema_many(weight_vectors[i:i + chunk])]
        count = len(weight_vectors)
        weights = np.array([[int(v) for v in w] for w in weight_vectors], dtype=np.int64)
        big = 1 << 62
        acc = self.d.accepting.astype(bool)[:, None]
        ones = np.ones(count, dtype=np.int64)
        lightest = np.where(acc, 0, big) * ones          # over winning continuations
        heaviest = np.where(acc, -big, 0) * ones         # over losing continuations
        zero = np.zeros(count, dtype=np.int64)
        for i in range(self.n - 1, -1, -1):
            h = self.level_class[i]
            kv = weights[:, h] if h >= 0 else zero
            e, t = self.d.else_children[i], self.d.then_children[i]
            lightest = np.minimum(lightest[e], lightest[t] + kv)
            heaviest = np.maximum(heaviest[e], heaviest[t] + kv)
        out = []
        for lo, wi in zip(heaviest[0].tolist(), lightest[0].tolist()):
            out.append((-1 if lo < -(1 << 61) else lo, float("inf") if wi > (1 << 61) else wi))
        return out

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


class _NeedDiagram(Exception):
    """The subset-sum oracle cannot answer: the point has left the start."""


class _Unbounded(Exception):
    """A search plane is unbounded under the known rows."""


#: Smallest n * quota for which the start is first certified without a diagram.
SUBSET_START_WORK = 1 << 24
#: Largest quota for the subset-sum bitsets.
SUBSET_QUOTA_LIMIT = 1 << 24
#: Diagrams with at most k q / (4 n) nodes per level count as narrow: they cost
#: less to build and use than the subset-sum knapsacks over the k classes.
#: A level has at most 2^min(i, n - i) nodes; beyond that the probe below
#: tries at most this many points or intervals per level.
SUBSET_PROBE_WIDTH = 1 << 12


class _SubsetOracle:
    """The oracle at the start point (v, Q) of an integer representation.

    There, a coalition is tight exactly when its weight is Q (winning) or
    Q - 1 (losing): a subset-sum condition.  Prefix and suffix bitsets of
    attainable class sums (the coefficients of prod (1 + z^v + ... + z^(m v))
    over GF(2)) give tight coalitions for :meth:`harvest`, and an exact-sum
    knapsack over the classes answers :meth:`extremes` while the point is
    the start.  Anything else raises :class:`_NeedDiagram`, and the caller
    falls back to the QOBDD.  No diagram is built.
    """

    def __init__(self, classes: _Classes, start, quota: int, max_calls: int = 32):
        self.classes = classes
        self.v = [int(x) for x in start]
        self.Q = int(quota)
        self.m = list(classes.multiplicities)
        self.calls, self.max_calls = 0, max_calls
        k, Q, v, m = classes.k, self.Q, self.v, self.m
        mask = (1 << (Q + 1)) - 1
        prefix = [1]                             # bit p: sum p from classes < h
        for h in range(k):
            cur, acc = prefix[-1], prefix[-1]
            for t in range(1, m[h] + 1):
                acc |= cur << (t * v[h])
            prefix.append(acc & mask)
        mirrored = [0] * k + [1 << Q]            # bit Q - s: sum s from classes >= h
        for h in range(k - 1, -1, -1):
            cur, acc = mirrored[h + 1], mirrored[h + 1]
            for t in range(1, m[h] + 1):
                acc |= cur >> (t * v[h])
            mirrored[h] = acc
        size = Q // 8 + 2
        self.prefix, self.mirrored = prefix, mirrored
        self.prefix_bits = [np.frombuffer(x.to_bytes(size, "little"), dtype=np.uint8) for x in prefix]
        self.mirrored_bits = [np.frombuffer(x.to_bytes(size, "little"), dtype=np.uint8) for x in mirrored]

    def without_dummies(self) -> bool:
        """True if every class is essential: some coalition without one of its
        members weighs between Q - v_h and Q - 1, so that member can swing."""
        v, m = self.v, self.m          # Q enters through the mirrored bits
        for h in range(self.classes.k):
            sums = 0                              # prefix plus up to m_h - 1 members of h
            for t in range(m[h]):
                sums |= self.prefix[h] << (t * v[h])
            window, span = sums << 1, 1           # bits x + 1 .. x + v_h
            while span < v[h]:
                step = min(span, v[h] - span)
                window |= window << step
                span += step
            if not window & self.mirrored[h + 1]:
                return False
        return True

    def _coalitions(self, requests):
        """Tight coalitions through a fixed choice on a level, as the diagram harvest.

        A request (h, last, bit, target) fixes one player of class h, on the
        first level of its run (``last`` False: the other m_h - 1 members
        follow in the suffix) or on the last one (they precede it in the
        prefix), in or out by ``bit``.  The prefix sum is the largest one
        that can be completed to ``target``; the prefix is traced back with
        the smallest counts first and the suffix with the largest counts
        first, for all requests at once.
        """
        k, Q, v, m = self.classes.k, self.Q, self.v, self.m
        mask = (1 << (Q + 1)) - 1
        partial_prefix, partial_suffix = {}, {}
        found = []
        for h, last, bit, target in requests:
            if last:
                if h not in partial_prefix:
                    acc = 0
                    for t in range(m[h]):
                        acc |= self.prefix[h] << (t * v[h])
                    partial_prefix[h] = acc & mask
                left, right = partial_prefix[h], self.mirrored[h + 1]
            else:
                if h not in partial_suffix:
                    acc = 0
                    for t in range(m[h]):
                        acc |= self.mirrored[h + 1] >> (t * v[h])
                    partial_suffix[h] = acc
                left, right = self.prefix[h], partial_suffix[h]
            options = left & (right >> ((Q - target) + bit * v[h]))
            if options:
                found.append((h, last, bit, target, options.bit_length() - 1))
        if not found:
            return []
        count = len(found)
        level = np.array([f[0] for f in found])
        last = np.array([f[1] for f in found], dtype=bool)
        counts = np.zeros((count, k), dtype=np.int64)
        counts[np.arange(count), level] = [f[2] for f in found]
        cap = np.array(m, dtype=np.int64)

        def trace(rest, active_of, bits_of, flip, upwards):
            for g in (range(k - 1, -1, -1) if flip else range(k)):
                active = active_of(g)
                if not active.any():
                    continue
                limit = np.where(level == g, cap[g] - 1, cap[g])
                bits = bits_of(g)
                chosen = np.full(count, -1, dtype=np.int64)
                for c in (range(0, m[g] + 1) if upwards else range(m[g], -1, -1)):
                    r = rest - c * v[g]
                    ok = active & (chosen < 0) & (r >= 0) & (c <= limit)
                    idx = np.where(ok, r if flip else Q - r, 0)
                    ok &= ((bits[idx >> 3] >> (idx & 7)) & 1).astype(bool)
                    chosen[ok] = c
                take = active & (chosen > 0)
                counts[take, g] += chosen[take]
                rest[take] -= chosen[take] * v[g]
            return rest

        # prefix: classes < h, and class h itself when the level is the last of its run
        trace(np.array([f[4] for f in found], dtype=np.int64),
              lambda g: (level > g) | ((level == g) & last), lambda g: self.prefix_bits[g],
              flip=True, upwards=True)
        # suffix: classes > h, and class h itself when the level is the first of its run
        trace(np.array([f[3] - f[4] - f[2] * v[f[0]] for f in found], dtype=np.int64),
              lambda g: (level < g) | ((level == g) & ~last), lambda g: self.mirrored_bits[g + 1],
              flip=False, upwards=False)
        weights = counts @ np.array(v, dtype=np.int64)
        if (not np.array_equal(weights, np.array([f[3] for f in found], dtype=np.int64))
                or (counts > cap).any()):
            raise AssertionError("subset-sum reconstruction failed")
        return [(bytes(row.astype(np.uint8)), f[3] == Q) for row, f in zip(counts, found)]

    def harvest(self, class_weights) -> set:
        """Tight coalitions through each choice on the first and the last level
        of every class run, for both outcomes: the rows of the diagram harvest,
        whose pool tends to give bases with small determinants."""
        if [int(x) for x in class_weights] != self.v:
            raise _NeedDiagram
        requests = [(h, last, bit, target) for h in range(self.classes.k)
                    for last in ((False, True) if self.m[h] > 1 else (False,))
                    for bit in (0, 1) for target in (self.Q, self.Q - 1)]
        return set(self._coalitions(requests))

    def extremes(self, P, S):
        """Lexicographic extremes of (c.v, c.S) among the tight coalitions."""
        if [int(x) for x in P] != self.v:
            raise _NeedDiagram
        self.calls += 1
        if self.calls > self.max_calls:
            raise _NeedDiagram
        S = [int(x) for x in S]
        if sum(abs(s) * m for s, m in zip(S, self.m)) >= INT64_SAFE:
            raise _NeedDiagram
        win = self._exact(S, self.Q, maximize=False)
        lose = self._exact(S, self.Q - 1, maximize=True)
        return ((self.Q, win[0], win[1]) if win else None,
                (self.Q - 1, lose[0], lose[1]) if lose else None)

    def _exact(self, S, target, maximize):
        """Best c.S over counts c (0 <= c_h <= m_h) with c.v = target (binary splitting).

        Unreachable sums start at -+2^62 and stay beyond -+2^61, since every
        sum of values is below 2^60 in size (checked by the caller); so the
        sentinel needs no mask, and a step is one in-place min or max.  A
        take bit set at an unreachable sum is never followed back.
        """
        if target < 0:
            return None
        none = -(1 << 62) if maximize else (1 << 62)
        best = np.full(target + 1, none, dtype=np.int64)
        best[0] = 0
        items, takes = [], []
        for h in range(self.classes.k):
            left, piece = self.m[h], 1
            while left > 0:
                p = min(piece, left)
                left -= p
                piece *= 2
                w = p * self.v[h]
                if w > target:
                    continue
                moved = best[:target + 1 - w] + p * S[h]
                here = best[w:]
                if maximize:
                    better = moved > here
                    np.maximum(here, moved, out=here)
                else:
                    better = moved < here
                    np.minimum(here, moved, out=here)
                items.append((h, p, w))
                takes.append(np.packbits(better))
        if abs(int(best[target])) > (1 << 61):
            return None
        counts, s = [0] * self.classes.k, target
        for (h, p, w), packed in zip(reversed(items), reversed(takes)):
            i = s - w
            if i >= 0 and (packed[i >> 3] >> (7 - (i & 7))) & 1:
                counts[h] += p
                s -= w
        if s != 0 or sum(c * x for c, x in zip(counts, self.v)) != target:
            raise AssertionError("exact-sum reconstruction failed")
        return int(best[target]), bytes(counts)


def _certify_start(classes: _Classes, start, quota: int) -> Optional[Fraction]:
    """q* = quota proven at the start (v, quota) without a diagram, or None."""
    if quota > SUBSET_QUOTA_LIMIT or max(classes.multiplicities) > 255:
        return None
    oracle = _SubsetOracle(classes, start, quota, max_calls=8)
    if not oracle.without_dummies():
        return None                               # the diagram path removes them
    solver = _ActiveSet(classes, oracle, start, quota)
    solver.bland = False
    try:                                          # one guided switch, then pricing
        solver.harvest()
        solver.guided_switch()
        if solver.price() is not None:
            return None
    except _NeedDiagram:
        return None
    qstar = solver.value()
    proof = solver.certificate()
    if proof is None or proof[0] != qstar or qstar != quota:
        return None
    return qstar


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
    """The best integer representation found near the LP optimum (x*, q*).

    (x*, q*) times the common denominator D of its coordinates is always a
    representation: winning rows keep their margin 0, losing rows get the
    margin D >= 1.  Besides it, x* and ceil(q*)/q* x* are rounded (nearest,
    down, up); one batched pass checks all candidates.  Returns (quota,
    class weights) or None.
    """
    k = classes.k
    X = [int(v) for v in X]
    common = reduce(gcd, X, int(delta))
    num = [v // common for v in X]                 # D (x*; q*)
    qn, den = num[k], int(delta) // common         # D q*, D
    candidates = {tuple(num[:k])}
    scales = [(qn, den)] + ([(_ceil(qstar), 1)] if qstar.denominator != 1 else [])
    for P, R in scales:                            # P num_h / (R qn) = (P / (R q*)) x*_h
        B = qn * R
        a = [v * P for v in num[:k]]
        candidates.add(tuple((2 * x + B) // (2 * B) for x in a))
        candidates.add(tuple(x // B for x in a))
        candidates.add(tuple(-(-x // B) for x in a))
    pool = [list(w) for w in candidates if min(w) >= 0]
    best = None
    for w, (losing, winning) in zip(pool, oracle.extrema_many(pool)):
        if losing < winning and (best is None or losing + 1 < best[0]):
            best = (losing + 1, w)
    return best


# ---------------------------------------------------------------------------
# integer gap: Gomory's group relaxation at the optimal labelled basis
# ---------------------------------------------------------------------------

#: Largest order of the quotient group for the group bound ...
GROUP_LIMIT = 1 << 20
#: ... and for listing the coset points.
GROUP_SEARCH_LIMIT = 1 << 16
#: Largest product of basis size and group order (the cost of the bound).
GROUP_WORK_LIMIT = 1 << 25
#: The second cheapest generator joins the search plane when it costs at
#: most 1/PLANE_RATIO of the third.
PLANE_RATIO = 16
_INF = 1 << 61


class _Group:
    """Gomory's group relaxation at an optimal basis of the labelled LP.

    Every integer representation z = (w, q) has integer slacks s = A z - b
    on the rows A of the basis: s_j >= 0 for coalition and sign rows, free
    for a temporary row.  z = adj (b + s) / delta is integral exactly when
    adj s = -adj b (mod delta), so s lies in one coset of the lattice A Z^K,
    whose quotient group has order delta.  With the optimal multipliers
    lam_j = adj[q, j] >= 0, the quota is (lam.b + lam.s) / delta.

    A cheapest coset point, a shortest path in the Cayley graph of the
    quotient, bounds q_min from below (:meth:`bound`); listing the coset
    points by cost and checking each on the diagram finds q_min
    (:meth:`search`).  The quotient is used as Z/delta when a linear map
    onto Z/delta exists (then it is an isomorphism, since the orders
    agree), otherwise as tuples when delta is small.
    """

    def __init__(self, solver):
        self.ok = False
        K, d = solver.K, int(solver.delta)
        self.K, self.d = K, d
        if d > GROUP_LIMIT or K * d > GROUP_WORK_LIMIT:
            return
        adj = [[int(v) for v in row] for row in solver.adj.tolist()]
        b = [int(v) for v in solver.b]
        costs = adj[K - 1]
        free = [key[0] == "F" for key in solver.rows]
        if any(c < 0 for c in costs) or any(f and c for f, c in zip(free, costs)):
            return                                    # not an optimal basis
        if max(costs) * 2 * d >= _INF:
            return
        self.adj, self.b, self.costs, self.free = adj, b, costs, free
        self.qnum = sum(c * v for c, v in zip(costs, b))          # delta * q*
        columns = [[adj[i][j] % d for i in range(K)] for j in range(K)]
        target = [(-sum(adj[i][j] * b[j] for j in range(K))) % d for i in range(K)]
        self.ok = self._encode(columns, target)

    # -- the quotient group ----------------------------------------------------
    def _encode(self, columns, target) -> bool:
        d, K = self.d, self.K
        rng = random.Random(d)
        for _ in range(16):
            u = [rng.randrange(d) for _ in range(K)]
            steps = [sum(a * b for a, b in zip(u, col)) % d for col in columns]
            if gcd(reduce(gcd, steps, 0), d) == 1:     # onto Z/d, hence bijective
                self.cyclic = True
                self.steps = steps
                self.target = sum(a * b for a, b in zip(u, target)) % d
                self.rest = (self.target - np.arange(d, dtype=np.int64)) % d
                return True
        if d > 4096:
            return False
        zero = (0,) * K                                # tuples, by breadth-first search
        index, elements, queue = {zero: 0}, [zero], [zero]
        while queue:
            nxt = []
            for e in queue:
                for col in columns:
                    t = tuple((x + y) % d for x, y in zip(e, col))
                    if t not in index:
                        index[t] = len(elements)
                        elements.append(t)
                        nxt.append(t)
            queue = nxt
        if len(elements) != d:
            return False
        self.cyclic = False
        self.perm = [np.array([index[tuple((x + y) % d for x, y in zip(e, col))] for e in elements],
                              dtype=np.int64) for col in columns]
        self.target = index[tuple(target)]
        self.rest = np.array([index[tuple((t - x) % d for t, x in zip(target, e))] for e in elements],
                             dtype=np.int64)
        return True

    def _step(self, g: int, j: int) -> int:
        return (g + self.steps[j]) % self.d if self.cyclic else int(self.perm[j][g])

    def _cycles(self, j: int) -> np.ndarray:
        """The elements of the group, one row per cycle of adding generator j."""
        d = self.d
        if self.cyclic:
            g = gcd(self.steps[j], d)
            order = d // g
            return (np.arange(g, dtype=np.int64)[:, None]
                    + self.steps[j] * np.arange(order, dtype=np.int64)[None, :]) % d
        perm, seen, rows = self.perm[j], np.zeros(d, dtype=bool), []
        for start in range(d):
            if not seen[start]:
                cycle, x = [], start
                while not seen[x]:
                    seen[x] = True
                    cycle.append(x)
                    x = int(perm[x])
                rows.append(cycle)
        return np.array(rows, dtype=np.int64)

    def _close(self, D: np.ndarray, j: int) -> np.ndarray:
        """min over t >= 0 of D(x - t g_j) + t cost_j (t in Z when the row is free)."""
        idx = self._cycles(j)
        values = D[idx]
        if self.free[j]:
            closed = np.repeat(values.min(axis=1, keepdims=True), idx.shape[1], axis=1)
        else:
            order = idx.shape[1]
            ramp = np.arange(2 * order, dtype=np.int64) * self.costs[j]
            both = np.concatenate((values, values), axis=1) - ramp
            closed = np.minimum(np.minimum.accumulate(both, axis=1) + ramp, _INF)[:, order:]
        out = np.empty_like(D)
        out[idx] = closed
        return out

    # -- bound and search ----------------------------------------------------
    def bound(self) -> int:
        """ceil((delta q* + gamma) / delta) for the cheapest coset point gamma."""
        D = np.full(self.d, _INF, dtype=np.int64)
        D[0] = 0
        for j in range(self.K):
            D = self._close(D, j)
        gamma = int(D[self.target])
        return -(-(self.qnum + gamma) // self.d)

    @property
    def searchable(self) -> bool:
        """At most two directions without cost (free or with zero multiplier):
        they span the plane of the search."""
        return self.ok and sum(c == 0 for c in self.costs) <= 2 and self.d <= GROUP_SEARCH_LIMIT

    def search(self, oracle, classes, upper, known=None, limit=10_000, work=400_000):
        """The cheapest valid coset point with quota below ``upper``.

        Returns ("found", (q, weights)), which is q_min; ("optimal", None) if
        no representation has a quota below ``upper``; or ("bound", lower)
        when ``limit`` passes over the diagram or ``work`` queue entries were
        spent, every point cheaper than ``lower`` being invalid.

        The r <= 2 cheapest generators (all zero-cost ones, and the cheapest
        positive one when it is far cheaper than the next) span a plane; the
        others form multisets rho.  On the plane of rho every coalition row
        is linear in (t_0, t_1), so the known rows cut out a polygon, and the
        coset condition a lattice.  The cheapest lattice point of the
        polygon is found exactly: Fourier-Motzkin elimination of t_0 gives
        the range of t_1, and a convex lower bound of the cost in t_1 limits
        the t_1 that are tried.  If the point fails on the diagram, the
        violated coalitions join the rows (cutting planes).  The multisets
        rho are listed by A*, with group distances as completion bound, and
        planes wait in the same queue under their current least cost, so the
        first valid point to leave the queue is optimal.
        """
        K, d, costs, k, free = self.K, self.d, self.costs, classes.k, self.free
        budget = (upper - 1) * d - self.qnum
        if budget < 0:
            return "optimal", None
        order = sorted(range(K), key=lambda j: (costs[j], j))
        r = max(1, sum(c == 0 for c in costs))
        if r == 1 and K >= 3 and costs[order[1]] * PLANE_RATIO <= costs[order[2]]:
            r = 2
        r = min(r, K)
        plane = order[:r]
        c = [costs[j] for j in plane] + [0] * (2 - r)
        lower_t = [None if free[j] else 0 for j in plane] + [0] * (2 - r)
        cyclic = self.cyclic
        steps = self.steps if cyclic else None
        perm = None if cyclic else [q.tolist() for q in self.perm]
        back = None
        if not cyclic and r == 2:
            back = [0] * d
            for x, y in enumerate(perm[plane[1]]):
                back[y] = x

        def step(g, j):
            return (g + steps[j]) % d if cyclic else perm[j][g]

        def minus1(g):
            return (g - steps[plane[1]]) % d if cyclic else back[g]

        dlog, x, t = [-1] * d, 0, 0            # t with t g_0 = e, for e in <g_0>
        while dlog[x] < 0:
            dlog[x] = t
            x, t = step(x, plane[0]), t + 1
        period = t
        length = 1                             # order of g_1 (1 without a second direction)
        if r == 2:
            x = step(0, plane[1])
            while x != 0:
                x, length = step(x, plane[1]), length + 1
        residues = {}

        def residue_table(e):
            """For u = t_1 mod ord g_1: t_0 mod ord g_0, or -1 if no t_0 fits."""
            if e not in residues:
                table, x = [], e
                for _ in range(length):
                    table.append(dlog[x])
                    if r == 2:
                        x = minus1(x)
                residues[e] = np.array(table, dtype=np.int64)
            return residues[e]

        D = np.full(d, _INF, dtype=np.int64)
        D[0] = 0
        for j in plane:
            D = self._close(D, j)
        dist = [None] * (K + 1)
        dist[K] = D
        for p in range(K - 1, r - 1, -1):
            D = self._close(D, order[p])
            dist[p] = D
        dist = [None if D is None else D.tolist() for D in dist]
        rest = self.rest.tolist()

        # rows a.z >= b become Gamma s >= beta in slack coordinates (sorted positions)
        adj, b = self.adj, self.b
        origin = [sum(adj[i][j] * b[j] for j in range(K)) for i in range(K)]
        columns = [[adj[i][order[p]] for i in range(K)] for p in range(K)]
        reach = budget // max(1, min((v for v in costs if v > 0), default=1)) + 1
        rows = {"G": None, "beta": None, "n": 0, "kind": np.int64}

        def add_row(a, rhs):
            g = [sum(x * y for x, y in zip(a, col)) for col in columns]
            bv = d * rhs - sum(x * y for x, y in zip(a, origin))
            n, G = rows["n"], rows["G"]
            if rows["kind"] is np.int64 and max(abs(v) for v in g) * reach * K * 4 + abs(bv) >= INT64_SAFE:
                rows["kind"] = object
                if G is not None:
                    rows["G"], rows["beta"] = G.astype(object), rows["beta"].astype(object)
            if G is None or n == len(rows["G"]):
                size = max(64, 2 * n)
                G2 = np.zeros((size, K), dtype=rows["kind"])
                b2 = np.zeros(size, dtype=rows["kind"])
                if n:
                    G2[:n], b2[:n] = rows["G"][:n], rows["beta"][:n]
                rows["G"], rows["beta"] = G2, b2
            rows["G"][n], rows["beta"][n] = g, bv
            rows["n"] = n + 1

        for h in range(K):                     # w_h >= 0 and q >= 1
            add_row([int(i == h) for i in range(K)], int(h == k))
        for h, value in enumerate(classes.values):
            if value < classes.quota:          # a losing single player: w_h <= q - 1
                add_row([-int(i == h) for i in range(k)] + [1], 1)
        if known is not None:
            for a, rhs in zip(known[0].tolist(), known[1].tolist()):
                add_row(a, rhs)

        def ceil_div(a, b):
            return -((-a) // b)

        def plane_point(s, cost, e):
            """Cheapest (t0, t1) on the plane of s under the known rows: (value, t0, t1)."""
            spare = budget - cost
            if spare < 0:
                return None
            table = residue_table(e)
            if not (table >= 0).any():
                return None
            n = rows["n"]
            G, bet = rows["G"][:n], rows["beta"][:n]
            used = [p for p in range(r, K) if s[p]]
            need = bet - G[:, used] @ np.array([s[p] for p in used], dtype=G.dtype) if used else bet.copy()
            a0 = G[:, 0]
            a1 = G[:, 1] if r == 2 else np.zeros(n, dtype=G.dtype)
            extra_a0, extra_a1, extra_n = [], [], []
            if lower_t[0] is not None:
                extra_a0.append(1), extra_a1.append(0), extra_n.append(0)
            if c[0] or c[1]:
                extra_a0.append(-c[0]), extra_a1.append(-c[1]), extra_n.append(-spare)
            if extra_a0:
                kind = G.dtype
                a0 = np.concatenate((a0, np.array(extra_a0, dtype=kind)))
                a1 = np.concatenate((a1, np.array(extra_a1, dtype=kind)))
                need = np.concatenate((need, np.array(extra_n, dtype=kind)))
            if a0.dtype != object:             # products of two rows must stay exact
                top = max(int(np.abs(a0).max()), int(np.abs(a1).max()), 1)
                if top * (top + int(np.abs(need).max())) * 4 >= INT64_SAFE:
                    a0, a1, need = a0.astype(object), a1.astype(object), need.astype(object)
            pos, neg, flat = a0 > 0, a0 < 0, a0 == 0
            lo1, hi1 = lower_t[1], None        # range of t_1
            for coef, rhs in ((a1[flat], need[flat]),) + (
                    ((a1[pos][:, None] * -a0[neg][None, :] + a1[neg][None, :] * a0[pos][:, None]).ravel(),
                     (need[pos][:, None] * -a0[neg][None, :] + need[neg][None, :] * a0[pos][:, None]).ravel()),):
                if not len(coef):
                    continue
                if np.any(rhs[coef == 0] > 0):
                    return None
                up, down = coef > 0, coef < 0
                if up.any():
                    v = int(ceil_div(rhs[up], coef[up]).max())
                    lo1 = v if lo1 is None else max(lo1, v)
                if down.any():
                    v = int((rhs[down] // coef[down]).min())
                    hi1 = v if hi1 is None else min(hi1, v)
            if r == 1:
                lo1, hi1 = 0, 0
            if lo1 is None or hi1 is None:
                raise _Unbounded
            if lo1 > hi1:
                return None
            P0, P1, PN = a0[pos], a1[pos], need[pos]
            M0, M1, MN = a0[neg], a1[neg], need[neg]

            def evaluate(T):
                """Exact least cost and t0 for the t1 in T (inf where none fits)."""
                T = np.asarray(T, dtype=a0.dtype)
                if len(P0):
                    lo = ceil_div(PN[:, None] - P1[:, None] * T[None, :], P0[:, None]).max(axis=0)
                else:
                    lo = np.full(len(T), -(1 << 62), dtype=a0.dtype)
                lo = np.asarray(lo)
                res = table[(T % length).astype(np.int64)]
                t0 = lo + (res - lo) % period
                ok = res >= 0
                if len(M0):
                    hi = ((MN[:, None] - M1[:, None] * T[None, :]) // M0[:, None]).min(axis=0)
                    ok &= t0 <= hi
                value = np.where(ok, c[0] * t0 + c[1] * T, _INF)
                return value, t0

            def bound(t1):
                """Convex lower bound of the cost at t1 (t0 at its real lower limit)."""
                if not len(P0) or c[0] == 0:
                    return c[1] * t1
                return c[1] * t1 + c[0] * max(Fraction(int(n_) - int(p1) * t1, int(p0))
                                              for p0, p1, n_ in zip(P0, P1, PN))

            if c[0] == 0:                      # cost grows with t1 (or is constant): first fit
                start = lo1
                while start <= hi1:
                    T = np.arange(start, min(hi1, start + 4095) + 1)
                    value, t0 = evaluate(T)
                    hit = np.flatnonzero(value < _INF)
                    if len(hit):
                        i = int(hit[0])
                        return int(value[i]), int(t0[i]), int(T[i])
                    start += 4096
                return None
            a, b_ = lo1, hi1                   # minimum of the convex bound (ternary search)
            while b_ - a > 2:
                m1, m2 = a + (b_ - a) // 3, b_ - (b_ - a) // 3
                if bound(m1) <= bound(m2):
                    b_ = m2
                else:
                    a = m1
            centre = min(range(a, b_ + 1), key=bound)
            best = None
            left, right = centre, centre - 1
            width = 256
            while True:                        # widen while the bound can still beat the best
                grow_left = left > lo1 and (best is None or bound(left - 1) < best[0])
                grow_right = right < hi1 and (best is None or bound(right + 1) < best[0])
                if not grow_left and not grow_right:
                    break
                chunk = []
                if grow_left:
                    new_left = max(lo1, left - width)
                    chunk.extend(range(new_left, left))
                    left = new_left
                if grow_right:
                    new_right = min(hi1, right + width)
                    chunk.extend(range(right + 1, new_right + 1))
                    right = new_right
                width = min(2 * width, 1 << 14)
                T = np.array(sorted(chunk))
                value, t0 = evaluate(T)
                for i in np.flatnonzero(value < _INF):
                    cand = (int(value[i]), int(T[i]), int(t0[i]))
                    if best is None or cand < best:
                        best = cand
            if best is None:
                return None
            return best[0], best[2], best[1]

        def point(s, t0, t1):
            """Weights and quota of the coset point s + t0 e_0 + t1 e_1."""
            z = list(origin)
            for p in range(K):
                m = t0 if p == 0 else (t1 if p == 1 and r == 2 else s[p])
                if m:
                    col = columns[p]
                    for i in range(K):
                        z[i] += m * col[i]
            if any(v % d for v in z):
                raise AssertionError("coset point is not integral")
            z = [v // d for v in z]
            return z[:k], z[k]

        push, pop = heapq.heappush, heapq.heappop
        heap, counter, checks, popped, floor = [], 0, 0, 0, 0
        stats = self.stats = {}

        def push_plane(cost, s, g):
            nonlocal counter
            found = plane_point(s, cost, rest[g])
            if found is not None and cost + found[0] <= budget:
                counter += 1
                push(heap, (cost + found[0], counter, 1, cost, s, g, found[1], found[2]))

        zero = (0,) * K
        try:
            if r == K:
                push_plane(0, zero, 0)
            elif dist[r][rest[0]] < _INF:
                heap.append((dist[r][rest[0]], 0, 0, 0, zero, 0, r, True))
            while heap and heap[0][0] <= budget:
                floor = heap[0][0]              # no unresolved point is cheaper
                popped += 1
                if popped > work:
                    stats.update(popped=popped, checks=checks, rows=rows["n"], r=r)
                    return "bound", -(-(self.qnum + heap[0][0]) // d)
                entry = pop(heap)
                if entry[2] == 0:               # a multiset of the other generators
                    _, _, _, cost, s, g, p, own = entry
                    if own:
                        push_plane(cost, s, g)
                    j = order[p]
                    g1, c1 = step(g, j), cost + costs[j]
                    v = dist[p][rest[g1]]
                    if v < _INF and c1 + v <= budget:
                        counter += 1
                        push(heap, (c1 + v, counter, 0, c1, s[:p] + (s[p] + 1,) + s[p + 1:], g1, p, True))
                    if p + 1 < K:
                        v = dist[p + 1][rest[g]]
                        if v < _INF and cost + v <= budget:
                            counter += 1
                            push(heap, (cost + v, counter, 0, cost, s, g, p + 1, False))
                    continue
                key, _, _, cost, s, g, t0, t1 = entry   # a plane and its cheapest point
                now = plane_point(s, cost, rest[g])
                if now is None:
                    continue
                if (now[1], now[2]) != (t0, t1) or cost + now[0] != key:
                    if cost + now[0] <= budget:          # rows were added meanwhile: requeue
                        counter += 1
                        push(heap, (cost + now[0], counter, 1, cost, s, g, now[1], now[2]))
                    continue
                if checks >= limit:
                    stats.update(popped=popped, checks=checks, rows=rows["n"], r=r)
                    return "bound", -(-(self.qnum + key) // d)
                checks += 1
                weights, q = point(s, t0, t1)
                win, lose = oracle.extremes(weights, [0] * k)
                valid = True
                if win is not None and win[0] < q:          # a winning coalition below q
                    add_row(list(win[2]) + [-1], 0)
                    valid = False
                if lose is not None and lose[0] >= q:       # a losing coalition at q or above
                    add_row([-x for x in lose[2]] + [1], 1)
                    valid = False
                if valid:
                    stats.update(popped=popped, checks=checks, rows=rows["n"], r=r)
                    return "found", (q, weights)
                counter += 1
                push(heap, (key, counter, 1, cost, s, g, t0, t1))   # the new rows move its point
        except _Unbounded:
            stats.update(popped=popped, checks=checks, rows=rows["n"], r=r, unbounded=True)
            return "bound", -(-(self.qnum + floor) // d)
        stats.update(popped=popped, checks=checks, rows=rows["n"], r=r)
        return "optimal", None


def _integer_gap(sources, lower, upper):
    """Raise the lower bound and search below ``upper`` at labelled LP optima.

    ``sources`` yields (classes, oracle, solver) with an optimal labelled
    active set (one weight per player).  Returns (lower, upper, player
    weights of a cheaper representation or None, stage or None).
    """
    weights = stage = None
    for classes, oracle, solver in sources:
        group = _Group(solver)
        if not group.ok:
            continue
        bound = group.bound()
        if bound > upper:
            raise AssertionError("group bound above a valid representation")
        if bound > lower:
            lower, stage = bound, "group relaxation"
        if lower >= upper:
            break
        if not group.searchable:
            continue
        solver._flush()
        status, value = group.search(oracle, classes, upper,
                                     known=(solver.pool_mat, solver.pool_b))
        if status == "found":
            upper, weights = value[0], classes.expand(value[1])
            lower, stage = upper, "integer gap enumeration"
        elif status == "optimal":
            lower, stage = upper, "integer gap enumeration"
        elif value > lower:
            lower, stage = value, "integer gap enumeration (limit)"
        break
    return lower, upper, weights, stage


def _labelled_sources(quota, weights, modes=(False, True)):
    """Optimal active sets of the labelled game, by default plain pivots first
    (that path tends to end in a basis with positive multipliers)."""
    classes = _Classes(quota, weights, labelled=True)
    order = [m[0] for m in classes.players]
    diagram = weighted_diagram(quota, weights, order)
    oracle = _Oracle(classes, diagram)
    clipped = [min(quota, v) for v in classes.values]
    common = reduce(gcd, clipped, 0)
    for guided in modes:
        solver = _ActiveSet(classes, oracle, [v // common for v in clipped],
                            _ceil(Fraction(quota, common)))
        if solver.run(guided=guided):
            yield classes, oracle, solver


#: Representations with a quota from this size on are first scaled down.
RESCALE_FROM = 1 << 20


def _rescale(oracle, start, quota):
    """A smaller representation of the same game, as a start for the search.

    The problem depends only on the game, so any representation is as good
    a start as the given one, and small numbers keep the dynamic programs
    in machine integers.  ``start`` is divided by 2, 4, 8, ... and rounded
    as long as one exact pass confirms that the heaviest losing coalition
    stays below the lightest winning one; its quota is then the heaviest
    losing weight plus one.  Returns (class weights, quota).
    """
    best, s = (list(start), quota), 2
    while s <= quota:
        weights = [(2 * v + s) // (2 * s) for v in start]
        losing, winning = oracle.integer_extrema(weights)
        if not losing < winning:
            break
        best, s = (weights, losing + 1), 2 * s
    weights, q = best
    g = reduce(gcd, weights, 0)
    if g > 1:                                       # all coalition weights are multiples of g
        weights, q = [v // g for v in weights], _ceil(Fraction(q, g))
    return weights, q


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
    diagram = None
    wide = len(weights) * quota >= SUBSET_START_WORK
    if wide:
        # a narrow diagram is built and used (see SUBSET_PROBE_WIDTH); the
        # probe gives up as soon as a level gets wider
        narrow = len({w for w in weights if w > 0}) * quota // (4 * len(weights))
        if (len(weights) + 1) // 2 < narrow.bit_length():
            wide = False                       # at most 2^ceil(n/2) nodes per level
        else:
            diagram = _meet_in_the_middle(quota, [weights[p] for p in order], order,
                                          limit=min(narrow, SUBSET_PROBE_WIDTH))
            wide = diagram is None
    if wide:
        # given weights that are already minimal are proven so by subset sums,
        # before a wide diagram is built
        classes = _Classes(quota, weights)
        clipped = [min(quota, v) for v in classes.values]
        common = reduce(gcd, clipped, 0)
        start, upper = [v // common for v in clipped], _ceil(Fraction(quota, common))
        if upper > 1:
            qstar = _certify_start(classes, start, upper)
            if qstar is not None:
                return MinimumQuotaResult(upper, tuple(classes.expand(start)), upper, upper,
                                          qstar, "certificate at the given weights")
    if diagram is None:
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
    upper = given = _ceil(Fraction(quota, common))
    if upper >= RESCALE_FROM:
        incumbent, upper = _rescale(oracle, incumbent, upper)
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
        at_start = optimal and not solver.moved and upper == given
        stage = "certificate at the given weights" if at_start else "active set"
        if lower > upper:
            raise AssertionError("active-set proof contradicts an integer representation")
    final_weights = classes.expand(incumbent)
    labelled = lp_bound is not None and all(m == 1 for m in classes.multiplicities)
    small = len(weights) <= gap_players and all(w > 0 for w in weights)
    if lower < upper and (labelled or small):
        # one weight per player: the class LP optimum is already labelled
        def sources():
            if labelled:
                yield classes, oracle, solver
            if small:
                # from the smallest representation known so far: same game, smaller numbers
                yield from _labelled_sources(upper, final_weights, (False,) if labelled else (False, True))
        lower, upper, found, gap_stage = _integer_gap(sources(), lower, upper)
        if found is not None:
            final_weights = found
        if gap_stage is not None:
            stage = gap_stage
    return MinimumQuotaResult(upper if lower == upper else None, tuple(final_weights),
                              lower, upper, lp_bound, stage)
