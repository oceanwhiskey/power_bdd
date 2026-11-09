"""Complete quasi-reduced ordered BDDs (QOBDDs) stored level by level.

A :class:`LevelDiagram` keeps, for every level ``i`` (0-based, one player
per level), two integer arrays: the else and then children of the nodes of
level ``i`` as positions on level ``i + 1``.  Level 0 holds only the root;
the positions after the last level are terminals, and ``accepting`` tells
which of them is the accepting terminal.  Every root-to-terminal path
passes through every level, as the level-wise evaluations require.

Diagrams are canonical: two nodes of one level never represent the same
residual game.  Weighted games are built from subset-sum bitsets (moderate
quotas) or with Bolus's interval tables (any quota); multi-tier games arise
by combining diagrams of the same player order with AND and OR.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from sortedcontainers import SortedList

#: Quotas up to this value use the bitset construction (memory ~ quota).
BITSET_QUOTA_LIMIT = 1 << 22

_FALSE, _TRUE = 0, 1


@dataclass(frozen=True)
class LevelDiagram:
    """A complete QOBDD; ``order[i]`` is the player on level ``i``."""

    order: Tuple[int, ...]
    else_children: Tuple[np.ndarray, ...]
    then_children: Tuple[np.ndarray, ...]
    accepting: np.ndarray

    @property
    def n(self) -> int:
        return len(self.order)

    @property
    def widths(self) -> List[int]:
        """Number of nodes on each level (level 0: the root)."""
        return [len(e) for e in self.else_children]

    @property
    def inner_nodes(self) -> int:
        """Number of inner nodes M."""
        return int(sum(len(e) for e in self.else_children))

    def is_constant(self) -> bool:
        return len(self.accepting) == 1

    def evaluate(self, coalition) -> bool:
        """Membership test for a set of players (original indices)."""
        members = set(coalition)
        node = 0
        for level, player in enumerate(self.order):
            children = self.then_children if player in members else self.else_children
            node = int(children[level][node])
        return bool(self.accepting[node])


# ---------------------------------------------------------------------------
# construction helpers
# ---------------------------------------------------------------------------

def constant_diagram(order: Sequence[int], value: bool) -> LevelDiagram:
    zero = np.zeros(1, dtype=np.int64)
    n = len(order)
    return LevelDiagram(tuple(order), tuple(zero.copy() for _ in range(n)),
                        tuple(zero.copy() for _ in range(n)),
                        np.array([bool(value)]))


def canonicalize(else_children: Sequence[np.ndarray], then_children: Sequence[np.ndarray],
                 accepting: np.ndarray, order: Sequence[int]) -> LevelDiagram:
    """Merge equal residual games level by level, bottom-up.

    The input must be a complete diagram in which every node is reachable
    from the root; the result is the canonical complete QOBDD.
    """
    n = len(order)
    values, cls = np.unique(np.asarray(accepting, dtype=bool), return_inverse=True)
    new_else: List[np.ndarray] = [None] * n
    new_then: List[np.ndarray] = [None] * n
    width_next = len(values)
    for i in range(n - 1, -1, -1):
        e = cls[np.asarray(else_children[i], dtype=np.int64)].astype(np.int64)
        t = cls[np.asarray(then_children[i], dtype=np.int64)].astype(np.int64)
        keys, cls = np.unique(e * width_next + t, return_inverse=True)
        cls = cls.reshape(-1)
        new_else[i] = keys // width_next
        new_then[i] = keys % width_next
        width_next = len(keys)
    return LevelDiagram(tuple(order), tuple(new_else), tuple(new_then), values)


def from_node_tables(n: int, root: int, T: dict, E: dict, levels: Sequence[Sequence[int]],
                     order: Sequence[int]) -> LevelDiagram:
    """Convert a node-table QOBDD (terminals 0 and 1, levels 1..n) to levels."""
    position = {_FALSE: 0, _TRUE: 1}
    else_children: List[np.ndarray] = [None] * n
    then_children: List[np.ndarray] = [None] * n
    for level in range(n, 0, -1):
        nodes = list(levels[level])
        if level == 1:
            nodes = [root]
        else_children[level - 1] = np.array([position[E[v]] for v in nodes], dtype=np.int64)
        then_children[level - 1] = np.array([position[T[v]] for v in nodes], dtype=np.int64)
        position = {v: j for j, v in enumerate(nodes)}
    return canonicalize(else_children, then_children, np.array([False, True]), order)


def run_deep(function: Callable, *args):
    """Run a deeply recursive function in a thread with a large stack."""
    result, error = [], []

    def target():
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 1_000_000))
        try:
            result.append(function(*args))
        except BaseException as exc:  # re-raised in the caller
            error.append(exc)
        finally:
            sys.setrecursionlimit(old)

    previous = threading.stack_size()
    for megabytes in (256, 128, 64, 32):
        try:
            threading.stack_size(megabytes * 1024 * 1024)
            break
        except (ValueError, RuntimeError):
            continue
    try:
        worker = threading.Thread(target=target)
        worker.start()
        worker.join()
    finally:
        threading.stack_size(previous)
    if error:
        raise error[0]
    return result[0]


# ---------------------------------------------------------------------------
# weighted games
# ---------------------------------------------------------------------------

def weighted_diagram(quota: int, weights: Sequence[int], order: Sequence[int]) -> LevelDiagram:
    """Winning QOBDD of ``[quota; weights]`` with players on levels ``order``."""
    level_weights = [int(weights[p]) for p in order]
    total = sum(level_weights)
    if quota <= 0:
        return constant_diagram(order, True)
    if quota > total:
        return constant_diagram(order, False)
    if quota <= BITSET_QUOTA_LIMIT:
        else_c, then_c, accepting = _bitset_levels(quota, level_weights)
        return canonicalize(else_c, then_c, accepting, order)
    return _interval_diagram(quota, level_weights, order)


def _bitset_levels(q: int, weights: List[int]):
    """Level arrays from subset-sum bitsets (nodes keyed by next attainable sum).

    A node on level ``i`` is identified by the smallest attainable suffix
    sum at least its residual quota; two residual quotas induce the same
    residual game exactly when no suffix sum lies between them.  O(nq) bit
    operations, any player order.  Requires ``0 < q <= sum(weights)``.
    """
    n = len(weights)
    suffix = [None] * (n + 2)
    attainable = np.zeros(q + 1, dtype=bool)
    attainable[0] = True
    suffix[n + 1] = np.packbits(attainable)
    for level in range(n, 0, -1):
        w = weights[level - 1]
        if 0 < w <= q:
            attainable[w:] |= attainable[:q + 1 - w].copy()
        suffix[level] = np.packbits(attainable)
    positions = np.arange(q + 1, dtype=np.int64)

    def next_attainable(level):
        if level == n + 1:
            return None
        att = np.unpackbits(suffix[level], count=q + 1).astype(bool)
        nxt = np.where(att, positions, q + 1)
        return np.minimum.accumulate(nxt[::-1])[::-1]

    def keys_of(nxt, residual):
        if nxt is None:
            return np.where(residual <= 0, 0, q + 1).astype(np.int64)
        out = np.zeros_like(residual)
        positive = residual > 0
        out[positive] = nxt[residual[positive]]
        return out

    def reachable_keys(nxt, prefix, reach_true):
        keys = keys_of(nxt, q - np.flatnonzero(prefix).astype(np.int64))
        mark = np.zeros(q + 2, dtype=bool)
        mark[keys] = True
        mark[0] |= reach_true
        return np.flatnonzero(mark)

    position = np.zeros(q + 2, dtype=np.int64)
    else_children, then_children = [], []
    prefix = np.zeros(q, dtype=bool)
    prefix[0] = True
    reach_true = False
    nxt = next_attainable(1)
    keys = reachable_keys(nxt, prefix, reach_true)
    for level in range(1, n + 1):
        w = weights[level - 1]
        if w < q:
            reach_true = reach_true or bool(prefix[q - w:].any())
            if w > 0:
                shifted = prefix.copy()
                shifted[w:] |= prefix[:q - w]
                prefix = shifted
        else:
            reach_true = reach_true or bool(prefix.any())
        nxt_child = next_attainable(level + 1)
        child_keys = reachable_keys(nxt_child, prefix, reach_true)
        rep = np.where(keys == 0, 0, np.minimum(keys, q)).astype(np.int64)
        else_keys = keys_of(nxt_child, rep)
        then_keys = keys_of(nxt_child, rep - w)
        else_keys[keys == 0] = 0
        then_keys[keys == 0] = 0
        position[child_keys] = np.arange(len(child_keys))
        else_children.append(position[else_keys].copy())
        then_children.append(position[then_keys].copy())
        keys, nxt = child_keys, nxt_child
    return else_children, then_children, keys == 0


def _interval_diagram(q: int, weights: List[int], order: Sequence[int]) -> LevelDiagram:
    """Bolus's construction with interval tables; memory independent of q.

    On every level, intervals ``(lower, upper]`` of residual quotas that
    induce the same residual game are kept sorted; a lookup replaces the
    recursion for every quota in a known interval.  Iterative, any order.
    """
    n = len(weights)
    T, E = {}, {}
    levels: List[List[int]] = [[] for _ in range(n + 1)]
    unique, constants = {}, {}
    tables = [SortedList() for _ in range(n + 2)]
    suffix = [0] * (n + 2)
    for i in range(n, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i - 1]
    next_id = [2]

    def mk(level, e, t):
        key = (level, e, t)
        node = unique.get(key)
        if node is None:
            node = next_id[0]
            next_id[0] += 1
            unique[key] = node
            T[node], E[node] = t, e
            levels[level].append(node)
        return node

    def constant(level, value):
        node = value
        for lv in range(n, level - 1, -1):
            key = (lv, value)
            cached = constants.get(key)
            if cached is None:
                cached = mk(lv, node, node)
                constants[key] = cached
            node = cached
        return node

    inf = float("inf")

    def base(level, rq):
        if level == n + 1:
            return (_TRUE, -inf, 0) if rq <= 0 else (_FALSE, 0, inf)
        if rq <= 0:
            return constant(level, _TRUE), -inf, 0
        if rq > suffix[level]:
            return constant(level, _FALSE), suffix[level], inf
        table = tables[level]
        pos = table.bisect_left((rq, -inf, -1))
        if pos < len(table):
            upper, lower, node = table[pos]
            if lower < rq <= upper:
                return node, lower, upper
        return None

    results = []
    stack = [(1, q, 0, None)]
    while stack:
        level, rq, stage, then_result = stack.pop()
        if stage == 0:
            found = base(level, rq)
            if found is not None:
                results.append(found)
                continue
            stack.append((level, rq, 1, None))
            stack.append((level + 1, rq - weights[level - 1], 0, None))
        elif stage == 1:
            then_result = results.pop()
            stack.append((level, rq, 2, then_result))
            stack.append((level + 1, rq, 0, None))
        else:
            e, e_lo, e_up = results.pop()
            t, t_lo, t_up = then_result
            w = weights[level - 1]
            lower, upper = max(t_lo + w, e_lo), min(t_up + w, e_up)
            node = mk(level, e, t)
            tables[level].add((upper, lower, node))
            results.append((node, lower, upper))
    root = results.pop()[0]
    return from_node_tables(n, root, T, E, levels, order)


# ---------------------------------------------------------------------------
# AND / OR of diagrams in the same order
# ---------------------------------------------------------------------------

def combine(a: LevelDiagram, b: LevelDiagram, operation: str) -> LevelDiagram:
    """Product construction of two diagrams with equal order, then reduction."""
    if a.order != b.order:
        raise ValueError("diagrams must use the same player order")
    n = a.n
    pa = np.zeros(1, dtype=np.int64)
    pb = np.zeros(1, dtype=np.int64)
    else_children, then_children = [], []
    for i in range(n):
        width_b = len(b.accepting) if i == n - 1 else len(b.else_children[i + 1])
        ek = a.else_children[i][pa] * width_b + b.else_children[i][pb]
        tk = a.then_children[i][pa] * width_b + b.then_children[i][pb]
        keys, inverse = np.unique(np.concatenate((ek, tk)), return_inverse=True)
        inverse = inverse.reshape(-1)
        else_children.append(inverse[:len(ek)])
        then_children.append(inverse[len(ek):])
        pa, pb = keys // width_b, keys % width_b
    if operation == "and":
        accepting = a.accepting[pa] & b.accepting[pb]
    elif operation == "or":
        accepting = a.accepting[pa] | b.accepting[pb]
    else:
        raise ValueError(operation)
    return canonicalize(else_children, then_children, accepting, a.order)


# ---------------------------------------------------------------------------
# completeness, minimal winning coalitions
# ---------------------------------------------------------------------------

def is_desirability_ordered(d: LevelDiagram) -> bool:
    """True if every player is at least as desirable as the next one.

    Player ``i`` is at least as desirable as player ``i + 1`` exactly when,
    for every node v on level i, the residual game after (reject i, select
    i+1) is contained in the one after (select i, reject i+1).  All these
    containment pairs are propagated down together; a pair (accepting,
    rejecting) at the terminals refutes the order.  Desirability is
    transitive, so adjacent pairs suffice.
    """
    n = d.n
    if n < 2:
        return True
    pending: List[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(n + 1)]
    pending_y: List[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(n + 1)]
    for i in range(n - 1):
        x = d.then_children[i + 1][d.else_children[i]]   # reject i, select i+1
        y = d.else_children[i + 1][d.then_children[i]]   # select i, reject i+1
        pending[i + 2] = np.concatenate((pending[i + 2], x))
        pending_y[i + 2] = np.concatenate((pending_y[i + 2], y))
    px = np.zeros(0, dtype=np.int64)
    py = np.zeros(0, dtype=np.int64)
    for level in range(2, n + 1):
        px = np.concatenate((px, pending[level]))
        py = np.concatenate((py, pending_y[level]))
        if level == n:
            break
        width = len(d.else_children[level])        # pairs are nodes of `level`
        keys = np.unique(px * width + py)
        px, py = keys // width, keys % width
        px, py = (np.concatenate((d.else_children[level][px], d.then_children[level][px])),
                  np.concatenate((d.else_children[level][py], d.then_children[level][py])))
    return not bool(np.any(d.accepting[px] & ~d.accepting[py]))


def minimal_winning_diagram(d: LevelDiagram, desirability_ordered: bool) -> LevelDiagram:
    """QOBDD of the minimal winning coalitions (Bolus's MinWin).

    With players in desirability order (every weighted game sorted by
    weight, every complete game), Bolus's Lemma 7 removes the set
    difference: the then-branch of a node is the then-branch's MinWin
    unless both children coincide.  Otherwise the general recursion
    MinWin(v) = (MinWin(E v), MinWin(T v) minus win(E v)) is used.
    """
    if desirability_ordered:
        return _minwin_ordered(d)
    return run_deep(_minwin_general, d)


def _minwin_ordered(d: LevelDiagram) -> LevelDiagram:
    n = d.n
    # classes of the result: terminal level holds FALSE/TRUE of the source;
    # level-wise false nodes are added as an extra class.
    acc = d.accepting.astype(bool)
    false_terminal = int(np.flatnonzero(~acc)[0]) if np.any(~acc) else None
    term_values = acc.copy()
    extra_false = false_terminal is None
    if extra_false:
        term_values = np.concatenate((acc, [False]))
    false_pos = int(np.flatnonzero(~term_values)[0])
    else_children: List[np.ndarray] = [None] * n
    then_children: List[np.ndarray] = [None] * n
    # result node j on level i corresponds to source node j (same positions),
    # plus one appended all-false node per level.
    for i in range(n - 1, -1, -1):
        e, t = d.else_children[i], d.then_children[i]
        redundant = e == t
        new_t = np.where(redundant, false_pos, t)
        width = len(e)
        else_children[i] = np.concatenate((e, [false_pos]))
        then_children[i] = np.concatenate((new_t, [false_pos]))
        false_pos = width          # the appended false node of this level
    return _reachable(else_children, then_children, term_values, 0, d.order)


def _minwin_general(d: LevelDiagram) -> LevelDiagram:
    n = d.n
    acc = [bool(x) for x in d.accepting]
    E_src = [arr.tolist() for arr in d.else_children]
    T_src = [arr.tolist() for arr in d.then_children]
    # Result store: per level lists of children; a child is a node index on
    # the next level, or a terminal "F"/"T" on the last level.
    store_e: List[List] = [[] for _ in range(n)]
    store_t: List[List] = [[] for _ in range(n)]
    unique: List[dict] = [dict() for _ in range(n)]
    false_cache: dict = {}
    minwin_memo: dict = {}
    diff_memo: dict = {}

    def mk(level, e, t):
        key = (e, t)
        node = unique[level].get(key)
        if node is None:
            node = len(store_e[level])
            store_e[level].append(e)
            store_t[level].append(t)
            unique[level][key] = node
        return node

    def false_node(level):
        if level == n:
            return "F"
        node = false_cache.get(level)
        if node is None:
            child = false_node(level + 1)
            node = mk(level, child, child)
            false_cache[level] = node
        return node

    def minwin(level, v):
        if level == n:
            return "T" if acc[v] else "F"
        key = (level, v)
        res = minwin_memo.get(key)
        if res is None:
            e, t = E_src[level][v], T_src[level][v]
            res = mk(level, minwin(level + 1, e),
                     diff(level + 1, minwin(level + 1, t), e))
            minwin_memo[key] = res
        return res

    def diff(level, x, y):
        """Result node x AND NOT win(source node y), both on `level`."""
        if level == n:
            return "F" if (x == "F" or acc[y]) else x
        if x == false_node(level):
            return x
        key = (level, x, y)
        res = diff_memo.get(key)
        if res is None:
            res = mk(level, diff(level + 1, store_e[level][x], E_src[level][y]),
                     diff(level + 1, store_t[level][x], T_src[level][y]))
            diff_memo[key] = res
        return res

    root = minwin(0, 0)
    terminal = {"F": 0, "T": 1}
    else_children: List[np.ndarray] = []
    then_children: List[np.ndarray] = []
    for level in range(n):
        if level == n - 1:
            else_children.append(np.array([terminal[c] for c in store_e[level]], dtype=np.int64))
            then_children.append(np.array([terminal[c] for c in store_t[level]], dtype=np.int64))
        else:
            else_children.append(np.array(store_e[level], dtype=np.int64))
            then_children.append(np.array(store_t[level], dtype=np.int64))
    return _reachable(else_children, then_children, np.array([False, True]), root, d.order)


def _reachable(else_children, then_children, accepting, root_index, order) -> LevelDiagram:
    n = len(order)
    current = np.array([root_index], dtype=np.int64)
    new_else, new_then = [], []
    for i in range(n):
        e, t = else_children[i][current], then_children[i][current]
        keys, inverse = np.unique(np.concatenate((e, t)), return_inverse=True)
        inverse = inverse.reshape(-1)
        new_else.append(inverse[:len(e)])
        new_then.append(inverse[len(e):])
        current = keys
    return canonicalize(new_else, new_then, accepting[current], order)
