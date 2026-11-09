"""Brute-force reference values from the definitions (small games only)."""
import itertools
import json
from fractions import Fraction
from math import factorial
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def brute_force(n, wins):
    """All four indices of the game with winning predicate ``wins``.

    Returns dict with absolute Banzhaf, Shapley-Shubik, normalized Johnston
    and Deegan-Packel, and the raw Johnston/Deegan-Packel values.
    """
    coalitions = [S for r in range(n + 1) for S in itertools.combinations(range(n), r)]
    winning = {S for S in coalitions if wins(S)}
    swings = [0] * n
    shapley = [Fraction(0)] * n
    johnston = [Fraction(0)] * n
    deegan = [Fraction(0)] * n
    for S in winning:
        critical = [i for i in S if tuple(x for x in S if x != i) not in winning]
        for i in critical:
            swings[i] += 1
            shapley[i] += Fraction(factorial(len(S) - 1) * factorial(n - len(S)), factorial(n))
            johnston[i] += Fraction(1, len(critical))
        if S and len(critical) == len(S):
            for i in S:
                deegan[i] += Fraction(1, len(S))

    def norm(values):
        total = sum(values)
        return [v / total if total else Fraction(0) for v in values]

    return dict(banzhaf=[Fraction(b, 2 ** (n - 1)) for b in swings], swings=swings,
                shapley_shubik=shapley, johnston=norm(johnston), johnston_raw=johnston,
                deegan_packel=norm(deegan), deegan_packel_raw=deegan)


def weighted_predicate(quota, weights):
    return lambda S: sum(weights[i] for i in S) >= quota


def same_game(quota_a, weights_a, quota_b, weights_b):
    """True if both weighted representations have the same winning coalitions."""
    import numpy as np
    from power_bdd._diagram import weighted_diagram
    order = sorted(range(len(weights_a)), key=lambda p: (-weights_a[p], p))
    a = weighted_diagram(quota_a, weights_a, order)
    b = weighted_diagram(quota_b, weights_b, order)
    return (all(np.array_equal(x, y) for x, y in zip(a.else_children, b.else_children))
            and all(np.array_equal(x, y) for x, y in zip(a.then_children, b.then_children))
            and np.array_equal(a.accepting, b.accepting))
