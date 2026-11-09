# power_bdd

Exact power indices and minimum quotas of simple games, computed with
quasi-reduced binary decision diagrams (QOBDDs) and dynamic programming.

Let $N=\{1,\dots,n\}$ be a set of voters with weights $w_i$. A coalition
$S\subseteq N$ wins if its weight reaches the quota $q$. Power indices
measure how much influence each voter has. `power_bdd` computes, exactly
(as fractions) or, on request, in float64,

* the **Banzhaf** index (absolute, $\beta_i = B_i/2^{n-1}$),
* the **Shapley–Shubik** index,
* the **Deegan–Packel** index,
* the **Johnston** index (power as exclusive control),
* the **minimum quota** of a weighted game, i.e. the smallest quota of any
  integer representation,

for weighted games and for multi-tier games built with AND and OR, such as
double-majority rules.

## Installation

    pip install power_bdd
    pip install "power_bdd[flint]"     # optional: faster Kakimura-Terai methods

Requires Python 3.10 or later and numpy.

## Quick start

You first build a game, then choose an index, then a method or `auto`:

```python
from power_bdd import Game

game = Game.weighted([3, 2, 1, 1], quota=5)

game.johnston().auto()          # Johnston index, method chosen by a heuristic
game.johnston().dp()            # ... with the O(n^2 q) dynamic program
game.banzhaf().kakimura()       # Banzhaf via Kakimura-Terai power series
game.deegan_packel().minwin()   # Deegan-Packel on the minimal-winning QOBDD
game.minimum_quota()            # exact minimum quota, here 5

print(game.johnston().auto())
# Johnston via qobdd
# player  value
# 0       56.6667 %
# 1       30.0000 %
# 2       6.6667 %
# 3       6.6667 %
```

## Building games

```python
from fractions import Fraction
from power_bdd import Game

Game.weighted([54, 33, 32, 25], quota=73)        # explicit quota
Game.weighted(population, share="65%")           # at least 65 % of the total weight
Game.weighted(weights, share=Fraction(2, 3))     # also 0.5, "1/2", ...
Game.at_least(15, of=27)                         # at least 15 of 27 members
Game.majority(9), Game.unanimity(5)
Game.from_minimal_winning(4, [{0, 1}, {0, 2, 3}, {1, 2, 3}])

game.named(["Germany", "France", ...])          # player names for the results
```

**Weights and quotas can be any real numbers:** `int`, `float`, `Fraction`,
`Decimal`, numeric strings such as `"18.47"` or `"1/3"`, and numpy scalars,
also mixed. The game is scaled exactly to an equivalent game with integer
weights (the smallest such factor), so every method applies unchanged:

```python
g = Game.weighted([0.7, 0.1, 0.2], quota=0.8)
g.weights, g.quota, g.scale        # (7, 1, 2), 8, Fraction(10, 1)
g.original_weights, g.original_quota   # (0.7, 0.1, 0.2), 0.8 as given
g.is_winning([0, 1])               # True: 0.7 + 0.1 reaches 0.8
```

Floats are read as written: `0.1` means 1/10, the number its shortest
representation shows. The float itself stores a nearby binary fraction,
and with those stored values 0.7 + 0.1 falls just short of 0.8. Pass
`floats="binary"` to use the stored values instead:

```python
Game.weighted([0.7, 0.1, 0.2], quota=0.8, floats="binary").is_winning([0, 1])   # False
```

Integer weights are kept as given (scale 1); a fractional quota with integer
weights is rounded up. Weights with many decimal places lead to large integer
quotas, which matters only for the quota-indexed methods (see below).

Combine games on the same players with `&` (all tiers must be won) and `|`
(one tier suffices), or with `Game.all_of(...)`, `Game.any_of(...)`,
`g.and_(...)`, `g.or_(...)`:

```python
# Council of the EU: 55 % of the members representing 65 % of the
# population; a blocking minority needs at least four members.
council = (Game.at_least(15, of=27)
           & (Game.weighted(population, share="65%") | Game.at_least(24, of=27))
          ).named(members)
council.is_complete()            # True: Johnston applies
council.johnston().auto()["Germany"]       # Fraction(...), about 24.808 %

# UN Security Council: all five permanent and at least four elected members
unsc = Game.weighted([1] * 5 + [0] * 10, quota=5) & Game.weighted([0] * 5 + [1] * 10, quota=4)
```

## Methods

Every method returns exact values unless you ask for floats (see
[Float64 results](#float64-results)). Choose one by name (`.method("qobdd")`)
or with the named shortcuts below; `.available()` lists the methods that
apply to a game, `.choose()` tells what `auto` would use.

| Index | Method | Applies to | Operations |
| --- | --- | --- | --- |
| Banzhaf | `qobdd` | every game | $O(M)$ after construction |
| | `kakimura` | weighted games | formal power series, near-linear in $q$ |
| Shapley–Shubik | `qobdd` | every monotone game | $O(nM)$ (weighted residual rows) |
| | `kakimura` | weighted games | formal power series |
| Deegan–Packel | `qobdd` | complete games | $O(nM)$ (last critical player) |
| | `minwin` | every monotone game | $O(nM_{\mathrm{mw}})$ on the minimal-winning QOBDD |
| Johnston | `qobdd` | complete games | $O(nM)$ (last critical player) |
| | `dp` | weighted games | $O(n^2q)$ time, $O(nq)$ storage |

$M$ is the number of nodes of the winning QOBDD, $M_{\mathrm{mw}}$ that of
the QOBDD of the minimal winning coalitions. For weighted games
$M\le n(q_{\min}+1)$, so the diagram methods depend on the *minimum* quota,
not on the given one.

**`auto`**:
* Banzhaf: Kakimura–Terai for weighted games with $nq \le 2\cdot10^6$ when
  python-flint 0.9 or later is installed (and no diagram has been built yet),
  else the QOBDD;
* Shapley–Shubik: QOBDD;
* Deegan–Packel: last-critical QOBDD for complete games, MinWin otherwise;
* Johnston: QOBDD.

These choices follow benchmark measurements: the power series are
fastest for small quotas with random weights, the diagram wins when the
quota is large compared with the structure of the game, and the Johnston
DP is a memory-light alternative rather than a faster one.

**Only valid methods are offered.** `Game.weighted(...)` returns a
`WeightedGame`, `&` and `|` return a `CompositeGame`, and each has its own
selectors: `kakimura()`, `dp()` and `minimum_quota()` exist only for
weighted games, so editors do not offer them for multi-tier games (asked for
by name, `.method("kakimura")` raises `MethodNotApplicable`).

```python
ec = Game.weighted(votes, quota=270)
ec.banzhaf().kakimura()          # fine
council.banzhaf().kakimura()     # AttributeError: multi-tier games have no kakimura()
council.banzhaf().qobdd()        # the QOBDD is the method for multi-tier games
```

**Johnston** and the `qobdd` Deegan–Packel method need a *complete* game
(all players comparable by desirability). Weighted games are always
complete. For multi-tier games, completeness is decided from the weights
when one order of the players is weakly decreasing in every tier (as for
the Council of the EU), and otherwise exactly on the diagram; the players
are then ordered by desirability automatically. Since Johnston has no other
method, `game.johnston()` raises `NotCompleteError` right away for games
that are not complete; for Deegan–Packel, `auto` uses `minwin` there.

**Quota-indexed methods** (`kakimura`, `dp`) work on tables with $n\cdot q$
entries and refuse games with $n\cdot q$ above
`power_bdd.game.QUOTA_INDEXED_LIMIT` (50 million) with a
`MethodNotApplicable` that points to `game.minimal()`: the same game with
its minimum quota, which is often far smaller than the given one. The QOBDD
methods are not affected, since the diagram depends only on the minimum
quota anyway. The limit is a module variable and can be changed.

```python
big = Game.weighted([300_000_000, 200_000_000, 100_000_000, 100_000_000], quota=500_000_000)
big.johnston().dp()             # MethodNotApplicable: ... use game.minimal() ...
big.minimal().johnston().dp()   # quota 5, same values
```

## Results

```python
r = council.johnston().qobdd()
r.values         # tuple of Fractions in player order
r["Germany"]     # by name (or by position)
r.as_dict(), r.to_floats(), r.table(digits=3)
r.raw            # unnormalized values (see below)
r.method, r.index, r.info
```

Raw values: swing counts $B_i$ (Banzhaf), numbers of pivotal orderings
$n!\,\varphi_i$ (Shapley–Shubik), $\sum 1/|\mathrm{Crit}(S)|$ over the winning
coalitions in which $i$ is critical (Johnston), $\sum 1/|S|$ over the minimal
winning coalitions containing $i$ (Deegan–Packel). Banzhaf `values` are
absolute; `r.normalized()` gives shares summing to one.

## Float64 results

Between the index and the method, `.floats()` switches to float64 results
and `.exact()` back to exact fractions (the default):

```python
council.johnston().floats().auto()       # values and raw values as floats
council.banzhaf().floats().qobdd()
council.johnston().floats().exact().auto()   # exact again
r.info["arithmetic"]                     # "float64", "exact, rounded to float64" or "exact"
```

* The **QOBDD methods** compute in float64 throughout. All their rows
  hold nonnegative numbers, so no cancellation occurs; on the benchmark games
  the relative errors stay below $10^{-14}$ (Shapley–Shubik, whose
  coefficients come from `lgamma`) and below $10^{-15}$ for the other indices.
* The **quota-indexed methods** (`kakimura`, `dp`) compute exactly and round
  the result: their alternating sums and prefix recoveries would be unstable
  in floating point.
* Banzhaf in float64 works for any number of players. The row readings
  (Shapley–Shubik, Deegan–Packel, Johnston) accept up to 1000 players in
  float64, beyond which their coefficients leave the float range; exact
  arithmetic has no such limit.

On diagrams with $10^5$ nodes and more, Banzhaf in float64 is about ten
times faster than exact, the row readings two to three times. For small
games the difference is negligible.

## Minimum quota

```python
idb = Game.weighted(idb_votes, quota=3_631_155)   # Inter-American Development Bank 1998
result = idb.minimum_quota()
result.quota          # 394
result.weights        # integer weights with [394; weights] = the same game
result.lower_bound, result.upper_bound, result.optimal, result.stage

idb.minimal()         # the same game with quota 394 ...
idb.minimal().johnston().dp()    # ... quota-indexed methods become fast
```

The computation needs no LP solver. Coalitions that are tight at the given
weights are harvested from the winning QOBDD and turned into an exact lower
bound; otherwise an exact simplex method moves on the polyhedron of all
representations, with the ratio test as a dynamic program on the QOBDD and
the inverse kept as an integer adjugate. When the LP bound and the best
integer representation still differ (an *integer gap*, e.g. the IMF game of
2002), all cheaper integer slack combinations of a positive basis are
enumerated (games with at most 32 players). Floating point only proposes
candidates; every bound is checked with exact integers.

## Version 2

Version 2 is a complete rewrite and not compatible with 1.x: the module
`power_bdd.bdd` with its classes `BDD` and `WeightedGame` has been removed.

## References

* S. Bolus, Power indices of simple games and vector-weighted majority
  games by means of binary decision diagrams, European Journal of
  Operational Research 210 (2011) 258–272.
* S. Bolus, A QOBDD-based approach to simple games, dissertation,
  Kiel University (2012).
* N. Kakimura, Y. Terai, Computing power indices in weighted majority games
  with formal power series, WALCOM 2026.
* J. Staudacher et al., Computing power indices for weighted voting games
  via dynamic programming, Operations Research and Decisions 31 (2021).
* L. M. Armijos-Toro et al., On generating functions to compute some power
  measures for weighted majority games, Annals of Operations Research (2026).
* T. Uno, Efficient computation of power indices for weighted majority
  games, ISAAC 2012.

## Development

    pip install -e ".[test,flint]"
    pytest                     # or: python -m unittest discover -s tests

The test suite covers a small example, the UN Security Council, the
Council of the EU with exact values for all members, real games and
benchmark instances for the minimum quota, and compares all methods with
exhaustive enumeration on random weighted and multi-tier games.
