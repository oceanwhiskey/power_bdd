"""Simple games and the fluent API.

Create a game, then choose an index, then a method::

    from power_bdd import Game

    ec = Game.weighted([55, 38, 29, ...], quota=270)
    ec.banzhaf().auto()                 # method chosen by a heuristic
    ec.johnston().dp()                  # aggregated O(n^2 q) dynamic program
    ec.minimum_quota()                  # exact minimum quota

    eu = Game.at_least(15, of=27) & (Game.weighted(population, share="65%")
                                     | Game.at_least(24, of=27))
    eu.named(members).johnston().qobdd()

Single weighted games (:class:`WeightedGame`) offer every method.
Multi-tier games (:class:`CompositeGame`, AND/OR of games) are evaluated on
their QOBDD only, so their selectors have no ``kakimura()`` or ``dp()`` and
they have no ``minimum_quota()``.  Johnston and the last-critical
Deegan-Packel method need a complete game: ``game.johnston()`` raises
:class:`NotCompleteError` right away for any other game.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, TypeVar

from . import _diagram as D
from . import _qobdd_indices as Q
from . import _weight_methods as W
from ._minimum_quota import MinimumQuotaResult, minimum_quota as _minimum_quota
from ._numbers import integer_game, to_fraction
from .errors import MethodNotApplicable, NotCompleteError
from .results import PowerIndex

__all__ = ["Game", "WeightedGame", "CompositeGame"]

_HAS_FLINT = W.HAS_FLINT      # python-flint installed and recent enough

#: auto uses Kakimura-Terai for Banzhaf below this n*q (with python-flint)
KAKIMURA_NQ_LIMIT = 2_000_000

G = TypeVar("G", bound="Game")


def _parse_share(share, floats: str = "decimal") -> Fraction:
    if isinstance(share, str):
        text = share.strip()
        if text.endswith("%"):
            return Fraction(text[:-1].strip()) / 100
        return Fraction(text)
    return to_fraction(share, floats)


class Game:
    """A monotone simple game on players ``0..n-1`` (optionally named)."""

    def __init__(self, n: int, players: Optional[Sequence[Hashable]] = None):
        if n < 1:
            raise ValueError("a game needs at least one player")
        self._n = n
        if players is not None:
            players = tuple(players)
            if len(players) != n:
                raise ValueError(f"expected {n} player names, got {len(players)}")
            if len(set(players)) != n:
                raise ValueError("player names must be distinct")
        self._players = players
        self._cache: Dict = {}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @staticmethod
    def weighted(weights: Sequence, quota=None, *, share=None,
                 players: Optional[Sequence[Hashable]] = None,
                 floats: str = "decimal") -> "WeightedGame":
        """Weighted majority game: a coalition wins if its weight reaches the quota.

        Weights and quota may be any real numbers: int, float, Fraction,
        Decimal, numeric strings or numpy numbers.  Give either ``quota`` or
        ``share``, a fraction of the total weight (``"65%"``, ``0.5``,
        ``Fraction(2, 3)``; "at least 65 %").

        Non-integer games are scaled exactly to integer weights (see
        :attr:`WeightedGame.scale`).  Floats are read as written
        (``floats="decimal"``: 0.1 is 1/10); ``floats="binary"`` uses the
        binary value a float stores instead.
        """
        weights = list(weights)
        if (quota is None) == (share is None):
            raise ValueError("give exactly one of quota or share")
        if quota is None:
            total = sum(to_fraction(w, floats) for w in weights)
            quota = _parse_share(share, floats) * total
        return WeightedGame(quota, weights, players, floats=floats)

    @staticmethod
    def at_least(k: int, of: int, *, players=None) -> "WeightedGame":
        """At least ``k`` of ``of`` players must agree (all players equal)."""
        return WeightedGame(int(k), [1] * int(of), players)

    @staticmethod
    def majority(n: int, *, players=None) -> "WeightedGame":
        """Simple majority: more than half of the ``n`` players."""
        return WeightedGame(n // 2 + 1, [1] * n, players)

    @staticmethod
    def unanimity(n: int, *, players=None) -> "WeightedGame":
        return WeightedGame(n, [1] * n, players)

    @staticmethod
    def all_of(*games: "Game") -> "CompositeGame":
        """Coalitions that win every game (conjunction of tiers)."""
        return CompositeGame("and", games)

    @staticmethod
    def any_of(*games: "Game") -> "CompositeGame":
        """Coalitions that win at least one game (disjunction of tiers)."""
        return CompositeGame("or", games)

    @staticmethod
    def from_minimal_winning(n: int, coalitions: Iterable[Iterable[int]], *,
                             players=None) -> "Game":
        """Monotone game given by its minimal winning coalitions."""
        leaves = []
        for coalition in coalitions:
            members = set(int(p) for p in coalition)
            if not members or min(members) < 0 or max(members) >= n:
                raise ValueError(f"invalid coalition {sorted(members)}")
            leaves.append(WeightedGame(len(members), [1 if p in members else 0 for p in range(n)]))
        if not leaves:
            raise ValueError("at least one winning coalition is required")
        game = leaves[0] if len(leaves) == 1 else CompositeGame("or", leaves)
        return game.named(players) if players is not None else game

    def __and__(self, other: "Game") -> "CompositeGame":
        return CompositeGame("and", (self, other))

    def __or__(self, other: "Game") -> "CompositeGame":
        return CompositeGame("or", (self, other))

    def and_(self, *others: "Game") -> "CompositeGame":
        return CompositeGame("and", (self,) + others)

    def or_(self, *others: "Game") -> "CompositeGame":
        return CompositeGame("or", (self,) + others)

    def named(self: G, players: Sequence[Hashable]) -> G:
        """The same game with player names."""
        clone = self._clone()
        Game.__init__(clone, self._n, players)
        return clone

    def _clone(self: G) -> G:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self._n

    @property
    def players(self) -> Tuple[Hashable, ...]:
        return self._players if self._players is not None else tuple(range(self._n))

    @property
    def is_weighted(self) -> bool:
        """True for a single weighted game (all weight-based methods apply)."""
        return False

    def is_winning(self, coalition: Iterable) -> bool:
        index = {p: i for i, p in enumerate(self.players)}
        members = [index[p] if p in index else int(p) for p in coalition]
        return self.diagram().evaluate(members)

    def is_complete(self) -> bool:
        """True if the desirability relation of the players is total.

        Decided from the weights alone when all tiers order the players the
        same way (no diagram needed), else exactly on the winning QOBDD.
        """
        if "complete" not in self._cache:
            if self._complete_by_weights():
                self._cache["complete"] = True
            else:
                self._prepare()
        return self._cache["complete"]

    def _complete_by_weights(self) -> bool:
        return False

    def diagram(self) -> D.LevelDiagram:
        """The winning QOBDD (players in desirability order if complete)."""
        self._prepare()
        return self._cache["diagram"]

    @property
    def qobdd_size(self) -> int:
        """Number of inner nodes M of the winning QOBDD."""
        return self.diagram().inner_nodes

    def _build(self, order: Sequence[int]) -> D.LevelDiagram:
        raise NotImplementedError

    def _initial_order(self) -> List[int]:
        raise NotImplementedError

    def _prepare(self) -> None:
        if "diagram" in self._cache:
            return
        order = self._initial_order()
        diagram = self._build(order)
        # with a common weight order the initial order is a desirability order
        complete = self._complete_by_weights() or D.is_desirability_ordered(diagram)
        if not complete:
            swings = Q.banzhaf_swings(diagram)
            position = {p: i for i, p in enumerate(order)}
            by_swings = sorted(order, key=lambda p: (-swings[position[p]], position[p]))
            if by_swings != order:
                candidate = self._build(by_swings)
                if D.is_desirability_ordered(candidate):
                    diagram, complete = candidate, True
        self._cache["diagram"] = diagram
        self._cache["complete"] = complete

    def _minwin(self) -> D.LevelDiagram:
        if "minwin" not in self._cache:
            self._cache["minwin"] = D.minimal_winning_diagram(self.diagram(), self.is_complete())
        return self._cache["minwin"]

    def _to_players(self, level_values) -> List:
        order = self.diagram().order
        result = [None] * self._n
        for level, player in enumerate(order):
            result[player] = level_values[level]
        return result

    # ------------------------------------------------------------------
    # fluent index selection
    # ------------------------------------------------------------------
    def banzhaf(self) -> "BanzhafSelector":
        """Absolute Banzhaf index beta_i = B_i / 2^(n-1)."""
        return BanzhafSelector(self)

    def shapley_shubik(self) -> "ShapleyShubikSelector":
        return ShapleyShubikSelector(self)

    def deegan_packel(self) -> "DeeganPackelSelector":
        return DeeganPackelSelector(self)

    def johnston(self) -> "JohnstonSelector":
        """Johnston index; raises :class:`NotCompleteError` unless the game is complete."""
        if not self.is_complete():
            raise _not_complete("johnston")
        return JohnstonSelector(self)

    def index(self, name: str) -> "_Selector":
        """Selector by name: banzhaf, shapley_shubik, deegan_packel, johnston."""
        key = name.lower().replace("-", "_").replace(" ", "_")
        aliases = {"shapley": "shapley_shubik", "ss": "shapley_shubik",
                   "dp": "deegan_packel", "deeganpackel": "deegan_packel",
                   "penrose": "banzhaf", "banzhaf_penrose": "banzhaf"}
        key = aliases.get(key, key)
        indices = ("banzhaf", "shapley_shubik", "deegan_packel", "johnston")
        if key not in indices:
            raise ValueError(f"unknown power index {name!r}; choose one of {list(indices)}")
        return getattr(self, key)()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} n={self._n}>"


class WeightedGame(Game):
    """Weighted majority game ``[quota; w_1, ..., w_n]``.

    ``weights`` and ``quota`` are integers.  For non-integer input they are
    the exactly scaled equivalent: ``weights[i] == original_weights[i] *
    scale`` and ``quota`` is the smallest integer at least
    ``original_quota * scale``; the winning coalitions are the same.
    """

    def __init__(self, quota, weights: Sequence, players=None, *, floats: str = "decimal"):
        original_weights = tuple(weights)
        integer_weights, integer_quota, scale = integer_game(original_weights, quota, floats)
        super().__init__(len(integer_weights), players)
        self.original_weights = original_weights
        self.original_quota = quota
        self.scale = scale
        self.quota = integer_quota
        self.weights = tuple(integer_weights)

    def _clone(self) -> "WeightedGame":
        clone = WeightedGame(self.quota, self.weights)
        clone.original_weights, clone.original_quota = self.original_weights, self.original_quota
        clone.scale = self.scale
        return clone

    @property
    def is_weighted(self) -> bool:
        return True

    def _complete_by_weights(self) -> bool:
        return True                             # weight order = desirability order

    # selectors with the weight-based methods
    def banzhaf(self) -> "WeightedBanzhafSelector":
        """Absolute Banzhaf index beta_i = B_i / 2^(n-1)."""
        return WeightedBanzhafSelector(self)

    def shapley_shubik(self) -> "WeightedShapleyShubikSelector":
        return WeightedShapleyShubikSelector(self)

    def johnston(self) -> "WeightedJohnstonSelector":
        """Johnston index (every weighted game is complete)."""
        return WeightedJohnstonSelector(self)

    def _initial_order(self) -> List[int]:
        return sorted(range(self._n), key=lambda p: (-self.weights[p], p))

    def _build(self, order):
        return D.weighted_diagram(self.quota, self.weights, order)

    def _prepare(self) -> None:
        if "diagram" not in self._cache:
            self._cache["diagram"] = self._build(self._initial_order())
            self._cache["complete"] = True      # weight order = desirability order

    def minimum_quota(self) -> MinimumQuotaResult:
        """Exact minimum quota of any integer representation (no LP solver)."""
        if "minimum_quota" not in self._cache:
            self._cache["minimum_quota"] = _minimum_quota(self.quota, self.weights)
        return self._cache["minimum_quota"]

    def minimal(self) -> "WeightedGame":
        """The same game with the smallest possible quota.

        Quota-indexed methods (Kakimura-Terai, the Johnston DP) take time
        proportional to the quota, so they profit directly: the
        Inter-American Development Bank game of 1998 has quota 3 631 155
        but minimum quota 394.  If the minimum quota is not proven, the best
        representation found is used; it is always the same game.
        """
        if self.quota < 1 or self.quota > sum(self.weights):
            return self
        result = self.minimum_quota()
        if result.upper_bound >= self.quota:
            return self
        game = WeightedGame(result.upper_bound, result.weights, self._players)
        game._cache["diagram"] = self.diagram()      # same game, same order
        game._cache["complete"] = True
        return game

    def __repr__(self) -> str:
        shown = ", ".join(map(str, self.weights[:8])) + (", ..." if self._n > 8 else "")
        return f"<WeightedGame [{self.quota}; {shown}]>"


class CompositeGame(Game):
    """AND/OR combination of games on the same players (multi-tier game)."""

    def __init__(self, operation: str, games: Sequence[Game], players=None):
        games = tuple(games)
        if operation not in ("and", "or"):
            raise ValueError(operation)
        if not games:
            raise ValueError("at least one game is required")
        n = games[0].n
        if any(g.n != n for g in games):
            raise ValueError("all tiers must have the same number of players")
        if players is None:
            named = [g._players for g in games if g._players is not None]
            players = named[0] if named else None
        super().__init__(n, players)
        # flatten nested combinations with the same operation
        flat = []
        for g in games:
            if isinstance(g, CompositeGame) and g.operation == operation:
                flat.extend(g.games)
            else:
                flat.append(g)
        self.operation = operation
        self.games = tuple(flat)

    def _clone(self) -> "CompositeGame":
        return CompositeGame(self.operation, self.games)

    def __getattr__(self, name: str):
        if name in ("minimum_quota", "minimal"):
            raise AttributeError(f"{name}() is defined for a single weighted game, "
                                 "not for a multi-tier game")
        raise AttributeError(f"'CompositeGame' object has no attribute {name!r}")

    def _leaves(self) -> List[WeightedGame]:
        leaves = []
        for g in self.games:
            leaves.extend(g._leaves() if isinstance(g, CompositeGame) else [g])
        return leaves

    def _initial_order(self) -> List[int]:
        leaves = self._leaves()
        return sorted(range(self._n), key=lambda p: (tuple(-leaf.weights[p] for leaf in leaves), p))

    def _complete_by_weights(self) -> bool:
        """True if one order of the players is weakly decreasing in every tier.

        Then every tier ranks i at least as desirable as j for i before j,
        AND and OR keep that relation, and the game is complete.  The
        converse fails (the two-tier UN Security Council is complete with
        opposed weights), so False means "check on the diagram".
        """
        if "comonotone" not in self._cache:
            order = self._initial_order()        # lexicographic: a common order if one exists
            self._cache["comonotone"] = all(
                leaf.weights[a] >= leaf.weights[b]
                for leaf in self._leaves() for a, b in zip(order, order[1:]))
        return self._cache["comonotone"]

    def _build(self, order):
        diagrams = [g._build(order) for g in self.games]
        result = diagrams[0]
        for other in diagrams[1:]:
            result = D.combine(result, other, self.operation)
        return result

    def __repr__(self) -> str:
        return f"<CompositeGame {self.operation.upper()} of {len(self.games)} tiers, n={self._n}>"


# ---------------------------------------------------------------------------
# index selectors
# ---------------------------------------------------------------------------
#
# Every index has a selector for any game (QOBDD methods only) and, where
# weight-based methods exist, a subclass for single weighted games that adds
# them.  Game.banzhaf() returns the former, WeightedGame.banzhaf() the
# latter, so editors and type checkers offer kakimura() and dp() only where
# they apply.

#: Quota-indexed methods (Kakimura-Terai, Johnston DP) refuse games with
#: n * quota above this value; use ``game.minimal()`` or the QOBDD instead.
QUOTA_INDEXED_LIMIT = 50_000_000

S = TypeVar("S", bound="_Selector")


class _Selector:
    """Chosen power index of a game; next choose a method (or ``auto``).

    ``.floats()`` switches to float64 results, ``.exact()`` back to exact
    fractions (the default).
    """

    index = ""
    methods: Tuple[str, ...] = ()
    #: methods that only the selector of a single weighted game has
    weighted_methods: Tuple[str, ...] = ()
    quota_indexed: Tuple[str, ...] = ()

    def __init__(self, game: Game, floating: bool = False):
        self.game = game
        self.floating = floating

    def floats(self: S) -> S:
        """Compute in float64: faster and leaner, with rounding errors.

        The QOBDD methods run in float64 (rows of nonnegative numbers); the
        quota-indexed methods compute exactly and round, because their
        alternating sums or prefix recoveries would be unstable in floats.
        """
        return type(self)(self.game, floating=True)

    def exact(self: S) -> S:
        """Compute with exact fractions (the default)."""
        return type(self)(self.game, floating=False)

    def available(self) -> Tuple[str, ...]:
        """Methods that apply to this game (``auto`` included)."""
        return ("auto",) + tuple(m for m in self.methods if self._why_not(m) is None)

    def choose(self) -> str:
        """The method that :meth:`auto` would use."""
        return "qobdd"

    def auto(self) -> PowerIndex:
        """Compute with the method chosen by the heuristic (see :meth:`choose`)."""
        return self.method(self.choose(), _auto=True)

    def method(self, name: str, _auto: bool = False) -> PowerIndex:
        """Compute with the method of the given name (or ``"auto"``)."""
        name = name.lower()
        if name == "auto":
            return self.auto()
        if name not in self.methods:
            if name in self.weighted_methods:
                raise MethodNotApplicable(
                    f"{self.index} via {name!r} needs a single weighted game; "
                    "multi-tier games are evaluated on their QOBDD")
            raise ValueError(f"{self.index}: unknown method {name!r}; choose one of "
                             f"{('auto',) + self.methods}")
        reason = self._why_not(name)
        if reason is not None:
            raise reason
        values, raw, info = getattr(self, "_compute_" + name)()
        info = dict(info, chosen_automatically=_auto)
        if self.floating:
            if info.get("arithmetic") != "float64":
                info["arithmetic"] = "exact, rounded to float64"
            values = [float(v) for v in values]
            raw = None if raw is None else [float(r) for r in raw]
        else:
            info["arithmetic"] = "exact"
        return PowerIndex(self.index, name, self.game.players, tuple(values),
                          None if raw is None else tuple(raw), info)

    def _why_not(self, name: str) -> Optional[Exception]:
        if name in self.quota_indexed:
            reason = _small_quota(self.game, name, self.index)
            if reason is not None:
                return reason
        return self._specific_reason(name)

    def _specific_reason(self, name: str) -> Optional[Exception]:
        return None

    def __getattr__(self, name: str):
        # only reached for missing attributes; keeps hidden methods out of
        # dir() and completion but explains them when called anyway
        if name in type(self).weighted_methods:
            raise AttributeError(
                f"{self.index} has no method {name}() for this game: it needs a single "
                f"weighted game; multi-tier games use qobdd()")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __repr__(self) -> str:
        mode = "float64" if self.floating else "exact"
        return f"<{self.index} ({mode}) of {self.game!r}: methods {self.available()}>"


def _small_quota(game: "WeightedGame", method: str, index: str) -> Optional[Exception]:
    size = game.n * max(game.quota, 1)
    if size > QUOTA_INDEXED_LIMIT:
        return MethodNotApplicable(
            f"{index} via {method!r} works on tables of n * quota = {size:,} entries "
            f"(limit {QUOTA_INDEXED_LIMIT:,}); use game.minimal() to compute with the "
            "minimum quota, or the qobdd method")
    return None


def _not_complete(index: str, method: Optional[str] = None) -> NotCompleteError:
    via = f" via {method!r}" if method else ""
    return NotCompleteError(
        f"{index}{via} needs a complete simple game; this game has players that "
        "are not comparable by desirability")


def _float_shares(numerators):
    total = sum(numerators)
    return [x / total if total else 0.0 for x in numerators]


# --- Banzhaf ----------------------------------------------------------------

class BanzhafSelector(_Selector):
    """Absolute Banzhaf index beta_i = B_i / 2^(n-1); raw = swing counts B_i."""

    index = "banzhaf"
    methods = ("qobdd",)
    weighted_methods = ("kakimura",)

    def qobdd(self) -> PowerIndex:
        """Swing counts on the winning QOBDD, O(M) per prime (any game)."""
        return self.method("qobdd")

    def _finish(self, swings, info):
        denominator = 2 ** (self.game.n - 1)
        return [Fraction(b, denominator) for b in swings], swings, info

    def _compute_qobdd(self):
        if self.floating:
            values = self.game._to_players(Q.banzhaf_float(self.game.diagram()))
            return values, None, {"qobdd_nodes": self.game.qobdd_size, "arithmetic": "float64"}
        swings = self.game._to_players(Q.banzhaf_swings(self.game.diagram()))
        return self._finish(swings, {"qobdd_nodes": self.game.qobdd_size})


class WeightedBanzhafSelector(BanzhafSelector):
    """Banzhaf index of a single weighted game: also Kakimura-Terai."""

    game: "WeightedGame"
    methods = ("qobdd", "kakimura")
    quota_indexed = ("kakimura",)

    def choose(self) -> str:
        """Kakimura-Terai for small n*q when python-flint is installed, no
        diagram exists yet and results are exact; the QOBDD otherwise.

        Measured on benchmark families: the power series win for random
        weights (n*q up to a few 10^5), the diagram wins once the quota is
        large compared with the structure of the game.
        """
        g = self.game
        if (_HAS_FLINT and not self.floating and "diagram" not in g._cache
                and g.n * max(g.quota, 1) <= KAKIMURA_NQ_LIMIT):
            return "kakimura"
        return "qobdd"

    def kakimura(self) -> PowerIndex:
        """Formal power series of Kakimura and Terai."""
        return self.method("kakimura")

    def _compute_kakimura(self):
        g = self.game
        positive = [p for p in range(g.n) if g.weights[p] > 0]
        swings = [0] * g.n
        if positive:
            values = W.kakimura_banzhaf(g.quota, [g.weights[p] for p in positive])
            dummies = g.n - len(positive)        # each dummy doubles every swing count
            for p, b in zip(positive, values):
                swings[p] = b << dummies
        return self._finish(swings, {"flint": _HAS_FLINT})


# --- Shapley-Shubik ---------------------------------------------------------

class ShapleyShubikSelector(_Selector):
    """Shapley-Shubik index; raw = numbers of pivotal orderings n! phi_i."""

    index = "shapley_shubik"
    methods = ("qobdd",)
    weighted_methods = ("kakimura",)

    def qobdd(self) -> PowerIndex:
        """Weighted residual rows on the winning QOBDD, O(nM) (any monotone game)."""
        return self.method("qobdd")

    def _finish(self, values, info):
        total = factorial(self.game.n)
        return values, [int(v * total) for v in values], info

    def _compute_qobdd(self):
        if self.floating:
            values = self.game._to_players(Q.row_readings(self.game.diagram(), "shapley", True))
            return values, None, {"qobdd_nodes": self.game.qobdd_size, "arithmetic": "float64"}
        values = self.game._to_players(Q.shapley_values(self.game.diagram()))
        return self._finish(values, {"qobdd_nodes": self.game.qobdd_size})


class WeightedShapleyShubikSelector(ShapleyShubikSelector):
    """Shapley-Shubik index of a single weighted game: also Kakimura-Terai."""

    game: "WeightedGame"
    methods = ("qobdd", "kakimura")
    quota_indexed = ("kakimura",)

    def kakimura(self) -> PowerIndex:
        """Formal power series of Kakimura and Terai."""
        return self.method("kakimura")

    def _compute_kakimura(self):
        g = self.game
        positive = [p for p in range(g.n) if g.weights[p] > 0]
        values = [Fraction(0)] * g.n
        if positive:
            sub = W.kakimura_shapley(g.quota, [g.weights[p] for p in positive])
            # removing dummies keeps every Shapley-Shubik value
            for p, v in zip(positive, sub):
                values[p] = v
        return self._finish(values, {"flint": _HAS_FLINT})


# --- Deegan-Packel ----------------------------------------------------------

class DeeganPackelSelector(_Selector):
    """Deegan-Packel index; raw = sum over minimal winning S of 1/|S|.

    ``qobdd`` needs a complete game and raises :class:`NotCompleteError`
    otherwise; ``minwin`` applies to every monotone game, and ``auto``
    chooses between them.
    """

    index = "deegan_packel"
    methods = ("qobdd", "minwin")

    def _specific_reason(self, name):
        if name == "qobdd" and not self.game.is_complete():
            return _not_complete(self.index, name)
        return None

    def choose(self) -> str:
        return "qobdd" if self.game.is_complete() else "minwin"

    def qobdd(self) -> PowerIndex:
        """Last-critical-player evaluation on the winning QOBDD, O(nM) (complete games)."""
        return self.method("qobdd")

    def minwin(self) -> PowerIndex:
        """Weighted rows on the minimal-winning QOBDD (Bolus), any monotone game."""
        return self.method("minwin")

    def _finish(self, diagram, mode, info):
        if self.floating:
            raw = self.game._to_players(Q.row_readings(diagram, mode, True))
            return _float_shares(raw), raw, dict(info, arithmetic="float64")
        numerators = self.game._to_players(Q.row_readings(diagram, mode))
        L = Q._lcm_upto(self.game.n)
        return Q.normalized_shares(numerators), [Fraction(x, L) for x in numerators], info

    def _compute_qobdd(self):
        return self._finish(self.game.diagram(), "deegan_packel",
                            {"qobdd_nodes": self.game.qobdd_size})

    def _compute_minwin(self):
        minwin = self.game._minwin()
        return self._finish(minwin, "member", {"minwin_nodes": minwin.inner_nodes})


# --- Johnston ---------------------------------------------------------------

class JohnstonSelector(_Selector):
    """Johnston index of a complete game; raw = sum over winning S with i
    critical of 1/|Crit(S)|.

    Only complete games get this selector: ``game.johnston()`` raises
    :class:`NotCompleteError` for any other game.
    """

    index = "johnston"
    methods = ("qobdd",)
    weighted_methods = ("dp",)

    def qobdd(self) -> PowerIndex:
        """Last-critical-player evaluation on the winning QOBDD, O(nM)."""
        return self.method("qobdd")

    def _compute_qobdd(self):
        if self.floating:
            raw = self.game._to_players(Q.row_readings(self.game.diagram(), "johnston", True))
            return _float_shares(raw), raw, {"qobdd_nodes": self.game.qobdd_size,
                                             "arithmetic": "float64"}
        numerators = Q.row_readings(self.game.diagram(), "johnston")
        L = Q._lcm_upto(self.game.n)
        players = self.game._to_players(numerators)
        return (Q.normalized_shares(players), [Fraction(x, L) for x in players],
                {"qobdd_nodes": self.game.qobdd_size})


class WeightedJohnstonSelector(JohnstonSelector):
    """Johnston index of a single weighted game: also the aggregated DP."""

    game: "WeightedGame"
    methods = ("qobdd", "dp")
    quota_indexed = ("dp",)

    def dp(self) -> PowerIndex:
        """Aggregated Johnston dynamic program, O(n^2 q) time and O(nq) storage."""
        return self.method("dp")

    def _compute_dp(self):
        g = self.game
        raw = W.johnston_dp(g.quota, g.weights)
        total = sum(raw)
        return [r / total if total else Fraction(0) for r in raw], raw, {}
