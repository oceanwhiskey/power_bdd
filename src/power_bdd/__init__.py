"""Exact power indices of simple games via QOBDDs and dynamic programming.

Quick start::

    from power_bdd import Game

    game = Game.weighted([4, 3, 2, 1], quota=6)
    game.johnston().auto()          # choose index, then method (or "auto")
    game.banzhaf().kakimura()
    game.minimum_quota()

See :mod:`power_bdd.game` for all constructors and methods.
"""
from .errors import MethodNotApplicable, NotCompleteError, PowerBDDError
from .game import CompositeGame, Game, WeightedGame
from .results import MinimumQuotaResult, PowerIndex

__version__ = "2.0.0"

__all__ = [
    "Game",
    "WeightedGame",
    "CompositeGame",
    "PowerIndex",
    "MinimumQuotaResult",
    "PowerBDDError",
    "MethodNotApplicable",
    "NotCompleteError",
    "__version__",
]
