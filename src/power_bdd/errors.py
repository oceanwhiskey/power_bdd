"""Exceptions of power_bdd."""


class PowerBDDError(Exception):
    """Base class of all errors raised by power_bdd."""


class MethodNotApplicable(PowerBDDError, ValueError):
    """The chosen method cannot evaluate this game (e.g. a weight-based
    method on a multi-tier game)."""


class NotCompleteError(MethodNotApplicable):
    """The index needs a complete simple game (a total desirability order),
    e.g. Johnston, but the game is not complete."""
