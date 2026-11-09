"""Result objects returned by the fluent API."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Hashable, Iterator, List, Optional, Tuple

from ._minimum_quota import MinimumQuotaResult

__all__ = ["PowerIndex", "MinimumQuotaResult"]

_TITLES = {
    "banzhaf": "Banzhaf (absolute)",
    "shapley_shubik": "Shapley-Shubik",
    "deegan_packel": "Deegan-Packel",
    "johnston": "Johnston",
}


@dataclass(frozen=True)
class PowerIndex:
    """Exact values of one power index for all players.

    ``values`` are exact fractions in player order.  For Banzhaf they are
    the absolute index beta_i = B_i / 2^(n-1); :meth:`normalized` gives
    shares summing to one.  The other indices already sum to one (unless
    no player has power).  ``raw`` holds the unnormalized quantities:
    swing counts B_i (Banzhaf), pivotal orderings n! phi_i
    (Shapley-Shubik), raw Johnston and Deegan-Packel values.
    """

    index: str
    method: str
    players: Tuple[Hashable, ...]
    values: Tuple[Fraction, ...]
    raw: Optional[Tuple] = None
    info: Dict = field(default_factory=dict, compare=False)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[Fraction]:
        return iter(self.values)

    def __getitem__(self, player) -> Fraction:
        """Value of a player, by name (if the game has names) or position."""
        if isinstance(player, int) and not isinstance(player, bool) and player not in self.players:
            return self.values[player]
        try:
            return self.values[self.players.index(player)]
        except ValueError:
            raise KeyError(player) from None

    def as_dict(self) -> Dict[Hashable, Fraction]:
        return dict(zip(self.players, self.values))

    def to_floats(self) -> List[float]:
        return [float(v) for v in self.values]

    def normalized(self) -> "PowerIndex":
        """Shares summing to one (useful for Banzhaf)."""
        total = sum(self.values)
        values = tuple(v / total if total else Fraction(0) for v in self.values)
        return PowerIndex(self.index, self.method, self.players, values, self.raw,
                          dict(self.info, normalized=True))

    def table(self, digits: int = 4, percent: bool = True) -> str:
        """Plain-text table of all players."""
        width = max([len(str(p)) for p in self.players] + [6])
        title = _TITLES.get(self.index, self.index)
        if self.info.get("normalized"):
            title += ", normalized"
        lines = [f"{title} via {self.method}", f"{'player':<{width}}  value"]
        for p, v in zip(self.players, self.values):
            shown = f"{100 * float(v):.{digits}f} %" if percent else f"{float(v):.{digits + 2}f}"
            lines.append(f"{str(p):<{width}}  {shown}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()

