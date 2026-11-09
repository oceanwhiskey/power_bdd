"""Example games: a small weighted game, the UN Security Council, the
Council of the EU, real games and benchmark instances with known minimum
quotas, and a multi-tier game that is not complete.
"""
import unittest
from fractions import Fraction

from power_bdd import Game, MethodNotApplicable, NotCompleteError

from .helpers import brute_force, load, same_game, weighted_predicate

F = Fraction

ALL_METHODS = {
    "banzhaf": ("qobdd", "kakimura"),
    "shapley_shubik": ("qobdd", "kakimura"),
    "deegan_packel": ("qobdd", "minwin"),
    "johnston": ("qobdd", "dp"),
}


def minimum_quota_games():
    return {g["id"]: g for g in load("minimum_quotas.json")["games"]}


class SmallExample(unittest.TestCase):
    """[5; 3, 2, 1, 1]."""

    expected = {
        "deegan_packel": [F(5, 12), F(1, 4), F(1, 6), F(1, 6)],
        "johnston": [F(17, 30), F(3, 10), F(1, 15), F(1, 15)],
        "banzhaf": [F(5, 8), F(3, 8), F(1, 8), F(1, 8)],
        "shapley_shubik": [F(7, 12), F(1, 4), F(1, 12), F(1, 12)],
    }
    raw = {
        "deegan_packel": [F(5, 6), F(1, 2), F(1, 3), F(1, 3)],
        "johnston": [F(17, 6), F(3, 2), F(1, 3), F(1, 3)],
    }

    def setUp(self):
        self.game = Game.weighted([3, 2, 1, 1], quota=5)

    def test_all_indices_all_methods(self):
        for index, methods in ALL_METHODS.items():
            for method in ("auto",) + methods:
                result = self.game.index(index).method(method)
                self.assertEqual(list(result.values), self.expected[index], (index, method))
                if index in self.raw:
                    self.assertEqual(list(result.raw), self.raw[index], (index, method))

    def test_minimum_quota(self):
        result = self.game.minimum_quota()
        self.assertTrue(result.optimal)
        self.assertEqual(result.quota, 5)

    def test_qobdd(self):
        self.assertTrue(self.game.is_complete())
        self.assertEqual(self.game.qobdd_size, 9)


class UnitedNationsSecurityCouncil(unittest.TestCase):
    """All five permanent and at least four of the ten elected members."""

    def test_ratios_permanent_to_elected(self):
        unsc = Game.weighted([7] * 5 + [1] * 10, quota=39)
        expected = {"shapley_shubik": F(421, 4), "banzhaf": F(212, 21),
                    "johnston": F(566, 35), "deegan_packel": F(5, 2)}
        for index, ratio in expected.items():
            for method in ALL_METHODS[index]:
                result = unsc.index(index).method(method)
                self.assertEqual(result[0] / result[5], ratio, (index, method))

    def test_same_game_as_a_two_tier_rule(self):
        permanent = Game.weighted([1] * 5 + [0] * 10, quota=5)
        elected = Game.weighted([0] * 5 + [1] * 10, quota=4)
        two_tier = permanent & elected
        unsc = Game.weighted([7] * 5 + [1] * 10, quota=39)
        self.assertTrue(two_tier.is_complete())
        for index in ALL_METHODS:
            self.assertEqual(two_tier.index(index).auto().values, unsc.index(index).auto().values)


class CouncilOfTheEU(unittest.TestCase):
    """Qualified majority 2026 (exact values)."""

    @classmethod
    def setUpClass(cls):
        data = load("eu27_2026.json")
        cls.data = data
        population = data["population"]
        cls.council = (Game.at_least(15, of=27)
                       & (Game.weighted(population, share="65%")
                          | Game.at_least(24, of=27))).named(data["members"])

    def test_rule(self):
        self.assertEqual(Game.weighted(self.data["population"], share="65%").quota,
                         self.data["rule"]["q"])

    def test_complete_and_diagram_size(self):
        self.assertTrue(self.council.is_complete())
        self.assertEqual(self.council.qobdd_size, self.data["qobdd_inner_nodes"])

    def test_exact_values_of_all_members(self):
        expected = {
            "banzhaf": [F(x) for x in self.data["banzhaf_swing_fraction"]],
            "shapley_shubik": [F(x) for x in self.data["shapley"]],
            "johnston": [F(x) for x in self.data["johnston"]],
            "deegan_packel": [F(x) for x in self.data["deegan_packel"]],
        }
        for index, values in expected.items():
            methods = ("auto", "qobdd") + (("minwin",) if index == "deegan_packel" else ())
            for method in methods:
                result = self.council.index(index).method(method)
                self.assertEqual(list(result.values), values, (index, method))

    def test_published_percentages(self):
        johnston = self.council.johnston().auto()
        self.assertEqual(round(100 * float(johnston["Germany"]), 3), 24.808)
        self.assertEqual(round(100 * float(self.council.deegan_packel().auto()["Malta"]), 3), 3.085)

    def test_weight_methods_are_not_offered(self):
        self.assertFalse(hasattr(self.council.johnston(), "dp"))
        with self.assertRaises(MethodNotApplicable):
            self.council.johnston().method("dp")
        self.assertEqual(self.council.banzhaf().available(), ("auto", "qobdd"))


class NonCompleteDisjunction(unittest.TestCase):
    """[2; 1,1,0,0] OR [2; 0,0,1,1]: weighted tiers, not complete."""

    def setUp(self):
        self.game = Game.weighted([1, 1, 0, 0], quota=2) | Game.weighted([0, 0, 1, 1], quota=2)

    def test_not_complete(self):
        self.assertFalse(self.game.is_complete())
        with self.assertRaises(NotCompleteError):
            self.game.johnston()
        with self.assertRaises(NotCompleteError):
            self.game.deegan_packel().qobdd()

    def test_other_indices(self):
        reference = brute_force(4, self.game.is_winning)
        self.assertEqual(self.game.deegan_packel().auto().method, "minwin")
        self.assertEqual(list(self.game.deegan_packel().auto().values), [F(1, 4)] * 4)
        self.assertEqual(list(self.game.banzhaf().auto().values), reference["banzhaf"])
        self.assertEqual(list(self.game.shapley_shubik().auto().values), reference["shapley_shubik"])


class RealGames(unittest.TestCase):
    """Inter-American Development Bank 1998, German Bundestag 2009, IMF 2002,
    US Electoral College 2016."""

    @classmethod
    def setUpClass(cls):
        cls.games = minimum_quota_games()

    def test_minimum_quotas(self):
        for name in ("idb_1998", "bundestag", "iwf_2002", "electoral_college_2016"):
            g = self.games[name]
            result = Game.weighted(g["weights"], quota=g["quota"]).minimum_quota()
            self.assertTrue(result.optimal, name)
            self.assertEqual(result.quota, g["q_min"], name)

    def test_idb_1998_against_enumeration(self):
        g = self.games["idb_1998"]              # Inter-American Development Bank, n = 14
        game = Game.weighted(g["weights"], quota=g["quota"])
        reference = brute_force(g["n"], weighted_predicate(g["quota"], g["weights"]))
        # quota-indexed methods on the minimal representation: quota 394 instead of 3 631 155
        minimal = game.minimal()
        self.assertEqual(minimal.quota, 394)
        for index, methods in ALL_METHODS.items():
            for method in methods:
                target = game if method in ("qobdd", "minwin") else minimal
                self.assertEqual(list(target.index(index).method(method).values), reference[index],
                                 (index, method))

    def test_bundestag_against_enumeration(self):
        g = self.games["bundestag"]
        game = Game.weighted(g["weights"], quota=g["quota"])
        reference = brute_force(g["n"], weighted_predicate(g["quota"], g["weights"]))
        for index in ALL_METHODS:
            self.assertEqual(list(game.index(index).auto().values), reference[index], index)

    def test_imf_2002_methods_agree(self):
        """IMF 2002 (Strand and Rapkin): q_min = 1 078 265 closes an integer gap.

        The quota stays above 10^6, so the quota-indexed Johnston DP and the
        bivariate series are left out; Banzhaf and Deegan-Packel are compared
        across methods, the other indices are covered by the other tests.
        """
        g = self.games["iwf_2002"]
        game = Game.weighted(g["weights"], quota=g["quota"])
        self.assertEqual(game.minimal().quota, 1078265)
        self.assertEqual(game.banzhaf().qobdd().values, game.banzhaf().kakimura().values)
        self.assertEqual(game.deegan_packel().qobdd().values, game.deegan_packel().minwin().values)
        self.assertEqual(sum(game.johnston().qobdd().values), 1)
        self.assertEqual(sum(game.shapley_shubik().qobdd().values), 1)

    def test_electoral_college_2016_methods_agree(self):
        g = self.games["electoral_college_2016"]
        game = Game.weighted(g["weights"], quota=g["quota"])
        for index, methods in ALL_METHODS.items():
            values = [game.index(index).method(m).values for m in methods]
            self.assertEqual(values[0], values[1], index)


class BenchmarkInstances(unittest.TestCase):
    """Minimum quotas of the benchmark fixtures with n <= 60."""

    def test_minimum_quotas(self):
        for name, g in minimum_quota_games().items():
            if g["group"] != "benchmark" or g["n"] > 60:
                continue
            with self.subTest(name):
                result = Game.weighted(g["weights"], quota=g["quota"]).minimum_quota()
                self.assertTrue(result.optimal)
                self.assertEqual(result.quota, g["q_min"])
                self.assertTrue(same_game(g["quota"], g["weights"], result.quota, result.weights))

    def test_staudacher_fixtures_methods_agree(self):
        games = minimum_quota_games()
        for name in ("uniform_n16", "linear_fixed_q600_n16", "uniform_n30"):
            g = games[name]
            game = Game.weighted(g["weights"], quota=g["quota"])
            for index, methods in ALL_METHODS.items():
                with self.subTest(name=name, index=index):
                    values = [game.index(index).method(m).values for m in methods]
                    self.assertEqual(values[0], values[1])

    def test_quota_does_not_change_the_diagram(self):
        """The three linear-range fixtures with n = 50 are one game."""
        games = minimum_quota_games()
        sizes = {Game.weighted(games[k]["weights"], quota=games[k]["quota"]).qobdd_size
                 for k in ("linear_q50k_n50", "linear_q100k_n50", "linear_q250k_n50")}
        self.assertEqual(sizes, {23481})


if __name__ == "__main__":
    unittest.main()
