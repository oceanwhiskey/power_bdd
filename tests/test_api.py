"""The fluent API: construction, method selection, results, errors."""
import unittest
from fractions import Fraction

from power_bdd import (CompositeGame, Game, MethodNotApplicable, NotCompleteError, PowerIndex,
                       WeightedGame)


class Construction(unittest.TestCase):
    def test_quota_or_share(self):
        self.assertEqual(Game.weighted([2, 2, 1], quota=3).quota, 3)
        self.assertEqual(Game.weighted([10, 20, 30, 40], share="65%").quota, 65)
        self.assertEqual(Game.weighted([10, 20, 30, 40], share=0.5).quota, 50)
        self.assertEqual(Game.weighted([1, 1, 1], share=Fraction(2, 3)).quota, 2)
        with self.assertRaises(ValueError):
            Game.weighted([1, 2])
        with self.assertRaises(ValueError):
            Game.weighted([1, 2], quota=2, share="50%")
        with self.assertRaises(ValueError):
            Game.weighted([1, -2], quota=1)

    def test_shorthands(self):
        self.assertEqual(Game.at_least(3, of=5).weights, (1,) * 5)
        self.assertEqual(Game.majority(5).quota, 3)
        self.assertEqual(Game.unanimity(4).quota, 4)

    def test_combinations(self):
        a, b, c = Game.at_least(2, of=3), Game.unanimity(3), Game.weighted([3, 1, 1], quota=3)
        self.assertIsInstance(a & b, CompositeGame)
        self.assertEqual(len((a & b & c).games), 3)                 # flattened
        self.assertEqual((a | b).operation, "or")
        self.assertEqual(len(Game.all_of(a, Game.any_of(b, c)).games), 2)
        self.assertEqual(len(a.and_(b, c).games), 3)
        with self.assertRaises(ValueError):
            a & Game.unanimity(4)

    def test_names(self):
        game = Game.weighted([3, 2, 1, 1], quota=5).named(["A", "B", "C", "D"])
        self.assertEqual(game.players, ("A", "B", "C", "D"))
        self.assertTrue(game.is_winning(["A", "B"]))
        self.assertFalse(game.is_winning(["B", "C", "D"]))
        result = game.johnston().auto()
        self.assertEqual(result["A"], Fraction(17, 30))
        self.assertEqual(result.as_dict()["D"], Fraction(1, 15))
        with self.assertRaises(KeyError):
            result["E"]
        with self.assertRaises(ValueError):
            Game.weighted([1, 2], quota=2).named(["only one"])
        tiers = Game.weighted([1, 1, 0], quota=2, players="xyz") & Game.unanimity(3)
        self.assertEqual(tiers.players, ("x", "y", "z"))


class MethodSelection(unittest.TestCase):
    def setUp(self):
        self.game = Game.weighted([3, 2, 1, 1], quota=5)

    def test_available_methods(self):
        self.assertEqual(self.game.banzhaf().available(), ("auto", "qobdd", "kakimura"))
        self.assertEqual(self.game.johnston().available(), ("auto", "qobdd", "dp"))
        composite = self.game & Game.at_least(2, of=4)
        self.assertEqual(composite.johnston().available(), ("auto", "qobdd"))
        self.assertEqual(composite.deegan_packel().available(), ("auto", "qobdd", "minwin"))

    def test_named_selectors_and_aliases(self):
        for alias in ("shapley", "SS", "Shapley-Shubik", "shapley_shubik"):
            self.assertEqual(self.game.index(alias).index, "shapley_shubik")
        with self.assertRaises(ValueError):
            self.game.index("holler")
        with self.assertRaises(ValueError):
            self.game.johnston().method("generating_functions")
        self.assertEqual(self.game.johnston().method("AUTO").method, "qobdd")

    def test_auto_records_choice(self):
        result = self.game.deegan_packel().auto()
        self.assertEqual(result.method, self.game.deegan_packel().choose())
        self.assertTrue(result.info["chosen_automatically"])
        self.assertFalse(self.game.deegan_packel().qobdd().info["chosen_automatically"])

    def test_weight_methods_only_on_weighted_games(self):
        composite = self.game | Game.unanimity(4)
        self.assertTrue(hasattr(self.game.banzhaf(), "kakimura"))
        self.assertTrue(hasattr(self.game.johnston(), "dp"))
        self.assertTrue(hasattr(self.game, "minimum_quota"))
        self.assertFalse(hasattr(composite.banzhaf(), "kakimura"))
        self.assertFalse(hasattr(composite.shapley_shubik(), "kakimura"))
        self.assertFalse(hasattr(composite.johnston(), "dp"))
        self.assertFalse(hasattr(composite, "minimum_quota"))
        self.assertFalse(hasattr(composite.banzhaf().floats(), "kakimura"))
        self.assertTrue(hasattr(self.game.named("abcd").banzhaf(), "kakimura"))
        with self.assertRaises(MethodNotApplicable):       # by name: a clear message
            composite.banzhaf().method("kakimura")
        with self.assertRaises(ValueError):
            composite.banzhaf().method("no_such_method")

    def test_johnston_fails_at_index_selection(self):
        incomplete = Game.weighted([1, 1, 0, 0], quota=2) | Game.weighted([0, 0, 1, 1], quota=2)
        with self.assertRaises(NotCompleteError):
            incomplete.johnston()
        with self.assertRaises(NotCompleteError):
            incomplete.index("johnston")
        with self.assertRaises(NotCompleteError):
            incomplete.deegan_packel().qobdd()
        self.assertEqual(incomplete.deegan_packel().choose(), "minwin")

    def test_completeness_from_weights(self):
        council = Game.at_least(15, of=27) & (Game.weighted(range(27, 0, -1), share="65%")
                                              | Game.at_least(24, of=27))
        self.assertTrue(council.is_complete())
        self.assertNotIn("diagram", council._cache)          # decided without the diagram
        # opposed weights: complete, but only the diagram can tell
        unsc = Game.weighted([1] * 5 + [0] * 10, quota=5) & Game.weighted([0] * 5 + [1] * 10, quota=4)
        self.assertFalse(unsc._complete_by_weights())
        self.assertTrue(unsc.is_complete())
        self.assertIn("diagram", unsc._cache)


class Results(unittest.TestCase):
    def test_result_helpers(self):
        result = Game.weighted([3, 2, 1, 1], quota=5).banzhaf().qobdd()
        self.assertIsInstance(result, PowerIndex)
        self.assertEqual(len(result), 4)
        self.assertEqual(list(result), [Fraction(5, 8), Fraction(3, 8), Fraction(1, 8), Fraction(1, 8)])
        self.assertEqual(result.raw, (5, 3, 1, 1))
        self.assertEqual(sum(result.normalized()), 1)
        self.assertAlmostEqual(result.to_floats()[0], 0.625)
        self.assertIn("Banzhaf (absolute) via qobdd", result.table())
        self.assertIn("62.5000 %", str(result))

    def test_minimal_representation(self):
        game = Game.weighted([6, 4, 2, 2], quota=10)        # [5; 3, 2, 1, 1], doubled
        minimal = game.minimal()
        self.assertIsInstance(minimal, WeightedGame)
        self.assertEqual(minimal.quota, 5)
        self.assertEqual(minimal.johnston().dp().values, game.johnston().qobdd().values)


class ValuesOfVersion1(unittest.TestCase):
    """The values of the test suite of power_bdd 1.x."""

    def test_electoral_college_1996(self):
        w = [54, 33, 32, 25, 23, 22, 21, 18, 15, 14, 13, 13, 12, 12, 11, 11, 11, 11, 10, 10, 9, 9,
             8, 8, 8, 8, 8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3]
        game = Game.weighted(w, quota=270)
        beta = game.banzhaf().auto().to_floats()
        phi = game.shapley_shubik().auto().to_floats()
        self.assertAlmostEqual(beta[0], 0.4664541346829614, places=14)
        self.assertAlmostEqual(beta[50], 0.0228538034563881, places=14)
        self.assertAlmostEqual(phi[0], 0.10812540951779513, places=14)
        self.assertAlmostEqual(phi[50], 0.005405549637386588, places=14)

    def test_us_federal_system(self):
        """(G1 and G2 and G3) or (G1 and G4 and G5 and G3) or (G6 and G7)."""
        house = [0, 0] + [0] * 100 + [1] * 435
        senate = [0, 0] + [1] * 100 + [0] * 435
        president = [1, 0] + [0] * 100 + [0] * 435
        vice = [0, 1] + [0] * 100 + [0] * 435
        g1, g2, g3 = (Game.weighted(house, quota=218), Game.weighted(senate, quota=51),
                      Game.weighted(president, quota=1))
        g4, g5 = Game.weighted(senate, quota=50), Game.weighted(vice, quota=1)
        g6, g7 = Game.weighted(house, quota=290), Game.weighted(senate, quota=67)
        federal = (g1 & g2 & g3) | (g1 & g4 & g5 & g3) | (g6 & g7)
        beta = federal.banzhaf().auto().to_floats()
        self.assertAlmostEqual(beta[0], 0.2499999999999993, places=14)
        self.assertAlmostEqual(beta[1], 0.019897309346794692, places=14)
        self.assertAlmostEqual(beta[2], 0.01989730934679494, places=14)
        self.assertAlmostEqual(beta[200], 0.009569407217589256, places=14)


if __name__ == "__main__":
    unittest.main()
