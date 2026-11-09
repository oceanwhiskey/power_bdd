"""Real-valued weights (exact scaling) and float64 results."""
import random
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np

from power_bdd import Game, MethodNotApplicable
from power_bdd import game as game_module

from .helpers import load, same_game


def close(a, b, tol=1e-12):
    return all(abs(float(x) - float(y)) <= tol * max(1.0, abs(float(y))) for x, y in zip(a, b))


class NumberInput(unittest.TestCase):
    def test_floats_are_read_as_written(self):
        game = Game.weighted([0.7, 0.1, 0.2], quota=0.8)
        self.assertEqual(game.weights, (7, 1, 2))
        self.assertEqual(game.quota, 8)
        self.assertEqual(game.scale, 10)
        self.assertTrue(game.is_winning([0, 1]))          # 0.7 + 0.1 reaches 0.8

    def test_binary_reading(self):
        game = Game.weighted([0.7, 0.1, 0.2], quota=0.8, floats="binary")
        self.assertFalse(game.is_winning([0, 1]))         # 0.7 + 0.1 < 0.8 in binary
        self.assertTrue(game.is_winning([0, 2]))
        with self.assertRaises(ValueError):
            Game.weighted([0.5, 0.5], quota=0.5, floats="hexadecimal")

    def test_mixed_exact_types(self):
        game = Game.weighted([Decimal("18.47"), Fraction(1, 3), np.float64(2.5), "0.25", np.int64(4)],
                             share="50%")
        self.assertEqual(game.weights, (5541, 100, 750, 75, 1200))
        self.assertEqual(game.scale, 300)
        self.assertEqual(game.original_weights[1], Fraction(1, 3))
        total = Fraction("18.47") + Fraction(1, 3) + Fraction(5, 2) + Fraction(1, 4) + 4
        self.assertEqual(game.quota, -(-(total / 2 * 300).numerator // (total / 2 * 300).denominator))

    def test_integers_stay_unchanged(self):
        game = Game.weighted([6, 4, 2, 2], quota=10)
        self.assertEqual((game.weights, game.quota, game.scale), ((6, 4, 2, 2), 10, 1))
        rounded = Game.weighted([6, 4, 2, 2], quota=9.5)
        self.assertEqual(rounded.quota, 10)

    def test_share_with_floats(self):
        game = Game.weighted([18.47, 15.18, 13.19, 0.13], share=0.65)
        self.assertEqual(game.weights, (1847, 1518, 1319, 13))
        self.assertEqual(game.quota, 3054)                 # ceil(0.65 * 4697)

    def test_invalid_numbers(self):
        for bad in (float("nan"), float("inf"), Decimal("NaN")):
            with self.assertRaises(ValueError):
                Game.weighted([1.0, bad], quota=1)
        with self.assertRaises(ValueError):
            Game.weighted([1.5, -0.5], quota=1)
        with self.assertRaises(TypeError):
            Game.weighted([True, 1], quota=1)
        with self.assertRaises(TypeError):
            Game.weighted([1 + 2j, 1], quota=1)

    def test_same_indices_as_integer_game(self):
        decimal_game = Game.weighted([0.3, 0.2, 0.1, 0.1], quota=0.5)
        integer_game = Game.weighted([3, 2, 1, 1], quota=5)
        for index in ("banzhaf", "shapley_shubik", "deegan_packel", "johnston"):
            for method in integer_game.index(index).available():
                self.assertEqual(decimal_game.index(index).method(method).values,
                                 integer_game.index(index).method(method).values)

    def test_minimum_quota_of_population_shares(self):
        """Population shares of the Council of the EU in percent, rounded to 0.01."""
        data = load("eu27_2026.json")
        total = sum(data["population"])
        shares = [round(100 * p / total, 2) for p in data["population"]]
        game = Game.weighted(shares, quota=65.0)
        result = game.minimum_quota()
        self.assertTrue(result.optimal)
        self.assertLessEqual(result.quota, game.quota)
        self.assertTrue(same_game(game.quota, list(game.weights), result.quota, list(result.weights)))


class QuotaIndexedLimit(unittest.TestCase):
    def test_large_quota_is_refused_with_a_hint(self):
        game = Game.weighted([300_000_000, 200_000_000, 100_000_000, 100_000_000], quota=500_000_000)
        self.assertEqual(game.johnston().available(), ("auto", "qobdd"))
        with self.assertRaises(MethodNotApplicable) as caught:
            game.johnston().dp()
        self.assertIn("minimal()", str(caught.exception))
        with self.assertRaises(MethodNotApplicable):
            game.banzhaf().kakimura()
        minimal = game.minimal()
        self.assertEqual(minimal.quota, 5)
        self.assertEqual(minimal.johnston().dp().values, game.johnston().qobdd().values)

    def test_limit_is_configurable(self):
        game = Game.weighted([3, 2, 1, 1], quota=5)
        old = game_module.QUOTA_INDEXED_LIMIT
        try:
            game_module.QUOTA_INDEXED_LIMIT = 10
            self.assertNotIn("dp", game.johnston().available())
        finally:
            game_module.QUOTA_INDEXED_LIMIT = old
        self.assertIn("dp", game.johnston().available())


class FloatResults(unittest.TestCase):
    def check(self, game):
        for index in ("banzhaf", "shapley_shubik", "deegan_packel", "johnston"):
            if index == "johnston" and not game.is_complete():
                continue
            exact = game.index(index)
            floating = exact.floats()
            for method in exact.available():
                with self.subTest(index=index, method=method):
                    a = floating.method(method)
                    b = exact.method(method)
                    self.assertTrue(all(isinstance(v, float) for v in a.values))
                    tol = 1e-9 if index == "shapley_shubik" else 1e-12
                    self.assertTrue(close(a.values, b.values, tol), (a.values, b.values))

    def test_random_weighted_games(self):
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(1, 9)
            weights = [rng.randint(0, 20) for _ in range(n)]
            if sum(weights) == 0:
                weights[0] = 1
            self.check(Game.weighted(weights, quota=rng.randint(1, sum(weights))))

    def test_random_multi_tier_games(self):
        rng = random.Random(4)
        for _ in range(25):
            n = rng.randint(2, 7)
            tiers = []
            for _ in range(2):
                weights = [rng.randint(0, 5) for _ in range(n)]
                if sum(weights) == 0:
                    weights[0] = 1
                tiers.append(Game.weighted(weights, quota=rng.randint(1, sum(weights))))
            self.check(tiers[0] & tiers[1] if rng.random() < 0.5 else tiers[0] | tiers[1])

    def test_council_of_the_eu(self):
        data = load("eu27_2026.json")
        population = data["population"]
        council = (Game.at_least(15, of=27)
                   & (Game.weighted(population, share="65%") | Game.at_least(24, of=27)))
        self.check(council)
        johnston = council.johnston().floats().auto()
        self.assertEqual(johnston.info["arithmetic"], "float64")
        self.assertAlmostEqual(johnston.values[0], 0.248082, places=6)

    def test_switching_back_and_forth(self):
        game = Game.weighted([3, 2, 1, 1], quota=5)
        self.assertIsInstance(game.johnston().floats().exact().auto().values[0], Fraction)
        self.assertEqual(game.banzhaf().floats().choose(), "qobdd")
        self.assertAlmostEqual(sum(game.banzhaf().floats().auto().normalized()), 1.0)

    def test_many_players(self):
        """Banzhaf in float64 has no limit on n; the row readings stop at 1000."""
        from math import comb
        game = Game.majority(1101)
        beta = game.banzhaf().floats().qobdd().values
        expected = Fraction(comb(1100, 550), 2 ** 1100)
        self.assertTrue(close([beta[0]], [expected], 1e-12))
        with self.assertRaises(ValueError):
            game.johnston().floats().qobdd()


if __name__ == "__main__":
    unittest.main()
