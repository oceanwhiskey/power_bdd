"""Every method against exhaustive enumeration on random games."""
import random
import unittest

from power_bdd import Game, NotCompleteError

from .helpers import brute_force, same_game, weighted_predicate


class RandomWeightedGames(unittest.TestCase):
    def test_all_methods(self):
        rng = random.Random(2026)
        for trial in range(120):
            n = rng.randint(1, 8)
            weights = [rng.randint(0, 9) for _ in range(n)]
            if sum(weights) == 0:
                weights[0] = 1
            quota = rng.randint(1, sum(weights))
            game = Game.weighted(weights, quota=quota)
            reference = brute_force(n, weighted_predicate(quota, weights))
            for index in ("banzhaf", "shapley_shubik", "deegan_packel", "johnston"):
                selector = game.index(index)
                for method in selector.available():
                    with self.subTest(trial=trial, index=index, method=method):
                        result = selector.method(method)
                        self.assertEqual(list(result.values), reference[index])
                        if index == "banzhaf":
                            self.assertEqual(list(result.raw), reference["swings"])
                        if index in ("johnston", "deegan_packel"):
                            self.assertEqual(list(result.raw), reference[index + "_raw"])

    def test_constant_games(self):
        for quota in (0, -3, 100):
            game = Game.weighted([3, 2, 1], quota=quota)
            for index in ("banzhaf", "shapley_shubik", "deegan_packel", "johnston"):
                self.assertEqual(list(game.index(index).auto().values), [0, 0, 0])

    def test_minimum_quota_witness(self):
        rng = random.Random(7)
        for trial in range(60):
            n = rng.randint(1, 9)
            weights = [rng.randint(0, 40) for _ in range(n)]
            if sum(weights) == 0:
                weights[0] = 1
            quota = rng.randint(1, sum(weights))
            result = Game.weighted(weights, quota=quota).minimum_quota()
            with self.subTest(trial=trial):
                self.assertTrue(result.optimal)
                self.assertLessEqual(result.quota, quota)
                self.assertTrue(same_game(quota, weights, result.quota, list(result.weights)))


class RandomMultiTierGames(unittest.TestCase):
    def test_and_or_combinations(self):
        rng = random.Random(11)
        for trial in range(80):
            n = rng.randint(2, 7)
            tiers = []
            for _ in range(rng.randint(2, 3)):
                weights = [rng.randint(0, 6) for _ in range(n)]
                if sum(weights) == 0:
                    weights[0] = 1
                tiers.append(Game.weighted(weights, quota=rng.randint(1, sum(weights))))
            game = tiers[0]
            for tier in tiers[1:]:
                game = game & tier if rng.random() < 0.5 else game | tier
            reference = brute_force(n, game.is_winning)
            for index in ("banzhaf", "shapley_shubik", "deegan_packel", "johnston"):
                if index == "johnston" and not game.is_complete():
                    with self.assertRaises(NotCompleteError):
                        game.index(index)
                    continue
                selector = game.index(index)
                methods = selector.available()
                for method in methods:
                    with self.subTest(trial=trial, index=index, method=method):
                        self.assertEqual(list(selector.method(method).values), reference[index])

    def test_completeness_against_enumeration(self):
        """is_complete(), with and without the weight-based shortcut, versus all pairs."""
        rng = random.Random(12)
        shortcut = 0
        for trial in range(150):
            n = rng.randint(2, 6)
            tiers = []
            for _ in range(rng.randint(2, 3)):
                # few distinct weights, so that common orders occur often
                weights = [rng.randint(0, 3) for _ in range(n)]
                if sum(weights) == 0:
                    weights[0] = 1
                tiers.append(Game.weighted(weights, quota=rng.randint(1, sum(weights))))
            game = tiers[0]
            for tier in tiers[1:]:
                game = game & tier if rng.random() < 0.5 else game | tier
            wins = {mask for mask in range(1 << n)
                    if game.is_winning([p for p in range(n) if mask >> p & 1])}

            def at_least_as_desirable(i, j):
                return all((S | 1 << i) in wins for S in range(1 << n)
                           if not S >> i & 1 and not S >> j & 1 and (S | 1 << j) in wins)

            complete = all(at_least_as_desirable(i, j) or at_least_as_desirable(j, i)
                           for i in range(n) for j in range(i + 1, n))
            fresh = game.named(list(range(n)))          # empty cache: shortcut path first
            with self.subTest(trial=trial):
                if fresh._complete_by_weights():
                    shortcut += 1
                    self.assertTrue(complete)
                self.assertEqual(fresh.is_complete(), complete)
                self.assertEqual(game.is_complete(), complete)
        self.assertGreater(shortcut, 10)

    def test_minimal_winning_constructor(self):
        coalitions = [{0, 1}, {0, 2, 3}, {1, 2, 3}]
        game = Game.from_minimal_winning(4, coalitions)
        for S in ([0, 1], [0, 2, 3], [1, 2, 3], [0, 1, 2]):
            self.assertTrue(game.is_winning(S))
        for S in ([0, 2], [2, 3], [1, 3], []):
            self.assertFalse(game.is_winning(S))
        reference = brute_force(4, game.is_winning)
        self.assertEqual(list(game.deegan_packel().auto().values), reference["deegan_packel"])


if __name__ == "__main__":
    unittest.main()
