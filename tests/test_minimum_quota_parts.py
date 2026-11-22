"""The building blocks of the minimum quota: subset sums at the start, upper
bounds from the LP optimum, and Gomory's group relaxation with the fibred
search for integer gaps."""
import itertools
import random
import unittest
from fractions import Fraction
from functools import reduce
from math import gcd

from power_bdd import Game
from power_bdd import _minimum_quota as MQ
from power_bdd._diagram import weighted_diagram

from .helpers import load, same_game


def games():
    return {g["id"]: g for g in load("minimum_quotas.json")["games"]}


def start_of(quota, weights):
    classes = MQ._Classes(quota, weights)
    clipped = [min(quota, v) for v in classes.values]
    common = reduce(gcd, clipped, 0)
    return classes, [v // common for v in clipped], MQ._ceil(Fraction(quota, common))


class SubsetSumStart(unittest.TestCase):
    def test_exact_sum_knapsack_against_enumeration(self):
        rng = random.Random(5)
        for trial in range(40):
            k = rng.randint(1, 4)
            values = sorted({rng.randint(1, 9) for _ in range(k)}, reverse=True)
            mult = [rng.randint(1, 3) for _ in values]
            weights = [v for v, m in zip(values, mult) for _ in range(m)]
            quota = rng.randint(1, sum(weights))
            classes, start, upper = start_of(quota, weights)
            oracle = MQ._SubsetOracle(classes, start, upper)
            S = [rng.randint(-5, 5) for _ in range(classes.k)]
            counts = list(itertools.product(*[range(m + 1) for m in classes.multiplicities]))
            for target, maximize in ((upper, False), (upper - 1, True)):
                with self.subTest(trial=trial, target=target):
                    values_at = [sum(c * s for c, s in zip(cnt, S)) for cnt in counts
                                 if sum(c * v for c, v in zip(cnt, start)) == target]
                    result = oracle._exact(S, target, maximize)
                    if not values_at:
                        self.assertIsNone(result)
                        continue
                    best, key = result
                    self.assertEqual(best, max(values_at) if maximize else min(values_at))
                    self.assertEqual(sum(c * v for c, v in zip(key, start)), target)
                    self.assertEqual(sum(c * s for c, s in zip(key, S)), best)

    def test_harvested_rows_are_tight(self):
        g = games()["uniform_n50"]
        classes, start, upper = start_of(g["quota"], g["weights"])
        rows = MQ._SubsetOracle(classes, start, upper).harvest(start)
        self.assertGreater(len(rows), classes.k)
        for counts, winning in rows:
            weight = sum(c * v for c, v in zip(counts, start))
            self.assertEqual(weight, upper if winning else upper - 1)
            self.assertEqual(classes.classify(counts), winning)

    def test_certificate_without_diagram(self):
        data = games()
        for name in ("uniform_n50", "linear_fixed_q600_n40", "electoral_college_2016"):
            g = data[name]
            with self.subTest(name):
                classes, start, upper = start_of(g["quota"], g["weights"])
                self.assertEqual(MQ._certify_start(classes, start, upper), g["q_min"])
        for name in ("linear_fixed_q600_n16", "uniform_n12", "linear_q100k_n80"):    # q_min < q
            g = data[name]
            with self.subTest(name):
                classes, start, upper = start_of(g["quota"], g["weights"])
                self.assertIsNone(MQ._certify_start(classes, start, upper))

    def test_dummies_are_left_to_the_diagram(self):
        # in [10; 6, 6, 1] the other players sum to 0, 6 or 12, never to 9: 1 is a dummy
        self.assertFalse(MQ._SubsetOracle(*start_of(10, [6, 6, 1])).without_dummies())
        self.assertTrue(MQ._SubsetOracle(*start_of(10, [6, 6, 4])).without_dummies())


class GroupRelaxation(unittest.TestCase):
    def labelled(self, quota, weights):
        return list(MQ._labelled_sources(quota, weights))

    def test_imf_2002(self):
        g = games()["iwf_2002"]
        sources = self.labelled(g["quota"], g["weights"])
        classes, oracle, solver = sources[0]
        group = MQ._Group(solver)
        self.assertTrue(group.ok)
        bound = group.bound()
        self.assertGreater(bound, MQ._ceil(solver.value()))      # stronger than ceil(q*)
        self.assertLessEqual(bound, g["q_min"])
        solver._flush()
        status, (q, weights) = group.search(oracle, classes, g["quota"],
                                            known=(solver.pool_mat, solver.pool_b))
        self.assertEqual((status, q), ("found", g["q_min"]))
        self.assertTrue(same_game(g["quota"], g["weights"], q, classes.expand(weights)))

    def test_random_games(self):
        """Bounds and witnesses stay valid; integer gaps are rare at this size."""
        rng = random.Random(17)
        for trial in range(40):
            n = rng.randint(6, 12)
            weights = [rng.randint(1, 10 ** 4) for _ in range(n)]
            quota = rng.randint(1, sum(weights))
            result = Game.weighted(weights, quota=quota).minimum_quota()
            with self.subTest(trial=trial):
                self.assertLessEqual(result.lower_bound, result.upper_bound)
                self.assertTrue(same_game(quota, weights, result.upper_bound, list(result.weights)))
                if result.lp_bound is not None:
                    self.assertGreaterEqual(result.lower_bound, MQ._ceil(result.lp_bound))

    def test_gap_with_a_free_direction(self):
        """30 random weights up to 10^12, quota at 17 % (stress suite, seed 193):
        the optimal basis has a zero multiplier, whose direction spans the
        search plane; the group bound and the search close the gap above
        ceil(q*)."""
        weights = [883590042106, 896131112599, 650049239908, 728907153481, 103959382304,
                   315358167879, 783569102846, 303024648050, 364079818806, 22523410717,
                   355537006217, 495501762979, 4105689769, 439356544728, 57988744547,
                   714111876010, 782336899358, 725538145403, 636144864751, 473565498551,
                   431764642554, 168428203181, 646203948551, 575063660663, 102257673970,
                   381407640366, 522173687891, 270585535358, 968749237999, 643278975768]
        quota = 2455699693942
        result = Game.weighted(weights, quota=quota).minimum_quota()
        self.assertTrue(result.optimal)
        self.assertEqual(result.quota, 182413)
        self.assertGreater(result.quota, MQ._ceil(result.lp_bound))
        self.assertTrue(same_game(quota, weights, result.quota, list(result.weights)))


    def test_gap_in_a_plane_of_two_free_directions(self):
        """30 random weights up to 10^6, quota at 17 % (stress suite, seed 71):
        the plain optimal basis has two zero multipliers, the guided one
        three; the search runs in the plane of the two."""
        weights = [414168, 599102, 849806, 208375, 720585, 443637, 980744, 646253, 97104,
                   14664, 602504, 658092, 786678, 634312, 840388, 201578, 567392, 688477,
                   228207, 465947, 708146, 368743, 77943, 351194, 148503, 450578, 31818,
                   108913, 563148, 129121]
        quota = 2309640
        result = Game.weighted(weights, quota=quota).minimum_quota()
        self.assertTrue(result.optimal)
        self.assertEqual(result.quota, 173728)
        self.assertGreater(result.quota, MQ._ceil(result.lp_bound))
        self.assertTrue(same_game(quota, weights, result.quota, list(result.weights)))


class UpperBounds(unittest.TestCase):
    def test_scaled_lp_optimum_is_a_representation(self):
        g = games()["linear_q100k_n60"]
        classes = MQ._Classes(g["quota"], g["weights"])
        order = sorted(range(len(g["weights"])), key=lambda p: (-g["weights"][p], p))
        oracle = MQ._Oracle(classes, weighted_diagram(g["quota"], g["weights"], order))
        _, start, upper = start_of(g["quota"], g["weights"])
        solver = MQ._ActiveSet(classes, oracle, start, upper)
        self.assertTrue(solver.run())
        quota, weights = MQ._round_candidates(classes, oracle, solver.X, solver.delta, solver.value())
        self.assertTrue(same_game(g["quota"], g["weights"], quota, classes.expand(weights)))
        self.assertGreaterEqual(quota, g["q_min"])


if __name__ == "__main__":
    unittest.main()
