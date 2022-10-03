# power_bdd
Let $N=\{1,2,..,n\}$ be a set of voters, each with a voting weight of $w_i$, who vote *yes* or *no* over a proposal. A coalition $S\in 2^N$ of yes-voters is considered winning if $\sum_{i\in S} w_i$ is greater or equal to a quota $q$. What is the voting power of each voter?

One very intuitive definition of the voting power of a voter is the probability that she/he influences the outcome by her/his vote given an assumed voting behaviour of all other voters. This leads to the notion of power indices. The most well known two are those according to Banzhaf/Penrose and Shapley/Shubik.

One very cool way to calculated them is by using binary decision diagrams.

This package creates the reduced ordered binary decision diagram ("ROBDD") of a weighted game and calculates power indices according to Banzhaf/Penrose and Shapley/Shubik.
This method allows to easily connect bdds with AND or OR and is also suited for voting systems with multiple layers.
The method was published by S. Bolus:
* [Bolus, S., 2011. Power indices of simple games and vector-weighted majority games by means of binary decision diagrams. European J. Oper. Res. 210 (2), 258–272.](https://www.sciencedirect.com/science/article/abs/pii/S0377221710006181)
* If you are interested in calculating power indices you should also check out the [website](https://www.informatik.uni-kiel.de/~progsys/simple_games/lab/lab.html), which offers a javascript-version with a lot more features.

# Usage

## Installation
    pip install power_bdd

## Import
    from power_bdd.bdd import BDD, WeightedGame

## one-tier games
This example calculates the power indices for the electoral college, 1996.

    w = [54, 33, 32, 25, 23, 22, 21, 18, 15, 14, 13, 13, 12, 12, 11, 11, 11, 11, 10, 10, 9, 9, 8, 8, 8, 8, 8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3]                
    q = 270
    game = WeightedGame(q, w)
    bdd = BDD(game)
    banzhaf = bdd.calc_banzhaf()
    shapley = bdd.calc_shapley()
    print(banzhaf)
    print(shapley)

## multi-tier games
This example calculates the power-indices for the U.S. federal system, which can represented by the weighted voting systems

$G$ = ($G_1$ and $G_2$ and $G_3$) or ($G_1$ and $G_4$ and $G_5$ and $G_3$) or ($G_6$ and $G_7$), 

see https://www.fernuni-hagen.de/stochastik/downloads/voting.pdf.

    game1 = WeightedGame(218, [0, 0] + [0] * 100 + [1] * 435)
    game2 = WeightedGame(51,  [0, 0] + [1] * 100 + [0] * 435)
    game3 = WeightedGame(1,   [1, 0] + [0] * 100 + [0] * 435)
    game4 = WeightedGame(50,  [0, 0] + [1] * 100 + [0] * 435)
    game5 = WeightedGame(1,   [0, 1] + [0] * 100 + [0] * 435)
    game6 = WeightedGame(290, [0, 0] + [0] * 100 + [1] * 435)
    game7 = WeightedGame(67,  [0, 0] + [1] * 100 + [0] * 435)

    federal_system = BDD(game1) & BDD(game2) & BDD(game3) | BDD(game1) & BDD(game4) & BDD(game5) & BDD(game3) | BDD(game6) & BDD(game7)
    banzhaf = federal_system.calc_banzhaf()
    print(banzhaf)

# Complexities
Listed complexities are expected complexities for the computation of all voters. At first glance, the complexities seem partly worse than other methods (e.g. using *dynamic programming*), but there are several hidden benefitial properties, for instance $q$ is not necessary the $q$ input $q$, but the smallest integer that is possible to use as quota to represent the game (with any weights).

## one-tier games:

power-index     |time            | space       
--------------- | -------------- | ---------
Banzhaf/Penrose | $O(nq \log(q))$ | $O(nq)$
Shapley/Shubik  | $O(n^3q)$      | $O(n^2q)$


## multi-tier games with _m_ tiers:

power-index     | time                       | space       
--------------- | -------------------------- | --------------------------
Banzhaf/Penrose | $O(n \prod\limits_{t=1}^m q^t)$   | $O(n \prod\limits_{t=1}^m q^t)$
Shapley/Shubik  | $O(n^3 \prod\limits_{t=1}^m q^t)$ | $O(n^2 \prod\limits_{t=1}^m q^t)$

# Remarks
* My java-version is much faster (somewhere between 10-100 times) so there should be plenty of room for optimization in python. Although beautiful is better than ugly, I guess fast is better than slow... and we need more trees! Not just to save the planet, but also in python.
* The original version uses an AVL-tree for the _create_-method. I have replaced that by a _SortedList_ form the _sortedcontainers_-library.    