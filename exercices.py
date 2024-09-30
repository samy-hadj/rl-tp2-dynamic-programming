"""
Ce fichier contient des exercices à compléter sur la programmation dynamique.
Il est évalué automatiquement avec pytest, vous pouvez le lancer avec la
commande `pytest exercices.py`.
"""

import os
import random
import typing as t

import gym
import numpy as np
import pytest
from gym import spaces

from dynamic_programming import MDP, GridWorldEnv, StochasticGridWorldEnv
from dynamic_programming.domino_paving import domino_paving
from dynamic_programming.fibonacci import fibonacci, fibonacci_memo
from dynamic_programming.values_iteration import (
    grid_world_value_iteration, mdp_value_iteration,
    stochastic_grid_world_value_iteration)


# Tests pour l'exercice 1
def test_mdp():
    mdp = MDP()
    assert mdp.P[0][0] == (1, -1, False)
    assert mdp.P[0][1] == (0, -1, False)
    assert mdp.P[1][0] == (0, -1, False)
    assert mdp.P[1][1] == (2, -1, False)
    assert mdp.P[2][0] == (2, 0, False)
    assert mdp.P[2][1] == (0, -1, False)

    mdp.reset()
    ret = mdp.step(0)
    assert ret[0] in [0, 1, 2]
    assert ret[1] in [0, -1]
    assert ret[2] in [True, False]
    assert isinstance(ret[3], dict)


def test_mdp_value_iteration(max_iter: int = 1000):
    mdp = MDP()
    values = mdp_value_iteration(mdp, max_iter=max_iter, gamma=1.0)
    assert np.allclose(values, [-2, -1, 0]), print(values)
    values = mdp_value_iteration(mdp, max_iter=max_iter, gamma=0.9)
    assert np.allclose(values, [-1.9, -1, 0])


def test_grid_world_value_iteration(max_iter=1000):
    env = GridWorldEnv()

    values = grid_world_value_iteration(env, max_iter, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert np.allclose(values, solution), print(values)

    values = grid_world_value_iteration(env, max_iter, gamma=0.9)
    solution = np.array(
        [
            [0.81, 0.9, 1.0, 0.0],
            [0.729, 0.0, 0.9, 0.0],
            [0.6561, 0.729, 0.81, 0.729],
            [0.59049, 0.6561, 0.729, 0.6561],
        ]
    )
    assert np.allclose(values, solution)


def test_stochastic_grid_world_value_iteration(max_iter=1000):
    env = StochasticGridWorldEnv()

    values = stochastic_grid_world_value_iteration(env, max_iter=max_iter, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(values, solution), print("  ", values)

    values = stochastic_grid_world_value_iteration(env, max_iter=max_iter, gamma=0.9)
    solution = np.array(
        [
            [0.77495822, 0.87063224, 0.98343293, 0.0],
            [0.68591168, 0.0, 0.77736888, 0.0],
            [0.60732544, 0.60891859, 0.68418232, 0.60570595],
            [0.54079452, 0.54500607, 0.60570595, 0.53914484],
        ]
    )
    assert np.allclose(values, solution), print("  ", values)


# Tests pour l'exercice 1
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci(n, expected):
    assert fibonacci(n) == expected


# Tests pour l'exercice 2
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci_memo(n, expected):
    assert fibonacci_memo(n) == expected


# Tests pour l'exercice 3
@pytest.mark.parametrize(
    "n,expected",
    [
        (1, 0),
        (2, 3),
        (3, 0),
        (4, 11),
        (5, 0),
        (6, 41),
        (7, 0),
        (8, 153),
        (9, 0),
        (10, 571),
    ],
)
def test_domino_paving(n, expected):
    assert domino_paving(n) == expected


def test_wall():
    env = GridWorldEnv()
    for i in range(2):
        env.step(0)
    old_position = env.current_position
    env.step(3)
    assert old_position == env.current_position, print(
        env.current_position, old_position
    )
