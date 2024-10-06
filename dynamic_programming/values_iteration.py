import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    iter_count = 0
    while iter_count < max_iter:
        pre_values = np.copy(values)
        s = 0
        while s < mdp.observation_space.n:
            value = max(
                [
                    mdp.P[s][a][1] + gamma * values[mdp.P[s][a][0]]
                    for a in range(mdp.action_space.n)
                ]
            )
            values[s] = value
            s += 1
        if not np.any(np.abs(values - pre_values) >= 1e-5):  # Inversion de la condition
            break
        iter_count += 1
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    iter_count = 0
    while iter_count < max_iter:
        prev_val = np.copy(values)
        delta = 0
        row = 0
        while row < 4:
            col = 0
            while col < 4:
                env.current_position = (row, col)
                delta += value_iteration_per_state(env, values, gamma, prev_val, delta)
                col += 1
            row += 1
        if not delta >= theta:  # Inversion de la condition
            break
        iter_count += 1
    return values
    # END SOLUTION


def value_iteration_per_state(env: GridWorldEnv, values, gamma, prev_val, delta):
    # BEGIN SOLUTION
    row, col = env.current_position
    values[row, col] = float("-inf")
    action = 0
    while action < env.action_space.n:
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
        action += 1
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta
    # END SOLUTION


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    iter_count = 0
    while iter_count < max_iter:
        prev_val = np.copy(values)
        delta = 0
        row = 0
        while row < 4:
            col = 0
            while col < 4:
                env.current_position = (row, col)
                delta += value_iteration_per_state(env, values, gamma, prev_val, delta)
                col += 1
            row += 1
        if not delta >= theta:  # Inversion de la condition
            break
        iter_count += 1
    return values
