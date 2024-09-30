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
    for i in range(max_iter):
        delta = 0
        new_values = np.copy(values)
        for state in range(mdp.observation_space.n):
            value_max = float("-inf")
            for action in range(mdp.action_space.n):
                next_state, reward, done = mdp.P[state][action]
                new_value = reward + gamma * (0 if done else values[next_state])
                value_max = max(value_max, new_value)
            new_values[state] = value_max
            delta = max(delta, abs(new_values[state] - values[state]))

        values = new_values
        if delta < 1e-5:  # Seuil de convergence
            break
    return values
    # END SOLUTION


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
    for i in range(max_iter):
        delta = 0
        new_values = np.copy(values)

        for row in range(env.height):
            for col in range(env.width):
                env.set_state(row, col)

                # Si c'est un mur ou un état terminal, sa valeur est constante
                if env.grid[row, col] == "W":
                    continue
                elif env.grid[row, col] == "P":
                    new_values[row, col] = 1.0
                    continue
                elif env.grid[row, col] == "N":
                    new_values[row, col] = -1.0
                    continue

                # Calcul de la valeur maximale pour les actions possibles
                value_max = float("-inf")
                for action in range(env.action_space.n):
                    next_state, reward, is_done, _ = env.step(action, make_move=False)
                    next_row, next_col = next_state
                    new_value = reward + gamma * values[next_row, next_col]
                    value_max = max(value_max, new_value)

                # Mettre à jour la valeur pour la case courante
                new_values[row, col] = value_max
                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        values = new_values

        # Arrêter si les valeurs ont convergé
        if delta < theta:
            break
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for i in range(max_iter):
        delta = 0
        prev_values = np.copy(values)
        for row in range(4):
            for col in range(4):
                env.set_state(row, col)

                # Skip walls and terminal states
                if env.grid[row, col] in {"W", "P", "N"}:
                    continue

                value_max = float("-inf")
                for action in range(env.action_space.n):
                    next_states = env.get_next_states(action)
                    value_sum = 0
                    for next_state, reward, probability, _, _ in next_states:
                        next_row, next_col = next_state
                        value_sum += probability * (
                            reward + gamma * prev_values[next_row, next_col]
                        )
                    value_max = max(value_max, value_sum)

                values[row, col] = value_max
                delta = max(delta, abs(values[row, col] - prev_values[row, col]))

        if delta < theta:
            break
    return values
    # END SOLUTION
