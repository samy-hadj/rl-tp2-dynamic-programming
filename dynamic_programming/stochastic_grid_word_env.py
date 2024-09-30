import random

import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv

# Exercice 4: GridWorld avec du bruit
# -----------------------------------
# Ecrire une fonction qui calcule la fonction de valeur pour le GridWorld
# avec du bruit.
# Le bruit est un mouvement aléatoire de l'agent vers sa gauche ou sa droite avec une probabilité de 0.1.


class StochasticGridWorldEnv(GridWorldEnv):
    """
    Stochastic version of the GridWorldEnv environment.
    Inherits from GridWorldEnv class.
    """

    def __init__(self):
        """
        Initializes the environment.
        """
        super().__init__()
        self.moving_prob = np.ones(shape=(self.height, self.width, self.action_space.n))
        zero_mask = (self.grid == "W") | (self.grid == "P") | (self.grid == "N")
        self.moving_prob[np.where(zero_mask)] = 0
        # self.moving_prob[np.where(~zero_mask)][:1] = 0.9
        # self.moving_prob[np.where(~zero_mask)][1:] = 0.05

    def _add_noise(self, action: int) -> int:
        """
        Adds stochasticity to the agent's actions.

        Args:
            action (int): The intended action.

        Returns:
            int: The actual action taken by the agent, which may differ from the
            intended action due to stochasticity.
        """
        prob = random.uniform(0, 1)
        if prob < 0.05:  # 5% chance to go left
            return (action - 1) % 4
        elif prob < 0.1:  # 5% chance to go right
            return (action + 1) % 4
        # 90% chance to go in the intended direction
        return action

    def get_next_states(self, action: int) -> list[tuple[int, float, float, bool]]:
        """
        Returns a list of possible next states, rewards, probabilities, and done
        flags for a given action.

        Args:
            action (int): The action to take.

        Returns:
            list[tuple[int, float, float, bool]]: A list of tuples, where each
            tuple contains the next state, reward, probability, and done flag
            for a possible next state resulting from the given action.
        """
        possible_actions = [(action - 1) % 4, (action + 1) % 4, action]
        probs = [0.05, 0.05, 0.9]
        res = []
        for action, prob in zip(possible_actions, probs):
            next_state, reward, is_done, _ = super().step(action, make_move=False)
            res.append((next_state, reward, prob, is_done, action))

        return res

    def step(self, action, make_move: bool = True):
        """
        Takes a step in the environment based on the given action and returns the resulting state, reward, and done flag.

        Args:
            action: The action to take in the environment.
            make_move: Whether or not to actually move the agent in the environment. If False, the state and reward will be calculated but the agent will not actually move.

        Returns:
            state: The resulting state after taking the given action.
            reward: The reward received for taking the given action.
            done: Whether or not the episode is done after taking the given action.
        """
        action = self._add_noise(action)
        return super().step(action, make_move)
