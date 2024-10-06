import typing as t

import gym
import numpy as np
from gym import spaces

# Exercice 3: Extension du MDP à un GridWorld (sans bruit)
# --------------------------------------------------------
# Implémenter un MDP simple avec la librairie `gym`. Ce MDP est formé
# d'un GridWorld de 3x4 cases, avec 4 actions possibles (haut, bas, gauche,
# droite). La case (1, 1) est inaccessible (mur), tandis que la case (1, 3)
# est un état terminal avec une récompense de -1. La case (0, 3) est un état
# terminal avec une récompense de +1. Tout autre état a une récompense de 0.
# L'agent commence dans la case (0, 0).


class GridWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    # F: Free, S: Start, P: Positive reward, N: negative reward, W: Wall
    grid: np.ndarray = np.array(
        [
            ["F", "F", "F", "P"],
            ["F", "W", "F", "N"],
            ["F", "F", "F", "F"],
            ["S", "F", "F", "F"],
        ]
    )
    current_position: tuple[int, int] = (3, 0)

    def __init__(self):
        super(GridWorldEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.width = 4
        self.height = 4
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.height), spaces.Discrete(self.width))
        )
        self.moving_prob = np.ones(shape=(self.height, self.width, self.action_space.n))
        zero_mask = (self.grid == "W") | (self.grid == "P") | (self.grid == "N")

        self.moving_prob[np.where(zero_mask)] = 0
        self.direction_table = [
            self.up_position,
            self.down_position,
            self.left_postion,
            self.right_postion,
        ]
        # self.current_position = (0, 0)

    def set_state(self, row: int, col: int) -> None:
        """
        Set the current position of the agent to the specified row and column.

        Args:
            row (int): The row index of the new position.
            col (int): The column index of the new position.

        Returns:
            None
        """
        self.current_position = (row, col)

    def up_position(self):
        return (
            max(0, self.current_position[0] - 1),
            self.current_position[1],
        )

    def down_position(self):
        return (
            min(self.height - 1, self.current_position[0] + 1),
            self.current_position[1],
        )

    def left_postion(self):
        return (
            self.current_position[0],
            max(0, self.current_position[1] - 1),
        )

    def right_postion(self):
        return (
            self.current_position[0],
            min(self.width - 1, self.current_position[1] + 1),
        )

    def step(self, action, make_move: bool = True):
        # new_pos = self.current_position
        old_pos = self.current_position
        new_pos = self.direction_table[action]()

        next_state = tuple(new_pos)
        # Check if the agent has hit a wall
        if self.grid[tuple(new_pos)] == "W":
            next_state = tuple(old_pos)
        # Check if the agent has reached the goal
        is_done = self.grid[tuple(new_pos)] in {"P", "N"}

        # Provide reward
        if old_pos != new_pos:
            if self.grid[tuple(new_pos)] == "N":
                reward = -1
            elif self.grid[tuple(new_pos)] == "P":
                reward = 1
            else:
                reward = 0
        else:
            reward = 0

        if make_move:  # self.grid[tuple(new_pos)] != "W" and make_move:
            self.current_position = next_state

        return next_state, reward, is_done, {}

    def get_next_states(self, action: int) -> list[tuple[int, float, float, bool]]:
        next_state, reward, is_done, _ = self.step(action, make_move=False)
        res = []

        i = 0
        while i < 1:
            res.append((next_state, reward, 1, is_done, action))
            i += 1

        return res

    def reset(self):
        self.current_position = (3, 0)  # Start Position
        return self.current_position

    def render(self):
        row = 0
        while row < 4:
            col = 0
            while col < 4:
                if self.current_position == tuple([row, col]):
                    print("X", end=" ")
                else:
                    print(self.grid[row, col], end=" ")
                col += 1
            row += 1
            print("")  # Newline at the end of the row
