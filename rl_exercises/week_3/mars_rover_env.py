from __future__ import annotations

import numpy as np

# Actions
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class MarsRover:
    """Simple Environment for a Mars Rover that can move in a 1D Space"""

    def __init__(
        self,
        transition_probabilities=np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
    ):
        """
        :param transition_probabilities: [Nx2] Array for N positions and 2 actions each.
        :param rewards: [Nx1] Array for rewards. rewards[pos] determines the reward for a given position `pos`.
        :param horizon: Number of total steps for this environment until it is done (e.g. battery drained)
        """
        self.rewards = rewards
        self.transition_probabilities = transition_probabilities
        self.current_steps = 0
        self.horizon = horizon
        self.position: int = 2

    def reset(self):
        self.current_steps = 0
        self.position = 2
        return self.position, {}

    def step(self, action: int) -> tuple[int, float, bool]:
        """
        Executes an action and return next_state, reward and whether the environment is done (horizon reached)

        :param action: Defines action. Has to be either 0 (go left) or 1 (go right)
        :return: tuple[next_state=position, reward, is_done]
        """
        # Determine move given an action and transition probabilities for environment
        action = int(action)
        self.current_steps += 1
        follow_action = np.random.random() < self.transition_probabilities[self.position][action]
        if not follow_action:
            action = 1 - action

        # Move and update position
        if action == 0:
            if self.position > 0:
                self.position -= 1
        elif action == 1:
            if self.position < 4:
                self.position += 1
        else:
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        # Get reward
        reward = self.rewards[self.position]
        return self.position, reward, self.current_steps >= self.horizon, False, {}