from __future__ import annotations

from collections import defaultdict
from typing import Any, Tuple

import numpy as np

from rl_exercises.agent import AbstractAgent


def to_bins(value: np.ndarray, bins: np.ndarray) -> float:
    """Put a single float value into closest bin"""
    return np.digitize(x=[value], bins=bins)[0]


def to_discrete_state(obs: Tuple[float, float, float, float], num_bins) -> Tuple[float, float, float, float]:
    """Transform an observation from continuous to discrete space"""
    x, v, theta, omega = obs
    CART_POSITION = np.linspace(-4.8, 4.8, num_bins)
    CART_VELOCITY = np.linspace(-1, 1, num_bins)
    POLE_ANGLE = np.linspace(-0.418, 0.418, num_bins)
    POLE_ANGULAR_VELOCITY = np.linspace(-3, 3, num_bins)
    state = (
        to_bins(x, CART_POSITION),
        to_bins(v, CART_VELOCITY),
        to_bins(theta, POLE_ANGLE),
        to_bins(omega, POLE_ANGULAR_VELOCITY),
    )
    return state


class TabularQAgent(AbstractAgent):
    """Q-Learning Agent Class."""

    def __init__(self, env, policy, learning_rate, gamma, num_bins, **kwargs) -> None:
        self.env = env
        self.Q = defaultdict(lambda: np.random.uniform(1, -1))
        self.policy = policy(self.env, self.Q)
        self.learning_rate = learning_rate
        self.gamma = gamma
        # This adds the option to pass a function via the kwargs
        # You'll only need this if you want to use a different environment
        if "discretize_state" in kwargs.keys():
            self.discretize_state = kwargs["discretize_state"]
        else:
            self.discretize_state = to_discrete_state
        self.num_bins = num_bins
 
    def predict(self, state, info) -> Any:
        discrete_state = to_discrete_state(state, self.num_bins)
        return self.policy(discrete_state)

    def save(self, path) -> Any:
        np.save(path, self.Q)

    def load(self, path) -> Any:
        self.Q = np.load(path)

    def update(
        self,
        transition: list[np.array],
    ) -> float:
        """Value Function Update.

        Parameters
        ----------
        transition : list[np.array]
            Transition to train on

        Returns
        -------
        float
            TD-Error
        """
        # TODO: Implement Q-Learning Update
        # alpha = 1
        td_error = 0
        return td_error
    
