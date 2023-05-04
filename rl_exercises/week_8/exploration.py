from __future__ import annotations
from functools import partial

import gymnasium as gym
import numpy as np
import torch.nn as nn
from rl_exercises.week_6 import EpsilonGreedyPolicy


class EpsilonDecayPolicy(EpsilonGreedyPolicy):
    """Policy implementing Epsilon Greedy Exploration with decaying epsilon."""

    def __init__(
        self,
        Q: nn.Module,
        env: gym.Env,
        epsilon: float,
        seed: int = None,
    ) -> None:
        """Init

        Parameters
        ----------
        Q : nn.Module
            State-Value function
        env : gym.Env
            Environment
        epsilon: float
            Exploration rate
        seed : int, optional
            Seed, by default None
        """
        self.Q = Q
        self.env = env
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)

    def update_epsilon(self):
        """Decay the epsilon value."""
        # TODO: implement decay
        self.epsilon = ...

    def __call__(self, state: np.array, exploration_rate: float = 0.0, eval: bool = False) -> int:
        """Select action

        Parameters
        ----------
        state : np.array
            State
        exploration_rate : float, optional
            exploration rate (epsilon), by default 0.0
        eval: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """
        action = ...
        # TODO implement algorithm

        return action
    

class EZGreedyPolicy(EpsilonGreedyPolicy):
    """Policy for Exploration with ε(z)-greedy"""

    def __init__(
        self,
        Q: nn.Module,
        env: gym.Env,
        duration_max: int = 100,
        mu: float = 3,
        seed: int = None,
    ) -> None:
        """Init

        Parameters
        ----------
        Q : nn.Module
            State-Value function
        env : gym.Env
            Environment
        duration_max : int, optional
            Maximum number of action repetition, by default 100
        mu : float, optional
            Zeta/Zipf distribution parameter, by default 2
        seed : int, optional
            Seed, by default None
        """
        self.Q = Q
        self.env = env
        self.duration_max = duration_max
        self.mu = mu

        self.n: int = 0  # number of times left to perform action
        self.w: int = -1  # random action in memory
        self.rng = np.random.default_rng(seed=seed)

    def sample_duration(self) -> int:
        """Sample duration from a zeta/zipf distribution

        The duration is capped at `self.duration_max`.

        Returns
        -------
        int
            duration (how often the action is repeated)
        """
        duration = ...
        # TODO implement sampling
        return duration

    def __call__(self, state: np.array, exploration_rate: float = 0.0, eval: bool = False) -> int:
        """Select action

        εz-greedy algorithm B.1 [Dabney et al., 2020].
        The while loop is happening outside, in the training loop.
        This is what is inside the while loop.

        Parameters
        ----------
        state : np.array
            State
        exploration_rate : float, optional
            exploration rate (epsilon), by default 0.0
        eval: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """
        action = ...
        # TODO implement algorithm

        return action