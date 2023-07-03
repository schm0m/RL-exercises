from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# This could be useful to you:
# import torch.nn.functional as F
import torch.optim as optim

from rl_exercises.agent import AbstractAgent


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""

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
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.Q(torch.from_numpy(state).float()).detach().numpy()
        action = np.argmax(q_values)
        return action


class VFAQAgent(AbstractAgent):
    """DQN Agent Class."""

    def __init__(self, env, policy, learning_rate, gamma, **kwargs) -> None:
        self.env = env
        self.Q = self.make_Q()
        self.policy = policy(self.env, self.Q)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = ...

    def make_Q(self) -> nn.Module:
        """
        The Q-function is of class `nn.Module`.

        Use `env.observation_space.shape` to get the shape of the input data.
        Use `env.action_space.n` to get number of possible actions for this environment.

        Parameters
        ----------
        env: gym.Env
            The environment the Q function is meant for

        Returns
        -------
        Q
        An intialized policy
        """
        # TODO: Create Q-Function from env.
        Q = ...
        return Q

    def predict(self, state, info) -> Any:
        return self.policy(state)

    def save(self, path) -> Any:
        train_state = {"parameters": self.Q.state_dict(), "optimizer_state": self.optimizer.state_dict()}
        torch.save(train_state, path)

    def load(self, path) -> Any:
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def update(
        self,
        training_batch: list[np.array],
    ) -> float:
        """
        Value Function Update for a Batch of Transitions.

        Use MSE loss.

        Parameters
        ----------
        training_batch : list[np.array]
            Batch to train on

        Returns
        -------
        float
            Loss
        """
        # TODO: Implement Value Function Update Step
        # Convert data into torch tensors

        # Compute MSE loss

        # Optimize the model

        loss = 0
        return float(loss)
