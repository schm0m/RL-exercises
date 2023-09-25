from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rl_exercises.agent import AbstractAgent
from typing import Tuple


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        seed: int = None,
    ) -> None:
        """Init

        Parameters
        ----------
        env : gym.Env
            Environment
        epsilon: float
            Exploration rate
        seed : int, optional
            Seed, by default None
        """
        self.env = env
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, Q: nn.Module, state: np.array, evaluate: bool = False) -> Tuple[int, dict]:
        """Select action

        Parameters
        ----------
        Q : nn.Module
            State-Value function
        state : np.array
            State
        evaluate: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        dict
            info
        """
        if not evaluate and np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        q_values = Q(torch.from_numpy(state).float()).detach().numpy()
        action = np.argmax(q_values)
        return action, {}


class VFAQAgent(AbstractAgent):
    """VFA Agent Class."""

    def __init__(self, env, policy, learning_rate, gamma, **kwargs) -> None:
        #todo add optimizer
        self.env = env
        self.Q = self.make_Q()
        self.policy = policy(self.env)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = ...

    def make_Q(self) -> nn.Module:
        """
        The Q-function is using linear function approximation for Q-value prediction.

        You can use tensors with 'requires_grad=True' to represent the weights of the linear function.
        Use `env.observation_space.shape` to get the shape of the input data.
        Use `env.action_space.n` to get number of possible actions for this environment.

        Returns
        -------
        Q
        An intialized policy
        """
        # TODO: Create Q-Function from env.
        Q = ...
        return Q

    def predict(self, state, info, evaluate=False) -> tuple[Any, dict]:
        # TODO: predict an action
        action = ...
        info = {}
        return action, info

    def save(self, path) -> Any:
        train_state = {"W": self.W, "b": self.b, "optimizer_state": self.optimizer.state_dict()} #todo Warum W und b?
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
