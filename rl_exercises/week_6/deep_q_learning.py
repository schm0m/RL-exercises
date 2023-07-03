from __future__ import annotations
from typing import Any

# Both of these can be useful to you:
# import random
# from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_exercises.agent import AbstractAgent, AbstractBuffer


# TODO: implement the replay buffer
class ReplayBuffer(AbstractBuffer):
    """Buffer for storing and sampling transitions."""

    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.states = np.empty()
        self.actions = np.empty()
        self.rewards = np.empty()
        self.next_states = np.empty()
        self.dones = np.empty()
        self.infos = np.empty()

    def add(self, state, action, reward, next_state, done, info):
        # TODO: add transitions to storage
        ...

    def sample(self, batch_size=32):
        # TODO: sample transitions
        batch_states = ...
        batch_actions = ...
        batch_rewards = ...
        batch_next_states = ...
        batch_dones = ...
        batch_infos = ...
        return (batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_infos)


class DQN(AbstractAgent):
    """DQN Agent Class."""

    def __init__(self, env, policy, learning_rate, gamma, **kwargs) -> None:
        self.env = env
        self.Q = self.make_Q()
        self.policy = policy(self.env, self.Q)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = ...

    def make_Q(self) -> nn.Module:
        """Create Q-Function from env.

        Use 1 hidden layer with 64 units.
        Use ReLU as an activation function after all layers except the last.
        Use `env.observation_space.shape` to get the shape of the input data.
        Use `env.action_space.n` to get number of possible actions for this environment.

        Returns
        -------
        nn.Module
            State-Value Function
        """
        # TODO create a deep network as a function approximator
        Q = nn.Sequential(
            [
                ("input_layer", nn.Linear(self.env.observation_space.low.shape[0], 64)),
                ...,
                ("output_layer", self.env.action_space.n),
            ]
        )

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
        """Value Function Update for a Batch of Transitions

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
        states = torch.from_numpy(np.array(training_batch[0])).float()
        actions = torch.from_numpy(np.array(training_batch[1])).unsqueeze(-1)
        rewards = torch.from_numpy(np.array(training_batch[2])).float()
        dones = torch.from_numpy(np.array(training_batch[3])).float()
        next_states = torch.from_numpy(np.array(training_batch[4])).float()

        # TODO: complete these lines to compute the loss
        q_values = ...
        target_q_values = ...
        loss = ...

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
