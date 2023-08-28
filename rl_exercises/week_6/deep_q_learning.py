from __future__ import annotations
from typing import Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_exercises.agent import AbstractAgent, AbstractBuffer


# TODO: implement the replay buffer
class ReplayBuffer(AbstractBuffer):
    """Buffer for storing and sampling transitions."""

    def __init__(self, capacity) -> None:
        self.capacity = int(capacity)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.infos = []

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

    def __len__(self):
        return len(self.states)


class DQN(AbstractAgent):
    """DQN Agent Class."""

    def __init__(self, env, policy, learning_rate, gamma, **kwargs) -> None:
        self.env = env
        self.Q = self.make_Q()
        self.policy = policy(self.env)
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

    def predict(self, state, info, evaluate=False) -> Any:
        return self.policy(self.Q, state, evaluate=evaluate), {}

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
        states, actions, rewards, next_states, dones, infos = training_batch
        # TODO: Implement update
        # Convert data into torch tensors

        # Compute MSE loss
        loss = ...
        # Optimize the model

        return float(loss.item())
