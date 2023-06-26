from __future__ import annotations
from typing import Any

import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.agent import AbstractAgent

from rich import print as printr


class ValueIteration(AbstractAgent):
    def __init__(self, env: MarsRover, gamma: float = 0.9, seed: int = 333, **kwargs) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.seed = seed

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        assert self.n_actions == 2, str(self.n_actions)

        # Get the MDP from the env
        self.S = states = np.arange(0, self.n_obs)
        self.A = actions = np.arange(0, self.n_actions)
        self.T = self.env.transition_matrix
        self.R = rewards = self.env.rewards
        self.gamma = gamma
        self.R_sa = self.env.get_reward_per_action()

        # Policy
        rng = np.random.default_rng(seed=self.seed)

        # Value Function
        self.V = np.zeros_like(self.S)

        self.policy_fitted: bool = False

    def predict(self, observation: int, info: dict | None = None) -> tuple[int, dict]:
        if not self.policy_fitted:
            self.update()
        action = self.pi(observation)
        info = {}
        return action, info

    def update(self, *args, **kwargs) -> None:
        if not self.policy_fitted:
            self.V, self.pi = do_value_iteration(
                V=self.V,
                MDP=(self.S, self.A, self.T, self.R_sa, self.gamma),
            )
            printr("V: ", self.V)
            printr("Final policy: ", self.pi)
            self.policy_fitted = True


def do_value_iteration(
    V: np.array, MDP: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)
) -> tuple(np.ndarray, np.ndarray):
    converged = False
    while not converged:
        V, converged = update_value_function(
            V=V,
            MDP=MDP,
        )

    pi = determine_pi(V=V)

    return V, pi


def determine_pi(V: np.array) -> callable:
    def pi(s):
        v_left = V[max(s - 1, 0)]
        v_right = V[min(s + 1, 0)]
        equal = v_left == v_right
        action = np.random.random_integers(0, 1) if equal else np.argmax([v_left, v_right])
        return action

    return pi


def update_value_function(
    V: np.array, MDP: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float), epsilon: float = 1e-8
) -> tuple[np.ndarray, bool]:
    S, A, T, R_sa, gamma = MDP
    delta = 0
    for s in S:
        v = V[s]
        vs = []
        for a in A:
            v_new = 0
            r = R_sa[s, a]
            for s_next in S:
                p = T[s, a, s_next]
                v_new += p * (r + gamma * V[s_next])
            vs.append(v_new)

        V[s] = max(vs)
        delta = max(delta, np.abs(v - V[s]))

    converged = delta < epsilon
    return V, converged
