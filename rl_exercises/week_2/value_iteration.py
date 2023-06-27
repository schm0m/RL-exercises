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
        self.S = self.env.S
        self.A = self.env.A
        self.T = self.env.transition_matrix
        self.R_sa = self.env.get_reward_per_action()
        self.gamma = gamma       

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
                seed=self.seed,
            )
            printr("V: ", self.V)
            printr("Final policy: ", self.pi)
            self.policy_fitted = True


def do_value_iteration(
    V: np.ndarray,
    MDP: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float),
    seed: int | None = None,
    epsilon: float = 1e-8,
) -> tuple(np.ndarray, callable):
    """Value Iteration

    First, update the value function until it is converged and then determine
    the policy.

    Parameters
    ----------
    V : np.ndarray
    MDP : tuple
        Markov Decision Process, defined as (S, A, T, R_sa, gamma).
        S: np.array: States, dim: |S|
        A: np.array: Actions, dim: |A|
        T: np.array: Transition matrix, dim: |S| x |A| x |S|
        R_sa: np.array: Reward being in state s and using action a, dim: |S| x |A|
        gamma: float: Discount factor.
    seed : int | None, optional
        Optional seed, by default None
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    np.ndarray, callable
        V, pi (The policy)
    """
    converged = False
    while not converged:
        V, converged = update_value_function(
            V=V,
            MDP=MDP,
            epsilon=epsilon,
        )

    pi = determine_pi(V=V, seed=seed)

    return V, pi


def determine_pi(V: np.array, seed: int | None = None) -> callable:
    """Generate the policy.

    Parameters
    ----------
    V : np.array
        Value function as value per state.
    seed : int | None, optional
        Optional seed, by default None

    Returns
    -------
    callable
        Policy receiving the state and outputting action.
    """
    # Seed the random generator
    rng = np.random.default_rng(seed=seed)

    def pi(s):
        # Values of possible follow-up states
        v_left = V[max(s - 1, 0)]
        v_right = V[min(s + 1, 0)]
        # What if values are equal?
        equal = v_left == v_right
        action = rng.integers(0, 1) if equal else np.argmax([v_left, v_right])
        return action

    return pi


def update_value_function(
    V: np.array, MDP: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float), epsilon: float = 1e-8
) -> tuple[np.ndarray, bool]:
    """Update Value Function.

    Parameters
    ----------
    V : np.array
        Value function of len |S|.
    MDP : tuple
        Markov Decision Process, defined as (S, A, T, R_sa, gamma).
        S: np.array: States, dim: |S|
        A: np.array: Actions, dim: |A|
        T: np.array: Transition matrix, dim: |S| x |A| x |S|
        R_sa: np.array: Reward being in state s and using action a, dim: |S| x |A|
        gamma: float: Discount factor.
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    tuple[np.ndarray, bool]
        V, converged
    """
    S, A, T, R_sa, gamma = MDP
    delta = 0
    for s in S:
        v = V[s]
        vs = []
        for a in A:
            r = R_sa[s, a]
            v_sum = 0
            for s_next in S:
                p = T[s, a, s_next]
                v_sum += p * gamma * V[s_next]
            vs.append(r + v_sum)

        V[s] = max(vs)
        delta = max(delta, np.abs(v - V[s]))

    converged = delta < epsilon
    return V, converged
