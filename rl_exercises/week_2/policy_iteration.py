from __future__ import annotations
from typing import Any

import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.agent import AbstractAgent

from rich import print as printr
import warnings


class PolicyIteration(AbstractAgent):
    """Policy Iteration

    Parameters
    ----------
    env : MarsRover
        Environment. Only applicable to the MarsRover env.
    gamma : float, optional
        Discount factor, by default 0.9
    seed : int, optional
        Seed, by default 333
    filename : str, optional
        Filename for the policy, by default "policy.npz"
    """

    def __init__(
        self, env: MarsRover, gamma: float = 0.9, seed: int = 333, filename: str = "policy.npy", **kwargs: dict
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.seed = seed
        self.filename = filename

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n  # type: ignore[attr-defined]
        self.n_actions = self.env.action_space.n  # type: ignore[attr-defined]

        # Get the MDP from the env
        self.S = self.env.states
        self.A = self.env.actions
        self.P = self.env.transition_probabilities
        self.R = self.env.rewards
        self.gamma = gamma
        self.R_sa = self.env.get_reward_per_action()

        # Policy
        rng = np.random.default_rng(seed=self.seed)
        self.pi: np.ndarray = rng.integers(0, self.n_actions, self.n_obs)

        # State-Value Function
        self.Q = np.zeros_like(self.P)

        self.policy_fitted: bool = False

    def predict(self, observation: int, info: dict | None = None) -> tuple[int, dict]:  # type: ignore[override]
        """Predict action based on observation

        Parameters
        ----------
        observation : int
            Observation.
        info : dict | None, optional
            Info dict, by default None

        Returns
        -------
        tuple[int, dict]
            Action, info dict.
        """
        action = self.pi[observation]
        info = {}
        return action, info

    def update(self, *args: tuple, **kwargs: dict) -> None:
        """Update policy

        In this case, determine the policy once by policy iteration.
        """
        if not self.policy_fitted:
            printr("Initial policy: ", self.pi)
            self.Q, self.pi = do_policy_iteration(
                Q=self.Q,
                pi=self.pi,
                MDP=(self.S, self.A, self.P, self.R_sa, self.gamma),
            )
            printr("Q: ", self.Q)
            printr("Final policy: ", self.pi)
            self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        if self.policy_fitted:
            np.save(self.filename, np.array(self.pi))
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        self.pi = np.load(self.filename)
        self.policy_fitted = True


def do_policy_evaluation(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Policy Evaluation

    Parameters
    ----------
    Q : np.ndarray
        State-value function.
    MDP : tuple
        Markov Decision Process, defined as (S, A, T, R_sa, gamma).
        S: np.ndarray: States, dim: |S|
        A: np.ndarray: Actions, dim: |A|
        T: np.ndarray: Transition matrix, dim: |S| x |A| x |S|
        R_sa: np.ndarray: Reward being in state s and using action a, dim: |S| x |A|
        gamma: float: Discount factor.
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    np.ndarray
        Q, state-value function
    """
    S, A, P, R_sa, gamma = MDP

    converged = False
    while not converged:
        Q_old = Q.copy()
        for s in S:
            a = pi[s]
            action = -1 if a == 0 else 1  # TODO choose according to probability
            s_next = s + action
            s_next = max(min(s_next, len(P) - 1), 0)
            # print(s, a, s_next)
            Q[s, a] = R_sa[s, a] + gamma * np.sum(P[s_next] * Q[s_next])
        converged = np.all(np.linalg.norm(Q - Q_old, 1) < epsilon)

    return Q


def do_policy_improvement(
    Q: np.ndarray,
    pi: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """Policy Improvement

    Parameters
    ----------
    Q : np.ndarray
        State-value function
    pi : np.ndarray
        Policy
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    tuple[np.ndarray, bool]
        Pi, converged.
    """

    pi_old = pi.copy()

    pi = np.argmax(Q, axis=1)

    converged = np.all(np.linalg.norm(pi - pi_old, 1) < epsilon)
    return pi, converged


def do_policy_iteration(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Policy Iteration

    Parameters
    ----------
    Q : np.ndarray
        State-value function.
    MDP : tuple
        Markov Decision Process, defined as (S, A, T, R_sa, gamma).
        S: np.ndarray: States, dim: |S|
        A: np.ndarray: Actions, dim: |A|
        T: np.ndarray: Transition matrix, dim: |S| x |A| x |S|
        R_sa: np.ndarray: Reward being in state s and using action a, dim: |S| x |A|
        gamma: float: Discount factor.
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    np.ndarray, callable
        Q, pi (the policy)
    """
    converged = False
    while not converged:
        Q = do_policy_evaluation(Q, pi, MDP, epsilon=epsilon)
        pi, converged = do_policy_improvement(Q, pi)

    return Q, pi


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update()
