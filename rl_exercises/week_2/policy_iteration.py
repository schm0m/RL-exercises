from __future__ import annotations

import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.agent import AbstractAgent

from rich import print as printr


class PolicyIteration(AbstractAgent):
    def __init__(
            self, 
            env: MarsRover,
            gamma: float = 0.9, 
            seed: int = 333,
            **kwargs
        ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.seed = seed

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        assert self.n_actions == 2, str(self.n_actions)

        # Get the MDP from the env
        self.S = states =  np.arange(0, self.n_obs)
        self.A = actions = np.arange(0, self.n_actions)
        self.P = transition_probabilites = self.env.transition_probabilities
        self.R = rewards = self.env.rewards
        self.gamma = gamma
        self.R_sa = self.get_reward_per_action()

        # Policy
        rng = np.random.default_rng(seed=self.seed)
        self.pi: np.array = rng.integers(0, self.n_actions, self.n_obs)

        # State-Value Function
        self.Q = np.zeros_like(self.P)

        self.policy_fitted: bool = False

    def get_reward_per_action(self):
        R_sa = np.zeros((self.n_obs, self.n_actions))  # same shape as P
        for s in range(R_sa.shape[0]):
            for a in range(R_sa.shape[1]):
                delta_s = -1 if a == 0 else 1
                s_index = max(0, min(self.n_obs-1, s + delta_s))
                printr(s_index, self.R)
                R_sa[s,a] = self.R[s_index]
        
        return R_sa

    def predict(self, observation: int, info: dict | None = None) -> tuple[int, dict]:
        action = self.pi[observation]
        info = {}
        return action, info
    
    def update(self, *args, **kwargs) -> None:
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


def do_policy_evaluation(Q: np.ndarray, pi: np.ndarray, MDP: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)) -> tuple(np.ndarray, np.ndarray):
    S, A, P, R_sa, gamma = MDP

    epsilon = 1e-8
    converged = False
    while not converged:
        Q_old = Q.copy()
        for s in S:
            a = pi[s]
            action = -1 if a == 0 else 1  # TODO choose according to probability
            s_next = s + action
            s_next = max(min(s_next, len(P) - 1), 0) 
            # print(s, a, s_next)
            Q[s, a] = R_sa[s, a] + gamma * np.sum( P[s_next] * Q[s_next])
        converged = np.all(np.linalg.norm(Q - Q_old, 1) < epsilon)

    return Q


def do_policy_improvement(Q: np.ndarray, pi: np.ndarray) -> tuple(np.ndarray, bool):
    pi_old = pi.copy()

    pi = np.argmax(Q, axis=1)

    epsilon = 1e-8
    converged = np.all(np.linalg.norm(pi - pi_old, 1) < epsilon)
    return pi, converged


def do_policy_iteration(Q: np.array, pi: np.array, MDP: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)) -> tuple(np.ndarray, np.ndarray):
    converged = False
    while not converged:
        Q = do_policy_evaluation(Q, pi, MDP)
        pi, converged = do_policy_improvement(Q, pi)

    return Q, pi


if __name__ == "__main__":
    algo = PolicyIteration(env=MarsRover())
    algo.update()
