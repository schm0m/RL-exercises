from __future__ import annotations
from collections import defaultdict
from typing import Callable, DefaultDict, Hashable, List, Tuple

import gymnasium as gym
import numpy as np

from rl_exercises.agent import AbstractAgent


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""
    
    def __init__(
        self,
        Q: DefaultDict,
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
        q_values = [self.Q[(state, action)] for action in range(self.env.action_space.n)]
        action = np.argmax(q_values).item()
        return action

#FIXME: I don't follow the AbstractAgent class
class SARSAAgent(AbstractAgent):
    """SARSA algorithm"""

    def __init__(self, env: gym.Env, policy, num_episodes: int, gamma: float = 1.0, alpha: float = 0.5, epsilon: float = 0.1):
        """Initialize the SARSA agent

        Parameters
        ----------
        env : gym.Env
            Environment for the agent
        num_episodes : int
            Number of episodes
        gamma : float, optional
            Discount Factor , by default 1.0
        alpha : float, optional
            Learning Rate, by default 0.5
        epsilon : float, optional
            Exploration Parameter, by default 0.1
        """

        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert 0 <= epsilon <= 1, "epsilon has to be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"

        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = policy
        self.n_actions = self.env.action_space.n

        # create Q structure
        self.Q: DefaultDict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))


    def update(
        Q: DefaultDict[int, np.ndarray],
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform a TD update

        Parameters
        ----------
        Q : DefaultDict[int, np.ndarray]
            List of Q values
        state : int
            Current state
        action : int
            action output by the policy
        reward : float
            reward for the transition
        next_state : int
            next state to be trainsitioned into
        next_action : int
            Next action for lookahead
        done : bool
            done flag

        Returns
        -------
        float
            New Q value for the state action pair
        """
        # TODO: Impelment the TD update
        return Q[state][action] + ...
