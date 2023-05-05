from __future__ import annotations
from collections import defaultdict
from typing import Callable, DefaultDict, Hashable, List, Tuple

import gymnasium as gym
import numpy as np

class SARSA(object):
    """ SARSA algorithm """
    
    def __init__ (
        self,
        env: gym.Env,
        num_episodes: int, 
        gamma: float = 1.0, 
        alpha: float = 0.5, 
        epsilon: float = 0.1
    ):
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
        
        self.env  = env
        self.num_episodes= num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self. epsilon = epsilon
        
        self.n_actions = self.env.action_space.n
        
        # create Q structure
        self.Q: DefaultDict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))

    def __call__(self, state: np.array, epsilon: float) -> int:
        """Implement an episolon greedy policy

        Parameters
        ----------
        probability_distribution : np.ndarray
            Probability distribution array

        Returns
        -------
        int
            Index of chosen action
        """
        
        # TODO: Implement a Epsilon-Greedy policy
        
        if np.random.random() > 1 - epsilon:
            p = np.random.random(self.n_actions)
        else:
            p = np.zeros(self.n_actions)
            p[np.where(Q[state] == np.max(Q[state]))] = 1
        
        # Make a probability distribution
        probs = p / np.sum(p)
        
        # Choose action from a probability distribution
        np.random.choice(range(len(p)), p=probs)
        
        return 0


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
            [description]
        action : int
            [description]
        reward : float
            [description]
        next_state : int
            [description]
        next_action : int
            [description]
        done : bool
            [description]

        Returns
        -------
        float
            [description]
        """
        delta = self.alpha * ((reward + self.gamma * Q[next_state][next_action] * (not done)) - Q[state][action])
        return Q[state][action] + delta