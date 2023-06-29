"""GridCore Env taken from https://github.com/automl/TabularTempoRL/"""
from __future__ import annotations

import sys
import time
from io import StringIO
from typing import Any, SupportsFloat, Tuple

import numpy as np
import gymnasium

# from gymnasium.envs.toy_text.discrete import DiscreteEnv

# Actions
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class MarsRover(gymnasium.Env):
    """Simple Environment for a Mars Rover that can move in a 1D Space

    Actions
    -------
    Discrete, 2:
    - 0: go left
    - 1: go right

    Observations
    ------------
    The current position of the rover (int).

    Reward
    ------
    Certain amount per field.

    Start/Reset State
    -----------------
    Position 2.
    """

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
    ):
        """Init the environment

        Parameters
        ----------
        transition_probabilities : np.ndarray, optional
            [Nx2] Array for N positions and 2 actions each, by default np.ones((5, 2)).
        rewards : list[float], optional
            [Nx1] Array for rewards. rewards[pos] determines the reward for a given position `pos`, by default [1, 0, 0, 0, 10].
        horizon : int, optional
            Number of total steps for this environment until it is done (e.g. battery drained), by default 10.
        """
        self.rewards: list[float] = rewards
        self.transition_probabilities: np.ndarray = transition_probabilities
        self.current_steps: int = 0
        self.horizon: int = horizon
        self.position: int = 2

        n = len(self.transition_probabilities)
        self.observation_space = gymnasium.spaces.Discrete(n=n)
        self.action_space = gymnasium.spaces.Discrete(n=2)

        self.states = np.arange(0, n)
        self.actions = np.arange(0, 2)
        self.transition_matrix = self.T = self.get_transition_matrix(
            S=self.states, A=self.actions, P=self.transition_probabilities
        )

    def get_reward_per_action(self):
        R_sa = np.zeros((len(self.states), len(self.actions)))  # same shape as P
        for s in range(R_sa.shape[0]):
            for a in range(R_sa.shape[1]):
                delta_s = -1 if a == 0 else 1
                s_index = max(0, min(len(self.states) - 1, s + delta_s))
                R_sa[s, a] = self.rewards[s_index]

        return R_sa

    def get_next_state(self, s: int, a: int, S: np.ndarray) -> int:
        delta_s = -1 if a == 0 else 1
        s_next = s + delta_s
        s_next = max(min(s_next, len(S) - 1), 0)
        return s_next

    def get_transition_matrix(self, S: np.ndarray, A: np.ndarray, P: np.ndarray) -> np.ndarray:
        T = np.zeros((len(S), len(A), len(S)))
        for s in S:
            for a in A:
                s_next = self.get_next_state(s, a, S)
                probability = P[s, a]
                T[s, a, s_next] = probability

        # T_ = np.ndarray(
        #     [
        #         [1, 0, 0, 0, 0],
        #         [0, 1, 0, 0, 0],
        #         [1, 0, 0, 0, 0],
        #         [0, 0, 1, 0, 0],
        #         [0, 1, 0, 0, 0],
        #         [0, 0, 0, 1, 0],
        #         [0, 0, 1, 0, 0],
        #         [0, 0, 0, 0, 1],
        #         [0, 0, 0, 1, 0],
        #         [0, 0, 0, 0, 1],
        #     ]
        # ).reshape((len(S), len(A), len(S)))

        # assert np.all(T == T_)
        return T

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.

        The rover will always be set to position 2.

        Parameters
        ----------
        seed : int | None, optional
            Seed, not used, by default None
        options : dict[str, Any] | None, optional
            Options, not used, by default None

        Returns
        -------
        tuple[Any, dict[str, Any]]
            Observation, info
        """
        self.current_steps = 0
        self.position = 2

        observation = self.position
        info = {}

        return observation, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Executes an action and return next_state, reward and whether the environment is done (horizon reached)

        :param action: Defines action. Has to be either 0 (go left) or 1 (go right)
        :return: tuple[next_state=position, reward, is_done]
        """
        # Determine move given an action and transition probabilities for environment
        action = int(action)
        self.current_steps += 1
        follow_action = np.random.random() < self.transition_probabilities[self.position][action]
        if not follow_action:
            action = 1 - action

        # Move and update position
        if action == 0:
            if self.position > 0:
                self.position -= 1
        elif action == 1:
            if self.position < 4:
                self.position += 1
        else:
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        # Get reward
        reward = self.rewards[self.position]

        terminated = False
        truncated = self.current_steps >= self.horizon

        info = {}

        return self.position, reward, terminated, truncated, info


# class GridCore(DiscreteEnv):
#     metadata = {"render.modes": ["human", "ansi"]}
#     action_to_name = ["Left", "Up", "Right", "Down"]

#     def __init__(
#         self,
#         shape: Tuple[int, int] = (5, 10),
#         start: Tuple[int, int] = (0, 0),
#         goal: Tuple[int, int] = (0, 9),
#         max_steps: int = 300,
#         percentage_reward: bool = False,
#         no_goal_rew: bool = False,
#     ):

#         try:
#             self.shape = self._shape
#         except AttributeError:
#             self.shape = shape

#         self.nS: int = np.prod(self.shape, dtype=int).item()
#         self.nA = 4
#         self.start = start
#         self.goal = goal
#         self.max_steps = max_steps
#         self._steps = 0
#         self._pr = percentage_reward
#         self._no_goal_rew = no_goal_rew
#         self.total_steps = 0

#         P = self._init_transition_probability()

#         # We always start in state (3, 0)
#         isd = np.zeros(self.nS)
#         isd[np.ravel_multi_index(start, self.shape)] = 1.0

#         super(GridCore, self).__init__(self.nS, self.nA, P, isd)

#     def step(self, a):
#         self._steps += 1
#         s, r, d, i = super(GridCore, self).step(a)
#         if self._steps >= self.max_steps:
#             d = True
#             i["early"] = True
#         self.total_steps += 1
#         return s, r, d, i

#     def reset(self):
#         self._steps = 0
#         return super(GridCore, self).reset()

#     def _init_transition_probability(self):
#         raise NotImplementedError

#     def _check_bounds(self, coord):
#         coord[0] = min(coord[0], self.shape[0] - 1)
#         coord[0] = max(coord[0], 0)
#         coord[1] = min(coord[1], self.shape[1] - 1)
#         coord[1] = max(coord[1], 0)
#         return coord

#     def print_T(self):
#         print(self.P[self.s])

#     def map_output(self, s, pos):
#         if self.s == s:
#             output = " x "
#         elif pos == self.goal:
#             output = " T "
#         else:
#             output = " o "
#         return output

#     def map_control_output(self, s, pos):
#         return self.map_output(s, pos)

#     def map_with_inbetween_goal(self, s, pos, in_between_goal):
#         return self.map_output(s, pos)

#     def render(self, mode="human", close=False, in_control=None, in_between_goal=None):
#         self._render(mode, close, in_control, in_between_goal)

#     def _render(self, mode="human", close=False, in_control=None, in_between_goal=None):
#         if close:
#             return
#         outfile = StringIO() if mode == "ansi" else sys.stdout
#         if mode == "human":
#             print("\033[2;0H")

#         for s in range(self.nS):
#             position = np.unravel_index(s, self.shape)
#             # print(self.s)
#             if in_control:
#                 output = self.map_control_output(s, position)
#             elif in_between_goal:
#                 output = self.map_with_inbetween_goal(s, position, in_between_goal)
#             else:
#                 output = self.map_output(s, position)
#             if position[1] == 0:
#                 output = output.lstrip()
#             if position[1] == self.shape[1] - 1:
#                 output = output.rstrip()
#                 output += "\n"
#             outfile.write(output)
#         outfile.write("\n")
#         if mode == "human":
#             if in_control:
#                 time.sleep(0.2)
#             else:
#                 time.sleep(0.05)


# class FallEnv(GridCore):
#     _pits: list[tuple[int, int]] | list[list[int]] = []

#     def _calculate_transition_prob(self, current, delta, prob):
#         transitions = []
#         for d, p in zip(delta, prob):
#             new_position = np.ndarray(current) + np.ndarray(d)
#             new_position = self._check_bounds(new_position).astype(int)
#             new_state = np.ravel_multi_index(tuple(new_position), self.shape)
#             reward = 0.0
#             is_done = False
#             if tuple(new_position) == self.goal:
#                 if self._pr:
#                     reward = 1 - (self._steps / self.max_steps)
#                 elif not self._no_goal_rew:
#                     reward = 1.0
#                 is_done = True
#             elif new_state in self._pits:
#                 reward = -1.0
#                 is_done = True
#             transitions.append((p, new_state, reward, is_done))
#         return transitions

#     def _init_transition_probability(self):
#         self.afp = 0.0  # todo: hotfix, check with Andre how to properly remove afp
#         for idx, p in enumerate(self._pits):
#             self._pits[idx] = np.ravel_multi_index(p, self.shape)
#         # Calculate transition probabilities
#         P = {}
#         for s in range(self.nS):
#             position = np.unravel_index(s, self.shape)
#             P[s] = {a: [] for a in range(self.nA)}
#             other_prob = self.afp / 3.0

#             tmp = [[UP, DOWN, LEFT, RIGHT], [DOWN, LEFT, RIGHT, UP], [LEFT, RIGHT, UP, DOWN], [RIGHT, UP, DOWN, LEFT]]
#             tmp_dirs = [
#                 [[-1, 0], [1, 0], [0, -1], [0, 1]],
#                 [[1, 0], [0, -1], [0, 1], [-1, 0]],
#                 [[0, -1], [0, 1], [-1, 0], [1, 0]],
#                 [[0, 1], [-1, 0], [1, 0], [0, -1]],
#             ]
#             tmp_pros = [
#                 [1 - self.afp, other_prob, other_prob, other_prob],
#                 [1 - self.afp, other_prob, other_prob, other_prob],
#                 [1 - self.afp, other_prob, other_prob, other_prob],
#                 [1 - self.afp, other_prob, other_prob, other_prob],
#             ]
#             for acts, dirs, probs in zip(tmp, tmp_dirs, tmp_pros):
#                 P[s][acts[0]] = self._calculate_transition_prob(position, dirs, probs)
#         return P

#     def map_output(self, s, pos):
#         if self.s == s:
#             output = " \u001b[33m*\u001b[37m "
#         elif pos == self.goal:
#             output = " \u001b[37mX\u001b[37m "
#         elif s in self._pits:
#             output = " \u001b[31m.\u001b[37m "
#         else:
#             output = " \u001b[30mo\u001b[37m "
#         return output

#     def map_control_output(self, s, pos):
#         if self.s == s:
#             return " \u001b[34m*\u001b[37m "
#         else:
#             return self.map_output(s, pos)

#     def map_with_inbetween_goal(self, s, pos, in_between_goal):
#         if s == in_between_goal:
#             return " \u001b[34mx\u001b[37m "
#         else:
#             return self.map_output(s, pos)


# class Bridge6x10Env(FallEnv):
#     _pits = [
#         [0, 2],
#         [0, 3],
#         [0, 4],
#         [0, 5],
#         [0, 6],
#         [0, 7],
#         [1, 2],
#         [1, 3],
#         [1, 4],
#         [1, 5],
#         [1, 6],
#         [1, 7],
#         [4, 2],
#         [4, 3],
#         [4, 4],
#         [4, 5],
#         [4, 6],
#         [4, 7],
#         [5, 2],
#         [5, 3],
#         [5, 4],
#         [5, 5],
#         [5, 6],
#         [5, 7],
#     ]
#     _shape = (6, 10)


# class Pit6x10Env(FallEnv):
#     _pits = [
#         [0, 2],
#         [0, 3],
#         [0, 4],
#         [0, 5],
#         [0, 6],
#         [0, 7],
#         [1, 2],
#         [1, 3],
#         [1, 4],
#         [1, 5],
#         [1, 6],
#         [1, 7],
#         [2, 2],
#         [2, 3],
#         [2, 4],
#         [2, 5],
#         [2, 6],
#         [2, 7],
#     ]
#     # [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7]]
#     _shape = (6, 10)


# class ZigZag6x10(FallEnv):
#     _pits = [
#         [0, 2],
#         [0, 3],
#         [1, 2],
#         [1, 3],
#         [2, 2],
#         [2, 3],
#         [3, 2],
#         [3, 3],
#         [5, 7],
#         [5, 6],
#         [4, 7],
#         [4, 6],
#         [3, 7],
#         [3, 6],
#         [2, 7],
#         [2, 6],
#     ]
#     _shape = (6, 10)


# class ZigZag6x10H(FallEnv):
#     _pits = [
#         [0, 2],
#         [0, 3],
#         [1, 2],
#         [1, 3],
#         [2, 2],
#         [2, 3],
#         [3, 2],
#         [3, 3],
#         [5, 7],
#         [5, 6],
#         [4, 7],
#         [4, 6],
#         [3, 7],
#         [3, 6],
#         [2, 7],
#         [2, 6],
#         [4, 4],
#         [5, 2],
#     ]
#     _shape = (6, 10)
