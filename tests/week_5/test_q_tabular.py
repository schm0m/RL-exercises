import unittest
from functools import partial

import gymnasium as gym
import numpy as np

from rl_exercises.week_4 import EpsilonGreedyPolicy
from rl_exercises.week_5.q_learning_tabular import TabularQAgent


class TestQLearningTabular(unittest.TestCase): #todo Tabular Testcases are broken
    def test_deterministic_case1(self):
        env = gym.make("CartPole-v0")
        _ = env.reset(seed=0)  # set the seed via reset once
        env.action_space.seed(0)
        policy = partial(EpsilonGreedyPolicy, epsilon=0.0, env = env, seed = 0)

        q = TabularQAgent(env, policy, 0.1, 0.99)
        q.policy.Q = q.Q
        rewards = []
        state, info = env.reset()
        terminated, truncated = False, False
        actions = []
        while not (terminated or truncated):
            action, info = q.predict(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            q.update((state, action, reward, next_state, (truncated or terminated), info))
            rewards.append(reward)
            actions.append(action)

        self.assertAlmostEqual(sum(rewards), 9)
        self.assertAlmostEqual(len(q.Q), 11)

    def test_deterministic_case2(self):
        env = gym.make("CartPole-v0")
        np.random.seed(42)
        _ = env.reset(seed=42)  # set the seed via reset once
        env.action_space.seed(42)
        policy = partial(EpsilonGreedyPolicy, epsilon=0.2, env = env, seed = 0)

        q = TabularQAgent(env, policy, 0.1, 0.99)
        q.policy.Q = q.Q
        rewards = []
        state, info = env.reset()
        terminated, truncated = False, False
        actions = []
        while not (terminated or truncated):
            action, info = q.predict(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            q.update((state, action, reward, next_state, (truncated or terminated), info))
            rewards.append(reward)
            actions.append(action)

        self.assertAlmostEqual(sum(rewards), 10)
        self.assertAlmostEqual(len(q.Q), 12)


if __name__ == "__main__":
    unittest.main()
