import unittest
from functools import partial

import gymnasium as gym
import numpy as np
import torch

from rl_exercises.week_5.q_learning_vfa import VFAQAgent, EpsilonGreedyPolicy


class TestQLearning(unittest.TestCase):
    def test_make_Q(self):
        # Test CartPole Environment
        env = gym.make("CartPole-v1")
        policy = partial(EpsilonGreedyPolicy, epsilon=0.0)
        q = VFAQAgent(env, policy, 0.1, 0.99)
        state, info = env.reset()
        v = q.predict(np.array([state]), info)
        self.assertEqual(2, len(v))

    def test_vfa_update(self):
        states = torch.tensor([[0, 0, 0, 0]]).float()
        actions = [0]
        rewards = [1]
        dones = [True]
        next_states = torch.tensor([[0, 0, 0, 0]]).float()
        infos = [{}]
        torch.manual_seed(0)

        env = gym.make("CartPole-v1")
        policy = partial(EpsilonGreedyPolicy, epsilon=0.0)
        q = VFAQAgent(env, policy, 0.1, 0.99)
        loss = q.update([states, actions, rewards, next_states, dones, infos])
        self.assertAlmostEqual(1, round(loss))

    def test_deterministic_case1(self):
        env = gym.make("CartPole-v0")
        np.random.seed(0)
        torch.manual_seed(0)
        _ = env.reset(seed=0)  # set the seed via reset once
        env.action_space.seed(0)
        policy = partial(EpsilonGreedyPolicy, epsilon=0.0)
        q = VFAQAgent(env, policy, 0.1, 0.99)

        rewards = []
        state, info = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, info = q.predict(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            q.update((state, action, reward, next_state, (truncated or terminated), info))
            rewards.append(reward)

        self.assertAlmostEqual(sum(rewards), 10)
        self.assertAlmostEqual(q.b.sum().item(), -1.1226010322)
        self.assertAlmostEqual(q.W.sum().item(), -2.3177227973)


if __name__ == "__main__":
    unittest.main()
