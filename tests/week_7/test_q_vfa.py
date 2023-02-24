import unittest

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from q_learning_vfa import make_Q, q_learning, vfa_update


class TestQLearning(unittest.TestCase):
    def test_make_Q(self):
        # Test CartPole Environment
        env = gym.make("CartPole-v1")
        Q = make_Q(env)
        state, info = env.reset()
        v = Q(torch.tensor([state]))
        self.assertEqual(2, len(v[0]))

    def test_vfa_update(self):
        states = [[0, 0, 0, 0]]
        actions = [0]
        rewards = [1]
        dones = [True]
        next_states = [[0, 0, 0, 0]]

        Q = make_Q(gym.make("CartPole-v1"))
        optimizer = optim.SGD(Q.parameters(), lr=1)

        loss = vfa_update(Q, optimizer, states, actions, rewards, dones, next_states)
        self.assertAlmostEqual(1, round(loss))

    def test_q_learning(self):
        THRESHOLD = 30
        mean_rewards = []
        for _ in range(5):
            env = gym.make("CartPole-v1")
            rewards, Q = q_learning(env, 10000)
            mean_latest_rewards = np.mean(rewards[-100:])
            mean_rewards.append(mean_latest_rewards)
            if mean_latest_rewards > THRESHOLD:
                break
        self.assertGreaterEqual(max(mean_rewards), THRESHOLD)


if __name__ == "__main__":
    unittest.main()
