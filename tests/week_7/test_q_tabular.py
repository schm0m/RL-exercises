import unittest

import gymnasium as gym
import numpy as np

from q_learning_tabular import q_learning


class TestQLearningTabular(unittest.TestCase):
    def test_reward_length(self):
        env = gym.make("CartPole-v0")

        rewards, Q = q_learning(env, 12)
        assert len(rewards) == 12

    def test_deterministic_case1(self):
        env = gym.make("CartPole-v0")
        np.random.seed(0)
        _ = env.reset(seed=0)  # set the seed via reset once
        env.action_space.seed(0)

        rewards, Q = q_learning(env, 10)
        print(Q)
        self.assertAlmostEqual(np.mean(rewards), 29)
        self.assertAlmostEqual(len(Q), 312)

        expected_Q = {  # List is incomplete
            ((10, 10, 10, 10), 0): 1.6030842364127578,
            ((10, 10, 10, 10), 1): 3.7417997188313343,
            ((10, 12, 10, 9), 0): 0.15269040132219058,
            ((10, 12, 10, 9), 1): 6.465764256160958,
            ((10, 14, 10, 8), 0): 1.0483426387618406,
            ((10, 14, 10, 8), 1): 0.8775197900309218,
            ((10, 16, 10, 7), 0): -0.13608912218786462,
            ((10, 16, 10, 7), 1): 1.8636068846074139,
            ((10, 18, 9, 6), 0): 3.4116044709637943,
            ((10, 18, 9, 6), 1): -0.665239691095876,
            ((10, 16, 9, 7), 0): -0.957236684465528,
            ((10, 16, 9, 7), 1): 3.0532779175514544,
            ((10, 16, 8, 7), 0): 2.522348917892566,
            ((10, 16, 8, 7), 1): -0.8893378340991678,
            ((10, 14, 8, 8), 0): 2.023595707402134,
        }

        for key, value in expected_Q.items():
            self.assertAlmostEqual(value, Q[key])

    def test_deterministic_case2(self):
        env = gym.make("CartPole-v0")
        np.random.seed(42)
        _ = env.reset(seed=42)
        env.action_space.seed(42)

        rewards, Q = q_learning(env, 10)
        print(Q)

        self.assertAlmostEqual(np.mean(rewards), 21.3)
        self.assertAlmostEqual(len(Q), 238)
        expected_Q = {  # List is incomplete
            ((10, 9, 11, 11), 0): 1.352363962672832,
            ((10, 9, 11, 11), 1): 3.3095403370366485,
            ((10, 10, 11, 10), 0): 2.325656478255184,
            ((10, 10, 11, 10), 1): 0.3645829801587519,
            ((10, 7, 11, 12), 0): 2.234998549849135,
            ((10, 7, 11, 12), 1): -0.04951286326447568,
            ((10, 5, 11, 13), 0): -0.22370578944475894,
            ((10, 5, 11, 13), 1): 4.146980470039365,
            ((10, 7, 12, 12), 0): 3.3712752982542433,
            ((10, 7, 12, 12), 1): -0.5703519227860272,
            ((10, 5, 12, 13), 0): -0.18482913772408494,
            ((10, 5, 12, 13), 1): 2.551771175309658,
            ((10, 7, 13, 12), 0): 1.6703554549679853,
            ((10, 7, 13, 12), 1): -0.8977710745066665,
            ((10, 5, 13, 13), 0): 0.39077246165325863,
        }

        for key, value in expected_Q.items():
            self.assertAlmostEqual(value, Q[key])


if __name__ == "__main__":
    unittest.main()
