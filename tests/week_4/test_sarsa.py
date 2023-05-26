import time
import unittest
from collections import defaultdict

import numpy as np

from rl_exercises.environments import FallEnv
from rl_exercises.week_4.sarsa import choose_action, make_epsilon_greedy_policy, sarsa, td_update


class TestSARSA(unittest.TestCase):
    def evaluate_probabilites(self, probability_factory, expected_distribution, error_margin=0.006, n=40000):
        np.random.seed(int(time.time()))  # Reset random seed for sampling
        samples_taken = np.zeros(len(expected_distribution))
        for i in range(n):
            p = probability_factory()
            samples_taken[choose_action(p)] += 1

        sample_distribution = samples_taken / np.sum(samples_taken)
        print("T:", expected_distribution)
        print("S:", sample_distribution)
        errors = np.abs(np.asarray(expected_distribution) - sample_distribution)
        print("E:", errors)
        for i in range(len(expected_distribution)):
            self.assertGreater(error_margin, errors[i])

    def test_td_update(self):
        Q = defaultdict(lambda: np.zeros(2))
        env = FallEnv()

        # Test single step
        state = env.reset()
        next_state, reward, done, _ = env.step(0)
        target = td_update(Q, state, 0, reward, next_state, next_action=1, gamma=0.9, alpha=0.1, done=done)
        self.assertEqual(target, 0)

        # Test 10 steps
        for i in range(10):
            state = next_state
            next_state, reward, done, _ = env.step(2)
        target = td_update(Q, state, 1, reward, next_state, next_action=1, gamma=0.9, alpha=0.1, done=done)
        self.assertNotEqual(target, 0)

        # Test done
        Q[0] = [0, 1]
        Q[1] = [2, 0]
        td_done = td_update(Q, 0, 1, 5, 1, 0, gamma=0.8, alpha=0.2, done=True)
        td_not_done = td_update(Q, 0, 1, 5, 1, 0, gamma=0.8, alpha=0.2, done=False)

        self.assertAlmostEqual(1.8, td_done)
        self.assertAlmostEqual(2.12, td_not_done)

    def test_sarsa(self):
        env = FallEnv()
        rewards, lengths, Q = sarsa(env, 10)
        self.assertEqual(len(lengths), len(rewards))
        self.assertGreater(sum(rewards), 0)
        q_unpacked = np.asarray([val for val in Q.values()])
        q_values_changed = np.sum(q_unpacked != 0)
        self.assertGreater(q_values_changed, 0)
        self.assertEqual(np.sum(Q[env.goal]), 0)

    def test_exploration(self):
        Q = defaultdict(lambda: np.zeros(2))
        env = FallEnv()
        state = env.reset()
        policy = make_epsilon_greedy_policy(Q, 0.5, 4)
        actions = []
        for _ in range(10):
            actions.append(policy(state))
        equalities = [np.array_equal(actions[0], actions[k]) for k in range(len(actions))]
        self.assertFalse(all(equalities))

    def test_make_epsilon_greedy_policy(self):
        print("Test make epsilon greedy policy")
        Q = defaultdict(lambda: np.zeros(2))
        n_actions = 5

        # Fill Q with pseudo-random values
        # np.random.seed(8)
        # n_states = 10
        # for i in range(n_states):
        #     Q[i] = np.random.random(n_actions)
        # print(Q)

        Q.update(
            {
                0: np.asarray([0.8734294, 0.96854066, 0.86919454, 0.53085569, 0.23272833]),
                1: np.asarray([0.0113988, 0.43046882, 0.40235136, 0.52267467, 0.4783918]),
                2: np.asarray([0.55535647, 0.54338602, 0.76089558, 0.71237457, 0.6196821]),
            }
        )

        # Evaluate policy with no exploration
        policy = make_epsilon_greedy_policy(Q, 0, n_actions)
        self.evaluate_probabilites(lambda: policy(0), [0, 1, 0, 0, 0])

        # Evaluate policy with some exploration
        policy = make_epsilon_greedy_policy(Q, 0.2, n_actions)
        self.evaluate_probabilites(lambda: policy(1), [0.04, 0.04, 0.04, 0.84, 0.04])

        # Evaluate policy with full exploration
        policy = make_epsilon_greedy_policy(Q, 1, n_actions)
        self.evaluate_probabilites(lambda: policy(2), [0.2, 0.2, 0.2, 0.2, 0.2])

    def test_choose_action(self):
        print("Test choose action")
        p = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.evaluate_probabilites(lambda: p, p)

        p = [0.2, 0.5, 0.1, 0.2]
        self.evaluate_probabilites(lambda: p, p)

        p = [0.2, 0.3, 0, 0.1, 0, 0.2, 0.2]
        self.evaluate_probabilites(lambda: p, p)

        p = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertRaises(Exception, lambda: self.evaluate_probabilites(lambda: p, p))


if __name__ == "__main__":
    unittest.main()
