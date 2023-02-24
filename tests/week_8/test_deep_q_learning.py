import unittest
import copy

import gym

from deep_q_learning import make_Q, q_learning


def check_nets(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False

    return True


# NOTE: This test simply checks whether the state of the Q network has changed or not.
class TestDeepQLearning(unittest.TestCase):
    def test_deep_q_learning(self):
        env = gym.make("LunarLander-v2")
        
        Q = make_Q(env)

        # Copy Q
        state_before_training = Q.state_dict()
        Q_before_training = make_Q(env)
        Q_before_training.load_state_dict(state_before_training)

        # Train
        Q, _ = q_learning(env, num_episodes=64, exploration_rate=0.1, Q=Q)

        # Check whether something was learned
        assert not check_nets(Q, Q_before_training)


if __name__ == "__main__":
    unittest.main()
