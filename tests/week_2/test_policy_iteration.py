import unittest
import numpy as np

from rl_exercises.environments import MarsRover
from rl_exercises.week_2.policy_iteration import PolicyIteration
from rl_exercises.train_agent import evaluate


def get_agent() -> PolicyIteration:
    agent = PolicyIteration(env=MarsRover())
    return agent


class TestPolicyIteration(unittest.TestCase):
    def test_policy_quality(self):
        seeds = range(1,11)
        r = []
        steps = []
        for seed in seeds:
            env = MarsRover()
            agent = PolicyIteration(env=env, seed=seed)
            agent.update()
            # Get mean reward per episode
            mean_r = evaluate(env=env, agent=agent, episodes=1)  # deterministic policy
            r.append(mean_r)
            # Get the number of policy improvement steps
            steps.append(agent.steps)

        self.assertTrue(np.mean(steps) > 1)   
        self.assertTrue(sum(r) > 0)     

    # def test_single_update(self):
    #     new_qs, new_pi, converged = update_policy([[0, 0], [0, 0]], [0, 1], 0, 1, 1, 10)
    #     self.assertTrue(new_pi[0] == 1)
    #     self.assertTrue(new_qs[0][1] > 0)
    #     self.assertFalse(converged)

    # def test_update_consistency(self):
    #     for _ in range(10):
    #         pos = np.random.randint(7) + 1
    #         qs = np.random.randint(5, size=(8, 2))
    #         new_qs, new_pi, _ = update_policy(
    #             qs,
    #             np.random.randint(2, size=8),
    #             pos,
    #             pos + (1 * -np.random.randint(2)),
    #             np.random.randint(2),
    #             np.random.randint(10),
    #         )
    #         self.assertTrue(
    #             all(
    #                 [
    #                     np.any(np.where(new_qs[state] == new_qs[state].max()) == new_pi[state])
    #                     for state in np.arange(len(new_qs))
    #                 ]
    #             )
    #         )

    # def test_convergence(self):
    #     _, _, converged = update_policy([[1, 0], [1, 0]], [0, 0], 0, 1, 1, 0)
    #     self.assertTrue(converged)


if __name__ == "__main__":
    unittest.main()
