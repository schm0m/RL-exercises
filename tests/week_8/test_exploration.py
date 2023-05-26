import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from rl_exercises.week_8.exploration import policy_classes


class MockActionSpace(Mock):
    def sample(self):
        return -2


class MockEnv(Mock):
    action_space = MockActionSpace()


class MockQ(Mock):
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        qvalues = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int)
        return qvalues


class TestExploration(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_sample_duration(self):
        solution = {
            "greedy": (0,),
            "ε-greedy": (0,),
            "εz-greedy": (4,),
        }

        for pname, pclass in policy_classes.items():
            policy = pclass(
                Q=MockQ(),
                env=MockEnv(),
                seed=333,
            )
            duration = policy.sample_duration()
            sol = solution[pname]
            self.assertEqual(duration, sol[0])

    def test_action_selection(self):
        solution = {
            # (n, w, action)
            "greedy": [
                (0, -1, 4),
                (0, -1, 4),
                (0, -1, 4),
            ],
            "ε-greedy": [
                (0, -2, -2),
                (0, -2, -2),
                (0, -2, -2),
            ],
            "εz-greedy": [
                (5, -2, -2),
                (4, -2, -2),
                (3, -2, -2),
            ],
        }

        for pname, pclass in policy_classes.items():
            policy = pclass(
                Q=MockQ(),
                env=MockEnv(),
                seed=333,
            )
            for i in range(3):
                action = policy(state=np.random.rand(3), exploration_rate=1)
                ret = (policy.n, policy.w, action)
                self.assertTupleEqual(ret, solution[pname][i])


class TestFileExistance(unittest.TestCase):
    def test_answer_file_exists(self):
        fname = "answers.txt"  # root dir is the parent folder of test
        self.assertTrue(Path(fname).is_file())

    def test_plot_folder_exists(self):
        dirname = "plots"
        self.assertTrue(Path(dirname).is_dir())
        self.assertTrue(any(Path(dirname).iterdir()))


if __name__ == "__main__":
    unittest.main()
