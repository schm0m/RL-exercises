import unittest
from pathlib import Path
import os


class TestAnswers(unittest.TestCase):
    def test_if_file_exists(self):
        path = Path("./plots/")

        self.assertTrue(path.is_dir())

        self.assertTrue(next(Path(path).iterdir(), None))
