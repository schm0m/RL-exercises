import unittest
from pathlib import Path


class TestAnswers(unittest.TestCase):
    def test_if_file_exists(self):
        fn = Path("./answers.txt")
        self.assertTrue(fn.exists())

        with open(fn, "r") as file:
            content = file.read()
        print(content)
        self.assertTrue(len(content) > 0)
