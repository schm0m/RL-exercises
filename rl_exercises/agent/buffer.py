from __future__ import annotations
from typing_extensions import TypeAlias
from abc import ABC, abstractmethod
import numpy as np

# TODO Do we use "done" here which is terminated | truncated or the latter two?
# state, action, reward, next_state, done, info
Transition = TypeAlias[tuple[np.ndarray, int | float, float, np.ndarray, bool, dict]]


class AbstractBuffer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def add(
        self, state: np.ndarray, action: int | float, reward: float, next_state: np.ndarray, done: bool, info: dict
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args: tuple, **kwargs: dict) -> list[Transition]:
        # TODO Should we require a batch size in the class and always return list[Transition] or Iterable[Transition]?
        raise NotImplementedError


class SimpleBuffer(AbstractBuffer):
    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        super().__init__()
        self.transition: Transition | None = None

    def __len__(self) -> int:
        return 1

    def add(
        self, state: np.ndarray, action: int | float, reward: float, next_state: np.ndarray, done: bool, info: dict
    ) -> None:
        self.transition = (state, action, reward, next_state, done, info)

    def sample(self, *args: tuple, **kwargs: dict) -> list[Transition]:
        # Batch size 1
        return [self.transition]
