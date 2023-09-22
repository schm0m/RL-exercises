from __future__ import annotations

from typing import Any
from abc import abstractmethod


class AbstractAgent(object):
    def __init__(self, *args: tuple[Any], **kwargs: dict) -> None:
        pass

    @abstractmethod
    def predict(self, *args: tuple[Any], **kwargs: dict) -> tuple[Any, dict]:
        """Predict action based on observation."""
        ...

    @abstractmethod
    def save(self, *args: tuple[Any], **kwargs: dict) -> Any:
        """Save agent to the disk."""
        ...

    @abstractmethod
    # TODO what is the return type? A callable?
    def load(self, *args: tuple[Any], **kwargs: dict) -> Any:
        """Load agent from the disk."""
        ...

    @abstractmethod
    def update(self, *args: tuple[Any], **kwargs: dict) -> Any | None:
        """Td update of the agent."""
        ...
