from __future__ import annotations

from typing import Any
from abc import abstractmethod


class AbstractAgent(object):
    def __init__(self, *args: tuple[Any], **kwargs: dict) -> None:
        pass

    @abstractmethod
    def predict(self, *args: tuple[Any], **kwargs: dict) -> tuple[Any, dict]:
        ...

    @abstractmethod
    def save(self, *args: tuple[Any], **kwargs: dict) -> Any:
        ...

    @abstractmethod
    # TODO what is the return type? A callable?
    def load(self, *args: tuple[Any], **kwargs: dict) -> Any:
        ...

    @abstractmethod
    def update(self, *args: tuple[Any], **kwargs: dict) -> Any | None:
        ...
