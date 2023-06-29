from __future__ import annotations

import numpy as np
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
    def load(self, *args: tuple[Any], **kwargs: dict) -> Any:
        ...

    @abstractmethod
    def update(self, *args: tuple[Any], **kwargs: dict) -> Any | None:
        ...