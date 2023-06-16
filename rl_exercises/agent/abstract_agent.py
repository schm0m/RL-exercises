from __future__ import annotations

import numpy as np
from typing import Any
from abc import abstractmethod

class AbstractAgent(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(*args, **kwargs) -> tuple[Any, dict]:
        ...

    @abstractmethod
    def save(*args, **kwargs) -> Any:
        ...

    @abstractmethod
    def load(*args, **kwargs) -> Any:
        ...

    @abstractmethod
    def update(*args, **kwargs) -> Any:
        ...