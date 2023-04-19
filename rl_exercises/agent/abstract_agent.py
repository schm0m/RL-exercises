from __future__ import annotations

import numpy as np
from abc import abstractmethod

class AbstractAgent(object):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def predict(
        observation
    ):
        ...

    @abstractmethod
    def save():
        ...

    @abstractmethod
    def load():
        ...

    @abstractmethod
    def update():
        ...