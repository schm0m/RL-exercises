class AbstractBuffer:
    def __init__(self) -> None:
        pass

    def add(self, state, action, reward, next_state, done, info):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    
class SimpleBuffer(AbstractBuffer):
    def __init__(self) -> None:
        super().__init__()
        self.transition = None

    def add(self, state, action, reward, next_state, done, info):
        self.transition = (state, action, reward, next_state, done, info)

    def sample(self):
        return self.transition