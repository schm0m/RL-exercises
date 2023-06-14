class AbstractBuffer:
    def __init__(self) -> None:
        pass

    def add(self, state, action, reward, next_state, done, info):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError
    
    
class SimpleBuffer(AbstractBuffer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.transition = None

    def __len__(self):
        return 1

    def add(self, state, action, reward, next_state, done, info):
        self.transition = (state, action, reward, next_state, done, info)

    def sample(self, *args, **kwargs):
        # Batch size 1
        return [self.transition]