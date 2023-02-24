import gym

class VacuumEnv(gym.Env):
    """
    So far, a non-functional env
    """
    def __init__(self):
        """
        Use this function to initialize the environment
        """
        self.action_space = None
        self.observation_space = None
        self.reward_range = None

    def reset(self):
        """
        Reset the environment
        """
        state = []
        return state

    def step(self, action):
        """
        This should move the vacuum
        """
        state = action
        reward = 0
        done = True
        meta_info = {}
        return state, reward, done, meta_info

    def close(self):
        """
        Make sure environment is closed
        """
        return True
