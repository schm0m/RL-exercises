import gymnasium as gym


class VacuumEnv(gym.Env):
    """So far, a non-functional env"""

    def __init__(self):
        """
        Use this function to initialize the environment
        """
        self.action_space: gym.spaces.Space = None
        self.observation_space: gym.spaces.Space = None
        self.reward_range: gym.spaces.Space = None

    def reset(self):
        """Reset the environment"""
        state = []
        info = {}
        return state, info

    def step(self, action):
        """This should move the vacuum"""
        state = action
        reward = 0
        terminated = True
        truncated = True
        info = {}
        return state, reward, terminated, truncated, info

    def close(self):
        """Make sure environment is closed"""
        return True
