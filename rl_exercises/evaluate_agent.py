import hydra
import gymnasium as gym

@hydra.main("rl_exercises/configs", "eval", version_base="1.1")
def evaluate(cfg):
    env = gym.make(cfg.env_name)
    for episode in cfg.num_episodes:
        pass

def load_policy(path):
    pass