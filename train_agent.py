import hydra
import gymnasium as gym

@hydra.main("rl_exercises/configs", "base", version_base="1.1")
def train(cfg):
    env = gym.make(cfg.env_name)
    if cfg.agent == 'sb3':
        pass
    else:
        # TODO: add your agent options here
        raise NotImplementedError
    
    for _ in range(cfg.training_steps):
        pass

def evaluate(env, policy, n_episodes):
    pass

if __name__ == "__main__":
    train()