import hydra
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import compiler_gym


@hydra.main("rl_exercises/configs", "base", version_base="1.1")
def train(cfg):
    env = make_env(cfg.env_name)
    if cfg.agent == 'sb3':
        pass
    else:
        # TODO: add your agent options here
        raise NotImplementedError
    
    for step in range(cfg.training_steps):
        pass
        
        if step % cfg.eval_every_n_steps == 0:
            evaluate(env, policy, cfg.n_eval_episodes)

def evaluate(env, policy, n_episodes):
    raise NotImplementedError

def make_env(env_name):
    if "compiler" in env_name:
        benchmark = "cbench-v1/dijkstra"
        env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark, reward_space="IrInstructionCountNorm", apply_api_compatibility=True)
        env = TimeLimit(env, max_episode_steps=100)
    else:
        env = gym.make(env_name)


if __name__ == "__main__":
    train()