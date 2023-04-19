import hydra
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import compiler_gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC


@hydra.main("rl_exercises/configs", "base", version_base="1.1")
def train(cfg):
    env = make_env(cfg.env_name)
    if cfg.agent == 'sb3':
        # Create agent
        model = SAC("MlpPolicy", env, verbose=cfg.verbose, tensorboard_log=cfg.log_dir, seed=cfg.seed)

        # Train agent
        model.learn(total_timesteps=cfg.total_timesteps)

        # Save agent
        model.save(cfg.model_fn)

        # Evaluate
        env = Monitor(gym.make(cfg.env_id))
        means = evaluate(env, model, n_episodes=cfg.n_eval_episodes)
        performance = np.mean(means)
        return performance
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
    env = Monitor(env)
    return env


if __name__ == "__main__":
    train()