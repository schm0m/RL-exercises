import hydra
from typing import Callable, List
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from train_agent import make_env


@hydra.main("rl_exercises/configs", "eval", version_base="1.1")
def evaluate_agent(cfg):
    env = make_env(cfg.env_name)
    policy = load_policy(cfg.policy_path)
    return_mean = evaluate(env, policy, cfg.num_episodes)
    print(return_mean)

def load_policy(path):
    raise NotImplementedError

def evaluate(env: gym.Env, policy: Callable[[np.ndarray], int], episodes=100):
    """
    Evaluate a given Policy on an Environment

    Parameters
    ----------
    env: gym.Env
        Environment to evaluate on
    policy: Callable[[np.ndarray], int]
        Policy to evaluate
    episodes: int
        Evaluation episodes

    Returns
    -------
    mean_rewards
        Mean evaluation rewards
    """
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                pbar.set_postfix({"episode reward": episode_rewards[-1], "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


if __name__ == "__main__":
    evaluate_agent()
