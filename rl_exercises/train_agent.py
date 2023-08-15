# Ignore "imported but unused"
# flake8: noqa: F401
import hydra
import numpy as np
from rich import print as printr
import gymnasium as gym
import rl_exercises
from gymnasium.wrappers import TimeLimit
import warnings

try:
    import compiler_gym
except:
    warnings.warn("Could not import compiler_gym. Probably it is not installed.")
from stable_baselines3.common.monitor import Monitor
from rl_exercises.environments import MarsRover
from stable_baselines3 import SAC, PPO
from typing import List
from tqdm import tqdm
from functools import partial

from rl_exercises.week_2 import PolicyIteration, ValueIteration
from rl_exercises.week_4 import EpsilonGreedyPolicy as TabularEpsilonGreedyPolicy
from rl_exercises.week_5 import TabularQAgent, VFAQAgent, EpsilonGreedyPolicy
from rl_exercises.week_6 import DQN


@hydra.main("configs", "base", version_base="1.1")
def train(cfg):
    env = make_env(cfg.env_name)
    printr(cfg)
    if cfg.agent == "sb3":
        return train_sb3(env, cfg)
    elif cfg.agent in ["policy_iteration", "value_iteration"]:
        agent = eval(cfg.agent_class)(env=env, **cfg.agent_kwargs)
    elif cfg.agent in ["tabular_q_learning", "linear_q_learning", "dqn"]:
        policy_class = eval(cfg.policy_class)
        policy = partial(policy_class, **cfg.policy_kwargs)
        agent_class = eval(cfg.agent_class)
        agent = agent_class(env, policy, **cfg.agent_kwargs)
    else:
        # TODO: add your agent options here
        raise NotImplementedError

    buffer_cls = getattr(rl_exercises.agent.buffer, cfg.buffer_cls, None)
    if buffer_cls is not None:
        buffer = buffer_cls(cfg.buffer_kwargs)
    state, info = env.reset()

    num_episodes = 0
    for step in range(cfg.training_steps):
        action, info = agent.predict(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, (truncated or terminated), info)

        if len(buffer) > cfg.batch_size or (cfg.update_after_episode_end and (terminated or truncated)):
            batch = buffer.sample(cfg.batch_size)
            agent.update(batch)

        state = next_state

        if terminated or truncated:
            state = env.reset()

        if step % cfg.eval_every_n_steps == 0:
            eval_performance = evaluate(env, agent, cfg.n_eval_episodes)
            print(f"Eval reward after {step} steps was {eval_performance}.")

    agent.save(cfg.outpath)
    final_eval = evaluate(env, agent, cfg.n_eval_episodes)
    print(final_eval)
    return final_eval


def train_sb3(env, cfg):
    # Create agent
    model = eval(cfg.agent_class)("MlpPolicy", env, verbose=cfg.verbose, tensorboard_log=cfg.log_dir, seed=cfg.seed)

    # Train agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save agent
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gym.make(cfg.env_id))
    means = evaluate(env, model, n_episodes=cfg.n_eval_episodes)
    performance = np.mean(means)
    return performance


def evaluate(env: gym.Env, agent, episodes=100):
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
        obs, info = env.reset()
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict(obs, info)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                done = True
                pbar.set_postfix({"episode reward": episode_rewards[-1], "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


def make_env(env_name, env_kwargs={}):
    if "compiler" in env_name:
        benchmark = "cbench-v1/dijkstra"
        env = gym.make(
            "llvm-autophase-ic-v0",
            benchmark=benchmark,
            reward_space="IrInstructionCountNorm",
            apply_api_compatibility=True,
        )
        env = TimeLimit(env, max_episode_steps=100)
    elif env_name == "MarsRover":
        env = MarsRover(**env_kwargs)
        # env = TimeLimit(env, max_episode_steps=env.horizon)
    else:
        env = gym.make(env_name, **env_kwargs)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    train()
