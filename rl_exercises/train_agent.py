import hydra
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import compiler_gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from typing import  List
from tqdm import tqdm


@hydra.main("rl_exercises/configs", "base", version_base="1.1")
def train(cfg):
    env = make_env(cfg.env_name)
    if cfg.agent == 'sb3':
        return train_sb3_sac(env, cfg)
    else:
        # TODO: add your agent options here
        raise NotImplementedError
    
    #TODO: make agent
    agent = None
    buffer = getattr(agent.buffer, cfg.buffer_cls)(cfg.buffer_kwargs)
    state, info = env.reset()

    num_episodes = 0
    for step in range(cfg.training_steps):
        action, info = agent.predict(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        if buffer is not None:
            buffer.add(state, action, reward, next_state, (truncated or terminated), info)

        if len(buffer) > cfg.batch_size or (cfg.update_after_episode_end and (terminated or truncated)):
            batch = buffer.sample(cfg.batch_size)
            agent.update(batch)

        state = next_state

        if terminated or truncated:
            state = env.reset()

        if step % cfg.eval_every_n_steps == 0:
            eval_performance = evaluate(env, policy, cfg.n_eval_episodes)
            print(f"Eval reward after {step} steps was {eval_performance}.")
    
    agent.save(cfg.outpath)
    final_eval = evaluate(env, policy, cfg.n_eval_episodes)
    print(final_eval)
    return final_eval

def train_sb3_sac(env, cfg):
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
                pbar.set_postfix({"episode reward": episode_rewards[-1], "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)

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