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
        return train_sb3_sac(cfg)
    else:
        # TODO: add your agent options here
        raise NotImplementedError
    
    agent = None
    TODO: make buffer/trajectory
    state, info = env.reset()

    num_episodes = 0
    for step in range(cfg.training_steps):
        action = agent.predict(state)
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
    
    final_eval = evaluate(env, policy, cfg.n_eval_episodes)
    print(final_eval)
    return final_eval

def train_sb3_sac(cfg):
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