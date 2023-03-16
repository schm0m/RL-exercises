from __future__ import annotations

from typing import Any, DefaultDict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# This could be useful to you:
# import torch.nn.functional as F
import torch.optim as optim


MAX_EPISODE_LENGTH = 200
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1
BATCH_SIZE = 64


def make_Q(env: gym.Env) -> nn.Module:
    """
    TODO: Create Q-Function from env.

    The Q-function is of class `nn.Module`.

    Use `env.observation_space.shape` to get the shape of the input data.
    Use `env.action_space.n` to get number of possible actions for this environment.

    Parameters
    ----------
    env: gym.Env
        The environment the Q function is meant for
    
    Returns
    -------
    Q
        An intialized policy
    """
    Q = ...
    return Q


def policy(env: gym.Env, Q: nn.Module, state: np.array, exploration_rate: float):
    """
    Act given a Q function

    Parameters
    ----------
    env: gym.Env
        The environment to act in
    Q: nn.Module
        The Q function
    state: np.array
        The current state
    exploration_rate: float
        Exploration epsilon

    Returns
    -------
    action_id
        ID of the action to take
    """
    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample()
    q_values = Q(torch.from_numpy(state).float()).detach().numpy()
    return np.argmax(q_values)


def vfa_update(Q: nn.Module, optimizer: optim.Optimizer, states: np.array, actions: np.array, rewards: np.array, dones: np.array, next_states: np.array) -> float:
    """
    TODO: Implement Value Function Training Step

    Parameters
    ----------
    Q: nn.Module
        Q function to update
    optimizer: optim.Optimizer
        Optimiyer to use
    states: np.array
        State Batch
    actions: np.array
        Action Batch
    rewards: np.array
        Reward Batch
    dones: np.array
        Termination Signal Batch
    next_states: np.array
        Next State Batch

    Returns
    -------
    loss
        Update loss
    """
    # Convert data into torch tensors

    # Compute MSE loss

    # Optimize the model

    loss = 0
    return float(loss)


def q_learning(
    env: gym.Env,
    num_episodes: int,
    exploration_rate: float = 0.5,
    exploration_rate_decay: float = 0.9,
    min_exploration_rate: float = 0.01,
) -> Tuple[List[float], DefaultDict[Tuple[Any, int], float]]:
    """
    TODO: Implement Q Learning

    Parameters
    ----------
    env: gym.Env
        Training Environment
    num_episodes: int
        Number of Training Episodes
    exploration_rate: float
        Epsilon
    exploration_rate_decay: float
        Epsilon Decay Rate
    min_exploration_rate: float
        Minimum Epsilon

    Returns
    -------
    rewards
        Training rewards
    Q
        Trained Q function
    """
    Q = make_Q(env)
    optimizer = optim.SGD(Q.parameters(), lr=LEARNING_RATE)

    rewards = []
    vfa_update_data = []
    for episode in range(num_episodes):
        rewards.append(0)
        obs, _ = env.reset()
        state = obs

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(env, Q, state, exploration_rate)

            obs, reward, terminated, truncated, _ = env.step(action)

            next_state = obs
            vfa_update_data.append((state, action, reward, terminated, next_state))

            state = next_state

            rewards[-1] += reward

            if len(vfa_update_data) >= BATCH_SIZE:
                vfa_update(Q, optimizer, *zip(*vfa_update_data))
                vfa_update_data.clear()

            if terminated or truncated:
                break

        exploration_rate = max(exploration_rate_decay * exploration_rate, min_exploration_rate)

        if episode % (num_episodes / 100) == 0:
            print(f"Episode {episode}:  Mean Reward: {np.mean(rewards[-int(num_episodes / 100):])}")
            print(f"Exploration rate: {exploration_rate:.4f}")

    print(f"{num_episodes:6d}: Mean Reward: ", np.mean(rewards[-int(num_episodes / 100) :]))
    return rewards, Q


def plot_rewards(rewards):
    """
    Plot Training Rewards

    Parameters
    ----------
    rewards: np.array
        Training Rewards
    """
    import matplotlib.pyplot as plt

    plt.plot(rewards, ".", label="Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    rew, q = q_learning(env, 10000)
    plot_rewards(rew)
