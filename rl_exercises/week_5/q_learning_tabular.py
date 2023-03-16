from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, List, Tuple

import gymnasium as gym
import numpy as np

MAX_EPISODE_LENGTH = 200
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 1

BINS = 20
NUM_STATES = BINS**4

CART_POSITION = np.linspace(-4.8, 4.8, BINS)
CART_VELOCITY = np.linspace(-1, 1, BINS)
POLE_ANGLE = np.linspace(-0.418, 0.418, BINS)
POLE_ANGULAR_VELOCITY = np.linspace(-3, 3, BINS)


def to_bins(value: np.ndarray, bins: np.ndarray) -> float:
    """Put a single float value into closest bin"""
    return np.digitize(x=[value], bins=bins)[0]


def to_discrete_state(obs: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Transform an observation from continuous to discrete space"""
    x, v, theta, omega = obs
    state = (
        to_bins(x, CART_POSITION),
        to_bins(v, CART_VELOCITY),
        to_bins(theta, POLE_ANGLE),
        to_bins(omega, POLE_ANGULAR_VELOCITY),
    )
    return state


def policy(env: gym.Env, Q: DefaultDict[Tuple[Any, int], float], state, exploration_rate: float) -> int:
    """
    Act given a Q function

    Parameters
    ----------
    env: gym.Env
        The environment to act in
    Q: DefaultDict[Tuple[Any, int], float]
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
    q_values = [Q[(state, action)] for action in range(env.action_space.n)]
    return np.argmax(q_values).item()


def q_learning(
    env: gym.Env,
    num_episodes: int,
    exploration_rate: float = 0.5,
    exploration_rate_decay: float = 0.9,
    min_exploration_rate: float = 0.01,
    gamma: float = 0.9,
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
    Q = defaultdict(lambda: np.random.uniform(1, -1))

    rewards: list[float] = []
    print(f"Performing Q-learning with {NUM_STATES:d} states")
    for episode in range(num_episodes):
        rewards.append(0)
        obs, _ = env.reset()
        state = to_discrete_state(obs)

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(env, Q, state, exploration_rate)  # non-greedy/exploring action
            # env.render()

            obs, reward, terminated, truncated, _ = env.step(action)

            next_state = to_discrete_state(obs)
            # optimal_next_action = policy(
            #    env, Q, next_state, exploration_rate=exploration_rate
            # )  # <- Greedy or not greedy?

            # TODO: Implement Q-Learning Update
            # alpha = 1

            state = next_state

            rewards[-1] += reward
            if terminated or truncated:
                break

        exploration_rate = max(exploration_rate_decay * exploration_rate, min_exploration_rate)
        if episode % (num_episodes / 100) == 0:
            print(f"Episode {episode}:  Mean Reward: {np.mean(rewards[-int(num_episodes / 100):])}")
    return rewards, Q


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    q_learning(env, 10000)
