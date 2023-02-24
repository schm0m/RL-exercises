from __future__ import annotations
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_EPISODE_LENGTH = 1000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64

"""Implement DQN"""



def make_Q(env: gym.Env) -> nn.Module:
    """Create Q-Function from env.

    Use 1 hidden layer with 64 units.
    Use ReLU as an activation function after all layers except the last.
    Use `env.observation_space.shape` to get the shape of the input data.
    Use `env.action_space.n` to get number of possible actions for this environment.

    Parameters
    ----------
    env : gym.Env
        Environment

    Returns
    -------
    nn.Module
        State-Value Function
    """
    # TODO create a deep network as a function approximator

    Q = ...

    return Q


def policy(Q: nn.Module, env: gym.Env, state: np.ndarray | torch.Tensor, exploration_rate: float = 0.) -> int:
    """Epsilon-greedy Policy

    Based on the Q function select an action in an epsilon-greedy way.

    Parameters
    ----------
    Q : nn.Module
        State-Value Function
    env : gym.Env
        Environment
    state : np.ndarray | torch.Tensor
        State
    exploration_rate : float
        Exploration rate (epsilon)

    Returns
    -------
    int
        action
    """

    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample()
    q_values = Q(torch.from_numpy(state).float()).detach().numpy()
    action = np.argmax(q_values)
    return action


def vfa_update(Q: nn.Module, optimizer: optim.Optimizer, states: list[np.array], actions: list[int], rewards: list[float], dones: list[bool], next_states: list[np.array]) -> float:
    """Value Function Update for a Batch of Transitions

    Use MSE loss.

    Parameters
    ----------
    Q : nn.Module
        State-Value Function represented by a DNN.
    optimizer : optim.Optimizer
        Torch optimizer.
    states : list[np.array]
        Batch  of states
    actions : list[int]
        Batch of actions
    rewards : list[float]
        Batch of rewards
    dones : list[bool]
        Batch of dones
    next_states : list[np.array]
        Batch of next states

    Returns
    -------
    float
        Loss
    """
    states = torch.from_numpy(np.array(states)).float()
    actions = torch.from_numpy(np.array(actions)).unsqueeze(-1)
    rewards = torch.from_numpy(np.array(rewards)).float()
    dones = torch.from_numpy(np.array(dones)).float()
    next_states = torch.from_numpy(np.array(next_states)).float()

    q_values = torch.gather(Q(states), dim=-1, index=actions).squeeze()
    target_q_values = (
        rewards + (1 - dones) * DISCOUNT_FACTOR * Q(next_states).max(dim=-1)[0].detach()
    )
    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def q_learning(env: gym.Env, num_episodes: int, exploration_rate: float = 0.1, Q: nn.Module | None = None) -> tuple[nn.Module, list[float]]:
    """Deep Q-Learning

    Parameters
    ----------
    env : gym.Env
        Environment
    num_episodes : int
        Number of episodes to train on.
    exploration_rate : float, optional
        Epsilon for epsilon-greedy policy, by default 0.1
    Q : nn.Module | None, optional
        State-Value Function as DNN, by default None

    Returns
    -------
    tuple[nn.Module, list[float]]
        Q, cumulative rewards per episode 
    """
    if Q is None:
        Q = make_Q(env)

    optimizer = optim.Adam(Q.parameters(), lr=5e-4)
    
    # TODO Create replay buffer with a maximum size of 100000
    

    rewards: list[float] = []
    for episode in range(num_episodes):
        rewards.append(0)
        obs = env.reset()
        state = obs

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(Q, env, state, exploration_rate)

            obs, reward, done, _ = env.step(action)

            next_state = obs

            #TODO Add transition to replay buffer 
            
            
            state = next_state

            rewards[-1] += reward

            if len(replay_buffer) >= BATCH_SIZE:
                # TODO Select batch from replay buffer with size BATCH_SIZE 
                

                # TODO Perform a value function update
                

            if done:
                break

        if episode % (num_episodes / 100) == 0:
            print(f"Episode {episode}, Mean Reward: {np.mean(rewards[-int(num_episodes / 100) :])}")
    return Q, rewards


def evaluate(Q: nn.Module, env: gym.Env, n_episodes: int = 1) -> list[float]:
    """Collect rewards on test episodes
    
    Parameters
    ----------
    Q : nn.Module
        State value function
    env : gym.Env
        Environment
    n_episodes : int, optional
        Number of episodes, by default 1

    Returns
    -------
    list[float]
        cumulative rewards on test episods
    """
    
    cumulative_rewards = []
    for i in range(n_episodes):
        cum_rew = 0
        obs = env.reset()
        done = False
        while not done:
            action = policy(Q, env, obs, exploration_rate=0.)  # greedy
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render(mode="human")
            if done:
                break
        cumulative_rewards.append(cum_rew)
                
    env.close()
    return cumulative_rewards


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    Q, rewards = q_learning(env, num_episodes=250)
    cumulative_rewards = evaluate(Q, env, n_episodes=1)

    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("Training")
    plt.show()

