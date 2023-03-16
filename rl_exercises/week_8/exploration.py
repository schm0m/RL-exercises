from __future__ import annotations

import random
from collections import deque
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper

from plotting import animate


def epsilon_decay(episode: int, epsilon: float) -> float:
    """
    Decay epsilon value across episodes

    Parameters
    ----------
    episode : int
        Current episode
    epsilon : float
        Base epsilon
    """
    # TODO check formula
    eps = epsilon * np.exp(-0.1 * episode)
    return eps


class QFunction(nn.Module):
    """State Value Function"""

    def __init__(self, env: gym.Env, n_hidden_units: int = 128) -> None:
        """Init

        Parameters
        ----------
        env : gym.Env
            Environment
        n_hidden_units : int, optional
            Number of units in hidden layers, by default 128
        """
        super().__init__()
        in_features = env.observation_space.shape[-1]
        out_features = env.action_space.n
        self.linear1 = nn.Linear(in_features, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.linear3 = nn.Linear(n_hidden_units, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward tensor through network

        Parameters
        ----------
        x : torch.Tensor
            State

        Returns
        -------
        torch.Tensor
            Action values
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class Policy(object):
    """Policy for Exploration with ε(z)-greedy"""

    def __init__(
        self,
        Q: nn.Module,
        env: gym.Env,
        duration_max: int = 100,
        mu: float = 3,
        seed: int = None,
        disable_exploration: bool = False,
    ) -> None:
        """Init

        Parameters
        ----------
        Q : nn.Module
            State-Value function
        env : gym.Env
            Environment
        duration_max : int, optional
            Maximum number of action repetition, by default 100
        mu : float, optional
            Zeta/Zipf distribution parameter, by default 2
        seed : int, optional
            Seed, by default None
        disable_exploration : bool, optional
            Act purely greedy if true, by default False
        """
        self.Q = Q
        self.env = env
        self.duration_max = duration_max
        self.mu = mu
        self.disable_exploration = disable_exploration

        self.n: int = 0  # number of times left to perform action
        self.w: int = -1  # random action in memory
        self.rng = np.random.default_rng(seed=seed)

    def sample_duration(self) -> int:
        """Sample duration from a zeta/zipf distribution

        The duration is capped at `self.duration_max`.

        Returns
        -------
        int
            duration (how often the action is repeated)
        """
        duration = ...
        # TODO implement sampling
        return duration

    def __call__(self, state: np.array, exploration_rate: float = 0.0) -> int:
        """Select action

        εz-greedy algorithm B.1 [Dabney et al., 2020].
        The while loop is happening outside, in the training loop.
        This is what is inside the while loop.

        Parameters
        ----------
        state : np.array
            State
        exploration_rate : float, optional
            exploration rate (epsilon), by default 0.0

        Returns
        -------
        int
            action
        """
        action = ...
        # TODO implement algorithm

        return action


def to_tensor(arr: np.ndarray, dtype=None) -> torch.Tensor:
    """Convert numpy array to tensor

    Parameters
    ----------
    arr : np.ndarray
        Numpy array
    dtype : _type_, optional
        dtype, by default None

    Returns
    -------
    torch.Tensor
        Converted array
    """
    return torch.tensor(arr, dtype=dtype)


def vfa_update(optimizer: torch.optim.Optimizer, Q: nn.Module, batch: list[np.array], gamma: float = 0.99) -> float:
    """Value-Function update for a batch

    Loss: MSE

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for Q function.
    Q : nn.Module
        State-Value function.
    batch : list[np.array]
        Batch. Must be in form [states, actions, rewards, next_states, dones].
    gamma : float
        Discount factor, default 0.99.

    Returns
    -------
    float
        Loss
    """
    states, actions, rewards, next_states, dones = zip(*batch)

    states = to_tensor(states)
    actions = to_tensor(actions, dtype=int).reshape((-1, 1))
    rewards = to_tensor(rewards)
    next_states = to_tensor(next_states)
    dones = to_tensor(dones, dtype=bool)

    # Calculate state values
    state_values = Q(states)
    state_values = torch.gather(state_values, -1, actions).squeeze()

    # Calculate next state values
    next_state_values = torch.max(Q(next_states))

    # Calculate targets
    targets = rewards + gamma * next_state_values * (~dones)

    # MSE loss
    loss = F.mse_loss(torch.squeeze(state_values), torch.squeeze(targets))

    # Step optimizer
    optimizer.zero_grad()  # reset gradients from previous iteration
    loss.backward()  # propagate new gradients
    optimizer.step()  # step optimizer

    return float(loss)


def evaluate(policy: Policy, env: gym.Env, render: bool = False, seed: int | None = None) -> list[float]:
    """Evaluate Policy (Acting Greedy)

    One episode.

    Parameters
    ----------
    policy : Policy
        Policy selecting actions.
    env : gym.Env
        Environment
    render : bool, optional
        Render if true and collect and animate frames, by default False

    Returns
    -------
    list[float]
        Reward per step.
    """
    observation, _ = env.reset(seed=seed)
    action: int = 0
    reward: float = 0
    truncated, terminated = False, False
    rewards: list[float] = []
    frames = []
    while not (truncated or terminated):
        action = policy(state=observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if render:
            frame = env.render()
            frames.append(frame)
    if render:
        animate(frames)
    return rewards


def make_env(env_id: str, max_episode_steps: int, deque_size_recorder: int = 500) -> MiniGridEnv:
    """Create grid environment and wrap

    Parameters
    ----------
    env_id : str
        Environment id, see `minigrid.__init__.py` for options.
    max_episode_steps : int
        Maximum number of steps per episode (cutoff).
    deque_size_recorder : int, optional
        Number of episodes to record rewards for, by default N_EPISODES

    Returns
    -------
    MiniGridEnv
        Grid environment
    """
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        max_steps=max_episode_steps,
    )
    env = FlatObsWrapper(env=env)  # flatten 7x7x3 observations to vector
    env = RecordEpisodeStatistics(env=env, deque_size=deque_size_recorder)

    return env


def train(
    policy_class: type,
    env_id: str = "MiniGrid-Empty-Random-5x5-v0",
    max_env_steps: int = 25000,
    max_episode_steps: int = 100,
    n_eval_episodes: int = 5,
    epsilon: float = 0.2,
    batch_size: int = 128,
    seed: int | None = 499,
    learning_rate: float = 0.001,
    buffer_size: int = 50000,
    gamma: float = 0.999,
):
    """
    Train a Policy

    Parameters
    ----------
    policy_class: 
        Policy Class to Train
    env_id: str 
        Environment ID
    max_env_steps: int
        Maximum number of steps
    max_episode_steps: int
        Maximum number of steps per episode
    n_eval_episodes: int
        Number of evaluation episodes
    epsilon: float
        Epsilon for exploration
    batch_size: int
        Batch Size
    seed: int | None
        Random seed
    learning_rate: float
        Learning Rate
    buffer_size: int
        Replay Buffer Size
    gamma: float
        Discount Factor

    Returns
    -------
    visited_positions 
        Positions the agent saw in training
    losses
        Losses during training
    cum_rewards
        Cummulative rewards
    env
        Environment
    actions
        Actions taken during training
    dones
        Episode ending signals
    """
    torch.manual_seed(seed)

    # Initialize env and display starting state and env info
    env = make_env(env_id=env_id, max_episode_steps=max_episode_steps)
    observation, _ = env.reset(seed=seed)
    # frame = env.render()
    # plt.imshow(frame)
    # plt.show()
    print("Obs shape", env.observation_space.shape, "N actions", env.action_space.n)

    # Initialize Replay Buffer
    replay_buffer = deque(maxlen=buffer_size)

    # Q Function and Policy
    Q = QFunction(env=env)
    policy = policy_class(Q=Q, env=env, seed=seed)

    # Optimizer
    optimizer = torch.optim.AdamW(params=Q.parameters(), lr=learning_rate)

    # Logging
    visited_positions = np.zeros((env.height, env.width))
    losses: list[tuple[int, float]] = []
    actions: list[tuple[int, int]] = []
    dones: list[tuple[int, bool]] = []

    step: int = 0
    action: int = 0
    reward: float = 0
    episode: int = 0
    # As long we have env interactions left
    while step < max_env_steps:
        # Reset
        losses_per_episode: list[float] = []
        observation, _ = env.reset(seed=seed)
        terminated: bool = False  # reached final state
        truncated: bool = False  # e.g. terminated by time limit
        while not terminated and not truncated:
            # Set exploration rate (could also be modified by an exploration schedule)
            eps = epsilon
            # Select action
            action = policy(state=observation, exploration_rate=eps)
            # Step env
            next_observation, reward, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated | truncated

            # Log
            actions.append((step, action))
            dones.append((step, done))
            visited_positions[env.agent_pos[1], env.agent_pos[0]] += 1

            replay_buffer.append([observation, action, reward, next_observation, done])
            observation = next_observation

            # Train Q function if we have a new batch
            if step % batch_size == 0 and len(replay_buffer) >= batch_size:
                batch = random.choices(replay_buffer, k=batch_size)
                loss = vfa_update(optimizer=optimizer, Q=Q, batch=batch, gamma=gamma)
                losses.append((step, loss))
                losses_per_episode.append(loss)

            if terminated or truncated:
                observation, info = env.reset()

            if step >= max_env_steps:
                break
        print(f"Episode {env.episode_count}, Returns", env.return_queue[-1], f"Loss: {np.mean(losses_per_episode):.6f}")
        episode += 1

    # Final evaluation
    cum_rewards = []
    for i in range(n_eval_episodes):
        rewards = evaluate(policy=policy, env=env, seed=None)
        cum_rewards.append(np.sum(rewards))
    print("Final evaluation", np.mean(cum_rewards), np.std(cum_rewards))

    return visited_positions, losses, cum_rewards, env, actions, dones


policy_classes = {
    "greedy": partial(Policy, duration_max=..., mu=..., disable_exploration=...),  # TODO set HPs
    "ε-greedy": partial(Policy, duration_max=..., mu=..., disable_exploration=...),  # TODO set HPs
    "εz-greedy": partial(Policy, duration_max=..., mu=..., disable_exploration=...),  # TODO set HPs
}


if __name__ == "__main__":
    from plotting import plot

    # Select environment
    env_id = "MiniGrid-DoorKey-5x5-v0"  # hardest
    env_id = "MiniGrid-Fetch-5x5-N2-v0"  # harder
    # env_id = "MiniGrid-Empty-Random-5x5-v0" # easy

    # Train different policies
    results = {}
    for policy_name, policy_class in policy_classes.items():
        ret = train(policy_class=policy_class, env_id=env_id)
        results[policy_name] = ret

    plot(results)
