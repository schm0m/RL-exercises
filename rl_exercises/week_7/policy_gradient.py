import gym
import numpy as np
import torch
import torch.nn as nn

# This could be useful to you:
# import torch.nn.functional as F
import torch.optim as optim

# This is potentially a helpful class:
# from torch.distributions import Categorical


from typing import List, Tuple

MAX_EPISODE_LENGTH = 1000
DISCOUNT_FACTOR = 0.99
SEED = 0
MIN_BATCH_SIZE = 128

env = gym.make("CartPole-v1")
env.seed(SEED)
torch.manual_seed(SEED)


class Policy(nn.Module):
    """Define policy network"""

    def __init__(
        self,
        state_space: gym.spaces.box.Box,
        action_space: gym.spaces.discrete.Discrete,
        hidden_size: int = 128,
    ):
        """Initialize the policy network

        Parameters
        ----------
        state_space : gym.spaces.box.Box
            Space for inputs to the network
        action_space : gym.spaces.discrete.Discrete
            Space for outputs of the network
        hidden_size : int, optional
            size of hidden layer, by default 128

        for more information about gym.spaces, please refer to https://www.gymlibrary.dev/api/spaces/

        """
        # TODO Initialize 2 linear layers to map the state to an output equal to the number of actions

        super().__init__()

    def forward(self, x: List[float]) -> torch.Tensor:
        """Forward pass of the policy network

        Parameters
        ----------
        x : List[float]
            State of the environment

        Returns
        -------
        torch.Tensor
            Probabilites over actions
        """

        # TODO pass the input through each layer

        # TODO compute the softmax to normalize the probabilities
        probs = ...

        return probs


policy = Policy(env.observation_space, env.action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def compute_returns(rewards: List[int], discount_factor: float = DISCOUNT_FACTOR) -> List[float]:
    """Compute discounted returns

    Parameters
    ----------
    rewards : List[int]
        rewards accumulated during trajectory sampling
    discount_factor : float, optional
        dsicount factor for computation, by default DISCOUNT_FACTOR

    Returns
    -------
    List[float]
        List of discounted returns
    """
    returns = []
    # TODO Compute the returns_to_go by discounting rewardsand add them sequentially to the list

    return returns


def policy_improvement(log_probs: torch.Tensor, rewards: List[int]) -> float:
    """Compute REINFORCE policy gradient and perform gradient ascent step

    Parameters
    ----------
    log_probs : torch.Tensor
        log probabilites of actions taken during the sampling procedure
    rewards : List[int]
        list of rewards for each of those actions

    Returns
    -------
    float
        loss computed using the log probabilities and advantages
    """

    # we need log probabilites for each reward in the list
    assert len(log_probs) == len(rewards)

    # TODO compute the returns
    # returns = ...

    # TODO compute advantages using returns
    # advantages = ...

    log_probs = torch.stack(log_probs)
    optimizer.zero_grad()

    # TODO Compute the loss as the sum of log probs weighted by advantages

    loss = ...

    loss.backward()
    optimizer.step()

    return loss.item()


def act(state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use policy to sample an action and return probability for gradient update

    Parameters
    ----------
    state : List[float]
        State of the environment -- 4D array

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        computed action and log probability of the action
    """
    # TODO pass the state through the policy network
    # probs = ...

    # TODO create the outupt into a categorical distribution and sample and action from it
    action = ...

    # TODO compute the log probabilitiy of the action
    log_prob = ...

    return action, log_prob


def policy_gradient(num_episodes: int) -> List[int]:
    """Compute Policy gradient for a given number of episodes

    Parameters
    ----------
    num_episodes : int
        Number of episodes for which the REINFORCE loop needs to run

    Returns
    -------
    List[int]
        List of accumulated rewards for each episode
    """
    rewards = []
    for episode in range(num_episodes):
        rewards.append(0)
        trajectory = []
        state = env.reset()

        for t in range(MAX_EPISODE_LENGTH):
            # Enable for rendering the environment
            # if episode % (num_episodes / 100) == 0:
            #     env.render()

            # Generate an action and its log_probability given a state
            action, log_prob = act(state)

            # Take a step in the environment using this action
            next_state, reward, done, _ = env.step(action.item())

            # Append the log probability and reward to the trajectory
            trajectory.append((log_prob, reward))

            state = next_state

            # accumulate the reward for the given episode
            rewards[-1] += reward

            if done:
                break

        # Policy improvement step
        loss = policy_improvement(*zip(*trajectory))

        # Do something with that loss
        loss.backward()

        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100) :]))
    return rewards


if __name__ == "__main__":
    policy_gradient(1000)
