from collections import defaultdict
from typing import List, Tuple, Callable, DefaultDict, Hashable

import gymnasium as gym
import numpy as np


# Most of this code is Code provided by Fabio Ferreira & Andre Biedenkapp
def sarsa(
        env: gym.Env, 
        num_episodes: int,
        gamma: float = 1.0, 
        alpha: float = 0.5, 
        epsilon: float = 0.1
    ) -> Tuple[List[float], List[int], DefaultDict[Hashable, np.ndarray]]:
    """
    Vanilla tabular SARSA algorithm
    
    :param num_episodes: number of episodes to train
    :param gamma: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    # Check hyperparameter boundaries
    assert 0 <= gamma <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be greater than 0'

    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []

    train_steps_list = []
    num_performed_steps = 0

    for i_episode in range(num_episodes + 1):
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state = env.reset()
        done = False
        episode_length = 0
        cumulative_reward = 0
        while not done:  # roll out episode
            num_performed_steps += 1
            action_dist = policy(state)
            action = choose_action(action_dist)
            next_state, reward, done, info = env.step(action)
            next_action_dist = policy(next_state)
            next_action = choose_action(next_action_dist)
            Q[state][action] = td_update(Q, state, action, reward, next_state, next_action, gamma, alpha, done)
            cumulative_reward += reward
            episode_length += 1
            state = next_state
        rewards.append(cumulative_reward)
        lens.append(episode_length)

        print(f'Done {i_episode:4d}/{num_episodes:4d} episodes, total_reward: {cumulative_reward}')

    return rewards, lens, Q


def choose_action(probability_distribution: np.ndarray) -> int:
    """
    Given a `probability_distribution` (shape=n_actions) choose an action according to these probabilities
    """
    return np.random.choice(range(len(probability_distribution)), p=probability_distribution)


def make_epsilon_greedy_policy(Q: DefaultDict[int, np.ndarray], epsilon: float, n_actions: int) -> Callable[
    [int], np.ndarray]:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param n_actions: size of action space to consider for this policy
    :return: Function that expects an observation and returns a policy
    """

    def policy_fn(observation: int) -> np.ndarray:
        """
        Returns an probability distribution for each action depending on an observation
        :param observation: Current observation for
        :return [Nx1]-Array: Policy
        """
        # TODO: Implement a Epsilon-Greedy policy
        # policy = np.ones(n_actions) * epsilon / n_actions
        # best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
        #     Q[observation] == Q[observation].max()
        # ))
        # policy[best_action] += (1 - epsilon)
        # return policy
        if np.random.random() > 1 - epsilon:
            p = np.random.random(n_actions)
        else:
            p = np.zeros(n_actions)
            p[np.where(Q[observation] == np.max(Q[observation]))] = 1
        p = p / np.sum(p)
        return p

    return policy_fn


def td_update(
        Q: DefaultDict[Hashable, np.ndarray],
        state: Hashable, 
        action: int, 
        reward: float, 
        next_state: Hashable, 
        next_action: int,
        gamma: float, 
        alpha: float, 
        done: bool
    ) -> float:
    """
    Simple TD update rule. if done: Do not use difference of estimation
    :param Q: [Nx2]-Array: State-Action Value Function
    :param state: Current State
    :param action: Current Action
    :param reward: Current Reward
    :param next_state: Next State
    :param next_action: Next Action
    :param gamma: Discount Factor
    :param alpha: Learning Rate
    :param done: Is episode finished
    :return: Value for TD-Update
    """
    # TODO: Calculate value for td-update
    delta = alpha * ((reward + gamma * Q[next_state][next_action] * (not done)) - Q[state][action])
    return Q[state][action] + delta

