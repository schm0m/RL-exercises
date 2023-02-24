from __future__ import annotations

import numpy as np
from env import MarsRover


def update_policy(
    qs: np.ndarray,
    pi: list[int] | np.ndarray,
    state: int,
    new_state: int,
    action: int,
    reward: float,
    gamma: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Single step for Policy iteration

    :param qs: 2D-Array of size (n, 2): State-Action Value function
    :param pi: 1D-Array of size n with integers defining the best action given a state
    :param state: Current state/position index
    :param new_state: New state/position index
    :param action: Selected action which leads to state -> new_state
    :param reward: Reward for this action
    :param gamma: Discount factor
    :return:
    """
    # TODO: complete this method
    new_qs = np.copy(qs)
    new_pi = np.copy(pi)
    converged = True
    return new_qs, new_pi, converged


# TODO: complete this method
def run_policy_iteration(
    transition_probabilities: np.ndarray = np.ones((5, 2)),
    rewards: list[float] = [1, 0, 0, 0, 10],
    horizon: int = 10,
) -> tuple[np.ndarray, int, float]:
    """
    :param transition_probabilities: [Nx2] Array for N positions and 2 actions each.
    :param rewards: [Nx1] Array for rewards. rewards[pos] determines the reward for a given position `pos`.
    :param horizon: Number of total steps for this environment until it is done (e.g. battery drained)
    :return: Tuple[Policy, number of update steps, reward]
    """
    env = MarsRover(transition_probabilities, rewards, horizon)
    n = len(env.rewards)

    done = False
    state = env.reset()

    pi = np.random.randint(0, 2, n)

    i = 0
    while not done:
        i += 1
        print(f"This is step {i}")
        action = pi[state]
        new_state, reward, done = env.step(action)

        # TODO: Use Policy iteration to update policy

        if done:
            new_state = env.reset()

    final_reward = evaluate_policy(pi, env)

    print(
        f"Your policy achieved a final accumulated reward of {final_reward} after {i} update steps."
    )

    return pi, i, final_reward


def evaluate_policy(pi: list[int] | np.ndarray, env: MarsRover) -> float:
    state = env.reset()
    done = False
    r_acc: float = 0
    while not done:
        action = pi[state]
        new_state, reward, done = env.step(action)
        r_acc += reward
    return r_acc


if __name__ == "__main__":
    print(run_policy_iteration())
