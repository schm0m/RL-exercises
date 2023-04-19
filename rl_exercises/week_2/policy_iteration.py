from __future__ import annotations

import numpy as np
from mars_rover_env import MarsRover


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

    Parameters
    ----------
    qs: [Nx2]-Array
        Q function
    pi: np.array
        Policy
    state: int
        Current state/position index
    new_state: int
        New state/position index
    action: int
        Selected action
    reward: float
        Reward for this action
    gamma: float
        Discount factor

    Returns
    -------
    new_qs
        Updated Q function
    new_pi
        Update policy
    converged
        Convergence Signal
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
    Run Policy Iteration

    Parameters
    ----------
    transition_probabilities: [Nx2]-Array
        Environment Transition Probabilities
    rewards: list[float]
        Reward structure for environment
    horizon: int
        Environment horizon

    Returns
    -------
    policy
        policy
    i
        Number of iterations
    final_reward
        Total reward
    """
    env = MarsRover(transition_probabilities, rewards, horizon)
    n = len(env.rewards)

    done = False
    state, _ = env.reset()

    pi = np.random.randint(0, 2, n)

    i = 0
    while not done:
        i += 1
        print(f"This is step {i}")
        action = pi[state]
        new_state, reward, done, _, _ = env.step(action)

        # TODO: Use Policy iteration to update policy

        if done:
            new_state, _ = env.reset()

        state = new_state

    final_reward = evaluate_policy(pi, env)

    print(f"Your policy achieved a final accumulated reward of {final_reward} after {i} update steps.")

    return pi, i, final_reward


def evaluate_policy(pi: list[int] | np.ndarray, env: MarsRover) -> float:
    """
    Evaluate Policy

    Parameters
    ----------
    pi: list[int], np.array
        Policy to evaluate
    env: MarsRover
        Evaluation environment

    Returns
    -------
    r_acc
        accumulated rewards
    """
    state, _ = env.reset()
    done = False
    r_acc: float = 0
    while not done:
        action = pi[state]
        state, reward, done, _, _ = env.step(action)
        r_acc += reward
    return r_acc


if __name__ == "__main__":
    print(run_policy_iteration())
