from __future__ import annotations
import numpy as np
from env import MarsRover


# TODO: complete this method
def update_value_function(
    v: np.ndarray, state: int, new_state: int, reward: float, gamma: float = 0.9
) -> tuple[np.ndarray, bool]:
    """
    Single step for Value iteration

    :param v: [Nx2]-Array: State Value function
    :param state: Current state/position index
    :param new_state: New state/position index
    :param reward: Reward for this action
    :param gamma: Discount factor
    :return: tuple[np.ndarray, bool] (updated value function, has converged)
    """
    new_pi = np.zeros(v.shape[0])
    converged = True
    return new_pi, converged


# TODO: complete this method
def run_value_iteration(
    transition_probabilities: np.ndarray = np.ones((5, 2)) * 0.5,
    rewards: list[float] = [1, 0, 0, 0, 10],
    horizon: int = 10,
):
    env = MarsRover(transition_probabilities, rewards, horizon)
    n = len(rewards)

    done = False
    state = env.reset()

    v = np.zeros(n)

    i = 0
    while not done:
        i += 1
        print(f"This is step {i}")
        action = 1
        new_state, reward, done = env.step(action)

        # TODO: Use Value iteration to update Value function
        update_value_function(v, state, ...)

        if done:
            new_state = env.reset()

        state = new_state

    final_reward = evaluate_agent(v, env)

    print(f"Your agent achieved a final accumulated reward of {final_reward} after {i} update steps.")

    return v, i, final_reward


def evaluate_agent(v: list[float] | np.ndarray, env: MarsRover) -> float:
    state = env.reset()
    done = False
    r_acc: float = 0
    while not done:
        action = np.argmax([v[max(state - 1, 0)], v[min(state + 1, 4)]])
        new_state, reward, done = env.step(action)
        r_acc += reward
    return r_acc


if __name__ == "__main__":
    print(run_value_iteration())
