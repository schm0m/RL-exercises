# Week 2: Policy and Value Iteration
This week you will implement the fundamental algorithms of policy and value iteration. You'll see how your agent's behaviour changes over time and hopefully have your first successful training runs.

⚠ Before you start, make sure to have read the general `README.md`.
You should add your solutions to the central train and eval script.

Run your solution with
```bash
python rl_exercises/train_agent.py +week/w2=policy_iteration
```

## Level 1
### 1. Policy Iteration for the MarsRover
In the `mars_rover_env.py` file you’ll find the first environment we’ll work with: the MarsRover. 
You have seen it as an example in the lecture: the agent can move left or right with each step and should ideally move to the rightmost state. In this first exercise, the environment will be deterministic, that means the rover will always execute the given action. 
Your task is to complete the given code stub in `policy_iteration.py` with the algorithm from the lecture.

### 2. Value Iteration for the probibalistic MarsRover
For this second exercise, we modify the MarsRover environment, now the rover may or may not execute the requested action, the probability is 50%. 
You will complete the code in `value_iteration.py` in order
to evaluate a policy on this variation of our environment.
What happens if you different initial policies? Will you always converge to the same policy? What if you vary gamma?

## Level 2
What happens if you only have access to `step()` instead of the dynamics and reward? Do both methods still work? This setting will be what we'll work with for the rest of the semester.

## Level 3
Implement Generalized Policy Iteration from the Sutton & Barto book. It is different from your Level 2 solution? Can you match the performance of policy and value iteration?