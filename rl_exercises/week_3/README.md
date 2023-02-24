# Week 2: Policy and Value Iteration
This week you will implement the fundamental algorithms of policy and value iteration. You'll see how your agent's behaviour changes over time and hopefully have your first successful training runs.

### 1. Policy Iteration for the MarsRover
In the env.py file you’ll find the first environment we’ll work with: the MarsRover. 
You have seen it as an example in the lecture: the agent can move left or right with each step and should ideally move to the rightmost state. In this first exercise, the environment will be deterministic, that means the rover will always execute the given action. 
Your task is to complete the given code stub in policy iteration.py with the algorithm from the lecture.

### 2. Value Iteration for the probibalistic MarsRover
For this second exercise, we modify the MarsRover environment, now the rover may or may not execute the requested action, the probability is 50%. 
You will complete the code in value iteration.py in order
to evaluate a policy on this variation of our environment.
