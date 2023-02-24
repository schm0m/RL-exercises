# Week 5: Q-Learning

This week you will implement Q-Learning, another model-free RL algorithm. By using linear function approximation, it is able to scale to infinitely large state spaces.

### 1. Tabular Q-Learning
Implement the Q-Learning update step in q learning tabular.py and try different state discretizations (BINS) and learning rates (LEARNING RATE). How does the number of states and learning rate affect the training of the RL algorithm?

### 2. Q-Learning with Linear Value Function Approximation
Implement Q-Learning with Linear Value Function Approximation. First create make Q that takes an environment as input and creates a PyTorch Model. 
Then implement the value function training step in q learning vfa.py using the Q module and the optimizer. How does the training differ from the tabular case? How sensitive is the algorithm to the weight initialization?
Update the hyperparameters and the model to achieve a mean reward of more than 50 for the CartPole environment.
For the open questions, please write your answers in ‘answers.txt‘. We will grade those manually.

