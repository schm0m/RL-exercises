# Week 4: Model-free Control
This week you will implement you first real model-free learning algorithm, SARSA, as well as conduct some experiments concerning its hyperparameters.

### 1. Model-free Control with SARSA
You will complete the code stubs in sarsa.py to implement the SARSA algorithm from the lecture. 
You should include epsilon greedy exploration, as exploration is an important part of model-free learning algorithms. 
As always, use the methods provided as guidance as to what is queried in the tests, but feel free to extend our suggestions in any way you like.

### 2. Hyperparameters of SARSA
Many concepts of SARSA also apply in more powerful RL algorithms, for example the effect of its hyperparameters. 
Therefore you now have an opportunity to experiment with different hyperparameter values and how they influence how successful the algorithm runs. 
If you want to know more about
SARSA, try answering these questions and report the results in the exercise:
- Does setting the learning rate to 0.8 increase or decrease the number of training steps?
- For which value of epsilon do you get the best result, 0.01, 0.1 or 0.9?
- Which works better for you, initializing Q to all 0 or initializing it randomly?

