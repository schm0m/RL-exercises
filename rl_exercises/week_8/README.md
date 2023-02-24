# Week 8: Exploration
This exercise showcases the impact of different exploration strategies. In this assignment you will implement a policy using no exploration, a policy using ε-greedy and one using εz-greedy [Dabney et al., 2020](https://arxiv.org/pdf/2006.01782.pdf). 
The εz-greedy policy samples not only a random action but also a duration for which the action will be played. You can find the algorithm in Appendix B of the linked paper. 
We will use grid environments. 
### 1. Implement ε(z)-greedy
Your task is to implement the (non)-ε(z) policy in Policy. call . Use the member variable disable exploration to enable complete greedy behavior. Hint: You can switch from εz-greedy to ε-greedy by setting duration max.
### 2. Implement Sampling of the Duration
Implement the sampling of the duration in Policy.sample duration. Hint: Check the paper for the hyperparameter μ.
### 3. Configure Policies
Add the hyperparameters to policy classes to create a greedy, ε-greedy and εz-greedy policy.
### 4. Run and Observe
Run exploration.py and note the differences in the results in answers.txt. Upload the figures to plots. Is the current algorithm well suited for the problem? What could be a way to improve it (think of the previous lectures)? You can also play with the hyperparameters (e.g., γand ε) and try different environments (e.g., bigger grid).