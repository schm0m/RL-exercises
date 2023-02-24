# Week 6: Deep Q-Learning

This week you will extend Q-Learning with linear function approximation to even more complex environments by implementing a Deep Q-Network (DQN).

## Setup

### First setup
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda-console (Miniconda prompt)
3. Create a new conda-environment using the `environment.yml`-file:
   * ``conda env create -f "environment.yml"``
4. Activate conda-environment:
   * ``conda activate rl-exercises``

### Update Environment
If you already created an environment and want to update it using this `environment.yml` you can use following command:
````shell
conda activate rl-exercises
conda env update --file "environment.yml" --prune
````

By appending ``--prune`` all existing packages which are not listed in `environment.yml` are removed. 
If you want to keep them, you can remove this flag.