# Week 2: Policy and Value Iteration
This week you will implement the fundamental algorithms of policy and value iteration. You'll see how your agent's behaviour changes over time and hopefully have your first successful training runs.

## Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda-console (Miniconda prompt)
3. Create a new conda-environment using the `environment.yml`-file:
   * ``conda env create -f "environment.yml"``
4. Activate conda-environment:
   * ``conda activate rl-exercises``

If you already created an environment and want to update it using this `environment.yml` you can use following command:
````shell
conda activate rl-exercises
conda env update --file "environment.yml" --prune
````

By appending ``--prune`` all existing packages which are not listed in `environment.yml` are removed. 
If you want to keep them, you can remove this flag.