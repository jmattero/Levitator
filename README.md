## Levitator RL

Gorkov potential, Singular value decomposition (SVD)

### Installation

Install PyTorch from https://pytorch.org/


### Instructions


#### Files
There are three things in this repository

`levitator_env.py` contains a [Gymnasium](https://gymnasium.farama.org/) implementation of the levitator with their new API, and a dense or sparse reward function. This should be compatible with standard RL frameworks. The observation space is a vector of 6 elements and the action space can be configured to be 'raw' or 'SVD'.

`levitator_goal_env.py` contains an OpenAI Gym GoalEnv implementation with a dictonary observation space, which contains not only the current state but the target we want to reach ("desired goal"), and a third "achieved goal" which is used by Hindsight Experience Replay. This environment should be compatible with GoalEnv-based RL frameworks.

`baselines_her_levitate.py` contains the code to train an SAC (or other) agent using Hindsight Experience Replay. The code is based on [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) which allows different algorithms to be used. 

The rest of the files are for a standard RL implementation of an algorithm called Soft Actor-Critic (SAC). To run it, run `main.py`.

### Viewing results, etc

The SAC implementation (`main.py`) uses Tensorboard to log results. Each run will go to a new folder in the `tb_runs` folder. You can run a Tensorboard server there by cding there and then `tensorboard --logdir=.`

The stable baselines thing also uses Tensorboard but you can also use [wandb](https://wandb.ai/) which makes collecting results easier. I recommend making an account and looking into it.

