from stable_baselines3 import HerReplayBuffer, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from levitator_goal_env import Levitator
import torch
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.callbacks import EvalCallback
import gym
from wandb.integration.sb3 import WandbCallback
import wandb

model_class = SAC  # works also with SAC, DDPG and TD3

#initial = torch.tensor([-0.02, 0.0, -0.01, 0, 0, 0])
initial = torch.tensor([0.00, 0.0, 0.00, 0, 0, 0])
target = torch.tensor([0.02, 0.0, 0.01])

delay = 1
config = "full"
planner_dynamics = "full"
max_steps = 200

env = Levitator(config, initial, target, delay)
env = gym.wrappers.TimeLimit(env, max_steps)
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes

#envs = DummyVecEnv([lambda : Levitator(config, initial, target, delay) for _ in range(4)])
#envs = SubprocVecEnv
# Initialize the model'
eval_env = Levitator(config, initial, target, delay)
eval_env = gym.wrappers.TimeLimit(eval_env, max_steps)
eval_callback = EvalCallback(eval_env, eval_freq=1000,
                             deterministic=True, render=False)

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=100,
        handle_timeout_termination=True
    ),
    verbose=10,
    tensorboard_log="./tb_logs/",
    batch_size=64,
    learning_starts=1000,
)

model = model_class.load("./her_levitator_randstart", env=env)

torch.set_printoptions(3, sci_mode=False)
import numpy as np
np.set_printoptions(3)

obs = env.reset()
for _ in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    dist = torch.linalg.norm(obs["observation"][:3] - target[:3], axis=-1)
    print('o', obs["observation"][:3], 'r', reward, 'a', action, 'dist', dist)
    if done:
        obs = env.reset()