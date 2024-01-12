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

config = {}
# wandb logging, can comment out if only using tensorboard
run = wandb.init(
    project="levitator",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
)

initial = torch.tensor([0.00, 0.0, 0.00, 0, 0, 0])
target = torch.tensor([0.01, 0.01, 0.01])

delay = 1
config = "full"
planner_dynamics = "full"
max_steps = 200

env = Levitator(config, initial, target, delay, sample_target=True)
env = gym.wrappers.TimeLimit(env, max_steps)

goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE
online_sampling = True

initial = torch.tensor([0.00, 0.0, 0.00, 0, 0, 0])
target = torch.tensor([0.01, 0.01, 0.01])
eval_env = Levitator(config, initial, target, delay, sample_target=False)
eval_env = gym.wrappers.TimeLimit(eval_env, max_steps)
eval_callback = EvalCallback(eval_env, eval_freq=2000,
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
        max_episode_length=200,
        handle_timeout_termination=True
    ),
    verbose=10,
    tensorboard_log="./tb_logs/",
    batch_size=1024,
    learning_starts=5000,
    tau=0.005,
    ent_coef=0.2
)

# Train the model
model.learn(500000, callback=[WandbCallback(verbose=10,), eval_callback])

run.finish()

model.save("./her_levitator_randstart")
model = model_class.load("./her_levitator_randstart", env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    print(reward)
    if done:
        obs = env.reset()