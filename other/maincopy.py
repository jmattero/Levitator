import torch
import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
from sac import SAC
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

#from levitator_goal_env import Levitator
from levitator_env import Levitator
from gymnasium.wrappers.time_limit import TimeLimit

from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Levitator",
                    help='Environment (default: Levitator)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic | Naive (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

#hyperparameter tuning
#automatic entropy tuning: True 
# -> at first big rewards, but afterwards really small rewards (-20 on episodes 300-1500)
#args.automatic_entropy_tuning = True

#Alpha: high -> exploration, low -> exploitations
# args.alpha = 1

#learning rate:
#args.lr = 0.0005

#Save:
save_ep = 100
save = False
save_name = '{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Environment
# env = NormalizedActions(gym.make(args.env_name))
#initial = torch.tensor([-0.02, 0.0, -0.01, 0, 0, 0])
initial = torch.tensor([0.0, 0.00, 0.0, 0, 0, 0])
target = torch.tensor([0.005, 0.005, 0.002])
#trying with diffrent target (closer)
#limit 0 - 0.001

#for naive:
#args.policy = 'Naive'

l = 0.0005
naive = [initial, target, l]

print(torch.linalg.norm(target))

# How many actions to take. Corresponds to max_steps * delay simulator timesteps
#100-> smaller to get faster runs
max_steps = 100
delay = 1
config = "full"
env = Levitator(config, initial, target, delay)
#This TimeLimit wrapper will issue a truncated signal if a maximum number of timesteps is exceeded:
env = TimeLimit(env, max_steps)

env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


#observation space shape for goalenv:
#print(env.observation_space['observation'].shape[0])

#observation space shape for levitator_env:
#env.observation_space.shape[0]

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args, naive)


if args.policy == 'Naive':
    args.env_name="l={}_delay{}".format(l, delay)
    folder = 'naive_runs' 
else:
    folder = 'runs'

#Tesnorboard
writer = SummaryWriter('{}/{}_SAC_{}_{}_{}'.format(folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state, info = env.reset()

    actions = []

    while not done:
        if episode_steps % delay == 0 or episode_steps == 0:
            if args.policy == 'Naive':
                action = agent.select_naive(state, episode_steps / delay)
            elif args.start_steps > total_numsteps:  #first hundred (if default) steps:
                action = env.action_space.sample()  # Sample random action
                #print(action)
            else:
                action = agent.select_action(state)  # Sample action from policy
            #print(action)
            actions.append(action)

            #print('     action: ', action)
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        #print(action)
        next_state, reward, terminated, truncated, info = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        
        done = terminated or truncated

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        #mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = not terminated

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        #print(action)
    #print(len(actions))
    actions = np.stack(actions)
    #print('action std', actions.std(0), actions.mean(0), actions.min(0), actions.max(0))

    #embed()
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if (i_episode % 3 == 0) and (args.eval is True):
        avg_reward = 0.
        episodes = 1
        for _ in range(episodes):
            state, info = env.reset()
            print('Starting at', state[:3])
            ss = state.clone()
            episode_reward = 0
            done = False
            eval_num_steps = 0

            actions = []
            test_states = []

            while not done:
                if eval_num_steps % delay == 0 or eval_num_steps == 0:
                    #select action
                    if args.policy == 'Naive':
                        action = agent.select_naive(state, eval_num_steps / delay)
                    else:
                        action = agent.select_action(state, evaluate=True)

                actions.append(action)
                eval_num_steps += 1
                test_states.append(state)
                
                next_state, reward, terminated, truncated, _  = env.step(action)
                
                episode_reward += reward

                done = terminated or truncated
                #print('_terminated:___', terminated, ' or truncated:__', truncated)
                
                state = next_state
                #print(np.linalg.norm(target - state[:3]))
            print('End at', state[:3], 'Distance travelled', torch.linalg.norm(state[:3] - ss[:3]))
            print('target was', target[:3], 'Distance to target', torch.linalg.norm(state[:3] - target[:3]))

            avg_reward += episode_reward
        avg_reward /= episodes
        
        #print(len(actions))
        actions = np.stack(actions)
        #print(actions)
        #print(np.linalg.norm(target - state[:3]))
        test_states = np.stack(test_states)
        #print(actions.shape)
        #print(test_states.shape)
        labels = ['x', 'y', 'z']
        fig = plt.figure(figsize=(12,6), dpi=100)
        ##########
        #plt.ylim([0, target.max().item()])
        #limits of action space:
        #plt.ylim([-0.05, 0.05])
        #########
        for i in range(3):
            plt.scatter(np.arange(actions.shape[0]), actions[:, i], label=labels[i])
        plt.legend()
        for i in range(3):
            plt.plot(np.arange(actions.shape[0]), test_states[:, i], label=labels[i])
        
        # reward plot
        plt.legend()

        writer.add_figure("test_episode", fig, i_episode)
        plt.clf()        
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 4)))
        print("----------------------------------------")

    #save the model:
    if i_episode % save_ep == 0 and save is True:
        agent.save_checkpoint(save_name)

env.close()


