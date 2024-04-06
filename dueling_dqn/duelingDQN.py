#Import required libraries

import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

import sys
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random

from pa2_utils import train

hyperparameters = {"gamma":0.99,
                   "hidden_size":256,
                   "learning_rate":1e-4,
                   "polyak_tau":1e-3,
                   "softmax_tau":0.1,
                   "max_memory_size":10000,
                   "dueling_type":"max",
                   "batch_size":512,
                   "total_episodes":10,
                   "log_interval":100,
                   "plot":False
                   }

env_name = "CartPole-v1"
# env_name = "Acrobot-v1"                                


# env_name = hyperparameter_set['env_name']

num_runs = 5
total_episodes = hyperparameters["total_episodes"]
soft_return = np.zeros((num_runs,total_episodes))
regrets = np.zeros((num_runs,))
length = []

for i in range(num_runs):
    returns, avg_returns, best_return, regrets[i] = train(env_name, hyperparameters)
    length.append(len(returns))
    if length[i]<total_episodes:
        returns = returns + (total_episodes-length[i])*[best_return]
    soft_return[i,:] = np.array(returns).reshape(-1,total_episodes)
    print(f"Regret in episode {i}: {regrets[i]:.4f}")

min_length = min(length)
avg_regret = sum(regrets)/num_runs
print(f"Average regret : {avg_regret:.4f}")
print(f"Average regret per episode: {avg_regret/total_episodes:.4f}")

np.savetxt(env_name+'_'+hyperparameters["dueling_type"]+'.csv', soft_return, delimiter=',')

mean_data = np.mean(soft_return, axis=0)
variance_data = np.var(soft_return, axis=0)
shade_data_1ow = mean_data - np.sqrt(variance_data)
shade_data_high = mean_data + np.sqrt(variance_data)

plt.plot(mean_data[:min_length])
plt.fill_between(range(min_length), shade_data_1ow[:min_length], shade_data_high[:min_length], alpha=0.3)
plt.plot(best_return*np.ones(total_episodes,)[:min_length])
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(f'Dueling DQN {hyperparameters["dueling_type"]} - Returns per Episode')
plt.savefig(env_name+'_'+hyperparameters["dueling_type"]+'.pdf')
#plt.show()