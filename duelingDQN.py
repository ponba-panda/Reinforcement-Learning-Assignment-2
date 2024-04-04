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
                   "hidden_size":64,
                   "learning_rate":1e-3,
                   "polyak_tau":1e-3,
                   "softmax_tau":0.1,
                   "max_memory_size":10000,
                   "dueling_type":"max",
                   "batch_size":256,
                   "total_episodes":500,
                   "log_interval":50,
                   "plot":False
                   }

env_name = "CartPole-v1"
# env_name = "Acrobot-v1"
returns, avg_returns = train(env_name, hyperparameters)

num_runs = 5
total_episodes = hyperparameters["total_episodes"]
soft_return = np.zeros((1,total_episodes))

for i in range(num_runs):
    returns, avg_returns = train(env_name, hyperparameters)
    soft_return = soft_return + (1/num_runs)*np.array(returns).reshape(-1,total_episodes)


plt.plot(soft_return[0])
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()

# print(soft_return)