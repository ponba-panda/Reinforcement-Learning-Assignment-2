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

# Replay Buffer
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

# Shared Network
class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_size, hidden_size, num_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        # self.affine2 = nn.Linear(hidden_size, hidden_size)
        # self.affine3 = nn.Linear(2*hidden_size, hidden_size)

        # advantage layer
        self.advantage_affine1 = nn.Linear(hidden_size,hidden_size)
        self.advantage_head = nn.Linear(hidden_size, num_actions)

        # value layer
        self.state_affine1 = nn.Linear(hidden_size,hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        # x = F.relu(self.affine2(x))
        # x = F.relu(self.affine3(x))

        # returns advantage values for each action at a given state
        advantage_values = F.relu(self.advantage_affine1(x))
        # advantage_values = x
        advantage_values = self.advantage_head(advantage_values)

        # returns state value
        state_values = F.relu(self.state_affine1(x))
        # state_values = x
        state_values = self.value_head(state_values)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return advantage_values, state_values

# Dueling DQN agent
class DDQNagent:
    def __init__(self, env, hidden_size=128, learning_rate=1e-4, gamma=0.99, polyak_tau=1e-2, softmax_tau=0.1, max_memory_size=50000, dueling_type='average', device='cpu'):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.tau = polyak_tau
        self.softmax_tau = softmax_tau
        self.dueling_type = dueling_type
        self.env = env
        self.device = device

        # Networks
        self.policy = Policy(self.num_states, hidden_size, self.num_actions)
        self.policy_target = Policy(self.num_states, hidden_size, self.num_actions)
        self.policy = nn.DataParallel(self.policy).to(self.device)
        self.policy_target = nn.DataParallel(self.policy_target).to(self.device)

        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # Training
        self.memory = Memory(max_memory_size)
        self.loss_criterion  = nn.MSELoss().to(self.device)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        advantage_values, state_value = self.policy.forward(state)
        advantage_values = advantage_values.to(self.device)
        state_value = state_value.to(self.device)
        if self.dueling_type=='average':
          q_values = state_value + advantage_values - (1/self.num_actions)*torch.sum(advantage_values)
        elif self.dueling_type=='max':
          q_values = state_value + advantage_values - torch.max(advantage_values)
        else:
          print('Error: Wrong dueling type specified. Specify either "average" or "max".')
        q_values = q_values.to(self.device)
        action_prob = F.softmax(q_values[0]/self.softmax_tau, dim=0).to(self.device) # .detach().numpy()
        return torch.multinomial(action_prob, 1, replacement=True)[0].item() # returns action as per softmax probabilities
        # return np.random.choice(np.arange(self.num_actions), p = action_prob)


    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        done = torch.Tensor(done).to(self.device)
        done_comp = torch.logical_not(done).int()

        # get state action values from target network
        advantage_values, state_values = self.policy_target.forward(next_states)
        advantage_values = advantage_values.to(self.device)
        state_values = state_values.to(self.device)
        if self.dueling_type=='average':
          q_values_target = torch.add(state_values, advantage_values)
          q_values_target = torch.add(q_values_target, (1/self.num_actions)*torch.sum(advantage_values, dim=1).reshape(batch_size,-1),alpha=-1)
          q_values_target = q_values_target.to(self.device)
        elif self.dueling_type=='max':
          q_values_target = torch.add(state_values, advantage_values)
          q_values_target = torch.add(q_values_target, torch.max(advantage_values, dim=1)[0].reshape(batch_size,-1),alpha=-1)
          q_values_target = q_values_target.to(self.device)
        else:
          print('Error: Wrong dueling type specified. Specify either "average" or "max".')
        q_value_target = rewards + self.gamma*(done_comp*torch.max(q_values_target,dim=1)[0]).reshape(batch_size,-1)
        q_value_target = q_value_target.to(self.device)

        # get state action values from online network
        advantage_values, state_values = self.policy.forward(states)
        advantage_values = advantage_values.to(self.device)
        state_values = state_values.to(self.device)
        if self.dueling_type=='average':
          q_values = torch.add(state_values, advantage_values)
          q_values = torch.add(q_values, (1/self.num_actions)*torch.sum(advantage_values, dim=1).reshape(batch_size,-1),alpha=-1)
          q_values = q_values.to(self.device)
        elif self.dueling_type=='max':
          q_values = torch.add(state_values, advantage_values)
          q_values = torch.add(q_values, torch.max(advantage_values, dim=1)[0].reshape(batch_size,-1),alpha=-1)
          q_values = q_values.to(self.device)
        else:
          print('Error: Wrong dueling type specified. Specify either "average" or "max".')
        q_value  = torch.gather(q_values, 1, actions.reshape(batch_size,-1).long())
        q_value = q_value.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss_criterion(q_value_target, q_value) # compute mean squared error loss
        loss.backward()
        self.optimizer.step()

        # update target networks
        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

        return loss

    def train(self, batch_size=128, total_episodes=50, log_interval=10, plot=True):
        # rewards = []
        # avg_rewards = []
        returns = []
        avg_returns = [-100]
        episode = 0
        avg_return = 0
        steps = []

        #while avg_returns[-1]<100:
        for episode in range(total_episodes):
            state = self.env.reset()
            episode_rewards = []
            loss = 0

            for step in range(500):
                action = self.choose_action(state)
                new_state, reward, done, _ = self.env.step(action)
                #if done and step!=499:
                  #reward = step - 499 + 1
                self.memory.push(state, action, reward, new_state, done)

                if len(self.memory) > batch_size:
                    loss += self.update(batch_size)

                state = new_state
                episode_rewards.append(reward)

                if done: # or step==499:
                    # avg_rewards.append(np.mean(rewards[-10:]))
                    # sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                    break

            if len(steps)>100 and np.array(steps[-100:]).mean()==499:
                  break

            R = 0
            for r in episode_rewards[::-1]:
                R = r + self.gamma*R
            # returns.insert(0,R)
            returns.append(R)
            avg_return = 0.05*R+0.95*avg_return
            # avg_returns.insert(0,avg_return)
            avg_returns.append(avg_return)
            # avg_returns.insert(0,np.mean(np.array(returns)))
            steps.append(step)
            if episode%log_interval==0:
                print(f"Episode: {episode}, Return: {returns[-1]:.4f}, Average Return: {avg_returns[-1]:.4f}, Steps: {step}, Average loss: {loss/step:.4f}")
            episode += 1

        print(f"Episode: {episode}, Return: {returns[-1]:.4f}, Average Return: {avg_returns[-1]:.4f}, Steps: {step}, Average loss per episode: {loss/step:.4f}")

        if plot:
          plt.plot(returns)
          plt.plot(avg_returns)
          plt.plot()
          plt.xlabel('Episode')
          plt.ylabel('Return')
          plt.show()

        return returns, avg_returns

def train(env_name, hyperparameters):

  seed = 543
  env = gym.make(env_name)
  env.reset(seed=seed)
  def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
      torch.cuda.manual_seed(seed)
      #torch.backends.cudnn.benchmark = True
      #torch.backends.cudnn.deterministic = False

  hidden_size = hyperparameters["hidden_size"]
  learning_rate = hyperparameters["learning_rate"]
  gamma = hyperparameters["gamma"]
  polyak_tau = hyperparameters["polyak_tau"]
  softmax_tau = hyperparameters["softmax_tau"]
  max_memory_size = hyperparameters["max_memory_size"]
  dueling_type = hyperparameters["dueling_type"]
  batch_size = hyperparameters["batch_size"]
  total_episodes = hyperparameters["total_episodes"]
  log_interval = hyperparameters["log_interval"]
  plot = hyperparameters["plot"]

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  agent = DDQNagent(env, hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma, polyak_tau=polyak_tau, softmax_tau=softmax_tau, max_memory_size=max_memory_size, dueling_type=dueling_type, device=device)
  returns, avg_returns = agent.train(batch_size, total_episodes, log_interval, plot)
  if device=='cuda':
    torch.cuda.empty_cache()

  return returns, avg_returns