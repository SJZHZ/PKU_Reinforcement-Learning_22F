import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
import tqdm
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear2 = nn.Linear(hidden_size, output_size)
        self.Linear2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.softmax(self.Linear2(x), dim=1)
        return x

class Reinforce(object):
    def __init__(self, envin, hidden_size):
        self.env = envin
        input_size, hidden_size, output_size = env.observation_space.shape[0], hidden_size, env.action_space.n
        self.net = Policy(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.net.parameters(), lr=0.01)
        self.GAMMA = 1.0

    def select_action(self, state):     # on-policy
        state = torch.Tensor(state).unsqueeze(0)
        probs = self.net(state)
        tmp = Categorical(probs)        # 依概率生成
        action = tmp.sample()
        log_prob = tmp.log_prob(action)
        return action.item(), log_prob  # 记录概率

    def update_parameters(self, rewards, log_probs):
        R = 0
        loss = 0
        for i in reversed(range(len(rewards))):     # 逆路径计算
            R = rewards[i] + self.GAMMA * R
            loss = loss - R * log_probs[i]          # 概率权重
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self, num_episode = 1000):
        average_reward = 0
        for i_episode in tqdm.trange(1, num_episode + 1):
            state = self.env.reset()
            log_probs = []
            rewards = []
            while True:
                # self.env.render()
                action, prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                log_probs.append(prob)
                rewards.append(reward)
                average_reward += reward
                state = next_state
                if done:
                    if i_episode % 100 == 0:
                        print('episode: ', i_episode, "tot_rewards: ", np.sum(rewards), 'average_rewards: ', average_reward / 100)
                        average_reward = 0
                    break
            self.update_parameters(rewards, log_probs)      # 蒙特卡洛

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    Agent = Reinforce(envin=env, hidden_size=16)
    Agent.train(num_episode=2000)

    total_reward = 0
    state = env.reset()
    while True:
        env.render()
        action, prob = Agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            print("total_reward: ", total_reward)
            break