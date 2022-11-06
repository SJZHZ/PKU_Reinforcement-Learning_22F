import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import gym
import random
import math
import numpy as np
import os
from typing import Dict, Tuple, List

class DQN(nn.Module):       # 模型
    def init(self, m:nn.Module):                                    # 初始化模型
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)                # 某种合适的初值
            nn.init.constant_(m.bias, 0)

    def __init__(self, in_dim, out_dim):                            # 构造模型
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.network = nn.Sequential                        # 三层线性变换，中间用relu连接
        (
            nn.Linear(in_dim, (in_dim+out_dim), bias=True),
            nn.ReLU(),
            nn.Linear((in_dim+out_dim), (in_dim+out_dim), bias=True),
            nn.ReLU(),
            nn.Linear((in_dim+out_dim), out_dim, bias=True)
        )
        self.network.apply(self, fn=self.init)


    def forward(self, state_batch:torch.Tensor) -> torch.Tensor:    # 前传
        return self.network(state_batch)

class DQNTrainer:           # 训练器
    DataType = Tuple[np.ndarray, int, float, np.ndarray]            # 类型定义
    def __init__(self, args:Dict):                                  # 构造函数
        for k,v in args.items():
            setattr(self, k, v)

        self.epsilon = self.epsilon_lower
        self.buffer = list()
        self.buffer_pointer = 0
        self.network.to(self.device)                # 选择GPU

    def add_to_buffer(self, data:DataType):                         # 循环放入buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size

    def sample_batch(self) -> List[DataType]:                       # 批量给出样本
        return random.sample(self.buffer, self.batch_size)

    def epsilon_greedy(self, state:np.ndarray):                     # e-贪心
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n-1)
        return self.greedy(state)

    def greedy(self, state):                                        # 贪心
        # network是神经网络模型
        # operator()即模型的作用
        # detach把输出张量从模型（计算图）分离出来
        # 返回的张量是一个action的概率分布，取argmax即贪心
        return self.network(torch.tensor(state, device=self.device)).detach().cpu().numpy().argmax()

    def update_model(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        state_batch, action_batch, reward_batch, new_state_batch = zip(*batch)

        target_values = (self.gamma * self.network(torch.tensor(np.array(new_state_batch), device=self.device)).detach().max(dim=1).values
            + torch.tensor(reward_batch, device=self.device))
        values = (self.network(torch.tensor(np.array(state_batch), device=self.device))
            .gather(dim=1, index=torch.tensor(action_batch, device=self.device).unsqueeze(dim=1))).squeeze()

        self.optimizer.zero_grad()
        loss = F.mse_loss(target_values, values)
        loss.backward()
        self.optimizer.step()

    def epsilon_decay(self, total_step):                            # epsilon衰减
        self.epsilon = self.epsilon_lower + (self.epsilon_upper-self.epsilon_lower) * math.exp(-total_step/self.epsilon_decay_freq)

    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = self.end_reward
                self.add_to_buffer((state, action, reward, new_state))
                self.update_model(total_step)
                state = new_state

                if self.render:
                    env.render()
                self.epsilon_decay(total_step)
                # self.save_model(i)

    def save_model(self, i):                                        # 保存模型
        if i % self.save_freq == 0:
            torch.save(self.network.state_dict(), os.path.join(self.save_path, f'{i}.pkl'))

if __name__ == '__main__':
    env_name = 'CartPole-v1'
    save_path = 'models'

    env = gym.make(env_name)
    network = DQN(in_dim=env.observation_space.shape[0], out_dim=env.action_space.n)
    latest_checkpoint = 0

    # if save_path not in os.listdir():
    #     os.mkdir(save_path)
    # elif len(os.listdir(save_path)) != 0:
    #     latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
    #     print(f'{latest_checkpoint}.pkl loaded')
    #     network.load_state_dict(torch.load(os.path.join(save_path, f'{latest_checkpoint}.pkl')))

    trainer = DQNTrainer({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'network':network,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint+1000,
        'gamma':0.8,
        'batch_size':64,
        'buffer_size':10000,
        'update_freq':1,
        'epsilon_lower':0.05,
        'epsilon_upper':0.9,
        'epsilon_decay_freq':200,
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer':optim.Adam(network.parameters()),
        'save_path':save_path,
        'save_freq':50
    })

    trainer.train()

    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = trainer.greedy(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        env.render()
    print(episode_reward)
    env.close()