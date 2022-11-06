import gym
import numpy as np
import tqdm
import random
import math
from typing import Tuple, Dict, Any
import os

# gym模拟Cartpole环境
class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals=16):
        self._env = gym.make('CartPole-v1')
        self.action_space = self._env.action_space
        self.intervals = intervals
        self.observation_space = gym.spaces.MultiDiscrete([intervals]*4)
        self._to_discrete = lambda x, a, b: int(min(max(0, (x-a)*self.intervals/(b-a)), self.intervals))

    def render(self):       # 渲染
        self._env.render()

    def reset(self):
        return self._discretize(self._env.reset())

    def _discretize(self, state:np.array)->Tuple:
        cart_pos, cart_v, pole_angle, pole_v = state
        cart_pos = self._to_discrete(cart_pos, -2.4, 2.4)
        cart_v = self._to_discrete(cart_v, -3.0, 3.0)
        pole_angle = self._to_discrete(pole_angle, -0.5, 0.5)
        pole_v = self._to_discrete(pole_v, -2.0, 2.0)
        return (cart_pos, cart_v, pole_angle, pole_v)

    def step(self, action:int)->Tuple[Tuple, float, bool, Any]:
        state, reward, done, info = self._env.step(action)
        state = self._discretize(state)
        return state, reward, done, info

# QLearning
class QLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()
        self.buffer_pointer = 0
        self.reward_table = list()

    def add_to_buffer(self, data):                  # 循环装入buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)                    # 第一次：加入
        else:
            self.buffer[self.buffer_pointer] = data     # 第二次及以后：覆盖
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size

    def sample_batch(self):                         # 随机从buffer中取
        return random.sample(self.buffer, self.batch_size)

    def greedy(self, state:Tuple)->int:             # 贪心
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:     # e贪心
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)

    def epsilon_decay(self, total_step):            # 随着步数增加从lower靠近upper
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)

    def lr_decay(self, total_step):                 # 随着步数增加从lower靠近upper
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)

    def update_q(self, total_step):                 # 每隔一段时间批量更新q
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        for state, action, reward, new_state in batch:
            self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])

    def train(self):
        total_step = 0
        total_reward = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = self.end_reward
                total_reward += reward
                self.add_to_buffer((state, action, reward, new_state))      # 四元组存入buffer
                self.update_q(total_step)

                if self.render:     # 渲染选项
                    self.env.render()
                self.epsilon_decay(total_step)
                self.lr_decay(total_step)
                # self.save_model(i)
                state = new_state
            self.reward_table.append(total_reward / (i + 1))

    def save_model(self, i):
        if i % self.save_freq == 0:
            np.save(os.path.join(self.save_path, f'{i}.npy'), self.q)

# n-TDLearning
class nTDLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()

        self.reward_table = list()
        self.NTD = 0

    def add_to_buffer(self, data):                  # 循环装入buffer
        self.buffer.append(data)                    # 第一次：加入

    def sample_batch(self):                         # 随机从buffer中取
        return random.sample(self.buffer, self.batch_size)

    def greedy(self, state:Tuple)->int:             # 贪心
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:     # e贪心
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)

    def epsilon_decay(self, total_step):            # 随着步数增加从lower靠近upper
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)

    def lr_decay(self, total_step):                 # 随着步数增加从lower靠近upper
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)

    # 重要
    def update_q(self):                             # 终止时刻顺序更新q
        # print(len(self.buffer))
        for T in range(len(self.buffer)):                                           # 每个开始步
            G = 0
            state = self.buffer[T][0]                                               # S
            action = self.buffer[T][1]                                              # A
            for i in range(self.NTD + 1):
                if i == self.NTD:                                                   # n步终止
                    latest_state = self.buffer[T + i][0]
                    latest_action = self.buffer[T + i][1]
                    G += self.q[latest_state][latest_action] * (self.gamma ** i)    # V
                    break
                G += self.buffer[T + i][2] * (self.gamma ** i)                      # R
                if self.buffer[T + i][3] == 1:                                      # done终止
                    break
            self.q[state][action] += self.lr * (G - self.q[state][action])

    # 重要
    def train(self):
        total_step = 0
        total_reward = 0
        for i in tqdm.trange(self.start_iter, self.iter):
            self.buffer.clear()                 # 每个事件单独作一次TD
            state = self.env.reset()
            done = False
            iter_reward = 0
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = self.end_reward
                total_reward += reward
                iter_reward += reward

                self.add_to_buffer((state, action, reward, done))   # 去掉new_state，加入done
                if done:
                    self.update_q()

                if self.render:     # 渲染选项
                    self.env.render()
                self.epsilon_decay(total_step)
                self.lr_decay(total_step)
                state = new_state
            if (i % self.update_freq == 0):
                self.reward_table.append(total_reward / (i + 1))

# 测试trainer效果
def print_avr(trainer, env):
    avr = 1000
    avr_reward = 0
    for i in range(avr):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = trainer.greedy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        avr_reward += episode_reward
    print('Test Average:', avr_reward / avr)

# 把reward输出到文件中
def write_reward(path, name, trainer):
    file = open(os.path.join(path, name), 'w+')
    for r in trainer.reward_table:
        file.write(str(r) + '\n')
    file.close()
    return



if __name__ == '__main__':
    env_name = 'DiscreteCartPole'
    intervals = 8
    env = DiscreteCartPoleEnv(intervals)

    latest_checkpoint = 0
    save_path = 'q_tables'
    '''
    if save_path not in os.listdir():
        os.mkdir(save_path)
    elif len(os.listdir(save_path)) != 0:
        latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
        print(f'{latest_checkpoint}.npy loaded')
        q_table = np.load(os.path.join(save_path, f'{latest_checkpoint}.npy'))
    '''

    reward_path = os.path.join('Reward')
    if reward_path not in os.listdir():
        os.mkdir(reward_path)

    # QLearner
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    trainer = QLearner({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint+1000,
        'batch_size':128,
        'buffer_size':10000,
        'gamma':0.9,
        'update_freq':1,
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':50
    })
    trainer.train()
    print("QL")
    print_avr(trainer=trainer, env=env)
    write_reward(reward_path, 'QL', trainer=trainer)

    TDiter = 10000

    # n-TDLearner - 5
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    TDtrainer = nTDLearner({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint + TDiter,
        'batch_size':128,
        'buffer_size':10000,
        'gamma':0.9,
        'update_freq': (int)(TDiter / 1000),
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':50,
    })
    TDtrainer.NTD = 5
    TDtrainer.train()
    print("5TD")
    print_avr(TDtrainer, env)
    write_reward(reward_path, str(TDtrainer.NTD) + 'TD', TDtrainer)

    # n-TDLearner - 20
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    TDtrainer = nTDLearner({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint + TDiter,
        'batch_size':128,
        'buffer_size':10000,
        'gamma':0.9,
        'update_freq': (int)(TDiter / 1000),
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':50,
    })
    TDtrainer.NTD = 20
    TDtrainer.train()
    print("20TD")
    print_avr(TDtrainer, env)
    write_reward(reward_path, str(TDtrainer.NTD) + 'TD', TDtrainer)


    # n-TDLearner - 100
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    TDtrainer = nTDLearner({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint + TDiter,
        'batch_size':128,
        'buffer_size':10000,
        'gamma':0.9,
        'update_freq': (int)(TDiter / 1000),
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':50,
    })
    TDtrainer.NTD = 100
    TDtrainer.train()
    print("100TD")
    print_avr(TDtrainer, env)
    write_reward(reward_path, str(TDtrainer.NTD) + 'TD', TDtrainer)


    # n-TDLearner - 1
    q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    TDtrainer = nTDLearner({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'start_iter':latest_checkpoint,
        'iter':latest_checkpoint + TDiter,
        'batch_size':128,
        'buffer_size':10000,
        'gamma':0.9,
        'update_freq': (int)(TDiter / 1000),
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':200,
        'lr_lower':0.05,
        'lr_upper':0.5,
        'lr_decay_freq':200,
        'save_path':save_path,
        'save_freq':50,
    })
    TDtrainer.NTD = 1
    TDtrainer.train()
    print("1TD")
    print_avr(TDtrainer, env)
    write_reward(reward_path, str(TDtrainer.NTD) + 'TD', TDtrainer)

    env.close()