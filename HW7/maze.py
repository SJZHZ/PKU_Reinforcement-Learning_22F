import numpy as np
import gym
import tqdm
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List

class MazeEnv(gym.Env):
    WALL = 1
    EMPTY = 0
    ACTIONS = {'x':[-1,1,0,0], 'y':[0,0,1,-1]}
    StateType = Tuple[int, int]

    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)

        self.max_y, self.max_x = self.maze.shape
        assert(self.maze[self.start_y, self.start_x] == MazeEnv.EMPTY)
        assert(self.maze[self.target_y, self.target_x] == MazeEnv.EMPTY)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([self.max_x, self.max_y])

    def is_valid_state(self, x, y):
        return  0 <= x < self.max_x and 0 <= y < self.max_y and self.maze[y, x] == MazeEnv.EMPTY

    def step(self, action:int) -> Tuple[StateType, float, bool, None]:
        new_x, new_y = self.x + MazeEnv.ACTIONS['x'][action], self.y + MazeEnv.ACTIONS['y'][action]

        if not self.is_valid_state(new_x, new_y):
            new_x, new_y = self.x, self.y
        self.x, self.y = new_x, new_y

        reach_target = ((self.x, self.y) == (self.target_x, self.target_y))
        reward = 1.0 if reach_target else 0.0

        return (self.x, self.y), reward, reach_target, None

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        return (self.x, self.y)

    def render(self):
        for i in range(self.max_y):
            for j in range(self.max_x):
                if (j, i) == (self.x, self.y):
                    print('*', end='')
                elif (j, i) == (self.target_x, self.target_y):
                    print('G', end='')
                else:
                    print(self.maze[i, j], end='')
            print()
        print()

    def close(self):
        pass

class DQN(nn.Module):       # 模型
    def init(self, m:nn.Module):                                    # 初始化模型
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)                # 某种合适的初值
            nn.init.constant_(m.bias, 0)

    def __init__(self, in_dim, out_dim):                            # 构造模型
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.network = nn.Sequential(                        # 三层线性变换，中间用relu连接
            nn.Linear(in_dim, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, out_dim, bias=True)
        )
        self.network.apply(self.init)

    def forward(self, state_batch:torch.Tensor) -> torch.Tensor:    # 前传
        return self.network(state_batch.to(torch.float32))  # 输入是整数，要变换成浮点数

class DQNTrainer:           # 训练器
    DataType = Tuple[np.ndarray, int, float, np.ndarray]            # 类型定义
    def __init__(self, args:Dict):                                  # 构造函数
        for k,v in args.items():
            setattr(self, k, v)

        self.network.to(self.device)                # 选择GPU
        self.epsilon = self.epsilon_lower
        self.buffer = list()
        self.buffer_pointer = 0
        self.reward_list = list()                   # 记录过程reward

    def add_to_buffer(self, data:DataType):                         # 循环放入buffer
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size

    def sample_batch(self) -> List[DataType]:                       # 批量随机抽样
        return random.sample(self.buffer, self.batch_size)

    def epsilon_greedy(self, state:np.ndarray):                     # e-贪心
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n-1)
        return self.greedy(state)

    def greedy(self, state):                                        # 贪心
        # network是神经网络模型，operator()即模型的作用
        # detach把输出张量从模型（计算图）分离出来
        # 返回的张量是一个action的概率分布，取argmax即贪心
        return self.network(torch.tensor(state, device=self.device)).detach().cpu().numpy().argmax()

    def epsilon_decay(self, total_step):                            # epsilon衰减
        self.epsilon = self.epsilon_lower + (self.epsilon_upper-self.epsilon_lower) * math.exp(-total_step/self.epsilon_decay_freq)

    def update_model(self, total_step):                             # 更新模型
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            return
        batch = self.sample_batch()
        state_batch, action_batch, reward_batch, new_state_batch = zip(*batch)      # 批量数据按元组解压

        # target_values是(S,A)对的观测价值Vm(S)[A] = gamma * max(Vm(S')) + R(S,A)
        target_values = (self.gamma * self.network(torch.tensor(np.array(new_state_batch), device=self.device)).detach().max(dim=1).values
            + torch.tensor(reward_batch, device=self.device))
        # values是模型在(S,A)对上的估计价值Vm(S)[A]
        #     gather通过索引取数据；unsqeeze在第dim位置插入维度为1的维；squeeze删除维度为1的维（默认全部，也可指定位置）
        values = (self.network(torch.tensor(np.array(state_batch), device=self.device))
            .gather(dim=1, index=torch.tensor(action_batch, device=self.device).unsqueeze(dim=1))).squeeze()

        self.optimizer.zero_grad()                  # 零化梯度（清零上一轮的冲量）
        loss = F.mse_loss(target_values, values)    # 计算损失函数loss
        loss.backward()                             # 回传loss计算梯度
        self.optimizer.step()                       # 使用optimizer对参数按梯度更新

    def train(self):                                                # 训练模型
        total_step = 0
        total_reward = self.total_reward * self.start_iter
        for i in tqdm.trange(self.start_iter, self.iter):
            state = self.env.reset()
            episode_step = 0
            done = False
            while not done:
                total_step += 1
                # float_state = np.array(state) * 1.0
                action = self.epsilon_greedy(state)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = self.end_reward
                reward -= 1
                self.add_to_buffer((state, action, reward, new_state))
                self.update_model(total_step)
                state = new_state

                total_reward += reward
                episode_step += 1

                self.epsilon_decay(total_step)
                if self.render:
                    env.render()
                # self.save_model(i)
            # self.reward_list.append(total_reward / (i + 1))
            print(episode_step)
            # self.reward_list.append(episode_reward)
            # if (i % 100 == 0):
            #     test_avr(self, self.env, 100)


def print_maze_policy(maze:np.ndarray, policy):
    max_y, max_x = maze.shape
    for i in range(max_y):
        for j in range(max_x):
            if maze[i, j] == MazeEnv.EMPTY:
                print('<>v^'[policy((j, i))], end='')
            else:
                print(maze[i, j], end='')
        print()
    print()

def random_maze_policy(state:MazeEnv.StateType):
    return random.randint(0, 3)

if __name__ == '__main__':
    maze = np.array([
        [0,0,0,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0]
    ])
    env = MazeEnv({
        'maze':maze,
        'start_x':0, 'start_y':2,
        'target_x':8, 'target_y':0
    })
    env_name = 'Maze'

    network = DQN(in_dim=env.observation_space.shape[0], out_dim=env.action_space.n)
    trainer = DQNTrainer({
        'env':env,
        'env_name':env_name,
        'render':False,
        'end_reward':10000,           # 行动奖励为0，终止奖励为正数
        'network':network,
        'start_iter':0,
        'iter':1000,
        'gamma':0.9,
        'batch_size':128,
        'buffer_size':20000,
        'update_freq':2,
        'epsilon_lower':0.05,
        'epsilon_upper':0.8,
        'epsilon_decay_freq':10000,
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer':optim.Adam(network.parameters()),
        # 'save_path':save_path,
        # 'save_freq':50,
        'total_reward':0
    })
    trainer.train()

    print_maze_policy(maze, trainer.greedy)


    env.close()