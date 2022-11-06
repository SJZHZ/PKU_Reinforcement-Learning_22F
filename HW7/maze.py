import numpy as np
import gym
from typing import Tuple, Dict
import random

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
    
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = random_maze_policy(state)
        new_state, reward, done, info = env.step(action)
        state = new_state
    
    print_maze_policy(maze, random_maze_policy)

    env.close()