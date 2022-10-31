#DP解简单的迷宫
import numpy as np

M = 100     #墙
T = 100     #终点

#可行域
active = np.zeros((8, 8))
active0 = np.array(
    [[1, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 1]])
active[1:-1, 1:-1] = active0
print(active)

#动作价值表
reward = np.ones((8,8)) * -M
reward0 = np.array(
    [[-1, -M, -1, -1, -1, -1],
    [-1, -M, -1, -1, -M, -1],
    [-1, -M, -1, -M, -1, -1],
    [-1, -M, -1, -M, -1, -1],
    [-1, -M, -1, -M, -1, -1],
    [-1, -1, -1, -M, -1, -1]])
reward[1:-1, 1:-1] = reward0
print(reward)

#状态价值表
value = np.ones_like(reward) * -M
value[6, 5] = T
print(value)

#优化上下左右
dX = [0, 1, 0, -1]
dY = [-1, 0, 1, 0]

for iter in range(20):                      #迭代轮数不小于最长路径长度（X*Y）
    for i in range(8):                      #每次迭代遍历整个矩阵
        for j in range(8):
            if (active[i, j] == 0):         #跳过可行域
                continue
            for action in range(4):         #上下左右
                ii = i + dX[action]
                jj = j + dY[action]
                value[i, j] = max(value[i, j], reward[ii, jj] + value[ii, jj])
    print(value)
