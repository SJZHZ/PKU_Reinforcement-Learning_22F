import numpy as np
from random import random, randint
from time import sleep
from matplotlib import pyplot as plt

class NormalDistBandit:
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must be the same length."
        self.n = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)
        assert all(self.stds >= 0), "Stds must be positive."

    def pull(self, k):
        assert 0 <= k < self.n, f"Invalid arm {k}."
        return np.random.normal(loc=self.means[k], scale=self.stds[k])

def epsilon_greedy(values, epsilon):
    assert len(values) > 1, "There should be 2 or more values."
    eps = epsilon * len(values) / (len(values) - 1)
    if random() <= eps:
        return randint(0, len(values)-1)
    return int(np.argmax(values))


if __name__ == "__main__":
    n = 5
    bandit = NormalDistBandit(means = np.array(range(-n, n+1)), stds = np.ones(11))

    # Task：把下面这几种epsilon的曲线画到一张图中，分析你所观察到的结果。
    epsilons = [0.01, 0.05, 0.1, 0.2]

    # 以下绘制epsilon=0.01的曲线图
    iter = 10000
    eps = 0.01

    plt.figure()

    for eps in epsilons:
        x = np.array(range(iter))
        y = np.zeros(iter, dtype=np.float64)

        values = np.zeros(n*2+1, dtype=np.float64)
        counts = np.zeros(n*2+1, dtype=np.int64)
        for i in range(1, iter):
            action = epsilon_greedy(values, eps)
            counts[action] += 1
            value = bandit.pull(action)
            values[action] = (values[action] * (counts[action] - 1) + value) / counts[action]
            y[i] = (y[i-1] * (i-1) + value) / i

        plt.plot(x, y, label = eps)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Average reward')
    plt.show()

# 观察到的现象：
#     对学习率为0.2的情况，很快就收敛，但充分迭代后收敛的值不如其他的好。
#     对学习率为0.01的情况，初始的效果不如其他的好，收敛速度缓慢。
#     0.1和0.05的情况则介于二者之间
# 一般来说：
#     如果步长太大，收敛速度很快，但收敛值可能会在最优解周围震荡而不够接近最优解
#     如果步长太小，收敛速度很慢，但收敛值在充分迭代后可以更加靠近最优解
