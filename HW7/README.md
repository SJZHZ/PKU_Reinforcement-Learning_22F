# 作业1.7
## CartPole
### 说明：
> cartpole.py中实现了DQN算法（with replay buffer），要求看懂代码并尝试调参（可以改任何你想改的部分），撰写实验报告（1-2页为宜）。
### 做法：
可调节的超参数有：
1. 网络结构
    > 层数<br>
    > 每层宽度<br>
    > （激活函数）<br>
    > （优化器）
2. 常数参数
    > γ, ε<br>
    > （batch_size, buffer_size, update_freq）

## Bonus：Maze
> maze.py中实现了迷宫环境。要求使用基于神经网络的值迭代算法（DQN/SARSA等）求解迷宫，并用print_maze_policy输出学习得到的策略。<br>
> 可以参考CartPole的DQN实现来完成此任务。<br>
> 按时正确实现总评+1分。不完成此任务不会扣分。