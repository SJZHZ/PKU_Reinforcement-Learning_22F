# 作业1.6
## Maze
  在Maze环境上实现并对比Dyna-Q算法和Dyna-Q+算法的表现，撰写实验报告并提交代码。

  使用Sutton RL book P166 Example 8.2 Blocking Maze 和 P167 Example 8.3 Shortcut Maze两个例子来对比上述两个算法。
### 提示：
  1. 样例中已经实现Q学习算法可供参考。
  2. 对于Dyna-Q / Q+算法的planning步骤，可以直接使用Maze环境，无需自己学习一个环境模型。
## DiscreteCartPole
  在DiscreteCartPole环境上实现n-step TD学习算法，对比不同n（至少3种）对算法效果的影响，撰写实验报告并提交代码。
### 提示：
  1. 需要安装gym包：pip install gym
  2. python样例程序中已经实现离散化的CartPole环境以及其上的Q学习算法。
  3. 可以尝试调整训练的各个超参数（包括离散化的粒度）来获得更好的训练效果。
