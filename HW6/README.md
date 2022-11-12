# 作业1.6
## Maze
### 说明
> 在Maze环境上实现并对比Dyna-Q算法和Dyna-Q+算法的表现，撰写实验报告并提交代码。<br>
> 使用Sutton RL book P166 Example 8.2 Blocking Maze 和 P167 Example 8.3 Shortcut Maze两个例子来对比上述两个算法。
### 提示
1. 样例中已经实现Q学习算法可供参考。
2. 对于Dyna-Q / Q+算法的planning步骤，可以直接使用Maze环境，无需自己学习一个环境模型。
### 做法
1. Dyna-Q实现不难，存储模型并用来更新即可
2. Dyna-Q+增加了一项KT_reward，在模拟模型（而非探索环境）时使用。但具体过程不是很明确：模拟中存在【动作选择】和【更新价值】两处可以使用KT_reward，我只在更新价值时使用了KT_reward

## DiscreteCartPole
### 说明
> DiscreteCartPole（倒立摆）是OpenAI gym包中的一个经典控制环境，通过action控制倒立摆向左/右的力度，尽量使其不倒<br>
> 在DiscreteCartPole环境上实现n-step TD学习算法，对比不同n（至少3种）对算法效果的影响，撰写实验报告并提交代码。
### 提示
1. 需要安装gym包：pip install gym
2. python样例程序中已经实现离散化的CartPole环境以及其上的Q学习算法。
3. 可以尝试调整训练的各个超参数（包括离散化的粒度）来获得更好的训练效果。
### 做法
1. 注意QLearner类的update_q函数是批量更新的，样本的消费-生产比为batch_size/update_freq
2. nTDLearner类的实现思路：
  > 对learn函数中的每个事件维护一个buffer（clear复用），每步探索环境时就把信息存入buffer。<br>
  > 如果事件结束，启动update_q函数。<br>
  > update_q函数负责用本事件的buffer更新q：对每个起始状态G=0，按事件发生顺序向后n步采集R，如果事件没有终止再补上q
3. 效果不好的原因：每个事件只作一次更新，利用率不高。可以考虑改成批量更新