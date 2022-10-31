#include <ctime>
#include <vector>
#include <cmath>
#include "maze.hpp"

class MazePolicyBase
{
    public:
        virtual int operator()(const MazeEnv::State& state) const = 0;
};

class MazePolicyQLearning : public MazePolicyBase
{
    public:
        int operator()(const MazeEnv::State& state) const           //基于状态动作价值表返回最佳动作
        {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action)
            {
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value)
                {
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazePolicyQLearning(const MazeEnv& e) : env(e)              //构造函数
        {
            epsilon = 0.1;
            alpha = 0.1;
            gamma = 0.95;
            q = new double[e.max_x * e.max_y * 4];
            srand(2022);
            for (int i = 0; i < e.max_x * e.max_y * 4; ++ i)
                q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
        }

        ~MazePolicyQLearning()                                      //析构函数
        {
            //env.~MazeEnv();
            delete []q;
        }

        void learn(int iter=1000, int verbose_freq=10)              //QL更新状态动作价值表
        {
            bool done;
            int action, next_action;
            double reward;
            int episode_step;
            MazeEnv::State state, next_state;
            MazeEnv::StepResult step_result;

            for (int i = 0; i < iter; ++ i)
            {
                state = env.reset();
                done = false;
                episode_step = 0;
                while (not done)
                {
                    action = epsilon_greedy(state);         //epsilon-greedy
                    step_result = env.step(action);         //返回结果
                    next_state = step_result.next_state;
                    reward = step_result.reward;
                    done = step_result.done;
                    ++ episode_step;
                    next_action = (*this)(next_state);      //选取下一个最优动作
                    q[locate(state, action)] +=             //更新动作价值
                        alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);
                    state = next_state;
                }
                if (i % verbose_freq == 0)
                    cout << "episode_step: " << episode_step << endl;
            }
        }

        int epsilon_greedy(MazeEnv::State state) const              //epsilon-greedy选取动作
        {
            if (rand() % 100000 < epsilon * 100000)
                return rand() % 4;
            return (*this)(state);      //最优策略
        }

        inline int locate(MazeEnv::State state, int action) const   //状态压缩：用一个int表示(x,y,action)
        {
            return state.second * env.max_x * 4 + state.first * 4 + action;
        }

        void print_policy() const                                   //打印最优动作矩阵
        {
            static const char action_vis[] = "<>v^";
            int action;
            MazeEnv::State state;
            for (int i = 0; i < env.max_y; ++ i)
            {
                for (int j = 0; j < env.max_x; ++ j)
                {
                    state = MazeEnv::State(j, i);
                    if (not env.is_valid_state(state))
                        cout << "#";
                    else if (env.is_goal_state(state))
                        cout << "G";
                    else
                    {
                        action = (*this)(MazeEnv::State(j, i));
                        cout << action_vis[action];
                    }
                }
                cout << endl;
            }
            cout << endl;
        }

    private:
        MazeEnv env;
        double *q;      //长宽不固定，不能直接生成三维矩阵
        double epsilon, alpha, gamma;
};


class MazePolicyDQ : public MazePolicyBase      //由于QLearning中有私有成员变量，不方便继承，干脆直接复制一份
{
    public:
        int operator()(const MazeEnv::State& state) const           //基于状态动作价值表返回最佳动作
        {
            int best_action = 0;
            double best_value = q[locate(state, 0)];
            double q_s_a;
            for (int action = 1; action < 4; ++ action)
            {
                q_s_a = q[locate(state, action)];
                if (q_s_a > best_value)
                {
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }

        MazePolicyDQ(const MazeEnv& e) : env(e)                     //构造函数
        {
            epsilon = 0.1;
            alpha = 0.1;
            gamma = 0.95;
            total_Reward = 0;
            total_Step = 0;

            q = new double[e.max_x * e.max_y * 4];
            model_Reward = new double[e.max_x * e.max_y * 4];           //给定状态动作对(S,A)，返回价值R
            model_NextState = new MazeEnv::State[e.max_x * e.max_y * 4];//给定状态动作对(S,A)，返回下一状态S'
            model_Visited = new int[e.max_x * e.max_y * 4];             //给定状态动作对(S,A)，返回是否访问过

            srand(2022);
            for (int i = 0; i < e.max_x * e.max_y * 4; ++ i)
                q[i] = 1.0 / (rand() % (e.max_x * e.max_y) + 1);
            memset(model_Visited, 0, sizeof(model_Visited));
            model_PrevState.clear();
        }

        ~MazePolicyDQ()                                             //析构函数
        {
            //env.~MazeEnv();
            delete []q;
            delete[] model_NextState;
            delete[] model_Reward;
            delete[] model_Visited;

        }

        void learn                                                  //DQ更新状态动作价值表
        (int iter=1000, int verbose_freq=10, int T = 0, int max_step = 20000)
        {
            bool done;
            int action, next_action;
            double reward;
            int episode_step;
            MazeEnv::State state, next_state;
            MazeEnv::StepResult step_result;

            for (int i = 0; i < iter; ++ i)
            {
                state = env.reset();
                done = false;
                episode_step = 0;
                while (not done)
                {
                    action = epsilon_greedy(state);         //epsilon-greedy
                    step_result = env.step(action);         //返回结果
                    next_state = step_result.next_state;
                    reward = step_result.reward;
                    done = step_result.done;
                    ++ episode_step;
                    next_action = (*this)(next_state);      //选取下一个最优动作
                    q[locate(state, action)] +=             //更新动作价值
                        alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);

                    total_Reward += reward;
                    total_Step++;
                    if (total_Step % verbose_freq == 0)
                        cout << total_Reward << ' ';
                    if (total_Step >= max_step)
                        return;

                    // model learning
                    model_Reward[locate(state, action)] = reward;
                    model_NextState[locate(state, action)] = next_state;
                    model_Visited[locate(state, action)]++;
                    model_PrevState.push_back(locate(state, action));

                    for (int j = 0; j < T; j++)
                    {
                        //search control
                            //随机State&Action
                        int temp = rand() % model_PrevState.size();
                        int temp_State_Action = model_PrevState[temp];
                            //获取Reward&NextState&NextAction
                        double temp_Reward = model_Reward[temp_State_Action];
                        MazeEnv::State temp_NextState = model_NextState[temp_State_Action];
                        int temp_NextAction = (*this)(temp_NextState);

                        //planning update
                        q[temp_State_Action] +=
                            alpha * (gamma * q[locate(temp_NextState, temp_NextAction)] + temp_Reward - q[temp_State_Action]);
                    }

                    state = next_state;
                }
                // if (i % verbose_freq == 0)
                //     cout << i << " episode_step : " << episode_step << endl;
            }
        }

        int epsilon_greedy(MazeEnv::State state) const              //epsilon-greedy选取动作
        {
            if (rand() % 100000 < epsilon * 100000)
                return rand() % 4;
            return (*this)(state);      //最优策略
        }

        inline int locate(MazeEnv::State state, int action) const   //状态压缩：用一个int表示(x,y,action)
        {
            return state.second * env.max_x * 4 + state.first * 4 + action;
        }

        void print_policy() const                                   //打印最优动作矩阵
        {
            static const char action_vis[] = "<>v^";
            int action;
            MazeEnv::State state;
            for (int i = 0; i < env.max_y; ++ i)
            {
                for (int j = 0; j < env.max_x; ++ j)
                {
                    state = MazeEnv::State(j, i);
                    if (not env.is_valid_state(state))
                        cout << "#";
                    else if (env.is_goal_state(state))
                        cout << "G";
                    else
                    {
                        action = (*this)(MazeEnv::State(j, i));
                        cout << action_vis[action];
                    }
                }
                cout << endl;
            }
            cout << endl;
        }

        MazeEnv env;
        //长宽不固定，不能直接生成三维矩阵
        double* q;
        // MODEL[S,A] <--> R, S'
        double* model_Reward;
        MazeEnv::State* model_NextState;
        int* model_Visited;
        vector<int> model_PrevState;

        double epsilon, alpha, gamma;
        double total_Reward;
        int total_Step;
};

class MazePolicyDQP : public MazePolicyDQ
{
    public:
        double KT(int S_A) const
        {
            return K * sqrt(global_Time - S_A_Time[S_A]);
        }
        int operator()(const MazeEnv::State& state) const
        {
            int best_action = 0;
            double best_value = q[locate(state, 0)] + KT(locate(state, 0));
            double q_s_a;
            for (int action = 1; action < 4; ++ action)
            {
                q_s_a = q[locate(state, action)] + KT(locate(state, action));
                if (q_s_a > best_value)
                {
                    best_value = q_s_a;
                    best_action = action;
                }
            }
            return best_action;
        }
        // int DQP(const MazeEnv::State& state)    //DQP最优
        // {
        //     int best_action = 0;
        //     double best_value = q[locate(state, 0)] + K * sqrt(global_Time - S_A_Time[locate(state, 0)]);
        //     double q_s_a;
        //     for (int action = 1; action < 4; ++ action)
        //     {
        //         q_s_a = q[locate(state, action)] + K * sqrt(global_Time - S_A_Time[locate(state, action)]);
        //         if (q_s_a > best_value)
        //         {
        //             best_value = q_s_a;
        //             best_action = action;
        //         }
        //     }
        //     return best_action;
        // }
        MazePolicyDQP(const MazeEnv& e, double KK) : MazePolicyDQ(e), K(KK)
        {
            global_Time = 0;
            S_A_Time = new int[e.max_y * e.max_x * 4];
            memset(S_A_Time, 0, sizeof(S_A_Time));
        }
        ~MazePolicyDQP()
        {
            delete[] S_A_Time;
        }

        void learn                                                  //DQ更新状态动作价值表
        (int iter=1000, int verbose_freq=10, int T = 0, int max_step = 20000)
        {
            bool done;
            int action, next_action;
            double reward;
            int episode_step;
            MazeEnv::State state, next_state;
            MazeEnv::StepResult step_result;

            for (int i = 0; i < iter; ++ i)
            {
                state = env.reset();
                done = false;
                episode_step = 0;
                while (not done)
                {
                    action = epsilon_greedy(state);         //epsilon-greedy
                    step_result = env.step(action);         //返回结果
                    next_state = step_result.next_state;
                    reward = step_result.reward;
                    done = step_result.done;
                    ++ episode_step;
                    next_action = (*this)(next_state);      //DQP选取最优动作
                    q[locate(state, action)] +=             //更新动作价值
                        alpha * (gamma * q[locate(next_state, next_action)] + reward - q[locate(state, action)]);

                    global_Time++;
                    S_A_Time[locate(state, action)] = global_Time;

                    total_Reward += reward;
                    total_Step++;
                    if (total_Step % verbose_freq == 0)
                        cout << total_Reward << ' ';
                    if (total_Step >= max_step)
                        return;

                    // model learning
                    model_Reward[locate(state, action)] = reward;
                    model_NextState[locate(state, action)] = next_state;
                    model_Visited[locate(state, action)]++;
                    model_PrevState.push_back(locate(state, action));

                    for (int j = 0; j < T; j++)
                    {
                        //search control
                            //随机State&Action
                        int temp = rand() % model_PrevState.size();
                        int temp_State_Action = model_PrevState[temp];
                            //获取Reward&NextState&NextAction
                        double temp_Reward = model_Reward[temp_State_Action];
                        MazeEnv::State temp_NextState = model_NextState[temp_State_Action];
                        int temp_NextAction = (*this)(temp_NextState);          //DQP

                        //planning update
                        q[temp_State_Action] +=
                            alpha * (gamma * q[locate(temp_NextState, temp_NextAction)] + temp_Reward - q[temp_State_Action]);
                    }

                    state = next_state;
                }
                // if (i % verbose_freq == 0)
                //     cout << i << " episode_step : " << episode_step << endl;
            }
        }







        double K;
        int global_Time;
        int *S_A_Time;
};

int main(){
    const int max_x = 9, max_y = 6;

/*
    const int start_x = 0, start_y = 2;
    const int target_x = 8, target_y = 0;
    int maze[max_y][max_x] =
    {
        {0,0,0,0,0,0,0,1,0},
        {0,0,1,0,0,0,0,1,0},
        {0,0,1,0,0,0,0,1,0},
        {0,0,1,0,0,0,0,0,0},
        {0,0,0,0,0,1,0,0,0},
        {0,0,0,0,0,0,0,0,0}
    };
    MazeEnv env(maze, max_x, max_y, start_x, start_y, target_x, target_y);

    MazePolicyQLearning policy(env);
    policy.learn(100, 1);
    printf("QL:\n");
    policy.print_policy();

    MazePolicyDQ DQpolicy0(env);
    DQpolicy0.learn(1000, 10, 0);
    printf("DQ0:\n");
    DQpolicy0.print_policy();

    MazePolicyDQ DQpolicy5(env);
    DQpolicy5.learn(1000, 10, 5);
    printf("DQ5:\n");
    DQpolicy5.print_policy();

    MazePolicyDQ DQpolicy50(env);
    DQpolicy50.learn(1000, 10, 50);
    printf("DQ50:\n");
    DQpolicy50.print_policy();
*/
    int Blocking_Maze1[max_y][max_x] =
    {
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {1,1,1,1,1,1,1,1,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}
    };
    MazeEnv Blocking_Env1(Blocking_Maze1, max_x, max_y, 3, 5, 8, 0);
/*
    int Blocking_Maze2[max_y][max_x] =
    {
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,1,1,1,1,1,1,1,1},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}
    };
    MazeEnv Blocking_Env2(Blocking_Maze1, max_x, max_y, 3, 5, 8, 0);
*/

    printf("Blocking DQ5:\n");
    MazePolicyDQ Blocking_DQ5(Blocking_Env1);
    Blocking_DQ5.learn(100, 10, 5, 20000);
    Blocking_DQ5.env.changebit(3 * 9 + 0);
    Blocking_DQ5.env.changebit(3 * 9 + 8);
    Blocking_DQ5.learn(100, 10, 5, 20000);

    printf("\nBlocking DQP5:\n");
    MazePolicyDQP Blocking_DQP5(Blocking_Env1, 0.0001);
    Blocking_DQP5.learn(100, 10, 5, 20000);
    Blocking_DQP5.env.changebit(3 * 9 + 0);
    Blocking_DQP5.env.changebit(3 * 9 + 8);
    Blocking_DQP5.learn(100, 10, 5, 20000);

    int Shortcut_Maze1[max_y][max_x] =
    {
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,1,1,1,1,1,1,1,1},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0}
    };
    MazeEnv Shortcut_Env1(Shortcut_Maze1, max_x, max_y, 3, 5, 8, 0);
    printf("\nShortcut DQ5:\n");
    MazePolicyDQ Shortcut_DQ5(Shortcut_Env1);
    Shortcut_DQ5.learn(1000, 10, 5, 20000);
    Shortcut_DQ5.env.changebit(3 * 9 + 8);
    Shortcut_DQ5.learn(1000, 10, 5, 20000);

    printf("\nShortcut DQP5:\n");
    MazePolicyDQP Shortcut_DQP5(Shortcut_Env1, 0.01);
    Shortcut_DQP5.learn(1000, 10, 5, 20000);
    Shortcut_DQP5.env.changebit(3 * 9 + 8);
    Shortcut_DQP5.learn(1000, 10, 5, 20000);
    printf("\n");
    Shortcut_DQP5.print_policy();

    return 0;
}