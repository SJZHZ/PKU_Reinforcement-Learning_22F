#include <utility>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;
class WindyGridWorld{
    public:
        static const int
            LEFT=0, RIGHT=1, UP=2, DOWN=3;
        static const char ACTION_NAME[][16];
        static const int WIND_X[];
        typedef pair<int, int> State;
        bool verbose;
        State state()                   //GET state
        {
            return make_pair(x, y);
        }
        void set_state(int x, int y)
        {
            this->x = x;
            this->y = y;
            if (verbose){
                cout << verbose << endl;
                cout << "State reset: (" << x << "," << y << ")" << endl;
            }
        }
        void reset(){
            set_state(0, 3);
        }
        pair<State, double> step(int action)    //状态转移
        {
            State old_state = state();
            double reward = state_transition(action);
            if (verbose){
                cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
                cout << "Action: " << ACTION_NAME[action] << endl;
                cout << "Wind: (0," << WIND_X[old_state.first] << ")" << endl;
                cout << "Reward: " << reward << endl;
                cout << "New State: (" << x << "," << y << ")" << endl << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action() const {
            return rand() % 4;
        }
        bool done() const {
            return x == 7 and y == 3;
        }
        WindyGridWorld(int x=0, int y=3, bool verbose=false){
            this->verbose = verbose;
            set_state(x, y);
        }

    private:
        int x, y;

        double state_transition(int action){
            int new_x = x, new_y = y;
            switch(action){
                case LEFT:
                    -- new_x;
                    break;
                case RIGHT:
                    ++ new_x;
                    break;
                case UP:
                    ++ new_y;
                    break;
                case DOWN:
                    -- new_y;
                    break;
            }
            new_y += WIND_X[x];
            x = max(0, new_x);
            x = min(x, 10-1);
            y = max(0, new_y);
            y = min(y, 7-1);
            return -1;
        }
};
const char WindyGridWorld::ACTION_NAME[][16] = {"LEFT(-1,0)", "RIGHT(1,0)", "UP(0,1)", "DOWN(0,-1)"};
const int WindyGridWorld::WIND_X[] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};

class WindyGridWorldPolicyBase
{
    public:
        virtual int operator() (const WindyGridWorld::State& state) const = 0;
        void print_path(void) const
        {
            WindyGridWorld env = WindyGridWorld();
            WindyGridWorld::State state;
            int episode_len = 0;
            while (not env.done())
            {
                state = env.state();
                cout << "(" << state.first << "," << state.second << ")->";
                env.step((*this)(state));
                ++ episode_len;
            }
            cout << "(" << env.state().first << "," << env.state().second << ")." << endl;
            cout << "Episode length: " << episode_len << endl;
        }
};

class WindyGridWorldPolicySarsa : public WindyGridWorldPolicyBase{
    public:
        WindyGridWorldPolicySarsa(){
            epsilon = 0.1;
            alpha = 0.5;
            gamma = 1.0;
            memset(q, 0, sizeof(q));
        }
        virtual int operator() (const WindyGridWorld::State& state) const {
            int best_action = 0;
            int x = state.first, y = state.second;
            double best_value = q[y][x][0];
            for (int i = 1; i < 4; ++ i){
                if (q[y][x][i] > best_value){
                    best_value = q[y][x][i];
                    best_action = i;
                }
            }
            return best_action;
        }

        //新增函数episode，用于策略递归产生episode，返回步数
        int episode(int cnt = 1, WindyGridWorld env = WindyGridWorld(), int action = -1)
        {
            WindyGridWorld::State state = env.state();
            if (action == -1)       //如果调用时未指定，则依据策略生成
                action = epsilon_greedy(state);         //e贪心
            int x = state.first, y = state.second;

            pair<WindyGridWorld::State, double> temp = env.step(action);
            WindyGridWorld::State ss = temp.first;
            double reward = temp.second;
            int aa = epsilon_greedy(ss);                //e贪心
            int xx = ss.first, yy = ss.second;

            q[y][x][action] = q[y][x][action] + alpha * (reward + gamma * q[yy][xx][aa] - q[y][x][action]);
            if (env.done())
                return cnt;                             //返回总步数
            else
                return episode(cnt + 1, env, aa);       //递归进入下一步，指定好了action
        }
        void learn(int iter = 1000000) // SARSA策略
        {
            // TODO
            while (iter > 0)            //不断新建episode
            {
                //我最初在learn函数中【直接递归】模拟episode：未done就递归继续下一步，done就递归新建环境
                //这样递归没有及时清除已结束的episode，迭代数万次就会【爆栈】
                //迭代必须按episode（调用）或者step（循环）分解
                iter -= episode();
            }
        }

        void print_path(void) const {
            cout << "Sarsa result:" << endl;
            this->WindyGridWorldPolicyBase::print_path();
        }
    private:
        // int visited[7][10][4] = {0};
        double q[7][10][4];         //动作-价值函数
        double epsilon, alpha, gamma;
        int epsilon_greedy(const WindyGridWorld::State& state)  //随机
        {
            if (rand() % 100000 < epsilon * 100000)
            {
                return rand() % 4;
            }
            return (*this)(state);
        }
};

class WindyGridWorldPolicyQLearning : public WindyGridWorldPolicyBase{
    public:
        WindyGridWorldPolicyQLearning()         //构造函数初始化数据
        {
            epsilon = 0.1;
            alpha = 0.5;
            gamma = 1.0;
            memset(q, 0, sizeof(q));
        }
        virtual int operator() (const WindyGridWorld::State& state) const   //重载运算符
        {
            int best_action = 0;
            int x = state.first, y = state.second;
            double best_value = q[y][x][0];
            for (int i = 1; i < 4; ++ i){
                if (q[y][x][i] > best_value){
                    best_value = q[y][x][i];
                    best_action = i;
                }
            }
            return best_action;
        }

        //新增函数episode，用于策略递归产生episode，返回步数
        int episode(int cnt = 1, WindyGridWorld env = WindyGridWorld())
        {
            WindyGridWorld::State state = env.state();
            int action = epsilon_greedy(state);         //e贪心
            int x = state.first, y = state.second;

            pair<WindyGridWorld::State, double> temp = env.step(action);
            WindyGridWorld::State ss = temp.first;
            double reward = temp.second;
            int aa = (*this)(ss);                       //最大值
            int xx = ss.first, yy = ss.second;

            q[y][x][action] = q[y][x][action] + alpha * (reward + gamma * q[yy][xx][aa] - q[y][x][action]);
            if (env.done())
                return cnt;                             //返回总步数
            else
                return episode(cnt + 1, env);           //递归进入下一步
        }
        void learn(int iter = 1000000)          //QLearning
        {
            //TODO
            while (iter > 0)
            {
                iter -= episode();
                //cout << iter << endl;
            }
        }

        void print_path(void) const {
            cout << "Q learning result:" << endl;
            this->WindyGridWorldPolicyBase::print_path();
        }
    private:
        double q[7][10][4];
        double epsilon, alpha, gamma;
        int epsilon_greedy(const WindyGridWorld::State& state){
            if (rand() % 100000 < epsilon * 100000){
                return rand() % 4;
            }
            return (*this)(state);
        }
};


#include <chrono>
#include <thread>

int main()
{
    //srand(time(nullptr));
    WindyGridWorldPolicySarsa policy_sarsa;
    policy_sarsa.learn();
    policy_sarsa.print_path();

    WindyGridWorldPolicyQLearning policy_q;
    policy_q.learn();
    policy_q.print_path();
    return 0;
}
/*
测试结果
Sarsa result:
(0,3)->(1,3)->(2,3)->(3,3)->(4,4)->(5,5)->(6,6)->(7,6)->(8,6)->(9,6)->(9,5)->(9,4)->(9,3)->(9,2)->(8,2)->(7,3).
Episode length: 15
Q learning result:
(0,3)->(1,3)->(2,3)->(3,3)->(4,4)->(5,5)->(6,6)->(7,6)->(8,6)->(9,6)->(9,5)->(9,4)->(9,3)->(9,2)->(8,2)->(7,3).
Episode length: 15
*/