#include <ctime>
#include <iostream>
#include <vector>
#include "tictactoe.hpp"
#include <chrono>
#include <thread>

using namespace std;

class TicTacToePolicyBase   //策略基类
{
    public:
        //接口重载运算符()，可以当函数用
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
};


// randomly select a valid action for the step.
class TicTacToePolicyRandom : public TicTacToePolicyBase
{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const       //实现
        {
            vector<TicTacToe::Action> actions = state.action_space();
            int n_action = actions.size();
            int action_id = rand() % n_action;          //随机
            if (state.turn == TicTacToe::PLAYER_X)
                return actions[action_id];
            else
                return actions[action_id];
        }
        TicTacToePolicyRandom(){
            srand(time(nullptr));                       //随机数初始化
        }
};


// select the first valid action.   修改
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    public:
        float value[1 << 18];       //状态对应的价值
        float rate = 0.01;
        double prob = 0.8;
        int t = 0;
        TicTacToe* env;
        TicTacToe* test;

        void init()                      //初始化
        {
            for (int i = 0; i < 1 << 18; i++)
                value[i] = 0.5;
        }
        TicTacToe::Action find_max(TicTacToe* env) const
        {
            bool vb = env->verbose;
            env->verbose = 0;
            float maxvalue = 0;
            TicTacToe::State state = env->get_state();
            vector<TicTacToe::Action> actions = state.action_space();

            int actionssize = actions.size();
            TicTacToe::Action ans = actions[0];
            for (int i = 0; i < actionssize; i++)
            {
                env->step(actions[i]);       //试下
                TicTacToe::State tempstate = env->get_state();
                int tempboard = tempstate.board;
                if (value[tempboard] >= maxvalue)
                {
                    maxvalue = value[tempboard];
                    ans = actions[i];
                }
                env->step_back();        //撤回
            }
            env->verbose = vb;
            return ans;
        }
        void go()           //下棋
        {
            TicTacToe::State state = test->get_state();
            if (test->done())
                return;

            double v = rand() / double(RAND_MAX);
            TicTacToe::Action action = find_max(test);      //贪心
            vector<TicTacToe::Action> actions = state.action_space();
            //cout << v << endl;
            if (v > prob)                                   //随机
            {
                int action_id = rand() % actions.size();
                action = actions[action_id];
            }
            if (state.turn != TicTacToe::PLAYER_X)
                action = actions[0];
            test->step(action);
            go();
        }
        void back()         //传递
        {
            TicTacToe::State tempstate = test->get_state();
            int tempboard = tempstate.board;
            if (tempboard == 0) return;

            if (test->get_state().test_win())
                value[tempboard] = 2 - test->winner();
            test->step_back();
            TicTacToe::State laststate = test->get_state();
            int lastboard = laststate.board;

            if (tempstate.turn == TicTacToe::PLAYER_X)
                value[lastboard] = value[tempboard];
            else
            {
                //cout << value[lastboard] << " -> ";
                value[lastboard] = value[lastboard] + rate * (value[tempboard] - value[lastboard]);
                //cout << value[lastboard] << endl;
            }
            back();
        }



        TicTacToe::Action operator()(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            if (state.turn == TicTacToe::PLAYER_X)
            {
                // TODO
                TicTacToe::Action action = find_max(env);
                return action;
            } else {
                return actions[0];
            }
        }
        TicTacToePolicyDefault()
        {
            //改动：初始化并训练
            TicTacToe t(false);
            test = &t;
            init();
            srand(time(nullptr));
            for (int i = 0; i < 10000; i++)
            {
                go();
                back();
            }

        }
};


// randomly select action
int main(){
    bool done = false;
    // set verbose true
    TicTacToe env(true);
    // TicTacToePolicyDefault policy;
    TicTacToePolicyDefault policy;
    policy.env = &env;
    while (not done){
        TicTacToe::State state = env.get_state();
        TicTacToe::Action action = policy(state);
        env.step(action);
        done = env.done();
        // env.step_back();
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    int winner = env.winner();
    return 0;
};


/*  可用函数
action_space
test_win
step
step_back


*/