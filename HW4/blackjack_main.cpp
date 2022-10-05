#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <utility>
#include <vector>
#include <cmath>
#include "blackjack.hpp"

class BlackjackPolicyBase                                           //基类提供接口
{
    public:
        virtual int operator() (const Blackjack::State& state)=0;
};

class BlackjackPolicyDefault : public BlackjackPolicyBase           //默认策略
{
    public:
        int operator() (const Blackjack::State& state)
        {
            if (state.turn == Blackjack::PLAYER)
                return state.player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
            else
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
        }
};

class BlackjackPolicyLearnableDefault : public BlackjackPolicyBase  //学习策略
{
    static constexpr const char *ACTION_NAME = "HS";
    public:
        int operator() (const Blackjack::State& state)
        {
            if (state.turn == Blackjack::DEALER)
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
            else
                return policy[state.dealer_shown][state.player_ace][state.player_sum];
        }
        bool update_policy()                                //依value更新policy，返回值显示修改
        {
            // TODO
            bool updated = false;
            for (int i = 1; i <= 10; i++)
                for (int j = 0; j <= 1; j++)
                    for (int k = 11; k <= 21; k++)
                    {
                        int temp = policy[i][j][k];
                        policy[i][j][k] = value[i][j][k][1] > value[i][j][k][0] ? Blackjack::STICK : Blackjack::HIT;
                        if (temp != policy[i][j][k])
                            updated = true;
                    }
            return updated;
            // simply take argmax
        }
        // n iterations for each start (initial state and first action), no need to modify.
        double MC(Blackjack& env, bool p, int action = 0)   //递归的MC搜索，返回时修改value
        {
            Blackjack::State temp = env.state();
            if (p)      //依据策略（否则依据指定action值）
                action = this->operator()(temp);
            double reward;
            Blackjack::StepResult result = env.step(action);
            if (result.done)            //局面结束，不递归
                reward = result.player_reward;
            else                        //局面未结束，依据策略MC继续递归
                reward = MC(env, 1);

            if (temp.turn == Blackjack::DEALER)     //对手轮无需回溯修改
                return reward;

            //无环的情况下visited自增和visited使用可以放在【递归】两边。有环情况下则必须放在一起
            int visited = ++state_action_count[temp.dealer_shown][temp.player_ace][temp.player_sum][action];
            double& tempvalue = value[temp.dealer_shown][temp.player_ace][temp.player_sum][action];
            tempvalue += (reward - tempvalue) / visited;
            return reward;
        }
        bool update_value(Blackjack& env, int n=10000)      //进行一轮MC估值和策略迭代
        {
            // TODO
            set_value_initial();    // REMEMBER to call set_value_initial() at the beginning
            for (int i = 0; i < n; i++)     //蒙特卡洛估值
            {
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown)
                    for (int player_ace = 0; player_ace <= 1; ++ player_ace)
                        for (int player_sum = 0; player_sum <= 20; ++ player_sum)       //20！！！
                        //之前写的是21，但随机初始化不应包含21天和（此时不能HIT）！
                            for (int action = 0; action < 2; action++)
                            {
                                env.reset(dealer_shown, player_ace, player_sum);
                                MC(env, 0, action);
                            }
            }
            bool ans = update_policy();     //策略提升
            if (!ans)
                cout << endl << "STABLE" << endl << endl;
            return ans;

            // simulate from every possible initial state:(dealer_shown, player_ace, player_sum) \
                (call Blackjack::reset(,,) to do this) and player's every possible first action
            // BE AWARE only use player's steps (rather than dealer's) to update value estimation.
        }
        void print_policy() const
        {
            cout << setw(10) << "Player Without Ace" << "\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum)
            {
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown)
                    cout << ACTION_NAME[policy[dealer_shown][0][player_sum]];
                cout << "\t\t\t\t" ;
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown)
                    cout << ACTION_NAME[policy[dealer_shown][1][player_sum]];
                cout << endl;
            }
            cout << endl;
        }
        void print_value() const
        {
            cout << setw(40) << "Player Without Ace" << setw(20) <<"\t\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][
                        policy[dealer_shown][0][player_sum]];
                }
                cout << "\t";
                for (int dealer_shown = 1; dealer_shown <= 10 ; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][
                        policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        void print_value2() const
        {
            for (int action = 0; action < 2; action++)
            {
                cout << setw(40) << "Player Without Ace \taction " << action << setw(20) <<"\t\t\t" << "Player With Usable Ace." << endl;
                for (int player_sum = 21; player_sum >= 11; -- player_sum)
                {
                    for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown)
                    {
                        cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][action];
                    }
                    cout << "\t";
                    for (int dealer_shown = 1; dealer_shown <= 10 ; ++ dealer_shown){
                        cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][action];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
        BlackjackPolicyLearnableDefault()                   //按Default策略初始化
        {
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 0; player_sum <= 21; ++ player_sum){
                        int& action = policy[dealer_shown][player_ace][player_sum];
                        action = player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
                    }
                }
            }
        }
    private:
        // 11: dealer_shown (A-10);
        // 2: player_usable_ace (true/false);
        // 22: player_sum (only need to consider 11-20, because HIT when sum<11 and STICK when sum=21 are always best action.)
        int policy[11][2][22];  //只有两个选择
        // 11:dealer_shown; 2:player_usable_ace; 22:player_sum; 2:action (0:HIT, 1:STICK)
        double value[11][2][22][2];
        int state_action_count[11][2][22][2];

        // record a episode sampled (only player's steps).
        struct EpisodeStep{
            // state: (dealer_shown, player_ace, player_sum)
            int dealer_shown;
            int player_ace;
            int player_sum;
            // the action taken at state
            int action;
            EpisodeStep(const Blackjack::State& state, int action)
            {
                dealer_shown = state.dealer_shown;
                player_ace = int(state.player_ace);
                player_sum = state.player_sum;
                this->action = action;
            }
            EpisodeStep(){}
        };
        vector<EpisodeStep> episode;

        void set_value_initial(){
            memset(value, 0, sizeof(value));
            memset(state_action_count, 0, sizeof(state_action_count));
        }
};

int main()
{
    Blackjack env(false);
    BlackjackPolicyDefault policy;      //替换成PP
    BlackjackPolicyLearnableDefault PP;
    srand(time(nullptr));
    for (int i = 0; i < 100; i++)
    {
        cout << "ITER : " << i << endl;
        if (!PP.update_value(env))
            break;
    }
    cout << "DETIALED ANSWER:" << endl << endl;
    PP.print_policy();
    PP.print_value();
    //PP.print_value2();


    // while (true) {
    //     done = false;
    //     env.reset();
    //     while (not done){
    //         Blackjack::State state = env.state();
    //         int action = policy(state);
    //         Blackjack::StepResult result = env.step(action);
    //         done = result.done;
    //     }
    //     cout << endl;
    //     this_thread::sleep_for(chrono::milliseconds(1000));
    // }
    return 0;
}

//策略迭代次数不确定，但基本都在二十次以下，运行时间不到一分钟。一次输出如下
/*

ITER : 0
ITER : 1
ITER : 2
ITER : 3
ITER : 4
ITER : 5
ITER : 6
ITER : 7
ITER : 8
ITER : 9
ITER : 10

STABLE

DETIALED ANSWER:

Player Without Ace              Player With Usable Ace.
SSSSSSSSSS                              SSSSSSSSSS
SSSSSSSSSS                              SSSSSSSSSS
SSSSSSSSSS                              SSSSSSSSSS
SSSSSSSSSS                              HSSSSSSSHH
SSSSSSSSSS                              HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HHHSSSHHHH                              HHHHHHHHHH
HHHHHHHHHH                              HHHHHHHHHH

                      Player Without Ace                                        Player With Usable Ace.
  0.63  0.88  0.88  0.89  0.89  0.90  0.93  0.93  0.94  0.89      0.64  0.88  0.89  0.89  0.89  0.90  0.93  0.93  0.94  0.89
  0.15  0.64  0.64  0.67  0.67  0.70  0.78  0.79  0.75  0.44      0.15  0.64  0.65  0.66  0.67  0.71  0.77  0.79  0.76  0.43
 -0.11  0.38  0.40  0.42  0.43  0.49  0.62  0.59  0.28 -0.01     -0.11  0.39  0.40  0.43  0.44  0.49  0.62  0.59  0.29 -0.01
 -0.37  0.12  0.15  0.17  0.20  0.29  0.40  0.10 -0.19 -0.24     -0.34  0.11  0.14  0.17  0.21  0.29  0.40  0.10 -0.10 -0.20
 -0.63 -0.15 -0.12 -0.07 -0.04  0.01 -0.11 -0.38 -0.42 -0.47     -0.39 -0.00  0.04  0.07  0.09  0.12  0.05 -0.07 -0.16 -0.25
 -0.65 -0.29 -0.26 -0.22 -0.17 -0.15 -0.41 -0.45 -0.51 -0.56     -0.38 -0.03  0.01  0.03  0.07  0.10 -0.01 -0.07 -0.15 -0.26
 -0.61 -0.30 -0.26 -0.20 -0.17 -0.16 -0.37 -0.42 -0.47 -0.54     -0.36 -0.01  0.02  0.07  0.09  0.11  0.05 -0.03 -0.11 -0.23
 -0.59 -0.29 -0.26 -0.21 -0.16 -0.16 -0.32 -0.36 -0.43 -0.50     -0.32  0.03  0.04  0.08  0.12  0.13  0.09  0.02 -0.08 -0.20
 -0.55 -0.30 -0.25 -0.22 -0.16 -0.15 -0.27 -0.33 -0.39 -0.46     -0.29  0.04  0.08  0.10  0.13  0.15  0.13  0.06 -0.05 -0.15
 -0.52 -0.25 -0.24 -0.21 -0.17 -0.16 -0.21 -0.27 -0.35 -0.42     -0.26  0.07  0.10  0.12  0.15  0.18  0.16  0.10  0.00 -0.13
 -0.11  0.23  0.26  0.29  0.30  0.34  0.30  0.23  0.16  0.08      0.03  0.36  0.38  0.40  0.43  0.45  0.45  0.41  0.31  0.19

*/