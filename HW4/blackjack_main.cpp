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
        void update_policy()
        {
            // TODO
            for (int i = 1; i <= 10; i++)
                for (int j = 0; j <= 1; j++)
                    for (int k = 0; k <= 21; k++)
                        policy[i][j][k] = value[i][j][k][1] > value[i][j][k][0] ? Blackjack::STICK : Blackjack::HIT;
            // simply take argmax
        }
        // n iterations for each start (initial state and first action), no need to modify.
        //int visitedtimes[11][2][22] = {0};
        int epsilon = 10;
        double MC(Blackjack& env)
        {
            Blackjack::State temp = env.state();
            if (temp.turn == Blackjack::DEALER)         //对手
            {
                int action = temp.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
                Blackjack::StepResult result = env.step(action);
                if (result.done)
                    return result.player_reward;
                return MC(env);
            }

            int action = policy[temp.dealer_shown][temp.player_ace][temp.player_sum];
            if (rand() % epsilon < 1)
                action = 1 - action;

            int visited = ++state_action_count[temp.dealer_shown][temp.player_ace][temp.player_sum][action];
            double reward;
            double& tempvalue = value[temp.dealer_shown][temp.player_ace][temp.player_sum][action];
            Blackjack::StepResult result = env.step(action);
            if (result.done)
                reward = result.player_reward;
            else
                reward = MC(env);
            tempvalue = reward / visited + (tempvalue * (visited - 1)) / visited;
            return reward;

        }
        void update_value(Blackjack& env, int n=1000)
        {
            // TODO
            set_value_initial();    // REMEMBER to call set_value_initial() at the beginning
            for (int i = 0; i < n; i++)
            {
                epsilon = 10 + n / 50;      //使用动态增大的epsilon以使得估计更贴近所求策略
                for (int j = 0; j < 1000; j++)
                    for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown)
                        for (int player_ace = 0; player_ace <= 1; ++ player_ace)
                            for (int player_sum = 0; player_sum <= 21; ++ player_sum)
                            {
                                env.reset(dealer_shown, player_ace, player_sum);
                                MC(env);
                            }
                update_policy();
            }
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

        BlackjackPolicyLearnableDefault()           //按Default策略初始化
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
    BlackjackPolicyDefault policy;
    BlackjackPolicyLearnableDefault PP;
    bool done;
    srand(time(nullptr));
    PP.update_value(env);
    PP.print_policy();
    PP.print_value();

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

/*  以下是一次运行的输出，运行时间约一分半
在Policy:Player Without Ace中
D_S = 3, P_S = 12时STICK
D_S = 10, P_S = 16时STICK
两处不符合答案
可能是epsilon的选择上还有问题

Player Without Ace              Player With Usable Ace.
SSSSSSSSSS                              SSSSSSSSSS
SSSSSSSSSS                              SSSSSSSSSS
SSSSSSSSSS                              SSSSSSSSSS
SSSSSSSSSS                              HSSSSSSSHH
SSSSSSSSSS                              HHHHHHHHHH
HSSSSSHHHS                            HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HSSSSSHHHH                              HHHHHHHHHH
HHSSSSHHHH                              HHHHHHHHHH
HHHHHHHHHH                              HHHHHHHHHH

                      Player Without Ace                                        Player With Usable Ace.
  0.65  0.94  0.95  0.95  0.96  0.96  0.95  0.95  0.96  0.90      0.65  0.92  0.92  0.92  0.93  0.93  0.95  0.95  0.96  0.90
  0.15  0.64  0.65  0.66  0.67  0.70  0.77  0.79  0.76  0.44      0.15  0.64  0.65  0.66  0.67  0.70  0.77  0.79  0.76  0.43
 -0.12  0.39  0.40  0.42  0.44  0.50  0.62  0.59  0.29 -0.02     -0.12  0.39  0.40  0.42  0.44  0.50  0.62  0.59  0.29 -0.02
 -0.38  0.12  0.15  0.18  0.20  0.28  0.40  0.11 -0.18 -0.24     -0.35  0.12  0.15  0.18  0.20  0.28  0.40  0.11 -0.12 -0.22
 -0.64 -0.15 -0.12 -0.08 -0.05  0.01 -0.11 -0.38 -0.42 -0.46     -0.41 -0.01  0.02  0.05  0.08  0.12  0.03 -0.09 -0.17 -0.26
 -0.65 -0.29 -0.25 -0.21 -0.17 -0.15 -0.43 -0.47 -0.52 -0.58     -0.39 -0.03 -0.00  0.03  0.06  0.09 -0.03 -0.09 -0.17 -0.27
 -0.62 -0.29 -0.25 -0.21 -0.17 -0.15 -0.39 -0.43 -0.49 -0.55     -0.37 -0.01  0.02  0.05  0.08  0.11  0.01 -0.05 -0.14 -0.24
 -0.60 -0.29 -0.25 -0.21 -0.17 -0.15 -0.34 -0.39 -0.45 -0.51     -0.34  0.01  0.04  0.07  0.10  0.13  0.05 -0.01 -0.10 -0.21
 -0.57 -0.29 -0.25 -0.21 -0.17 -0.15 -0.29 -0.34 -0.40 -0.48     -0.32  0.03  0.06  0.09  0.12  0.15  0.09  0.03 -0.06 -0.18
 -0.53 -0.27 -0.25 -0.21 -0.17 -0.15 -0.24 -0.29 -0.36 -0.44     -0.29  0.06  0.09  0.11  0.14  0.17  0.13  0.07 -0.03 -0.15
 -0.13  0.21  0.23  0.25  0.28  0.30  0.25  0.19  0.12  0.03     -0.01  0.35  0.37  0.39  0.41  0.44  0.43  0.37  0.29  0.17
*/