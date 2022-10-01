#include <ctime>
#include <random>
#include <utility>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace std;

class JackCarRental{
    static const int
        MAX_CAR_1,
        MAX_CAR_2,
        MOVE_LIMIT;
    static const double
        MOVE_COST,
        RENT_PRICE,
        MEAN_REQUEST_1,
        MEAN_REQUEST_2,
        MEAN_RETURN_1,
        MEAN_RETURN_2;
    static poisson_distribution<int>
        request_1, request_2, return_1, return_2;
    public:
        typedef pair<int, int> State;
        bool verbose;
        State state(){
            return make_pair(car_1, car_2);
        }
        void set_state(int car_1, int car_2){
            if (verbose){
                cout << "State set to (" << car_1 << ", " << car_2 << ")" << endl;
            }
            this->car_1 = car_1;
            this->car_2 = car_2;
        }
        void reset(){
            if (verbose){
                cout << "Environment reset." << endl;
            }
            day = 0;
            car_1 = 0;
            car_2 = 0;
        }
        pair<State, double> step(int action)        //第day天从1号移到2号
        {
            day ++;
            if (verbose){
                cout << "\nDay: " << day
                    << " State: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            double reward = state_transition(action);
            if (verbose){
                cout << "\tReward: " << reward << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action(){
            int action_low = max(-car_2, -MOVE_LIMIT);
            int action_high = min(car_1, MOVE_LIMIT);
            uniform_int_distribution<int> random_action(action_low, action_high);
            int action = random_action(e);
            if (verbose){
                cout << "\tAction " << action
                    << " sampled from uniform[" << action_low << ", " << action_high << "]" << endl;
            }
            return action;
        }
        JackCarRental(int car_1=0, int car_2=0, bool verbose=false){
            this->day = 0;
            this->verbose = verbose;
            set_state(car_1, car_2);
            e.seed(time(nullptr));
        }
    private:
        int car_1, car_2, day;
        default_random_engine e;

        double state_transition(int action){
            car_1 = min(car_1 - action, MAX_CAR_1);
            car_2 = min(car_2 + action, MAX_CAR_2);
            double total_move_cost = abs(action) * MOVE_COST;
            if (verbose){
                cout << "\tMove: (" << -action << ", " << action
                    << "), cost: " << total_move_cost << endl;
                cout << "\tAfter movement, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            int req_1 = this->request_1(e);
            int req_2 = request_2(e);
            if (verbose){
                cout << "\tRental request: (" << req_1 << ", " << req_2 << ")" << endl;
            }
            int rent_1 = min(car_1, req_1);
            int rent_2 = min(car_2, req_2);
            double total_income = (rent_1 + rent_2) * RENT_PRICE;
            car_1 -= rent_1;
            car_2 -= rent_2;
            if (verbose){
                cout << "\tRent: (" << rent_1 << ", " << rent_2
                    << "), income: " << total_income << endl;
                cout << "\tAfter rent, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            int ret_1 = return_1(e);
            int ret_2 = return_2(e);
            if (verbose){
                cout << "\tCars to return: (" << ret_1 << ", " << ret_2 << ")" << endl;
            }
            car_1 = min(car_1 + ret_1, MAX_CAR_1);
            car_2 = min(car_2 + ret_2, MAX_CAR_2);
            if (verbose){
                cout << "\tAfter return, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            return total_income - total_move_cost;
        }
};

const int
    JackCarRental::MAX_CAR_1 = 20,
    JackCarRental::MAX_CAR_2 = 20,
    JackCarRental::MOVE_LIMIT = 5;
const double
    JackCarRental::MOVE_COST = 2.0,
    JackCarRental::RENT_PRICE = 10.0,
    JackCarRental::MEAN_REQUEST_1 = 3.0,
    JackCarRental::MEAN_REQUEST_2 = 4.0,
    JackCarRental::MEAN_RETURN_1 = 3.0,
    JackCarRental::MEAN_RETURN_2 = 2.0;
poisson_distribution<int>
    JackCarRental::request_1(JackCarRental::MEAN_REQUEST_1),
    JackCarRental::request_2(JackCarRental::MEAN_REQUEST_2),
    JackCarRental::return_1(JackCarRental::MEAN_RETURN_1),
    JackCarRental::return_2(JackCarRental::MEAN_RETURN_2);

double PoissonProb[5][21];                      //Poisson分布
double JointProb[21][21][21][21] = {0};         //联合分布
double V[21][21] = {0};                         //价值矩阵
int S[21][21] = {0};                            //策略矩阵
double Gamma = 0.9, rent = 10;
void calcPoisson(int lam)                       //计算Poisson分布
{
    PoissonProb[lam][0] = exp(-lam);
    double sum = PoissonProb[lam][0];
    for (int i = 1; i < 21; i++)
    {
        PoissonProb[lam][i] = PoissonProb[lam][i - 1] * lam / i;
        sum += PoissonProb[lam][i];
    }
}
void calcJoint()                                //计算联合分布
{
    for (int i = 2; i <= 4; i++)
        calcPoisson(i);
    for (int rq1 = 0; rq1 < 21; rq1++)
        for (int rq2 = 0; rq2 < 21; rq2++)
            for (int rt1 = 0; rt1 < 21; rt1++)
                for (int rt2 = 0; rt2 < 21; rt2++)
                    JointProb[rq1][rq2][rt1][rt2] = PoissonProb[3][rq1] * PoissonProb[4][rq2] * PoissonProb[3][rt1] * PoissonProb[2][rt2];
    double sum = 0;
    for (int rq1 = 0; rq1 < 21; rq1++)
        for (int rq2 = 0; rq2 < 21; rq2++)
            for (int rt1 = 0; rt1 < 21; rt1++)
                for (int rt2 = 0; rt2 < 21; rt2++)
                    sum += JointProb[rq1][rq2][rt1][rt2];
}
void printS()                                   //输出策略
{
    for (int i = 0; i < 21; i++)
    {
        for (int j = 0; j < 21; j++)
            cout << S[i][j] << ' ';
        cout << endl;
    }
}
void printV()                                   //输出价值
{
    for (int i = 0; i < 21; i++)
    {
        for (int j = 0; j < 21; j++)
            cout << V[i][j] << ' ';
        cout << endl;
    }
}
void updatevalue(bool verbose)                  //依据概率状态转移更新价值矩阵
{
    double temp[21][21] = {0};
    for (int i = 0; i < 21; i++)
        for (int j = 0; j < 21; j++)
            temp[i][j] = V[i][j];
    for (int i = 0; i < 21; i++)                        //1
        for (int j = 0; j < 21; j++)                    //2
        {
            double reward = 0, value = 0;
            for (int rq1 = 0; rq1 < 21; rq1++)
                for (int rq2 = 0; rq2 < 21; rq2++)
                    for (int rt1 = 0; rt1 < 21; rt1++)
                        for (int rt2 = 0; rt2 < 21; rt2++)      //遍历搜索状态转移，没有归类剪枝
                        {
                            double Prob = JointProb[rq1][rq2][rt1][rt2];
                            int rest1 = max(i - rq1, 0), rest2 = max(j - rq2, 0);
                            reward += Prob * (i - rest1 + j - rest2) * rent;                //租金奖励
                            int new1 = min(rest1 + rt1, 20), new2 = min(rest2 + rt2, 20);
                            value += Prob * Gamma * temp[new1][new2];                       //状态转移价值
                        }
            V[i][j] = (reward + value);             //value是已经衰减过的
        }
    if (verbose)
        printV();
}
void updatevalueasstrategy(bool verbose)        //依据已有最优策略更新价值，但没想好如何使用！
{
    double temp[21][21] = {0};
    for (int i = 0; i < 21; i++)
        for (int j = 0; j < 21; j++)
            temp[i][j] = V[i][j];
    for (int i = 0; i < 21; i++)                        //1
        for (int j = 0; j < 21; j++)                    //2
        {
            int action = S[i][j];
            V[i][j] = temp[i - action][j + action] - 2 * action;
        }
    if (verbose)
        printV();
}

bool updatestrategy(bool verbose)               //依据价值矩阵更新最优策略，并校正（最优）价值
{
    bool updated = false;
    double temp[21][21] = {0};
    for (int i = 0; i < 21; i++)
        for (int j = 0; j < 21; j++)
            temp[i][j] = V[i][j];

    for (int i = 0; i < 21; i++)                        //1
        for (int j = 0; j < 21; j++)                    //2
        {
            int lb = max(-j, -5), ub = min(i, 5), bestk;
            double maxvalue = -1000;
            for (int k = lb; k <= ub; k++)              //遍历搜索动作k
            {
                if (i - k > 20 || j + k > 20)           //车的效用是正面的，不必多花钱消除车
                    continue;
                double value = temp[i - k][j + k] - 2 * abs(k);     //状态转移价值
                if (value > maxvalue)                   //最优化动作以取得最优价值
                {
                    maxvalue = value;
                    bestk = k;
                }
            }
            V[i][j] = maxvalue;                         //更新为最优价值
            if (S[i][j] != bestk)                       //不更新则认为收敛
            {
                S[i][j] = bestk;
                updated = true;
            }
        }
    if (verbose)
        printS();
    return updated;
}



#include <chrono>
#include <thread>
int main()
{
    cout << setprecision(1) << fixed;
    calcJoint();
    for (int i = 0; i < 20; i++)                //更类似值迭代，即在值估计不充分时就先优化策略
    {
        bool flag = 1;
        cout << "IT:" << i << endl;
        updatevalue(0);
        flag = updatestrategy(0);
        // for (int j = 0; j < 10; j++)
        //     updatevalueasstrategy(0);
        if (!flag) break;
    }
    // IT:17
    // 编译时间大概10s，运行时间大概45s
    printV();
    printS();
    //system("pause");
    return 0;
}
/*
0 0 0 0 0 0 0 0 -1 -1 -2 -2 -2 -3 -3 -3 -3 -3 -4 -4 -4
0 0 0 0 0 0 0 0 0 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3
0 0 0 0 0 0 0 0 0 0 0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -1 -1 -1 -1 -2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -1
1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 3 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 3 3 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
4 4 3 3 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 4 4 3 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 5 4 3 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 5 4 3 3 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 5 4 4 3 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 5 5 4 3 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
5 5 5 4 3 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
5 5 5 4 3 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0
5 5 5 4 3 3 2 2 1 1 1 0 0 0 0 0 0 0 0 0 0
5 5 5 4 4 3 3 2 2 2 1 1 1 1 1 0 0 0 0 0 0
5 5 5 5 4 4 3 3 3 2 2 2 2 2 1 1 1 0 0 0 0
*/
/*
    JackCarRental env(0, 0, true);
    while (true){
        int action = env.sample_action();
        env.step(action);
        this_thread::sleep_for(chrono::milliseconds(1000));
    }
*/