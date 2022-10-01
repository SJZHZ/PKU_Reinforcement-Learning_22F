#include <utility>
#include <cstdlib>
#include <iostream>
#include <iomanip>
using namespace std;

class GridWorld{
    public:
        static const int
            NORTH=0, SOUTH=1, EAST=2, WEST=3;
        static const char ACTION_NAME[][16];
        typedef pair<int, int> State;
        bool verbose;
        State state(){
            return make_pair(x, y);
        }
        void set_state(int x, int y){
            this->x = x;
            this->y = y;
            if (verbose){
                cout << "State reset: (" << x << "," << y << ")" << endl;
            }
        }
        void reset(){
            set_state(0, 0);
        }
        pair<State, double> step(int action){
            State old_state = state();
            double reward = state_transition(action);
            if (verbose){
                cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
                cout << "Action: " << ACTION_NAME[action] << endl;
                cout << "Reward: " << reward << endl;
                cout << "New State: (" << x << "," << y << ")" << endl << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action(){
            return rand() % 4;
        }
        GridWorld(int x=0, int y=0, bool verbose=false) //样例顺序写错了，我修改了一下
        {
            this->verbose = verbose;
            set_state(x, y);
        }

    private:
        int x, y;
        double state_transition(int action){
            if (state() == make_pair(1, 0)){
                x = 1;
                y = 4;
                return 10.0;
            }
            if (state() == make_pair(3, 0)){
                x = 3;
                y = 2;
                return 5.0;
            }
            if (action == NORTH and y == 0 or
                action == SOUTH and y == 4 or
                action == EAST and x == 4 or
                action == WEST and x == 0){
                return -1.0;
            }
            switch (action){
                case NORTH:
                    y --; break;
                case SOUTH:
                    y ++; break;
                case EAST:
                    x ++; break;
                case WEST:
                    x --; break;
            }
            return 0.0;
        }
};
const char GridWorld::ACTION_NAME[][16] = {"NORTH(0,-1)", "SOUTH(0,1)", "EAST:(1,0)", "WEST:(-1,0)"};


double V[2][10][10] = {0};          //估值矩阵（滚动数组）
double Gamma = 0.9;                 //衰变系数
bool stepable[5][5][4];             //行动空间矩阵

void printvalue(bool i)             //打印矩阵V[i]的值
{
    cout << i << endl;
    for (int j = 0; j < 5; j++)
    {
        for (int k = 0; k < 5; k++)
            cout << V[i][j][k] << ' ';
        cout << endl;
    }
}
void initstepable(bool b)           //按b统一初始化stepable
{
    for (int i= 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 4; k++)
                stepable[i][j][k] = b;
}
bool updatestepable()               //贪心更新stepable
{
    bool updated = false;
    for (int i= 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            double maxvalue = -100;
            bool maxaction[4] = {0};
            for (int k = 0; k < 4; k++)         //探索
            {
                GridWorld tempenv = GridWorld(i, j, false);
                pair<GridWorld::State, double> temppair = tempenv.step(k);
                GridWorld::State tempstate = temppair.first;
                double stepvalue = temppair.second + Gamma * V[1][tempstate.first][tempstate.second];
                if (stepvalue == maxvalue)
                    maxaction[k] = 1;
                if (stepvalue > maxvalue)
                {
                    *(int*)maxaction = 0;
                    maxaction[k] = 1;
                    maxvalue = stepvalue;
                }
            }
            for (int k = 0; k < 4; k++)         //更新
            {
                if (stepable[i][j][k] != maxaction[k])
                {
                    updated = true;
                    stepable[i][j][k] = maxaction[k];
                }
                //cout << maxaction[k];
            }
            //cout << ' ';
        }
        //cout << endl;
    }
    return updated;
}
void RD(int t, bool verbose)             //随机迭代计算价值
{
    for (int i = 0; i < 20; i++)                    //反复迭代
    {
        for (int j = 0; j < 5; j++)                 //x
            for (int k = 0; k < 5; k++)             //y
            {
                double stepreward = 0, stepvalue = 0;
                int cnt = 0;
                for (int l = 0; l < 4; l++)         //action
                {
                    if (stepable[j][k][l])
                        cnt++;
                    else
                        continue;
                    GridWorld tempenv = GridWorld(j, k, false);                     //临时环境
                    pair<GridWorld::State, double> temppair = tempenv.step(l);
                    stepreward += temppair.second;
                    GridWorld::State tempstate = temppair.first;
                    stepvalue += V[i % 2][tempstate.first][tempstate.second];
                }
                V[(i + 1) % 2][j][k] = (stepreward + Gamma * stepvalue) / cnt;      //状态转移
            }
    }
    if (verbose)
        printvalue(1);
}


#include <chrono>
#include <thread>
int main()
{
    cout << setprecision(2) << fixed;
    initstepable(1);
    RD(50, true);
    while(updatestepable())
    {
        RD(20, false);
    }
    printvalue(1);
    //system("pause");
    return 0;
}
/*
    GridWorld env = GridWorld(0, 0, true);
    while (true)
    {
        int action = env.sample_action();
        auto state_reward = env.step(action);

        //this_thread::sleep_for(chrono::milliseconds(1000));
    }
*/