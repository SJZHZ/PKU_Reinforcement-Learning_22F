#include <iostream>
#include <vector>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;

class SoapBubble{
    public:
        typedef vector<vector<double>> Bubble;
        SoapBubble(int max_x, int max_y,
            const vector<int> &x_lower, const vector<int> &x_upper,
            const vector<int> &y_lower, const vector<int> &y_upper,
            const Bubble &bubble)
        {
            SoapBubble(max_x, max_y, x_lower, x_upper, y_lower, y_upper);
            set_bubble(bubble);
        }
        SoapBubble(int max_x, int max_y,
            const vector<int> &x_lower, const vector<int> &x_upper,
            const vector<int> &y_lower, const vector<int> &y_upper)
        {
            this->max_x = max_x;
            this->max_y = max_y;
            this->x_lower = x_lower;
            this->x_upper = x_upper;
            this->y_lower = y_lower;
            this->y_upper = y_upper;
        }
        void set_bubble(const Bubble &bubble)       //外部矩阵导入内部矩阵
        {
            this->bubble = bubble;
            for (int y = 0; y < max_y; ++ y)
                for (int x = 0; x < max_x; ++ x)
                    if (not at_border(x, y))
                        this->bubble[y][x] = 0;
        }
        bool inside_bubble(int x, int y) const {
            return x_lower[y] <= x and x <= x_upper[y]
                and y_lower[x] <= y and y <= y_upper[x];
        }
        bool at_border(int x, int y) const {
            return inside_bubble(x, y)
                and (x_lower[y] == x or x == x_upper[y]
                or y_lower[x] == y or y == y_upper[x]);
        }
        void print_bubble(void) const
        {
            for (int y = 0; y < max_y; ++ y)
            {
                for (int x = 0; x < max_x; ++ x)

                    if (inside_bubble(x, y))
                        cout << setw(8) << setprecision(2) << bubble[y][x];
                    else
                        cout << setw(8) << " ";
                cout << endl;
            }
            cout << endl;
        }

        void inner_heights_dp(int iter = 500000)                        //边界值不断扩散直至收敛到解
        {
            for (int i = 0; i < iter; ++ i)
                for (int y = 0; y < max_y; ++ y)
                    for (int x = 0; x < max_x; ++ x)
                        if (inside_bubble(x, y) and not at_border(x, y))
                        {
                            double average = 0;
                            int inside_neighbor = 0;
                            for (int d = 0; d < DIR_NUM; ++ d)
                            {
                                int new_x = x + DX[d];
                                int new_y = y + DY[d];
                                if (inside_bubble(new_x, new_y))
                                {
                                    ++ inside_neighbor;
                                    average += bubble[new_y][new_x];
                                }
                            }
                            bubble[y][x] = average / inside_neighbor;
                        }
        }
        pair<int, int> randomsearch(int x, int y, int last)             //随机化搜索
        {
            if (at_border(x, y))
                return pair<int, int>(x, y);
            int d = rand() % DIR_NUM;       //下一步动作
            x += DX[d];
            y += DY[d];
            return randomsearch(x, y, d);   //期望有限
        }
        void updatevalue(pair<int, int> term, int x, int y, int iter)   //按边缘更新价值，每次只更新一个点
        {
            //本来想做的是沿路径更新的，但是没想明白有环的时候first visit怎么设计
            int xx = term.first;
            int yy = term.second;
            bubble[y][x] = (bubble[yy][xx] + bubble[y][x] * iter) / (iter + 1);
        }
        void inner_heights_mc(int iter=500000)                          //不断探索更新直到收敛到解
        {
            srand(time(nullptr));
            for (int i = 0; i < iter; i++)
                for (int y = 0; y < max_y; ++ y)
                    for (int x = 0; x < max_x; ++ x)            //求整个矩阵则批量处理。也可以求单点
                    {
                        if (!inside_bubble(x, y) || at_border(x, y))
                            continue;
                        pair<int, int> term = randomsearch(x, y, 10);
                        updatevalue(term, x, y, i);
                    }
        }
    private:                                    //数据结构
        static const int DIR_NUM = 4;
        static const int DX[DIR_NUM];
        static const int DY[DIR_NUM];
        int max_x, max_y;
        vector<int> x_lower, x_upper, y_lower, y_upper;                 //上下左右边界用vector保存（铁丝是给定的凸曲线）
        Bubble bubble;                                                  //每个点的高度（二维矩阵）

};
const int SoapBubble::DX[] = {0,1,0,-1}, SoapBubble::DY[] = {1,0,-1,0}; //四个方向


SoapBubble default_bubble_generator(int max_x=10, int max_y=10)
{
    vector<int> x_lower, x_upper, y_lower, y_upper;
    for (int y = 0, lower, upper; y < max_y; ++ y)
    {
        lower = y >> 2;                                 //0，0，1，1，2，2，3，3，4，4
        upper = max_y - 1 - ((max_y - y) >> 2);         //5，5，6，6，7，7，8，8，9，9
        x_lower.push_back(lower);
        x_upper.push_back(upper);
    }
    for (int x = 0, lower, upper; x < max_x; ++ x)
    {
        lower = x >> 2;                                 //0，0，1，1，2，2，3，3，4，4
        upper = max_x - 1 - ((max_x - x) >> 2);         //5，5，6，6，7，7，8，8，9，9
        y_lower.push_back(lower);
        y_upper.push_back(upper);
    }
    SoapBubble soap_bubble(max_x, max_y, x_lower, x_upper, y_lower, y_upper);
    SoapBubble::Bubble bubble(max_y);
    for (int y = 0; y < max_y; ++ y)                    //第一维是y，第二维是x
        for (int x = 0; x < max_x; ++ x)
            if (soap_bubble.at_border(x, y))
                bubble[y].push_back(log((sin(x)+1) / (cos(y)+1)));      //指定函数，非随机化
            else
                bubble[y].push_back(0);
    soap_bubble.set_bubble(bubble);
    return soap_bubble;
}

int main(){
    SoapBubble soap_bubble = default_bubble_generator();
    soap_bubble.print_bubble();
    soap_bubble.inner_heights_dp();     //动态规划
    soap_bubble.print_bubble();
    soap_bubble = default_bubble_generator();
    soap_bubble.inner_heights_mc();     //蒙特卡洛
    soap_bubble.print_bubble();
    return 0;
}
/*  运行结果

   -0.69  -0.083  -0.046   -0.56
   -0.43       0       0       0    -1.8    -3.6   -0.76   0.073
    0.54       0       0       0       0       0       0       1
     4.6       0       0       0       0       0       0       0     5.3
             1.7       0       0       0       0       0       0     1.7
            0.36       0       0       0       0       0       0    0.44
          -0.062       0       0       0       0       0       0   0.015
           0.049   0.085       0       0       0       0       0       0   -0.22
                            0.29    -1.3      -3   -0.17       0       0     0.5
                                                             2.9     3.1     2.8

   -0.69  -0.083  -0.046   -0.56
   -0.43   0.012   -0.12   -0.71    -1.8    -3.6   -0.76   0.073
    0.54    0.68    0.25   -0.31    -0.9    -1.2  -0.088       1
     4.6     1.9    0.76    0.12   -0.24    -0.2    0.57       2     5.3
             1.7    0.76    0.25   0.017   0.086    0.53     1.2     1.7
            0.36    0.33    0.13  -0.033 -0.0076    0.26    0.53    0.44
          -0.062   0.079  -0.036   -0.27   -0.34  -0.021    0.22   0.015
           0.049   0.085  -0.083   -0.67    -1.1   -0.22    0.36    0.36   -0.22
                            0.29    -1.3      -3   -0.17     1.1     1.3     0.5
                                                             2.9     3.1     2.8

   -0.69  -0.083  -0.046   -0.56
   -0.43   0.014   -0.12    -0.7    -1.8    -3.6   -0.76   0.073
    0.54    0.68    0.26   -0.29   -0.88    -1.2   -0.11       1
     4.6     1.9    0.75    0.12   -0.25   -0.19    0.56       2     5.3
             1.7    0.74    0.25   0.015   0.086    0.53     1.2     1.7
            0.36    0.32    0.13  -0.035  -0.012    0.26    0.54    0.44
          -0.062    0.07  -0.041   -0.27   -0.34  -0.024    0.23   0.015
           0.049   0.085  -0.087   -0.67    -1.1   -0.23    0.36    0.35   -0.22
                            0.29    -1.3      -3   -0.17     1.1     1.3     0.5
                                                             2.9     3.1     2.8
*/