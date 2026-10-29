#include <ceres/ceres.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

struct CurveFunction{
    
};


int main(int argc, char const *argv[]){

    // 真值：y = exp(1.0 * x^2 + 2.0 * x + 0.5)
    const double a_true = 1.0, b_true = 2.0, c_true = 0.5;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);

    std::vector<double> xs, ys;
    for (int i = 0; i < 100; ++i) {
        const double x = -1.0 + 2.0 * i / 99.0;
        double y = std::exp(a_true * x * x + b_true * x + c_true) + noise(rng);

        // 每 10 个点制造一个外点（错误关联 / 误匹配的类比）
        if (i % 10 == 0) {
            y += 0.4 * y;
        }
        xs.push_back(x);
        ys.push_back(y);
    }

    return 0;
}
