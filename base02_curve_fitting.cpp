// 第 2 课配套代码：用 Ceres 做带外点的曲线拟合
//
// 目的：把「残差块 + 自动微分 + 鲁棒核 + 白化」这四件事讲清楚。
// 这些正是 SLAM 后端（位姿图 / BA）里每个残差块都要做的事。
//
// 模型：y = exp(a * x^2 + b * x + c)
// 数据：100 个点，高斯噪声 + 每 10 个点里放一个 40% 的外点。

#include <ceres/ceres.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// 一个残差块：给定观测 (x, y)，计算预测值与观测值的差。
// 模板 operator() 是自动微分的前提：Ceres 会把 T 换成 Jet 类型，
// 在计算残差的同时把对 abc 的偏导一起算出来。
struct CurveResidual {
    CurveResidual(double x, double y, double sigma) : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        // 白化：残差除以观测标准差。等价于在代价里乘信息矩阵 Sigma^{-1}。
        // SLAM 里重投影误差除以像素噪声 sigma、IMU 残差除以测量噪声，都是这一步。
        residual[0] = (T(y_) - ceres::exp(abc[0] * T(x_ * x_) + abc[1] * T(x_) + abc[2])) /
                      T(sigma_);
        return true;
    }

    const double x_;
    const double y_;
    const double sigma_;
};

// 求解一次：use_huber 决定是否用鲁棒核。
void Solve(bool use_huber, const std::vector<double>& xs, const std::vector<double>& ys) {
    double abc[3] = {0.8, 1.8, 0.2};  // 初值

    ceres::Problem problem;
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
            new CurveResidual(xs[i], ys[i], 0.02));

        // 关键：LossFunction 作用在「白化后的残差」上。
        // 阈值 1.0 表示：残差超过 1 sigma 后，代价从二次增长退化为线性增长，
        // 从而不让少数外点主导整个解。
        ceres::LossFunction* loss = use_huber ? new ceres::HuberLoss(1.0) : nullptr;

        problem.AddResidualBlock(cost, loss, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  // 3 个参数的小问题
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << (use_huber ? "[Huber]   " : "[No loss] ") << summary.BriefReport() << "\n";
    std::cout << "          abc = " << abc[0] << ", " << abc[1] << ", " << abc[2] << "\n";
}

int main() {
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

    std::cout << "truth:    abc = " << a_true << ", " << b_true << ", " << c_true << "\n";
    Solve(false, xs, ys);
    Solve(true, xs, ys);

    return 0;
}
