// 第 1 课课后作业参考答案
//
// 任务：直线拟合 y = m * x + b
//   1) 自动微分，联合估计 m, b
//   2) 固定 b（SetParameterBlockConstant），只估计 m
//   3) 数值微分，与自动微分比较精度和耗时
//
// 编译后运行：./build/hw01_basics

#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// 自动微分版本：m 和 b 是两个独立的 1 维参数块。
struct LineResidual {
    LineResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m, const T* const b, T* residual) const {
        residual[0] = T(y_) - (m[0] * T(x_) + b[0]);
        return true;
    }

    const double x_, y_;
};

// 数值微分版本：参数必须是 double，operator() 不是模板。
struct LineResidualNum {
    LineResidualNum(double x, double y) : x_(x), y_(y) {}

    bool operator()(const double* const m, const double* const b, double* residual) const {
        residual[0] = y_ - (m[0] * x_ + b[0]);
        return true;
    }

    const double x_, y_;
};

struct Result {
    double m;
    double b;
    double cost;
    double ms;
};

Result SolveAuto(const std::vector<double>& xs,
                 const std::vector<double>& ys,
                 bool fix_b,
                 double b_value) {
    double m = 0.0;
    double b = fix_b ? b_value : 0.0;

    ceres::Problem problem;
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<LineResidual, 1, 1, 1>(
                new LineResidual(xs[i], ys[i]));
        problem.AddResidualBlock(cost, nullptr, &m, &b);
    }

    // 固定 b：它是一个独立的参数块，可以直接固定。
    if (fix_b) {
        problem.SetParameterBlockConstant(&b);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const auto t0 = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, &problem, &summary);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {m, b, summary.final_cost, ms};
}

Result SolveNumeric(const std::vector<double>& xs,
                    const std::vector<double>& ys) {
    double m = 0.0;
    double b = 0.0;

    ceres::Problem problem;
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::NumericDiffCostFunction<LineResidualNum,
                                               ceres::CENTRAL, 1, 1, 1>(
                new LineResidualNum(xs[i], ys[i]));
        problem.AddResidualBlock(cost, nullptr, &m, &b);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const auto t0 = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, &problem, &summary);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {m, b, summary.final_cost, ms};
}

int main() {
    const double m_true = 2.0;
    const double b_true = -0.5;
    const int n = 200;

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.05);

    std::vector<double> xs, ys;
    for (int i = 0; i < n; ++i) {
        const double x = -1.0 + 2.0 * i / (n - 1);
        xs.push_back(x);
        ys.push_back(m_true * x + b_true + noise(rng));
    }

    // 闭式最小二乘解，用来对照 Ceres 的结果。
    double x_sum = 0.0, y_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x_sum += xs[i];
        y_sum += ys[i];
    }
    const double x_bar = x_sum / n;
    const double y_bar = y_sum / n;

    double sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        sxx += (xs[i] - x_bar) * (xs[i] - x_bar);
        sxy += (xs[i] - x_bar) * (ys[i] - y_bar);
    }
    const double m_ls = sxy / sxx;
    const double b_ls = y_bar - m_ls * x_bar;

    const Result a = SolveAuto(xs, ys, false, 0.0);
    const Result b = SolveAuto(xs, ys, true, b_true);
    const Result c = SolveNumeric(xs, ys);

    std::cout << "truth:           m=" << m_true << " b=" << b_true << "\n";
    std::cout << "closed form:     m=" << m_ls << " b=" << b_ls << "\n";
    std::cout << "[A] auto free:   m=" << a.m << " b=" << a.b
              << " cost=" << a.cost << " time=" << a.ms << " ms\n";
    std::cout << "[B] auto fix b:  m=" << b.m
              << " cost=" << b.cost << " time=" << b.ms << " ms\n";
    std::cout << "[C] numeric:     m=" << c.m << " b=" << c.b
              << " cost=" << c.cost << " time=" << c.ms << " ms\n";
    std::cout << "|A-C|: " << std::abs(a.m - c.m) << " (m), "
              << std::abs(a.b - c.b) << " (b)\n";

    return 0;
}
