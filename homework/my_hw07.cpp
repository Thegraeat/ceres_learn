// 第 7 课 课后作业模板（my_hw07.cpp）
//
// 任务：把工程工具用一遍
//   1) Ceres Covariance 与理论协方差对比
//   2) GradientChecker 检查自动微分雅可比
//   3) DENSE_QR vs SPARSE_NORMAL_CHOLESKY 性能对比
//   4) 设计题（不用写代码，见 assignments/07_homework.md）
//
// 这个文件已经准备好：数据生成、协方差标准差对比的打印、
// GradientChecker 的 Probe 流程、计时和结果输出。需要你手写的地方：
//   TODO 1  CurveResidual::operator() —— 白化残差
//   TODO 2  BuildProblem / SolveWith  —— Ceres 建模与求解
//   TODO 3  TheoreticalCovariance     —— 雅可比 J 与 H = sum J^T J
//   TODO 4  Ceres Covariance          —— Compute / GetCovarianceBlock
//   TODO 5  GradientChecker 的 cost   —— 创建自动微分 cost
//
// 编译运行：
//   cmake --build build --target my_hw07 -j && ./build/my_hw07
//
// 验收标准见 assignments/07_homework.md。

#include <ceres/ceres.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

// ----------------------------- 残差块 -----------------------------
struct CurveResidual {
    CurveResidual(double x, double y, double sigma)
        : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        // TODO 1: 白化残差
        //   residual[0] = (y - exp(a*x^2 + b*x + c)) / sigma
        //   注意用 T 版本的 exp。
        (void)abc;
        residual[0] = T(0.0);
        return true;
    }

    const double x_, y_, sigma_;
};

// ---------------------------------------------------------------------------
// 建模与求解
// ---------------------------------------------------------------------------
bool BuildProblem(ceres::Problem* problem,
                  double* abc,
                  const std::vector<double>& xs,
                  const std::vector<double>& ys) {
    // TODO 2a: 对每个观测创建
    //   AutoDiffCostFunction<CurveResidual, 1, 3>(
    //       new CurveResidual(xs[i], ys[i], 0.02))
    // 并 AddResidualBlock(cost, nullptr, abc)
    (void)problem;
    (void)abc;
    (void)xs;
    (void)ys;
    return false;  // TODO 2a 完成后替换
}

double SolveWith(ceres::LinearSolverType type,
                 const std::vector<double>& xs,
                 const std::vector<double>& ys,
                 double abc[3]) {
    abc[0] = 0.8;
    abc[1] = 1.8;
    abc[2] = 0.2;

    // TODO 2b: 用给定的 type 构建并求解问题，返回耗时（毫秒）。
    //   - ceres::Problem problem; BuildProblem(&problem, abc, xs, ys)
    //   - Options: linear_solver_type = type, max_num_iterations = 100
    //   - 用 std::chrono::high_resolution_clock 包住 ceres::Solve
    //   - 检查 summary.IsSolutionUsable()
    (void)type;
    return -1.0;  // TODO 2b 完成后替换
}

// ---------------------------------------------------------------------------
// 理论协方差：Sigma = sigma^2 * (J^T J)^{-1}
// ---------------------------------------------------------------------------
Eigen::Matrix3d TheoreticalCovariance(const std::vector<double>& xs,
                                      const double abc[3],
                                      double sigma) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    // TODO 3: 对每个 x：
    //   e = exp(a*x^2 + b*x + c)
    //   J = [-x^2*e, -x*e, -e]
    //   H += J * J.transpose()
    (void)xs;
    (void)abc;

    if (H.isZero(0.0)) return Eigen::Matrix3d::Zero();  // TODO 3 完成后删除
    return sigma * sigma * H.ldlt().solve(Eigen::Matrix3d::Identity());
}

int main() {
    const double truth[3] = {1.0, 2.0, 0.5};
    const int n = 10000;  // 任务 3 可以再改到 100000 观察差距
    const double sigma = 0.02;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, sigma);

    std::vector<double> xs, ys;
    for (int i = 0; i < n; ++i) {
        const double x = -1.0 + 2.0 * i / (n - 1);
        xs.push_back(x);
        ys.push_back(std::exp(truth[0] * x * x + truth[1] * x + truth[2]) +
                     noise(rng));
    }

    // 1) 求解
    double abc[3] = {0.8, 1.8, 0.2};
    ceres::Problem problem;
    if (!BuildProblem(&problem, abc, xs, ys)) {
        std::cout << "TODO: 先完成 TODO 1/2，问题才能构建和求解。\n";
        return 0;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    // TODO 2c: ceres::Solve(options, &problem, &summary);
    //          并检查 summary.IsSolutionUsable()。
    bool solved = false;
    if (!solved) {
        std::cout << "TODO: 还没有调用 ceres::Solve。\n";
        return 0;
    }

    std::cout << summary.BriefReport() << "\n";
    std::cout << "abc = (" << abc[0] << ", " << abc[1] << ", " << abc[2] << ")\n\n";

    // 2) Ceres 协方差
    ceres::Covariance::Options cov_options;
    ceres::Covariance covariance(cov_options);
    std::vector<std::pair<const double*, const double*>> blocks;
    blocks.push_back(std::make_pair(abc, abc));

    double cov_ceres[9] = {0.0};
    // TODO 4: 设置 cov_options（例如 DENSE_SVD、min_reciprocal_condition_number），
    //   然后调用
    //     covariance.Compute(blocks, &problem)
    //     covariance.GetCovarianceBlock(abc, abc, cov_ceres)
    bool cov_ok = false;
    if (!cov_ok) {
        std::cout << "TODO: 还没有计算 Ceres 协方差。\n";
        return 0;
    }

    // 3) 理论协方差
    const Eigen::Matrix3d cov_theory = TheoreticalCovariance(xs, abc, sigma);

    std::cout << "stddev comparison:\n";
    for (int i = 0; i < 3; ++i) {
        const double std_ceres = std::sqrt(std::max(0.0, cov_ceres[i * 3 + i]));
        const double std_theory = std::sqrt(cov_theory(i, i));
        const double rel = std::fabs(std_ceres - std_theory) /
                           std::max(1e-12, std_theory);
        std::cout << "  param " << i << ": ceres=" << std_ceres
                  << " theory=" << std_theory << " rel=" << rel << "\n";
    }
    std::cout << "covariance ok: " << (cov_ok ? "yes" : "no") << "\n\n";

    // 4) 梯度检查
    {
        // TODO 5: 创建自动微分 cost：
        //   new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
        //       new CurveResidual(0.3, 1.0, 1.0))
        ceres::CostFunction* cost = nullptr;

        if (cost == nullptr) {
            std::cout << "TODO: 还没有创建 GradientChecker 用的 cost。\n";
            return 0;
        }

        std::vector<const ceres::Manifold*> manifolds = {nullptr};
        ceres::NumericDiffOptions ndo;

        ceres::GradientChecker checker(cost, &manifolds, ndo);
        double* params[1] = {abc};
        ceres::GradientChecker::ProbeResults results;

        const bool ok = checker.Probe(params, 1e-6, &results);
        std::cout << "gradient check: " << (ok ? "PASS" : "FAIL")
                  << ", max relative error = " << results.maximum_relative_error
                  << "\n\n";

        delete cost;
        if (!ok) return 1;
    }

    // 5) 性能对比
    double abc_dense[3] = {0.0, 0.0, 0.0};
    double abc_sparse[3] = {0.0, 0.0, 0.0};

    const double ms_dense = SolveWith(ceres::DENSE_QR, xs, ys, abc_dense);
    const double ms_sparse =
        SolveWith(ceres::SPARSE_NORMAL_CHOLESKY, xs, ys, abc_sparse);

    if (ms_dense < 0.0 || ms_sparse < 0.0) {
        std::cout << "TODO: 还没有完成 SolveWith 的求解与计时。\n";
        return 0;
    }

    std::cout << n << " residuals, 3 parameters:\n";
    std::cout << "  DENSE_QR:               " << ms_dense << " ms\n";
    std::cout << "  SPARSE_NORMAL_CHOLESKY: " << ms_sparse << " ms\n";
    std::cout << "  solution diff: "
              << std::fabs(abc_dense[0] - abc_sparse[0]) << ", "
              << std::fabs(abc_dense[1] - abc_sparse[1]) << ", "
              << std::fabs(abc_dense[2] - abc_sparse[2]) << "\n";

    return 0;
}
