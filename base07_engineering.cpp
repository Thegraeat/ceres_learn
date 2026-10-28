// 第 7 课配套代码：工程实践
//
// 三个工具：
//   1. GradientChecker：用有限差分验证自动微分雅可比
//   2. Covariance：估计参数的不确定性
//   3. 线性求解器对比：小问题上看不出差异，大问题才有区别
//
// 编译后运行：./build/base07_engineering

#include <ceres/ceres.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

struct CurveResidual {
    CurveResidual(double x, double y, double sigma)
        : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] =
            (T(y_) - ceres::exp(abc[0] * T(x_ * x_) + abc[1] * T(x_) + abc[2])) /
            T(sigma_);
        return true;
    }

    const double x_, y_, sigma_;
};

void BuildProblem(ceres::Problem* problem,
                  double* abc,
                  const std::vector<double>& xs,
                  const std::vector<double>& ys) {
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
                new CurveResidual(xs[i], ys[i], 0.02));
        problem->AddResidualBlock(cost, nullptr, abc);
    }
}

double SolveWith(ceres::LinearSolverType type,
                 const std::vector<double>& xs,
                 const std::vector<double>& ys,
                 double abc[3]) {
    abc[0] = 0.8;
    abc[1] = 1.8;
    abc[2] = 0.2;

    ceres::Problem problem;
    BuildProblem(&problem, abc, xs, ys);

    ceres::Solver::Options options;
    options.linear_solver_type = type;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const auto t0 = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) throw std::runtime_error(summary.BriefReport());
    const auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const double truth[3] = {1.0, 2.0, 0.5};
    const int n = 1000;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);

    std::vector<double> xs, ys;
    for (int i = 0; i < n; ++i) {
        const double x = -1.0 + 2.0 * i / (n - 1);
        xs.push_back(x);
        ys.push_back(std::exp(truth[0] * x * x + truth[1] * x + truth[2]) +
                     noise(rng));
    }

    // 1) 求解 + 协方差
    double abc[3] = {0.8, 1.8, 0.2};
    ceres::Problem problem;
    BuildProblem(&problem, abc, xs, ys);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    if (!summary.IsSolutionUsable()) return 1;
    std::cout << "abc = (" << abc[0] << ", " << abc[1] << ", " << abc[2] << ")\n\n";

    ceres::Covariance::Options cov_options;
    cov_options.algorithm_type = ceres::DENSE_SVD;
    cov_options.min_reciprocal_condition_number = 1e-14;

    ceres::Covariance covariance(cov_options);
    std::vector<std::pair<const double*, const double*>> blocks;
    blocks.push_back(std::make_pair(abc, abc));

    if (covariance.Compute(blocks, &problem)) {
        double cov[9];
        if (covariance.GetCovarianceBlock(abc, abc, cov)) {
            std::cout << "parameter stddev from covariance:\n";
            for (int i = 0; i < 3; ++i) {
                std::cout << "  sigma_" << i << " = "
                          << std::sqrt(std::max(0.0, cov[i * 3 + i])) << "\n";
            }
            std::cout << "  corr(a,b) = "
                      << cov[1] / std::sqrt(cov[0] * cov[4]) << "\n\n";
        } else {
            return 1;
        }
    } else {
        std::cout << "covariance computation failed\n\n";
        return 1;
    }

    // 2) 梯度检查
    {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
                new CurveResidual(0.3, 1.0, 1.0));

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

    // 3) 线性求解器对比
    double abc_dense[3];
    double abc_sparse[3];

    const double ms_dense =
        SolveWith(ceres::DENSE_QR, xs, ys, abc_dense);
    const double ms_sparse =
        SolveWith(ceres::SPARSE_NORMAL_CHOLESKY, xs, ys, abc_sparse);

    std::cout << "1000 residuals, 3 parameters:\n";
    std::cout << "  DENSE_QR:                 " << ms_dense << " ms\n";
    std::cout << "  SPARSE_NORMAL_CHOLESKY:   " << ms_sparse << " ms\n";
    std::cout << "  |abc_dense - abc_sparse| = "
              << std::fabs(abc_dense[0] - abc_sparse[0]) << ", "
              << std::fabs(abc_dense[1] - abc_sparse[1]) << ", "
              << std::fabs(abc_dense[2] - abc_sparse[2]) << "\n";
    return 0;
}
