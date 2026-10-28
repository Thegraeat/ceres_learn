// 第 7 课课后作业参考答案
//
// 任务：
//   1) Ceres Covariance 与理论协方差对比
//   2) GradientChecker 检查自动微分雅可比
//   3) DENSE_QR vs SPARSE_NORMAL_CHOLESKY 性能对比
//
// 编译后运行：./build/hw07_engineering

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

// 理论协方差：Sigma = sigma^2 * (J^T J)^{-1}
Eigen::Matrix3d TheoreticalCovariance(const std::vector<double>& xs,
                                      const double abc[3],
                                      double sigma) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    for (double x : xs) {
        const double e = std::exp(abc[0] * x * x + abc[1] * x + abc[2]);
        const Eigen::Vector3d J(-x * x * e, -x * e, -e);
        H += J * J.transpose();
    }

    return sigma * sigma * H.ldlt().solve(Eigen::Matrix3d::Identity());
}

int main() {
    const double truth[3] = {1.0, 2.0, 0.5};
    const int n = 10000;
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
    BuildProblem(&problem, abc, xs, ys);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    if (!summary.IsSolutionUsable()) return 1;
    std::cout << "abc = (" << abc[0] << ", " << abc[1] << ", " << abc[2] << ")\n\n";

    // 2) Ceres 协方差
    ceres::Covariance::Options cov_options;
    cov_options.algorithm_type = ceres::DENSE_SVD;
    cov_options.min_reciprocal_condition_number = 1e-14;

    ceres::Covariance covariance(cov_options);
    std::vector<std::pair<const double*, const double*>> blocks;
    blocks.push_back(std::make_pair(abc, abc));

    double cov_ceres[9] = {0.0};
    bool cov_ok = covariance.Compute(blocks, &problem) &&
                  covariance.GetCovarianceBlock(abc, abc, cov_ceres);
    if (!cov_ok) {
        std::cerr << "covariance computation failed\n";
        return 1;
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

    // 5) 性能对比
    double abc_dense[3];
    double abc_sparse[3];

    const double ms_dense = SolveWith(ceres::DENSE_QR, xs, ys, abc_dense);
    const double ms_sparse =
        SolveWith(ceres::SPARSE_NORMAL_CHOLESKY, xs, ys, abc_sparse);

    std::cout << n << " residuals, 3 parameters:\n";
    std::cout << "  DENSE_QR:               " << ms_dense << " ms\n";
    std::cout << "  SPARSE_NORMAL_CHOLESKY: " << ms_sparse << " ms\n";
    std::cout << "  solution diff: "
              << std::fabs(abc_dense[0] - abc_sparse[0]) << ", "
              << std::fabs(abc_dense[1] - abc_sparse[1]) << ", "
              << std::fabs(abc_dense[2] - abc_sparse[2]) << "\n";

    return 0;
}
