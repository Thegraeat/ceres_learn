// 第 6 课课后作业参考答案
//
// 任务：一般线性化点下的边缘化
//   x_lin = (1.2, 1.8, 3.1)，旧残差在 x_lin 处不为零
//   先验中心必须用 x_prior = x_lin - Hmm^{-1} gm
//   验证全量问题与边缘化问题在 x0,x1 上一致
//
// 编译后运行：./build/hw06_marginalization

#include <ceres/ceres.h>

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>

struct Linear2 {
    Linear2(double a, double b, double c) : a_(a), b_(b), c_(c) {}

    template <typename T>
    bool operator()(const T* const x0, const T* const x1, T* residual) const {
        residual[0] = T(a_) * x0[0] + T(b_) * x1[0] - T(c_);
        return true;
    }

    double a_, b_, c_;
};

struct Linear1 {
    Linear1(double a, double c) : a_(a), c_(c) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(a_) * x[0] - T(c_);
        return true;
    }

    double a_, c_;
};

struct PriorCost {
    PriorCost(const Eigen::Matrix2d& L, const Eigen::Vector2d& x_prior)
        : L_(L), x_prior_(x_prior) {}

    template <typename T>
    bool operator()(const T* const x0, const T* const x1, T* residual) const {
        const T d0 = x0[0] - T(x_prior_(0));
        const T d1 = x1[0] - T(x_prior_(1));

        residual[0] = T(L_(0, 0)) * d0 + T(L_(1, 0)) * d1;
        residual[1] = T(L_(0, 1)) * d0 + T(L_(1, 1)) * d1;
        return true;
    }

    Eigen::Matrix2d L_;
    Eigen::Vector2d x_prior_;
};

bool SolveFull(double obs, double* x0, double* x1, double* x2) {
    *x0 = 0.0;
    *x1 = 0.0;
    *x2 = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear2, 1, 1, 1>(new Linear2(1, 1, 3)),
        nullptr, x0, x1);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear2, 1, 1, 1>(new Linear2(1, 1, 5)),
        nullptr, x1, x2);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear1, 1, 1>(new Linear1(1, 2)),
        nullptr, x2);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear1, 1, 1>(new Linear1(1, obs)),
        nullptr, x0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 20;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 1e-14;
    options.parameter_tolerance = 1e-14;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    return summary.IsSolutionUsable();
}

bool SolveMarginalized(const Eigen::Matrix2d& L,
                       const Eigen::Vector2d& x_prior,
                       double obs,
                       double* x0,
                       double* x1) {
    *x0 = 0.0;
    *x1 = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PriorCost, 2, 1, 1>(
            new PriorCost(L, x_prior)),
        nullptr, x0, x1);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear1, 1, 1>(new Linear1(1, obs)),
        nullptr, x0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 20;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 1e-14;
    options.parameter_tolerance = 1e-14;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    return summary.IsSolutionUsable();
}

int main() {
    std::cout << std::setprecision(12);
    // 1) 非最优线性化点
    const Eigen::Vector3d x_lin(1.2, 1.8, 3.1);

    const Eigen::Vector3d J1(1.0, 1.0, 0.0);
    const Eigen::Vector3d J2(0.0, 1.0, 1.0);
    const Eigen::Vector3d J3(0.0, 0.0, 1.0);

    const double r1 = x_lin(0) + x_lin(1) - 3.0;  // 0.0
    const double r2 = x_lin(1) + x_lin(2) - 5.0;  // -0.1
    const double r3 = x_lin(2) - 2.0;             // 1.1

    const Eigen::Matrix3d H = J1 * J1.transpose() + J2 * J2.transpose() +
                              J3 * J3.transpose();
    const Eigen::Vector3d g = J1 * r1 + J2 * r2 + J3 * r3;

    std::cout << "r = (" << r1 << ", " << r2 << ", " << r3 << ")\n";
    std::cout << "H =\n" << H << "\n\ng = " << g.transpose() << "\n\n";

    // 2) Schur 补
    const Eigen::Matrix2d Hmm = H.topLeftCorner<2, 2>() -
                                H.topRightCorner<2, 1>() * (1.0 / H(2, 2)) *
                                    H.bottomLeftCorner<1, 2>();
    const Eigen::Vector2d gm =
        g.head<2>() - H.topRightCorner<2, 1>() * (1.0 / H(2, 2)) * g(2);

    std::cout << "Hmm =\n" << Hmm << "\n\ngm = " << gm.transpose() << "\n";

    // 3) 先验：中心必须是旧信息下的最优解，而不是 x_lin
    const Eigen::Vector2d x_prior = x_lin.head<2>() - Hmm.ldlt().solve(gm);
    std::cout << "x_prior = " << x_prior.transpose() << "\n";

    const Eigen::LLT<Eigen::Matrix2d> llt(Hmm);
    const Eigen::Matrix2d L = llt.matrixL();
    std::cout << "L =\n" << L << "\n\n";

    // 4) 两个新观测
    for (double obs : {1.1, 1.5}) {
        double x0 = 0.0, x1 = 0.0, x2 = 0.0;
        double y0 = 0.0, y1 = 0.0;

        if (!SolveFull(obs, &x0, &x1, &x2) ||
            !SolveMarginalized(L, x_prior, obs, &y0, &y1)) return 1;

        std::cout << "obs x0 = " << obs << "\n";
        std::cout << "  full:   x0=" << x0 << " x1=" << x1 << " x2=" << x2 << "\n";
        std::cout << "  marg:   x0=" << y0 << " x1=" << y1 << "\n";
        std::cout << "  diff:   " << std::fabs(x0 - y0) << ", "
                  << std::fabs(x1 - y1) << "\n\n";
        if (std::fabs(x0 - y0) >= 1e-10 || std::fabs(x1 - y1) >= 1e-10 ||
            std::fabs(x0 - 0.75 * obs) >= 1e-10 ||
            std::fabs(x1 - (3.0 - 0.5 * obs)) >= 1e-10 ||
            std::fabs(x2 - (2.0 + 0.25 * obs)) >= 1e-10) return 1;
    }

    return 0;
}
