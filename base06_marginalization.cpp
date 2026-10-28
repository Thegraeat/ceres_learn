// 第 6 课配套代码：滑动窗口与边缘化
//
// 用一个线性最小二乘例子，把「旧信息压缩成先验」这件事讲清楚。
//
// 变量：x0, x1, x2
// 旧残差：
//   r1 = x0 + x1 - 3
//   r2 = x1 + x2 - 5
//   r3 = x2 - 2
// 新观测：
//   r4 = x0 - 1.1
//
// 全量问题：x0, x1, x2 一起解。
// 边缘化：用 Schur 补消掉 x2，得到 x0,x1 的先验，再加新观测。
// 对线性问题，两者在 x0,x1 上的解完全一致。
//
// 编译后运行：./build/base06_marginalization

#include <ceres/ceres.h>

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>

// ----------------------------- 残差块 -----------------------------
// 两变量线性残差：a*x0 + b*x1 - c
struct Linear2 {
    Linear2(double a, double b, double c) : a_(a), b_(b), c_(c) {}

    template <typename T>
    bool operator()(const T* const x0, const T* const x1, T* residual) const {
        residual[0] = T(a_) * x0[0] + T(b_) * x1[0] - T(c_);
        return true;
    }

    double a_, b_, c_;
};

// 单变量线性残差：a*x - c
struct Linear1 {
    Linear1(double a, double c) : a_(a), c_(c) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(a_) * x[0] - T(c_);
        return true;
    }

    double a_, c_;
};

// 边缘化先验：r = L^T * (x - x_lin)，H_prior = L * L^T
struct PriorCost {
    PriorCost(const Eigen::Matrix2d& L, const Eigen::Vector2d& x_lin)
        : L_(L), x_lin_(x_lin) {}

    template <typename T>
    bool operator()(const T* const x0, const T* const x1, T* residual) const {
        const T d0 = x0[0] - T(x_lin_(0));
        const T d1 = x1[0] - T(x_lin_(1));

        // L 是下三角，L^T * d
        residual[0] = T(L_(0, 0)) * d0 + T(L_(1, 0)) * d1;
        residual[1] = T(L_(0, 1)) * d0 + T(L_(1, 1)) * d1;
        return true;
    }

    Eigen::Matrix2d L_;
    Eigen::Vector2d x_lin_;
};

int main() {
    // 1) 旧残差的 H 和 g（在 x_lin = (0,3,2) 处，残差全为 0）
    const Eigen::Vector3d x_lin(0.0, 3.0, 2.0);
    std::cout << std::setprecision(12);

    const Eigen::Vector3d J1(1.0, 1.0, 0.0);
    const Eigen::Vector3d J2(0.0, 1.0, 1.0);
    const Eigen::Vector3d J3(0.0, 0.0, 1.0);

    const double r1 = x_lin(0) + x_lin(1) - 3.0;  // 0
    const double r2 = x_lin(1) + x_lin(2) - 5.0;  // 0
    const double r3 = x_lin(2) - 2.0;             // 0

    Eigen::Matrix3d H = J1 * J1.transpose() + J2 * J2.transpose() +
                        J3 * J3.transpose();
    const Eigen::Vector3d g = J1 * r1 + J2 * r2 + J3 * r3;

    std::cout << "H =\n" << H << "\n\ng = " << g.transpose() << "\n\n";

    // 2) Schur 补：边缘化 x2
    const Eigen::Matrix2d Hmm = H.topLeftCorner<2, 2>() -
                                H.topRightCorner<2, 1>() * (1.0 / H(2, 2)) *
                                    H.bottomLeftCorner<1, 2>();
    const Eigen::Vector2d gm =
        g.head<2>() - H.topRightCorner<2, 1>() * (1.0 / H(2, 2)) * g(2);

    std::cout << "Hmm =\n" << Hmm << "\n\ngm = " << gm.transpose() << "\n\n";

    // Hmm = L * L^T
    const Eigen::LLT<Eigen::Matrix2d> llt(Hmm);
    const Eigen::Matrix2d L = llt.matrixL();

    // 边缘化后，x0,x1 在旧信息下的最优解：
    // x_prior = x_lin.head<2>() - Hmm^{-1} gm
    // 本例 gm=0，所以先验中心就是 x_lin.head<2>() = (0,3)。
    const Eigen::Vector2d x_prior = x_lin.head<2>() - Hmm.ldlt().solve(gm);

    std::cout << "prior mean = " << x_prior.transpose() << "\n\n";

    // 3) 全量问题：x0,x1,x2 + 旧残差 + 新观测
    double x0 = 0.0, x1 = 0.0, x2 = 0.0;

    ceres::Problem full;
    full.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear2, 1, 1, 1>(new Linear2(1, 1, 3)),
        nullptr, &x0, &x1);
    full.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear2, 1, 1, 1>(new Linear2(1, 1, 5)),
        nullptr, &x1, &x2);
    full.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear1, 1, 1>(new Linear1(1, 2)),
        nullptr, &x2);
    full.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear1, 1, 1>(new Linear1(1, 1.1)),
        nullptr, &x0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 20;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 1e-14;
    options.parameter_tolerance = 1e-14;

    ceres::Solver::Summary summary_full;
    ceres::Solve(options, &full, &summary_full);

    std::cout << "[full]       x0=" << x0 << " x1=" << x1 << " x2=" << x2 << "\n";

    // 4) 边缘化问题：x0,x1 + 先验 + 新观测
    double y0 = 0.0, y1 = 0.0;

    ceres::Problem marg;
    marg.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PriorCost, 2, 1, 1>(
            new PriorCost(L, x_prior)),
        nullptr, &y0, &y1);
    marg.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Linear1, 1, 1>(new Linear1(1, 1.1)),
        nullptr, &y0);

    ceres::Solver::Summary summary_marg;
    ceres::Solve(options, &marg, &summary_marg);

    std::cout << "[marginalized] x0=" << y0 << " x1=" << y1 << "\n";
    std::cout << "diff: |dx0|=" << std::fabs(x0 - y0)
              << " |dx1|=" << std::fabs(x1 - y1) << "\n\n";

    std::cout << summary_full.BriefReport() << "\n";
    std::cout << summary_marg.BriefReport() << "\n";
    return summary_full.IsSolutionUsable() && summary_marg.IsSolutionUsable() &&
                   std::fabs(x0 - y0) < 1e-10 && std::fabs(x1 - y1) < 1e-10 &&
                   std::fabs(x0 - 0.825) < 1e-10 &&
                   std::fabs(x1 - 2.45) < 1e-10 &&
                   std::fabs(x2 - 2.275) < 1e-10
               ? 0 : 1;
}
