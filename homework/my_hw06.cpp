// 第 6 课 课后作业模板（my_hw06.cpp）
//
// 任务：一般线性化点下的边缘化
//   x_lin = (1.2, 1.8, 3.1)，旧残差在 x_lin 处不再为零
//   先验中心必须用 x_prior = x_lin - Hmm^{-1} gm
//   验证全量问题与边缘化问题在 x0, x1 上一致（误差 < 1e-10）
//
// 这个文件已经准备好：三个线性残差块的结构、两个 Ceres 求解函数的签名、
// 打印格式和两个新观测（1.1 / 1.5）的验证循环。需要你手写的地方：
//   TODO 1  三个 functor 的 operator() —— 残差
//   TODO 2  SolveFull / SolveMarginalized —— Ceres 建模与求解
//   TODO 3  r、H、g 的计算
//   TODO 4  Schur 补 Hmm / gm 与先验中心 x_prior
//
// 编译运行：
//   cmake --build build --target my_hw06 -j && ./build/my_hw06
//
// 验收标准见 assignments/06_homework.md。

#include <ceres/ceres.h>

#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <iostream>

// ---------------------------------------------------------------------------
// 残差块
// ---------------------------------------------------------------------------
struct Linear2 {
    Linear2(double a, double b, double c) : a_(a), b_(b), c_(c) {}

    template <typename T>
    bool operator()(const T* const x0, const T* const x1, T* residual) const {
        // TODO 1a: residual[0] = a*x0 + b*x1 - c
        residual[0] = T(0.0);
        (void)x0;
        (void)x1;
        return true;
    }

    double a_, b_, c_;
};

struct Linear1 {
    Linear1(double a, double c) : a_(a), c_(c) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        // TODO 1b: residual[0] = a*x - c
        residual[0] = T(0.0);
        (void)x;
        return true;
    }

    double a_, c_;
};

struct PriorCost {
    PriorCost(const Eigen::Matrix2d& L, const Eigen::Vector2d& x_prior)
        : L_(L), x_prior_(x_prior) {}

    template <typename T>
    bool operator()(const T* const x0, const T* const x1, T* residual) const {
        // TODO 1c: 先验残差 r = L * (x - x_prior)
        //   d0 = x0 - x_prior(0), d1 = x1 - x_prior(1)
        //   residual[0] = L(0,0)*d0 + L(1,0)*d1
        //   residual[1] = L(0,1)*d0 + L(1,1)*d1
        //   注意 Eigen 是列主序，L(行, 列)。
        residual[0] = T(0.0);
        residual[1] = T(0.0);
        (void)x0;
        (void)x1;
        return true;
    }

    Eigen::Matrix2d L_;
    Eigen::Vector2d x_prior_;
};

// ---------------------------------------------------------------------------
// 全量问题：x0, x1, x2 + 三条旧残差 + 新观测 x0 = obs
// ---------------------------------------------------------------------------
bool SolveFull(double obs, double* x0, double* x1, double* x2) {
    *x0 = 0.0;
    *x1 = 0.0;
    *x2 = 0.0;

    // TODO 2a: 构建全量问题并求解。
    //   - Linear2(1, 1, 3) 作用在 x0, x1
    //   - Linear2(1, 1, 5) 作用在 x1, x2
    //   - Linear1(1, 2)    作用在 x2
    //   - Linear1(1, obs)  作用在 x0
    //   - Options: DENSE_QR + DOGLEG
    //     max_num_iterations = 20
    //     function_tolerance = 0.0
    //     gradient_tolerance = 1e-14
    //     parameter_tolerance = 1e-14
    //   - 返回 summary.IsSolutionUsable()
    (void)obs;
    return false;  // TODO 2a 完成后替换
}

// ---------------------------------------------------------------------------
// 边缘化问题：x0, x1 + 先验 + 新观测 x0 = obs
// ---------------------------------------------------------------------------
bool SolveMarginalized(const Eigen::Matrix2d& L,
                       const Eigen::Vector2d& x_prior,
                       double obs,
                       double* x0,
                       double* x1) {
    *x0 = 0.0;
    *x1 = 0.0;

    // TODO 2b: 构建边缘化问题并求解。
    //   - PriorCost(L, x_prior) 作用在 x0, x1
    //   - Linear1(1, obs)       作用在 x0
    //   - Options 与 SolveFull 相同
    (void)L;
    (void)x_prior;
    (void)obs;
    return false;  // TODO 2b 完成后替换
}

int main() {
    std::cout << std::setprecision(12);

    // 1) 非最优线性化点
    const Eigen::Vector3d x_lin(1.2, 1.8, 3.1);

    const Eigen::Vector3d J1(1.0, 1.0, 0.0);
    const Eigen::Vector3d J2(0.0, 1.0, 1.0);
    const Eigen::Vector3d J3(0.0, 0.0, 1.0);

    // TODO 3a: 在 x_lin 处计算三条旧残差：
    //   r1 = x0 + x1 - 3
    //   r2 = x1 + x2 - 5
    //   r3 = x2 - 2
    double r1 = 0.0;
    double r2 = 0.0;
    double r3 = 0.0;

    // TODO 3b: H = J1*J1^T + J2*J2^T + J3*J3^T
    //          g = J1*r1 + J2*r2 + J3*r3
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();

    std::cout << "r = (" << r1 << ", " << r2 << ", " << r3 << ")\n";
    std::cout << "H =\n" << H << "\n\ng = " << g.transpose() << "\n\n";

    // 2) Schur 补
    // TODO 4a: Hmm = H(0:1,0:1) - H(0:1,2) * (1/H(2,2)) * H(2,0:1)
    //          gm  = g(0:1)     - H(0:1,2) * (1/H(2,2)) * g(2)
    Eigen::Matrix2d Hmm = Eigen::Matrix2d::Zero();
    Eigen::Vector2d gm = Eigen::Vector2d::Zero();

    std::cout << "Hmm =\n" << Hmm << "\n\ngm = " << gm.transpose() << "\n";

    // 3) 先验中心：旧信息下的最优解，而不是 x_lin
    // TODO 4b: x_prior = x_lin.head<2>() - Hmm.ldlt().solve(gm)
    Eigen::Vector2d x_prior = Eigen::Vector2d::Zero();

    std::cout << "x_prior = " << x_prior.transpose() << "\n";

    const Eigen::LLT<Eigen::Matrix2d> llt(Hmm);
    const Eigen::Matrix2d L = llt.matrixL();
    std::cout << "L =\n" << L << "\n\n";

    // 4) 两个新观测
    for (double obs : {1.1, 1.5}) {
        double x0 = 0.0, x1 = 0.0, x2 = 0.0;
        double y0 = 0.0, y1 = 0.0;

        if (!SolveFull(obs, &x0, &x1, &x2) ||
            !SolveMarginalized(L, x_prior, obs, &y0, &y1)) {
            std::cout << "TODO: 先完成 TODO 1/2，全量与边缘化问题才能求解。\n";
            return 0;
        }

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
