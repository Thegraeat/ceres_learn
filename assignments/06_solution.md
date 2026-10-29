# 第 6 课 课后作业参考答案

> 对应题目：`assignments/06_homework.md`
> 可编译代码：`assignments/hw06_marginalization.cpp`

---

## 1. 关键公式

### 1.1 非最优线性化点

旧残差在 $x_{lin} = (1.2, 1.8, 3.1)$ 处：

$$
r_1 = 0.0,\qquad r_2 = -0.1,\qquad r_3 = 1.1
$$

梯度：

$$
g = \sum_i J_i^\top r_i
= \begin{bmatrix}0\\-0.1\\1.0\end{bmatrix}
$$

$H$ 不变（线性问题的雅可比是常数）：

$$
H =
\begin{bmatrix}
1 & 1 & 0 \\
1 & 2 & 1 \\
0 & 1 & 2
\end{bmatrix}
$$

### 1.2 Schur 补

$$
H_{prior} =
\begin{bmatrix}
1 & 1 \\
1 & 1.5
\end{bmatrix},
\qquad
g_{prior} =
\begin{bmatrix}
0 \\ -0.6
\end{bmatrix}
$$

### 1.3 先验中心

$$
x_{prior}
= x_{lin,m} - H_{prior}^{-1}g_{prior}
= \begin{bmatrix}1.2\\1.8\end{bmatrix}
- \begin{bmatrix}3 & -2\\-2 & 2\end{bmatrix}
\begin{bmatrix}0\\-0.6\end{bmatrix}
= \begin{bmatrix}0\\3\end{bmatrix}
$$

**这就是本题的核心**：$x_{prior}$ 不是 $(1.2, 1.8)$。

几何上，旧信息在边缘化 $x_2$ 后的最优解是 $(0, 3)$：

- $x_2 = 2$ 让 $r_3 = 0$；
- $x_1 = 3$、$x_0 = 0$ 同时让 $r_2 = r_1 = 0$，这里并不需要折中。

### 1.4 Cholesky

$$
H_{prior} = LL^\top,
\qquad
L =
\begin{bmatrix}
1 & 0 \\
1 & \sqrt{0.5}
\end{bmatrix}
$$

先验残差：

$$
r_{prior} = L^\top\left(x_{0:1} - x_{prior}\right)
$$

### 1.5 把消元梯度也手算出来

保留 $x_0,x_1$、删除 $x_2$，原始矩阵块为 $H_{mr}=(0,1)^\top$、$H_{rr}=2$，原始梯度块为 $g_m=(0,-0.1)^\top$、$g_r=1$。所以：

$$
g_{prior}=\begin{bmatrix}0\\-0.1\end{bmatrix}
-\begin{bmatrix}0\\1\end{bmatrix}\frac12\cdot1
=\begin{bmatrix}0\\-0.6\end{bmatrix}.
$$

负号来自将被删除变量的方程代回保留侧，不是任意选的梯度方向。再解 $H_{prior}d=g_{prior}$，得到 $d=(1.2,-1.2)^\top$，于是中心为 $(1.2,1.8)^\top-d=(0,3)^\top$。

这里 $x_{0:1}$ 表示将 $x_0,x_1$ 堆叠成二维列向量。数学记号统一使用 $H_{prior},g_{prior}$；参考代码仍把它们命名为 `Hmm,gm`，不要与消元前的 $H_{mm},g_m$ 混淆。

同一组线性旧残差在任意展开点都能被一阶式**精确**表示，所以改变 $x_{lin}$ 只会改变增量坐标与梯度，不改变绝对状态中的旧信息。最终中心仍是 $(0,3)$，不是偶然算出了相同数字。

---

## 2. 完整答案代码

```cpp
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
```

---

## 3. 预期输出

```
r = (0, -0.1, 1.1)
H =
1 1 0
1 2 1
0 1 2

g = 0 -0.1 1

Hmm =
  1   1
  1 1.5

gm = 0 -0.6
x_prior = 0 3
L =
        1        0
        1 0.707107

obs x0 = 1.1
  full:   x0=0.825 x1=2.45 x2=2.275
  marg:   x0=0.825 x1=2.45
  diff:   ~1e-16, ~1e-16

obs x0 = 1.5
  full:   x0=1.125 x1=2.25 x2=2.375
  marg:   x0=1.125 x1=2.25
  diff:   ~1e-16, ~1e-16
```

`diff` 在浮点精度范围内，说明边缘化对线性问题是精确的。

---

## 4. 思考题答案

### 4.1 为什么先验中心不是 $x_{lin}$？

$x_{lin}$ 只是泰勒展开点。边缘化保留的是旧信息在 $x_m$ 上的**边缘分布**，它的均值（最优解）是：

$$
x_{prior} = x_{lin,m} - H_{prior}^{-1}g_{prior}
$$

在先验正定时，中心等于展开点的条件是 Schur 消元后的 $g_{prior}=0$；旧完整问题处于驻点是充分条件，但不是必要条件。

### 4.2 $H_{prior}$ 的物理含义

$H_{prior} = H_{mm} - H_{mr} H_{rr}^{-1} H_{rm}$ 是 Schur 补：

- $H_{mm}$：保留变量自身的直接信息；
- $H_{mr} H_{rr}^{-1} H_{rm}$：允许 $x_r$ 随 $x_m$ 调整后，相对“把 $x_r$ 视为已知常量”减少的信息；
- Schur 补体现被消元变量的不确定性，不能误解为减去重复观测；消元还可能在剩余变量之间形成稠密耦合。

结果 $H_{prior}$ 是旧观测对 $x_m$ 的**等效信息矩阵**。

### 4.3 被丢掉的变量还能回来吗？

仅靠先验不能。边缘化后的先验不再包含原始观测和被删除变量的联合表示；这不意味着磁盘上的原始数据被自动删除。若仍保存了原始观测和状态，可以重建原问题再优化。

要恢复只有两种办法：

- 保留原始观测，重新做全量优化；
- 从地图/关键帧里重新提取约束。

所以边缘化是**不可逆**的。

### 4.4 FEJ 解决什么问题？

FEJ（First Estimate Jacobian）在相关雅可比项采用一致的首次估计，目的是保持正确的不可观测零空间，缓解历史先验与当前活动因子线性化不一致引入的虚假信息；它不是删除重复观测的办法。冻结边缘化先验也不等于实现了完整 FEJ。本题是精确线性问题，没有该非线性可观测性问题，不能用它证明 FEJ 的效果。

---

## 5. 常见错误

1. **先验中心直接用 $x_{lin}$**：本题会得到 $(1.2, 1.8)$，全量与边缘化差很多。
2. **`gm` 符号错**：$g_m - H_{mr} H_{rr}^{-1} g_r$，注意是减。
3. **Cholesky 方向错**：残差是 $L^\top d$，不是 $L d$。
4. **全量问题从 $x_{lin}$ 出发**：线性问题无所谓，但非线性问题会掩盖等价性。
5. **新观测变化时重算先验**：先验只依赖旧信息，不变。
