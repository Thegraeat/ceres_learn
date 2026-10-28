# 第 7 课 课后作业参考答案

> 对应题目：`assignments/07_homework.md`
> 可编译代码：`assignments/hw07_engineering.cpp`

---

## 1. 关键公式

### 1.1 理论协方差

未白化残差：

$$
r_i = y_i - \exp(a x_i^2 + b x_i + c)
$$

雅可比：

$$
J_i =
\begin{bmatrix}
-x_i^2 e^{u_i} & -x_i e^{u_i} & -e^{u_i}
\end{bmatrix}
$$

白化残差是 $r_i / \sigma$，所以白化后的 $H$：

$$
H_w = \frac{1}{\sigma^2}\sum_i J_i^\top J_i
$$

高斯-牛顿协方差：

$$
\Sigma = H_w^{-1}
= \sigma^2\left(\sum_i J_i^\top J_i\right)^{-1}
$$

### 1.2 标准差与样本数

$H \propto N$，所以：

$$
\sigma_{\text{param}} \propto \frac{1}{\sqrt{N}}
$$

点数翻 100 倍，标准差大约降到 `1/10`。

---

## 2. 完整答案代码

```cpp
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
```

---

## 3. 预期输出

```
stddev comparison:
  param 0: ceres=... theory=... rel=1e-12 ~ 1e-8
  param 1: ceres=... theory=... rel=1e-12 ~ 1e-8
  param 2: ceres=... theory=... rel=1e-12 ~ 1e-8
covariance ok: yes

gradient check: PASS, max relative error = 1e-8 ~ 1e-6

1000 residuals, 3 parameters:
  DENSE_QR:               ... ms
  SPARSE_NORMAL_CHOLESKY: ... ms
  solution diff: ~1e-10
```

说明：

- 协方差的相对误差在 `1e-8` 以内，说明 Ceres 和解析公式在同一个线性化点、同一个 $H$ 上一致；
- 梯度检查通过，说明自动微分雅可比正确；
- 两种求解器解一致，耗时差异取决于问题规模和 Ceres 内部开销。

---

## 4. 设计题参考答案

一个可落地的 VIO 后端模块图：

```
┌─────────────────────────────────────────────────────────┐
│                        main / dataset                    │
│  读取关键帧、2D 观测、3D 点、IMU 测量、真值（可选）        │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ types.h      │   │ imu.h/.cpp   │   │ factors.h/.cpp   │
│ Pose/State   │   │ 预积分        │   │ Reprojection     │
│ Observation  │   │ 零偏雅可比    │   │ ImuError         │
│ Preintegration│  │ 协方差传播    │   │ PriorCost        │
└──────────────┘   └──────┬───────┘   └────────┬─────────┘
                          │                    │
                          ▼                    ▼
                 ┌────────────────────────────────────┐
                 │        backend.h/.cpp              │
                 │  BuildProblem()                    │
                 │  AddVisualFactors()                │
                 │  AddImuFactors()                   │
                 │  AddMarginalizationPrior()         │
                 │  FixGauge()                        │
                 │  Solve()                           │
                 └──────────────┬─────────────────────┘
                                │
                                ▼
                 ┌────────────────────────────────────┐
                 │        eval.h/.cpp                 │
                 │  ATE / RPE / 重投影 RMSE / NEES     │
                 └────────────────────────────────────┘
```

### 状态

```cpp
struct State {
    double q[4];   // w,x,y,z
    double p[3];
    double v[3];
    double bg[3];
    double ba[3];
};
```

### 因子

| 因子 | 连接 | 维度 |
|---|---|---|
| 重投影 | $T_i$, $X_k$ | 2 |
| IMU 预积分 | $T_i, v_i, b_i, T_j, v_j, b_j$ | 9 或 15 |
| 边缘化先验 | 窗口内状态 | 取决于状态维度 |
| 零偏随机游走 | $b_i, b_j$ | 6 |

### 边缘化策略

- 窗口固定大小（例如 10 个关键帧）；
- 最老帧的视觉观测、IMU 因子、先验一起构造成 $H, g$；
- Schur 补消去最老帧，得到剩余状态的先验；
- 先验中心用 $x_{prior} = x_{lin} - H^{-1}g$；
- 固定边缘化先验的参考点；完整 FEJ 另需针对活动因子与不可观测零空间推导，不能仅凭冻结先验声称实现。

### 评估

- 重投影 RMSE ≈ 像素噪声；
- ATE（对齐后）；
- RPE（1 秒间隔）；
- 零偏误差；
- NEES（如果同时估协方差）。

### 最容易出错的两个地方

1. **坐标系和四元数顺序**：$T_{wc}$ 还是 $T_{cw}$、`wxyz` 还是 `xyzw`，一处写反，残差就完全错，而且优化还能“收敛”。
2. **预积分与零偏的符号**：$a_m = R^\top(a_w - g)$，预积分里 $a_m - b_a$，残差里 $R_i^\top(\ldots) - \Delta v$。符号错一处，IMU 因子会把解拉向错误方向。

---

## 5. 思考题答案

### 5.1 为什么协方差随 N 增大而减小？

$H = \sum J_i^\top J_i$ 随样本数线性增长，$\Sigma \approx H^{-1}$ 随 $N$ 线性减小，所以标准差按 $1/\sqrt{N}$ 下降。这和“多测几次更准”的直觉一致。

### 5.2 为什么 loss function 会影响协方差？

Ceres 默认应用 loss 对局部雅可比的修正，因此改变了协方差近似。`apply_loss_function = false` 可以关闭这项修正，但不会自动恢复真实不确定性；含外点问题不能简单去掉核就当作正确的高斯模型。应明确内点筛选、噪声模型与稳健统计协方差的区别；详见第 7 课。

### 5.3 固定参数块的协方差为什么是 0？

固定参数没有自由度，$H$ 中对应的行列被移除，协方差自然是 0。它表示“我们假设这个量完全已知”，不是“估计得很准”。

### 5.4 为什么 3 个参数时两种求解器差距不大？

$H$ 只有 $3\times 3$，稀疏结构没有意义；`SPARSE_NORMAL_CHOLESKY` 的额外开销（稀疏矩阵分析、排序）反而可能让它更慢。稀疏求解器的优势要到几百上千个参数才显现。

---

## 6. 常见错误

1. **理论协方差忘记乘 $\sigma^2$**：得到的是白化后的协方差，标准差偏大 $1/\sigma$ 倍。
2. **Ceres 协方差在固定参数上计算**：固定块返回 0，别拿它和理论值比。
3. **`GradientChecker` 的 manifolds 个数不对**：每个参数块一个，欧氏空间用 `nullptr`。
4. **`Probe` 的参数数组生命周期**：`params` 指向的 `abc` 必须在 `Probe` 期间有效。
5. **性能对比复用同一个 `Problem`**：第二次求解从第一次的解出发，耗时和结果都不可比。
