# 第 1 课 课后作业参考答案

> 对应题目：`assignments/01_homework.md`
> 可编译代码：`assignments/hw01_basics.cpp`

---

## 1. 解题思路

### 1.1 两个独立参数块

不要把 $m$ 和 $b$ 塞进一个 `double mb[2]`。分成两块：

```cpp
double m = 0.0;
double b = 0.0;
problem.AddResidualBlock(cost, nullptr, &m, &b);
```

好处：

- 模板维度是 `<F, 1, 1, 1>`，语义清楚；
- 可以单独固定 $b$，不需要 `SubsetManifold`；
- 调试时可以分别打印。

### 1.2 残差与雅可比

$$
r_i = y_i - (m x_i + b)
$$

$$
J_i = \begin{bmatrix}
\dfrac{\partial r_i}{\partial m} & \dfrac{\partial r_i}{\partial b}
\end{bmatrix}
= \begin{bmatrix}
-x_i & -1
\end{bmatrix}
$$

正规方程：

$$
H = \sum_i J_i^\top J_i =
\begin{bmatrix}
\sum x_i^2 & \sum x_i \\
\sum x_i & n
\end{bmatrix},
\qquad
g = \sum_i J_i^\top r_i =
\begin{bmatrix}
-\sum x_i r_i \\
-\sum r_i
\end{bmatrix}
$$

这就是思考题 2 的答案。

### 1.3 闭式解

这里要求的是参数的**绝对值**，不能把本轮的增量方程 $H\Delta x=-g$ 直接当成绝对参数方程。令代价对 $m,b$ 的导数为零，有：

$$
\sum_i x_i(y_i-mx_i-b)=0,\qquad
\sum_i(y_i-mx_i-b)=0.
$$

整理得到：

$$
\begin{bmatrix}\sum_i x_i^2&\sum_i x_i\\\sum_i x_i&n\end{bmatrix}
\begin{bmatrix}m\\b\end{bmatrix}
=\begin{bmatrix}\sum_i x_iy_i\\\sum_i y_i\end{bmatrix}.
$$

第二行除以 $n$，得到 $b=\bar y-m\bar x$，其中 $\bar x=\frac1n\sum_i x_i$、$\bar y=\frac1n\sum_i y_i$。将它代入第一行并整理，得到：

$$
\hat m =
\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}
     {\sum_i (x_i-\bar x)^2},
\qquad
\hat b = \bar y - \hat m \bar x
$$

代码里顺手算一遍，用来验证 Ceres。

分母 $\sum_i(x_i-\bar x)^2$ 描述采样横坐标的分散程度。它为零时，所有 $x_i$ 相同，无法分别确定斜率和截距。分子描述 $x,y$ 一起增减的程度；两者相除就是斜率。

例如数据为 $(0,1),(1,3),(2,5)$，有 $\bar x=1,\bar y=3$。分子为 $2+0+2=4$，分母为 $1+0+1=2$，所以 $\hat m=2,\hat b=1$。

### 1.4 固定 $b$

```cpp
if (fix_b) {
    problem.SetParameterBlockConstant(&b);
}
```

注意：$b$ 仍然要通过 `AddResidualBlock` 传进去，只是它的增量被固定为 0。

---

## 2. 完整答案代码

```cpp
#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

struct LineResidual {
    LineResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m, const T* const b, T* residual) const {
        residual[0] = T(y_) - (m[0] * T(x_) + b[0]);
        return true;
    }

    const double x_, y_;
};

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

    // 闭式解
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
```

---

## 3. 预期输出（数值形态）

```
truth:           m=2 b=-0.5
closed form:     m=1.99xx b=-0.49xx
[A] auto free:   m=1.99xx b=-0.49xx cost=... time=... ms
[B] auto fix b:  m=1.99xx cost=... time=... ms
[C] numeric:     m=1.99xx b=-0.49xx cost=... time=... ms
|A-C|: 1e-10 ~ 1e-07 (m), 1e-10 ~ 1e-07 (b)
```

具体数字取决于随机数，但应该满足：

- `A` 与闭式解几乎完全一致（差异 < `1e-6`）；
- `B` 的 `m` 与 `A` 接近，但不会完全相同；
- `C` 与 `A` 的差在 `1e-4` 以内；
- `C` 的耗时通常比 `A` 大（200 个残差、2 个参数，中心差分每个残差要算 4 次）。

---

## 4. 思考题答案

### 4.1 固定 $b$ 后 $m$ 为什么几乎不变？

固定截距为 $b_0$ 后，只需令关于 $m$ 的导数为零：

$$
\hat m_{fixed}=\frac{\sum_i x_i(y_i-b_0)}{\sum_i x_i^2}.
$$

将联合最小二乘结果 $\hat m,\hat b$ 代入正规方程，可进一步得到：

$$
\hat m_{fixed}-\hat m
=(\hat b-b_0)\frac{\sum_i x_i}{\sum_i x_i^2}.
$$

因此变化是否小，取决于固定截距与原估计差多少，以及采样位置的分布。若 $\sum_i x_i=0$ 且斜率可辨识，固定任意截距都不会改变最优斜率；若固定了错误截距且采样偏向一侧，斜率就会调整来补偿。不能仅凭样本数多就保证几乎不变。

### 4.2 $H$ 和 $g$

$$
H =
\begin{bmatrix}
\sum x_i^2 & \sum x_i \\[4pt]
\sum x_i & n
\end{bmatrix},
\qquad
g =
\begin{bmatrix}
-\sum x_i r_i \\[4pt]
-\sum r_i
\end{bmatrix}
$$

### 4.3 所有 $x_i$ 相同时会怎样？

$H$ 的两列线性相关：

$$
H =
\begin{bmatrix}
n x^2 & n x \\
n x & n
\end{bmatrix},
\qquad \det H = 0
$$

$H$ 奇异，$m$ 和 $b$ 不可分辨（$m x + b$ 只确定一个组合）。某些求解器/配置可能报：

```
Termination: FAILURE
```

但不能把 `FAILURE` 当作必然结果：LM 阻尼或 QR 也可能给出一组拟合相同的解并报告收敛。此时不可辨识性仍存在，不能解释单个参数的唯一值或直接求满秩协方差。位姿图固定节点也是处理类似零空间，而不只是“防止报错”。

### 4.4 数值微分慢多少？

中心差分对每个参数需两次扰动求值，2 个参数对应 4 次扰动，另需基础残差评估；自动微分一次 Jet 求值传播全部导数，但 Jet 运算也有随导数维度增长的开销。因此不能从调用次数推出固定的 4 倍耗时。该小例子还受内部调度、线性求解和计时波动影响，应预热并重复测量。

---

## 5. 常见错误

1. **把 $m$ 和 $b$ 放进一个数组，然后试图固定其中一个**：`SetParameterBlockConstant` 只能固定整个参数块，不能固定块内单个元素（除非用 `SubsetManifold`）。
2. **`AutoDiffCostFunction<F, 1, 1, 1>` 写成 `<F, 1, 2>`**：维度不匹配，运行时崩溃。
3. **数值微分 functor 写成模板**：`NumericDiffCostFunction` 要求 `double` 版本。
4. **`AddResidualBlock` 参数顺序写反**：`&b, &m` 会让结果完全错，但不报错。
5. **闭式解公式写错**：用 $\sum x_i y_i - n\bar{x}\bar{y}$ 时注意和 $\sum (x_i-\bar{x})(y_i-\bar{y})$ 等价。
