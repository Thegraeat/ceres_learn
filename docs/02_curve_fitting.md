# 第 2 课：曲线拟合——白化、鲁棒核与自动微分

> 对应代码：`base02_curve_fitting.cpp`
> 目标：理解「残差白化 → 鲁棒核 → 自动微分」这条链路，它是后面所有 SLAM 残差块的模板。
> 前置：第 1 课 `01_ceres_basics.md`。

---

## 1. 问题定义

用模型

$$
f(x;\,a,b,c) = \exp\!\left(a x^2 + b x + c\right)
$$

拟合 100 个带噪声的观测 $(x_i, y_i)$，其中每 10 个点里有一个**外点**（outlier）。

真值：

$$
a^\star = 1.0,\qquad b^\star = 2.0,\qquad c^\star = 0.5
$$

外点模拟的是 SLAM 里的**误匹配**：大部分观测是对的，少数观测错得很离谱。

---

## 2. 从最大似然到最小二乘

假设观测模型

$$
y_i = f(x_i;\,a,b,c) + \varepsilon_i,\qquad
\varepsilon_i \sim \mathcal{N}(0,\sigma^2)
$$

负对数似然（去掉常数）：

$$
-\log p(y \mid a,b,c)
= \frac{1}{2\sigma^2}\sum_{i=1}^{N}\left(y_i - f(x_i)\right)^2
$$

所以最大似然等价于

$$
\min_{a,b,c}\;\sum_{i=1}^{N}
\left(\frac{y_i - f(x_i)}{\sigma}\right)^2
$$

定义**白化残差**：

$$
r_i(a,b,c) = \frac{y_i - \exp(a x_i^2 + b x_i + c)}{\sigma}
$$

目标函数：

$$
\min_{a,b,c}\; \sum_{i=1}^{N} \rho\!\left(r_i^2\right)
$$

- $1/\sigma$ 是白化，等价于乘信息矩阵 $\Sigma^{-1} = 1/\sigma^2$；
- $\rho$ 是鲁棒核，默认 $\rho(s) = s$。

> **为什么推荐白化**：白化使阈值 $\delta$ 可以按“几个 sigma”解释。未白化时阈值仍可按原始观测单位设置，例如 $\delta_{\mathrm{raw}} = \delta_{\mathrm{white}}\sigma$；二者不能混淆。对固定 $\sigma$ 的本例，省略目标函数前的 $1/2$ 不改变最优解，但 Ceres 的 `final_cost` 包含 $1/2$。

---

## 3. 雅可比推导

Ceres 的自动微分会在内部算出下面的雅可比。理解它，才能判断自动微分有没有写错。

记

$$
u_i = a x_i^2 + b x_i + c,\qquad f_i = e^{u_i}
$$

残差对三个参数的偏导：

$$
\frac{\partial r_i}{\partial a}
= -\frac{x_i^2 e^{u_i}}{\sigma},
\qquad
\frac{\partial r_i}{\partial b}
= -\frac{x_i e^{u_i}}{\sigma},
\qquad
\frac{\partial r_i}{\partial c}
= -\frac{e^{u_i}}{\sigma}
$$

所以每个残差块的雅可比是 $1\times 3$：

$$
J_i =
\begin{bmatrix}
-\dfrac{x_i^2 e^{u_i}}{\sigma} &
-\dfrac{x_i e^{u_i}}{\sigma} &
-\dfrac{e^{u_i}}{\sigma}
\end{bmatrix}
$$

堆叠起来就是 $N\times 3$ 的 $J$，正规方程是 $3\times 3$：

$$
\left(\sum_i J_i^\top J_i\right)
\begin{bmatrix}\Delta a\\ \Delta b\\ \Delta c\end{bmatrix}
= -\sum_i J_i^\top r_i
$$

因为只有 3 个参数，$H$ 是稠密的 $3\times 3$，用 `DENSE_QR` 正合适。

### 3.1 自动微分怎么得到这个雅可比

Ceres 把 `T` 替换成 `Jet`：

$$
x_i = x_i + 0\cdot\epsilon,\quad
a = a + 1\cdot\epsilon,\quad
b = b + 0\cdot\epsilon,\quad
c = c + 0\cdot\epsilon
$$

然后 `exp` 的 Jet 重载会按链式法则传播：

$$
\exp(u + u'\epsilon) = e^{u} + e^{u}u'\epsilon
$$

最后 $\varepsilon$ 的系数就是 $\partial r/\partial a$。这是单方向示意；实际 `Jet<double, 3>` 用三个独立的无穷小分量，在一次 functor 求值中传播三列导数，不需要分别调用三次。依赖参数的中间量必须保留 `T` 类型，常量可以是 `double`。

---

## 4. 鲁棒核

### 4.1 Huber

Ceres 的 `HuberLoss`($\delta$) 定义在 $s = r^2$ 上：

$$
\rho(s) =
\begin{cases}
s, & s \le \delta^2 \\[4pt]
2\delta\sqrt{s} - \delta^2, & s > \delta^2
\end{cases}
$$

影响函数（代价对残差的导数）：

$$
\psi(r) = \rho'(r^2)\cdot 2r =
\begin{cases}
2r, & |r| \le \delta \\
2\delta\,\mathrm{sign}(r), & |r| > \delta
\end{cases}
$$

含义：内点（$|r| \le \delta$）保持二次代价；外点（$|r| > \delta$）退化成线性代价，不会让一个错观测产生巨大的梯度。

### 4.2 Cauchy

$$
\rho(s) = \delta^2\log\!\left(1+\frac{s}{\delta^2}\right),
\qquad
\psi(r) = \frac{2r}{1+r^2/\delta^2}
$$

外点的权重按 $1/r^2$ 衰减，比 Huber 更激进。

### 4.3 阈值怎么选

经验规则：

- 残差已经白化（单位是 sigma）：$\delta$ 取 $1.0 \sim 3.0$；
- 外点比例高：用更小的 $\delta$，或用 Cauchy；
- 外点比例低、希望保持精度：用较大的 $\delta$，或 Huber。

代码里 $\sigma = 0.02$，内点残差大约是 $\mathcal{N}(0,1)$，外点残差是几百个 sigma，所以 `HuberLoss(1.0)` 能干净地把外点压住。

---

## 5. 完整代码

```cpp
// 第 2 课配套代码：用 Ceres 做带外点的曲线拟合
//
// 目的：把「残差块 + 自动微分 + 鲁棒核 + 白化」这四件事讲清楚。
// 这些正是 SLAM 后端（位姿图 / BA）里每个残差块都要做的事。
//
// 模型：y = exp(a * x^2 + b * x + c)
// 数据：100 个点，高斯噪声 + 每 10 个点里放一个 40% 的外点。

#include <ceres/ceres.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// 一个残差块：给定观测 (x, y)，计算预测值与观测值的差。
// 模板 operator() 是自动微分的前提：Ceres 会把 T 换成 Jet 类型，
// 在计算残差的同时把对 abc 的偏导一起算出来。
struct CurveResidual {
    CurveResidual(double x, double y, double sigma) : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        // 白化：残差除以观测标准差。等价于在代价里乘信息矩阵 Sigma^{-1}。
        // SLAM 里重投影误差除以像素噪声 sigma、IMU 残差除以测量噪声，都是这一步。
        residual[0] = (T(y_) - ceres::exp(abc[0] * T(x_ * x_) + abc[1] * T(x_) + abc[2])) /
                      T(sigma_);
        return true;
    }

    const double x_;
    const double y_;
    const double sigma_;
};

// 求解一次：use_huber 决定是否用鲁棒核。
void Solve(bool use_huber, const std::vector<double>& xs, const std::vector<double>& ys) {
    double abc[3] = {0.8, 1.8, 0.2};  // 初值

    ceres::Problem problem;
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
            new CurveResidual(xs[i], ys[i], 0.02));

        // 关键：LossFunction 作用在「白化后的残差」上。
        // 阈值 1.0 表示：残差超过 1 sigma 后，代价从二次增长退化为线性增长，
        // 从而不让少数外点主导整个解。
        ceres::LossFunction* loss = use_huber ? new ceres::HuberLoss(1.0) : nullptr;

        problem.AddResidualBlock(cost, loss, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  // 3 个参数的小问题
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << (use_huber ? "[Huber]   " : "[No loss] ") << summary.BriefReport() << "\n";
    std::cout << "          abc = " << abc[0] << ", " << abc[1] << ", " << abc[2] << "\n";
}

int main() {
    // 真值：y = exp(1.0 * x^2 + 2.0 * x + 0.5)
    const double a_true = 1.0, b_true = 2.0, c_true = 0.5;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);

    std::vector<double> xs, ys;
    for (int i = 0; i < 100; ++i) {
        const double x = -1.0 + 2.0 * i / 99.0;
        double y = std::exp(a_true * x * x + b_true * x + c_true) + noise(rng);

        // 每 10 个点制造一个外点（错误关联 / 误匹配的类比）
        if (i % 10 == 0) {
            y += 0.4 * y;
        }
        xs.push_back(x);
        ys.push_back(y);
    }

    std::cout << "truth:    abc = " << a_true << ", " << b_true << ", " << c_true << "\n";
    Solve(false, xs, ys);
    Solve(true, xs, ys);

    return 0;
}
```

---

## 6. 代码逐段讲解

### 6.1 残差 functor

```cpp
struct CurveResidual {
    CurveResidual(double x, double y, double sigma) : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] = (T(y_) - ceres::exp(abc[0] * T(x_ * x_) + abc[1] * T(x_) + abc[2])) /
                      T(sigma_);
        return true;
    }

    const double x_;
    const double y_;
    const double sigma_;
};
```

- `x_`、`y_`、`sigma_` 是**观测常量**，用 `double` 存储没问题；
- 但进入表达式时全部转成 `T`：`T(x_ * x_)`、`T(sigma_)`；
- `ceres::exp` 是 Ceres 提供的重载，能接受 `Jet`；如果用 `std::exp`，模板推导会失败；
- 残差维度是 1，所以 `residual[0]`。

**常见错误**：写成

```cpp
const double u = abc[0] * x_ * x_ + ...;   // 错！abc[0] 是 T
```

这行根本编译不过（`T` 不能赋给 `double`）。另一种更隐蔽的写法是先把参数转成 `double`，能编译，但雅可比全变 0。

### 6.2 构建代价函数

```cpp
ceres::CostFunction* cost =
    new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
        new CurveResidual(xs[i], ys[i], 0.02));
```

- `1`：残差维度；
- `3`：参数块 `abc` 的维度；
- 每个观测点一个残差块，共用同一组参数 `abc`。

100 个残差块、3 个参数，$H$ 是 $3\times 3$ 稠密矩阵。

### 6.3 鲁棒核

```cpp
ceres::LossFunction* loss = use_huber ? new ceres::HuberLoss(1.0) : nullptr;
problem.AddResidualBlock(cost, loss, abc);
```

- `nullptr`：普通最小二乘；
- `HuberLoss(1.0)`：白化残差超过 1σ 后降权。

### 6.4 求解器

```cpp
options.linear_solver_type = ceres::DENSE_QR;
options.max_num_iterations = 100;
```

参数只有 3 个，稠密 QR 足够。`DENSE_QR` 直接分解雅可比（LM 时是增广系统），并不通过显式构造 $J^\top J$ 来解正规方程，因此能避免条件数平方带来的精度损失。

### 6.5 两次求解

```cpp
Solve(false, xs, ys);   // 普通最小二乘
Solve(true, xs, ys);    // Huber
```

两次都从同一个初值 `{0.8, 1.8, 0.2}` 出发，方便对比。

---

## 7. 预期结果

- `[No loss]`：10 个外点的残差白化后高达几百，二次代价会把参数明显拉偏；
- `[Huber]`：外点被压成线性代价，参数应该更接近真值 `(1.0, 2.0, 0.5)`。

具体数值取决于随机数，不必逐位一致。重点比较两行的参数误差；不同核对应不同目标函数，不能用谁的 `final_cost` 更小判断谁拟合更好。

> 如果 `[No loss]` 的参数跑得很远甚至发散，不是代码错，而是外点确实破坏了最小二乘。这正是鲁棒核存在的理由。

---

## 8. 练习

1. **改变噪声尺度**：把数据生成噪声和残差白化中的 `sigma` 同时从 `0.02` 改成 `0.2`，保持白化阈值 `1.0`。解释为什么此时原始观测单位下的阈值自动从 `0.02` 变成 `0.2`，而白化阈值不用再乘 `sigma`。
2. **提高外点比例**：把 `i % 10` 改成 `i % 3`，观察失效情况。一次合成实验的失败比例不等于估计器的理论 breakdown point；非线性回归还受初值、杠杆点和几何条件影响。
3. **换核函数**：把 `HuberLoss(1.0)` 换成 `CauchyLoss(1.0)`，比较最终参数和迭代次数。
4. **去掉白化**：把残差里的 `/ T(sigma_)` 删掉，阈值仍用 `1.0`。解释为什么 Huber 基本失效。
5. **思考题**：如果 `sigma` 未知，为什么不能只把它加入 $r/\sigma$ 中一起优化？写出完整高斯负对数似然 $N\log\sigma + \sum e_i^2/(2\sigma^2)$，并用 $\log\sigma$ 参数化保证正值；或者使用作业中的“鲁棒拟合 + MAD”交替估计。若缺少 $N\log\sigma$，令 $\sigma \to \infty$ 就能虚假地降低代价。

---

## 9. 这一课和 SLAM 的对应关系

| 曲线拟合 | SLAM |
|---|---|
| `abc` | 位姿 / 路标点 |
| $(x_i, y_i)$ | 一次观测（像素坐标、IMU 读数、激光点） |
| $\exp(a x^2 + b x + c)$ | 观测模型（投影、预积分、点到面距离） |
| `sigma` | 观测噪声 / 协方差 |
| 外点 | 误匹配、动态物体、错误回环 |
| `HuberLoss` | 后端鲁棒核 |

下一课 `03_pose_graph.md` 会把这套结构搬到 SE(3) 上，残差从 1 维变成 6 维，参数从 3 个标量变成成百上千个位姿。
