# 第 2 课 课后作业参考答案

> 对应题目：`assignments/02_homework.md`
> 可编译代码：`assignments/hw02_robust.cpp`

---

## 1. 关键公式

### 1.1 白化

$$
r_i = \frac{y_i - \exp(a x_i^2 + b x_i + c)}{\sigma}
$$

Huber / Cauchy 的阈值 $\delta$ 作用在 $r_i$ 上，所以 $\sigma$ 必须和残差一起进入代价函数。

### 1.2 MAD 估计 `sigma`

残差 $r_i = y_i - \hat y_i$。MAD：

$$
\mathrm{MAD} = \mathrm{median}\!\left(\left|r_i - \mathrm{median}(r)\right|\right)
$$

正态分布下：

$$
\mathrm{MAD} \approx 0.6745\,\sigma
\quad\Longrightarrow\quad
\hat\sigma = \frac{\mathrm{MAD}}{0.6745} \approx 1.4826\,\mathrm{MAD}
$$

MAD 作为尺度统计量具有约 50% 的污染崩溃点，但这并不保证有限污染下无偏，更不意味着对错误回归模型的残差仍能估出真实观测噪声。应先得到合理的鲁棒模型，再估计原始残差尺度。

#### 为什么要乘 $1.4826$

令 $Z$ 为标准正态变量，$\Phi$ 为它的累积分布函数。绝对值的中位数 $c$ 满足：

$$
P(|Z|\le c)=0.5
\quad\Longrightarrow\quad 2\Phi(c)-1=0.5
\quad\Longrightarrow\quad c=\Phi^{-1}(0.75)\approx0.67449.
$$

所以正态噪声的绝对偏差中位数只有标准差的约 $0.67449$ 倍，要除以它才能回到标准差尺度。$1.4826$ 是正态模型下的校准系数，不是任意噪声分布都适用的常数。

#### 手算一次 MAD

取原始残差 $[-0.03,-0.02,-0.01,0,0.01,0.02,2]$。中位数是 $0$；绝对偏差排序后为 $[0,0.01,0.01,0.02,0.02,0.03,2]$，其中位数为 $0.02$，所以 $\hat\sigma\approx0.02965$。

这个例子只演示计算顺序：一个很大的数没有像平方和那样主导结果。七个样本不足以保证估计无偏，也不能由此知道其真实噪声一定是 $0.02965$。

### 1.3 3σ 剔除

保留 $|r_i| < 3\hat\sigma$ 的点。正态分布下，内点落在这个区间外的概率约 0.27%，所以阈值 3 不会误删太多内点。

本节的 $r_i$ 指**原始残差**，与讲义里默认白化后的 $r_i$ 记法不同。等价判断是 $|r_i/\hat\sigma|<3$。上面的手算例子阈值约为 $0.08896$，前六个值保留，最后的 $2$ 被排除。

$0.27\%$ 的解释要求模型正确、零均值且噪声尺度准确；实际残差来自拟合，可能有偏、相关或尺度估计误差，因此不是硬剔除流程的无条件误删率。

---

## 2. 完整答案代码

```cpp
// 第 2 课课后作业参考答案
//
// 任务：带外点的曲线拟合 y = exp(a x^2 + b x + c)
//   1) 比较 No loss / Huber / Cauchy
//   2) 用 MAD 从残差估计 sigma，再重新白化
//   3) 用 3 sigma 阈值剔除外点，重新做普通最小二乘
//
// 编译后运行：./build/hw02_robust

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
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

enum class LossType { NONE, HUBER, CAUCHY };

struct Result {
    double abc[3];
    double cost;
};

Result Solve(LossType type,
             double sigma,
             const std::vector<double>& xs,
             const std::vector<double>& ys,
             const double init[3]) {
    double abc[3] = {init[0], init[1], init[2]};
    if (xs.size() != ys.size() || xs.size() < 3 ||
        !std::isfinite(sigma) || sigma <= 0.0) {
        throw std::invalid_argument("need matching data, at least 3 points and positive sigma");
    }

    ceres::Problem problem;
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
                new CurveResidual(xs[i], ys[i], sigma));

        ceres::LossFunction* loss = nullptr;
        if (type == LossType::HUBER) {
            loss = new ceres::HuberLoss(1.0);
        } else if (type == LossType::CAUCHY) {
            loss = new ceres::CauchyLoss(1.0);
        }

        problem.AddResidualBlock(cost, loss, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) throw std::runtime_error(summary.BriefReport());

    return {abc[0], abc[1], abc[2], summary.final_cost};
}

double Median(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of empty data");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 用残差的中位数绝对偏差（MAD）估计噪声 sigma。
// 1.4826 是正态分布下 MAD 与标准差的换算系数。
double EstimateSigma(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     const double abc[3]) {
    std::vector<double> residuals;
    residuals.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        const double pred =
            std::exp(abc[0] * xs[i] * xs[i] + abc[1] * xs[i] + abc[2]);
        residuals.push_back(ys[i] - pred);
    }

    const double med = Median(residuals);
    for (double& r : residuals) {
        r = std::abs(r - med);
    }
    return 1.4826 * Median(residuals);
}

Result SolveInliers(double sigma,
                    double threshold,
                    const std::vector<double>& xs,
                    const std::vector<double>& ys,
                    const double init[3]) {
    // 先做鲁棒拟合，避免被外点拉偏的模型误删内点。
    const Result rough = Solve(LossType::HUBER, sigma, xs, ys, init);

    std::vector<double> xi, yi;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double pred = std::exp(rough.abc[0] * xs[i] * xs[i] +
                                     rough.abc[1] * xs[i] + rough.abc[2]);
        if (std::abs(ys[i] - pred) < threshold * sigma) {
            xi.push_back(xs[i]);
            yi.push_back(ys[i]);
        }
    }

    std::cout << "[inlier] kept " << xi.size() << " / " << xs.size() << " points\n";
    return Solve(LossType::NONE, sigma, xi, yi, rough.abc);
}

void PrintRow(const char* name, const Result& r, const double truth[3]) {
    double err2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double d = r.abc[k] - truth[k];
        err2 += d * d;
    }
    std::cout << name << " abc=(" << r.abc[0] << ", " << r.abc[1] << ", "
              << r.abc[2] << ") |err|=" << std::sqrt(err2)
              << " cost=" << r.cost << "\n";
}

int main() {
    const double truth[3] = {1.0, 2.0, 0.5};
    const double init[3] = {0.8, 1.8, 0.2};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);

    std::vector<double> xs, ys;
    for (int i = 0; i < 100; ++i) {
        const double x = -1.0 + 2.0 * i / 99.0;
        double y = std::exp(truth[0] * x * x + truth[1] * x + truth[2]) + noise(rng);
        if (i % 10 == 0) {
            y += 0.4 * y;  // 外点
        }
        xs.push_back(x);
        ys.push_back(y);
    }

    std::cout << "truth abc=(" << truth[0] << ", " << truth[1] << ", "
              << truth[2] << ")\n\n";

    const Result none = Solve(LossType::NONE, 0.02, xs, ys, init);
    const Result huber = Solve(LossType::HUBER, 0.02, xs, ys, init);
    const Result cauchy = Solve(LossType::CAUCHY, 0.02, xs, ys, init);

    PrintRow("[none  ]", none, truth);
    PrintRow("[huber ]", huber, truth);
    PrintRow("[cauchy]", cauchy, truth);

    // 假设 sigma 未知：粗尺度 -> 鲁棒拟合 -> MAD -> 重新白化。
    const double initial_scale = std::max(1e-8, EstimateSigma(xs, ys, none.abc));
    const Result preliminary = Solve(LossType::CAUCHY, initial_scale, xs, ys, none.abc);
    const double sigma_hat = std::max(1e-8, EstimateSigma(xs, ys, preliminary.abc));
    std::cout << "\nMAD sigma_hat = " << sigma_hat << " (true 0.02)\n";

    const Result mad = Solve(LossType::HUBER, sigma_hat, xs, ys, preliminary.abc);
    PrintRow("[huber+MAD]", mad, truth);

    // 用 3 sigma 阈值剔除外点后，再做普通最小二乘
    const Result inlier = SolveInliers(sigma_hat, 3.0, xs, ys, mad.abc);
    PrintRow("[3sigma LS]", inlier, truth);

    return 0;
}
```

---

## 3. 预期输出形态

```
truth abc=(1, 2, 0.5)

[none  ] abc=(...) |err|=较大 cost=较大
[huber ] abc=(...) |err|=较小 cost=较小
[cauchy] abc=(...) |err|=较小 cost=较小
MAD sigma_hat = 0.02 左右 (true 0.02)
[huber+MAD] abc=(...) |err|=较小
[inlier] kept 90 / 100 points
[3sigma LS] abc=(...) |err|=最小或接近
```

不要追求和这里完全一样的数字。判断标准：

1. `none` 的 `|err|` 明显大于 `huber`；
2. `sigma_hat` 落在 `0.015 ~ 0.03`；
3. 内点保留数约 90。

> 如果 `sigma_hat` 偏离很大，检查是不是用了白化后的残差去估 sigma。MAD 要用**原始残差** $y - pred$。

---

## 4. 代码讲解

### 4.1 `Solve` 的通用化

把 `sigma` 和 `LossType` 作为参数传进去，就能复用同一套残差代码。这是工程上的常见写法：

```cpp
Result Solve(LossType type, double sigma, ...);
```

真实系统里，`sigma` 往往来自传感器标定，`LossType` 来自配置。

### 4.2 为什么不能只用 `none` 的残差估 sigma

普通最小二乘被外点拉偏后，内点残差也包含随 $x$ 变化的模型误差；减去中位数不能消除这种误差，因此直接 MAD 会高估噪声。本机该做法得到约 `0.154`，远大于真实 `0.02`。参考代码改成“粗尺度 → Cauchy 鲁棒拟合 → MAD → Huber”，得到约 `0.0237`。

更严谨的做法是迭代：

```
拟合 -> 估 sigma -> 重新白化拟合 -> 再估 sigma -> ...
```

可迭代到模型与尺度稳定，但没有普适的轮数保证。交替更新模型/尺度与 IRLS 有联系，却不等同于 IRLS；后者特指根据残差更新权重、反复解加权最小二乘子问题。

### 4.3 硬剔除的实现细节

先用粗拟合算残差，再用 $|r| < 3\sigma$ 建内点集，最后在内点集上重新拟合。

两个注意点：

- 阈值 $3\sigma$ 里的 $\sigma$ 必须是**原始残差尺度**，不是白化后的；
- 如果第一轮拟合被外点拉得太偏，内点也会被误删。工程上会先用鲁棒核做一轮，再剔除。

---

## 5. 思考题答案

### 5.1 Huber 和 Cauchy 为什么不同？

Huber 的影响函数在大残差处是常数（线性代价），Cauchy 的影响函数按 $1/r$ 衰减（对数代价）。外点很大时，Cauchy 给它的权重更小，所以抑制更强；但代价函数更非凸，更容易陷入局部极小。

### 5.2 外点比例升到 40% 会怎样？

不能把 MAD 或位置估计的 50% 崩溃点直接移植到非线性 Huber 回归。其抗污染能力取决于设计点、杠杆值、污染方向、尺度估计和初值；40% 时可能失败，也可能仍有有效解。Cauchy 对大残差抑制更强，但更非凸。高外点率下可结合 RANSAC、几何验证和前端数据关联，不过这些方法也不是无条件成功的保证。

### 5.3 硬剔除 vs 鲁棒核，哪个适合在线 SLAM？

在线 SLAM 通常用**鲁棒核**：

- 代价函数连续可微，优化稳定；
- 不需要离散的“删/留”决策，避免信息矩阵突变；
- 可以渐进降权，回环误检时不会直接把约束删掉。

硬剔除更适合离线 BA 或后端清理，因为删点会改变问题结构，且阈值难以在线设定。

### 5.4 sigma 随观测变化怎么办？

MAD 假设噪声同方差。如果噪声随深度/距离变化：

- 用参数化的噪声模型 $\sigma(z)$，先验标定；
- 用 Student-t 分布或自适应核；
- 在 BA 中直接使用每个观测的协方差 $\Sigma_i$，而不是一个全局 $\sigma$。

---

## 6. 常见错误

1. **用白化后的残差估 sigma**：会把 `sigma_hat` 估成 1 左右。
2. **忘记 `1.4826`**：`sigma_hat` 会偏小约 32%。
3. **剔除阈值用了白化残差**：$|r/\sigma| < 3$ 和 $|r| < 3\sigma$ 是等价的，但别混着写。
4. **在剔除后的子集上仍用原来的 `sigma` 白化**：这没问题，`sigma` 是观测噪声，不随样本数变。
5. **Huber/Cauchy 的阈值设成 0.1**：内点也会被降权，结果会变差。
