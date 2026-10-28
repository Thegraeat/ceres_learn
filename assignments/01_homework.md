# 第 1 课 课后编程作业

> 对应讲义：`docs/01_ceres_basics.md`
> 对应答案：`assignments/01_solution.md`（做完再看）
> 建议先自己新建 `my_hw01.cpp`，不要直接改答案文件。

---

## 任务：直线拟合 $y = m x + b$

数据生成方式（固定随机种子 7）：

```cpp
const double m_true = 2.0, b_true = -0.5;
const int n = 200;
std::mt19937 rng(7);
std::normal_distribution<double> noise(0.0, 0.05);
for (int i = 0; i < n; ++i) {
    double x = -1.0 + 2.0 * i / (n - 1);
    double y = m_true * x + b_true + noise(rng);
}
```

### 1. 联合估计 $m$ 和 $b$（自动微分）

要求：

- 把 $m$ 和 $b$ 写成**两个独立的 1 维参数块**；
- 残差为 $r_i = y_i - (m x_i + b)$；
- 用 `AutoDiffCostFunction`；
- 输出 $m$、$b$、`summary.final_cost`。

### 2. 固定 $b$，只估计 $m$

要求：

- 初值把 $b$ 设为真值 `-0.5`；
- 用 `problem.SetParameterBlockConstant(&b)` 固定它；
- 只优化 $m$，输出结果。

### 3. 用数值微分重做任务 1

要求：

- 写一个**非模板**的 functor；
- 用 `NumericDiffCostFunction<Functor, ceres::CENTRAL, 1, 1, 1>`；
- 输出 $m$、$b$，并与自动微分的结果比较绝对误差。

### 4. 对照闭式解

对 $r_i = y_i - (m x_i + b)$，最小二乘闭式解是：

$$
\hat m =
\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}
     {\sum_i (x_i-\bar x)^2},
\qquad
\hat b = \bar y - \hat m \bar x
$$

把闭式解也打印出来，验证 Ceres 的结果与它一致。

---

## 验收标准

| 检查项 | 标准 |
|---|---|
| 任务 1 的 $m$、$b$ | 与闭式解相差 < `1e-6` |
| 任务 2 的 $m$ | 与真值 `2.0` 相差 < `0.05` |
| 任务 3 的 $m$、$b$ | 与任务 1 相差 < `1e-4` |
| 程序 | 能编译、能运行、输出四项结果 |

> 任务 2 的 $m$ 不会和任务 1 完全相同。固定 $b$ 改变了问题，$m$ 的估计会略有变化，这是正常的。

---

## 提示

1. `AutoDiffCostFunction<F, 1, 1, 1>` 的两个 `1` 分别对应 $m$ 和 $b$。
2. `AddResidualBlock(cost, nullptr, &m, &b)` 的参数顺序必须和模板一致。
3. 固定参数要在**添加完残差块之后**调用 `SetParameterBlockConstant`，两种顺序其实都行，但先加残差更直观。
4. 数值微分的 functor 参数是 `const double* const`，不要写成模板。
5. 计时可以用：

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
ceres::Solve(options, &problem, &summary);
auto t1 = std::chrono::high_resolution_clock::now();
double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
```

---

## 思考题（不用写代码，但要能回答）

1. 为什么把 $b$ 固定成真值后，$m$ 的估计几乎不变？
2. 这个问题的 $H$ 是 $2\times 2$ 的。写出 $H$ 和 $g$ 的表达式。
3. 如果所有 $x_i$ 都相同，$H$ 会怎样？Ceres 会报什么？
4. 数值微分在这个问题上比自动微分慢多少？为什么小问题上差距可能不明显？
