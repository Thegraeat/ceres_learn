# 第 7 课 课后编程作业

> 对应讲义：`docs/07_engineering.md`
> 对应答案：`assignments/07_solution.md`（做完再看）
> 可编译答案：`assignments/hw07_engineering.cpp`

---

## 任务：把工程工具用一遍

以 `base07_engineering.cpp` 为起点。

### 1. 协方差：Ceres vs 理论

曲线拟合模型：

$$
y_i = \exp(a x_i^2 + b x_i + c) + \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

未白化残差的雅可比：

$$
J_i =
\begin{bmatrix}
-x_i^2 e^{u_i} & -x_i e^{u_i} & -e^{u_i}
\end{bmatrix},
\qquad u_i = a x_i^2 + b x_i + c
$$

理论协方差（高斯-牛顿近似）：

$$
\Sigma \approx \sigma^2\left(\sum_i J_i^\top J_i\right)^{-1}
$$

要求：

- 用 Eigen 构建 $3\times 3$ 的 $H = \sum J_i^\top J_i$；
- 计算 $\Sigma = \sigma^2 H^{-1}$；
- 用 `ceres::Covariance` 估计协方差；
- 输出两者的对角线标准差，比较相对误差。

验收：相对误差 < `1e-6`（同一线性化点、同一问题）。

### 2. 梯度检查

用 `ceres::GradientChecker` 检查 `CurveResidual` 的自动微分雅可比。

要求：

- 参数块是 `abc[3]`，manifolds 传 `{nullptr}`；
- `Probe` 的 `relative_precision` 用 `1e-6`；
- 输出 `PASS/FAIL` 和 `maximum_relative_error`。

### 3. 性能对比

把点数从 `1000` 改到 `10000`，比较：

| 求解器 | 耗时 |
|---|---|
| `DENSE_QR` | ? |
| `SPARSE_NORMAL_CHOLESKY` | ? |

要求：

- 记录耗时；
- 验证两种求解器的解一致；
- 解释为什么 3 个参数的问题差距不大。

### 4. 设计题（不用写代码）

按讲义第 6 节，写出你的 VIO 后端模块图，至少包含：

- 状态定义；
- 三类因子（视觉、IMU、先验）；
- 边缘化策略；
- 评估指标；
- 你认为最容易出错的两个地方。

---

## 验收标准

| 检查项 | 标准 |
|---|---|
| 协方差 | Ceres 与理论标准差相对误差 < `1e-6` |
| 梯度检查 | `PASS`，最大相对误差 < `1e-6` |
| 性能 | 两种求解器解一致，耗时合理 |
| 设计题 | 模块图完整，能说清数据流 |

---

## 提示

1. 理论协方差里的 $\sigma^2$ 是**观测噪声方差**，不是白化后的 1。
2. Ceres 的协方差是在当前参数处、对**白化残差**计算的，所以等价于 $\sigma^2 H^{-1}$。
3. 避免显式 `inverse()`，用 LDLT/SVD 等分解求解；行列式受尺度影响，不能用它代替秩/条件数判断。若需诊断，用奇异值或特征值谱。
4. `GradientChecker` 不接管 cost 的所有权，记得 `delete`。
5. 可以复用同一个 `Problem`，但每次必须恢复相同参数初值、配置和数据；分别构建更不易误操作。不要让第二次从第一次优化结果出发，再把耗时差当作求解器优势。

---

## 提交

把三组输出贴回来：

- Ceres 标准差 vs 理论标准差；
- 梯度检查结果；
- 两种求解器的耗时和解差。

再附上设计题的模块图。
