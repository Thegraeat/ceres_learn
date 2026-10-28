# 第 7 课：工程实践与综合项目

> 对应代码：`base07_engineering.cpp`
> 目标：把前六课的数学落到工程上——怎么验证雅可比、怎么估协方差、怎么调求解器、怎么评估、怎么组织一个 VIO 后端。
> 前置：第 1~6 课。

---

## 1. 调试 Ceres 问题的顺序

遇到“优化不收敛 / 收敛到错解”，按这个顺序查，不要一上来调 `Solver::Options`：

```
1. 残差几何对不对？        -> 无噪声一致数据的真值残差应接近 0
2. 雅可比对不对？          -> GradientChecker / 有限差分
3. 参数化对不对？          -> 四元数归一化、Manifold、固定 gauge
4. 权重对不对？            -> 白化、协方差、鲁棒核阈值
5. 初值够不够好？          -> 多初值、前端、RANSAC
6. 求解器合不合适？        -> 线性求解器、ordering、迭代上限
```

前五步占 90% 的问题。求解器选项通常不是根因。

---

## 2. 雅可比检查

### 2.1 原理

有限差分（中心）：

$$
J^{num}_{ij} \approx
\frac{r_i(x + h e_j) - r_i(x - h e_j)}{2h}
$$

和自动微分/解析雅可比比较，相对误差应在 `1e-6` 量级。

### 2.2 Ceres 的 GradientChecker

```cpp
ceres::CostFunction* cost =
    new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
        new CurveResidual(0.3, 1.0, 1.0));

std::vector<const ceres::Manifold*> manifolds = {nullptr};
ceres::NumericDiffOptions ndo;

ceres::GradientChecker checker(cost, &manifolds, ndo);
double* params[1] = {abc};
ceres::GradientChecker::ProbeResults results;

bool ok = checker.Probe(params, 1e-6, &results);
```

- `manifolds` 的个数等于参数块个数；`nullptr` 表示欧氏空间；
- `Probe` 返回 `false` 表示雅可比不匹配；
- `results.maximum_relative_error` 给出最大相对误差；
- `GradientChecker` **不接管** `cost` 的所有权。

### 2.3 什么时候必须检查

- 手写解析雅可比之后；
- 新增一种残差之后；
- 改了坐标系/扰动定义之后；
- 优化“能收敛但结果不对”时。

自动微分可以避免手推导数错误，但无法验证几何模型、坐标系、白化和符号是否正确。`GradientChecker` 检查的是“所写函数的导数是否匹配数值差分”，即使通过，也要用已知几何与无噪数据独立验证模型。带噪观测在真值处一般不为零。

---

## 3. 协方差估计

### 3.1 数学

高斯-牛顿在最优点的近似：

$$
\Sigma \approx \left(J^\top J\right)^{-1}
$$

如果残差已经白化。参数标准差：

$$
\sigma_i = \sqrt{\Sigma_{ii}}
$$

相关系数：

$$
\rho_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}
$$

### 3.2 Ceres API

```cpp
ceres::Covariance::Options options;
options.algorithm_type = ceres::DENSE_SVD;
options.min_reciprocal_condition_number = 1e-14;

ceres::Covariance covariance(options);
std::vector<std::pair<const double*, const double*>> blocks;
blocks.push_back(std::make_pair(abc, abc));

double cov[9];   // row-major
if (!covariance.Compute(blocks, &problem) ||
    !covariance.GetCovarianceBlock(abc, abc, cov)) {
    return 1;
}
```

### 3.3 注意事项

1. **鲁棒核改变协方差的含义**。`Covariance::Options::apply_loss_function` 默认 `true`；返回的是应用核函数修正后的局部雅可比近似，不是一般 M 估计器的完整统计协方差。设为 `false` 只关闭该修正，不会消除外点或自动恢复“真实协方差”。可在可信内点与正确噪声模型上重新评估，或研究相应的稳健统计估计。
2. **固定参数块的协方差为 0**。固定 gauge 后，被固定变量的不确定性不可估。
3. **协方差依赖线性化点**。强非线性问题在远处估的协方差没有意义。
4. **`DENSE_SVD` 适合小问题**；大问题用 `SPARSE_QR`，但要求 $H$ 满秩。
5. **协方差不是置信区间**。它只反映局部二次近似。
6. **流形块区分 ambient 与 tangent 协方差**。`GetCovarianceBlockInTangentSpace` 可获取局部协方差；Ceres 四元数切空间是半角，换成物理小旋转角协方差还需乘 4，并明确左右扰动与坐标系。

### 3.4 NEES

评估协方差是否可信：

$$
\mathrm{NEES} = e^\top \Sigma^{-1} e
$$

$e$ 是估计误差。如果滤波器/优化一致，$\mathrm{NEES}$ 的期望等于状态维度。NEES 远大于维度，说明协方差过小（过度自信）；远小于维度，说明协方差过大（保守）。

---

## 4. 求解器调优

### 4.1 线性求解器

| 求解器 | 适用 | 备注 |
|---|---|---|
| `DENSE_QR` | 参数 < 500 | 稳，慢 |
| `DENSE_NORMAL_CHOLESKY` | 稠密、残差多 | 需要 H 正定 |
| `SPARSE_NORMAL_CHOLESKY` | 位姿图 | H 稀疏 |
| `DENSE_SCHUR` | 小 BA | 显式 Schur |
| `SPARSE_SCHUR` | 大 BA | 需要 SuiteSparse |
| `ITERATIVE_SCHUR` | 大 BA | 内置，无需 SuiteSparse |
| `CGNR` | 超大规模 | 迭代 |

### 4.2 参数分组

BA 里要让 Ceres 知道哪些是位姿、哪些是点：

```cpp
auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
for (auto& point : points) {
    ordering->AddElementToGroup(point.data(), 0);
}
for (auto& pose : poses) {
    ordering->AddElementToGroup(pose.q, 1);
    ordering->AddElementToGroup(pose.t, 1);
}
options.linear_solver_ordering = ordering;
```

`ITERATIVE_SCHUR` 会消去 group 0 的变量（点），只解 group 1（位姿）。

group 0 必须是独立集：同一个残差不能连接两个不同的待消元点。仅创建空 ordering 而不添加参数并不能完成分组；分开存储的 `q`、`t` 都要加入位姿组，且必须使用 Problem 内真实参数地址。

### 4.3 常用开关

```cpp
options.max_num_iterations = 50;
options.function_tolerance = 1e-6;
options.gradient_tolerance = 1e-10;
options.parameter_tolerance = 1e-8;
options.num_threads = 4;
options.use_nonmonotonic_steps = true;   // 允许代价偶尔上升
options.max_linear_solver_iterations = 500;
```

### 4.4 性能判断

- `Summary::iterations`：迭代次数异常多，通常是初值/尺度问题；
- `Summary::final_cost`：和残差数量、噪声水平对比；
- `FullReport()`：看线性求解器耗时、雅可比评估耗时；
- 稀疏分解成本由图连接、排序、填充和后端决定，不能只按变量数承诺线性或 $N^{1.5}$。计时时应固定数据与初值、预热并重复运行，同时检查解是否可用以及实际使用的线性求解器。

---

## 5. 评估指标

### 5.1 重投影误差

$$
\mathrm{RMSE}_{reproj} =
\sqrt{\frac{1}{|\mathcal{O}|}\sum_{(c,k)}
\left\|\pi(T_c^{-1}X_k)-\mathbf{uv}_{ck}\right\|^2}
$$

该定义按二维观测范数计算，真值处的 RMS 尺度是 $\sqrt{2}\,\sigma_{\mathrm{px}}$；若希望与单坐标噪声直接对比，可改用分母 $2\lvert O\rvert$。训练数据上的拟合残差还受自由度影响，不应把 $\mathrm{RMSE} = \sigma_{\mathrm{px}}$ 当作普适验收等式。

### 5.2 ATE

$$
\mathrm{ATE} = \sqrt{\frac{1}{N}\sum_i
\left\|t_i - t_i^{gt}\right\|^2}
$$

仅在确有坐标系 gauge 时对齐。米制轨迹通常用 SE(3) 对齐、不允许缩放；缺少尺度观测的单目轨迹可用 Sim(3) 对齐。已经锚定到真值坐标系和尺度的合成示例可直接比较。不要用 Sim(3) 对齐掩盖米制系统的尺度误差，也不要把时变漂移当成单一全局变换消掉。

### 5.3 RPE

相对位姿误差：

$$
E_{ij} = \left(T_i^{-1}T_j\right)^{-1}
\left(T_i^{gt\,-1}T_j^{gt}\right)
$$

$$
\mathrm{RPE}_{trans} = \|E_{ij}.\mathbf t\|,
\qquad
\mathrm{RPE}_{rot} = \angle(E_{ij}.\mathbf R)
$$

RPE 对局部漂移更敏感，ATE 对全局一致性更敏感。

### 5.4 其他

- 零偏误差（VIO）；
- 协方差一致性（NEES）；
- 运行时间 / 内存；
- 失败率（跟踪丢失、优化失败）。

---

## 6. 综合项目：小型 VIO 后端

目标：把前六课拼成一个可运行的视觉惯性 BA。

### 6.1 输入

- 关键帧位姿初值 $T_i$、速度 $v_i$、零偏 $b_g$, $b_a$；
- 每帧的 2D 观测 $(u,v)$ 和对应的 3D 点 $X_k$；
- 相邻关键帧之间的 IMU 预积分量 $\Delta R$, $\Delta v$, $\Delta p$ 与协方差；
- 明确 gauge 与初始条件：典型充分激励 VIO 的 gauge 是全局平移和 yaw；固定完整首帧姿态还额外给定 roll/pitch，固定速度则另加已知初始条件。

### 6.2 模块划分

```
vio_backend/
├── types.h          // Pose, State, Observation, Preintegration
├── imu.h/.cpp       // 预积分、零偏雅可比、协方差传播
├── factors.h/.cpp   // ReprojectionError, ImuError, PriorCost
├── backend.h/.cpp   // 构建 Problem、求解、边缘化
├── eval.h/.cpp      // ATE/RPE/重投影
└── main.cpp         // 合成数据或 EuRoC 读取
```

### 6.3 分阶段实现

| 阶段 | 内容 | 验收 |
|---|---|---|
| 1 | 视觉-only BA（第 4 课） | 明确尺度规范，重投影 RMSE 与二维噪声/自由度相符 |
| 2 | IMU-only（第 5 课） | 位姿/速度误差下降 |
| 3 | 视觉 + IMU 联合 | 无视觉时短时传播；有足够运动激励与观测时检验零偏可观性和收敛 |
| 4 | 滑动窗口 + 边缘化（第 6 课） | 边缘化前后线性问题一致 |
| 5 | 评估与调优 | ATE/RPE/NEES 合理 |

### 6.4 关键设计决策

1. **状态顺序**：`[q, p, v, bg, ba]`，q 用四元数 + Manifold；
2. **因子图**：视觉因子连 $(T_i, X_k)$，IMU 因子连 $(T_i, v_i, b_i, T_j, v_j, b_j)$；
3. **零偏**：先固定，跑通后加一阶修正；
4. **边缘化**：先做视觉-only 的 Schur 补，再加 IMU 先验；
5. **协方差**：先用教学权重，随后区分 2×2 视觉、9×9 运动预积分和含零偏随机游走的 15×15 联合模型；
6. **FEJ**：冻结边缘化先验是必要的历史线性化处理，但不等于完整 FEJ；需检查活动因子与先验的不可观测零空间是否一致。

### 6.5 验收清单

- [ ] 无噪声一致数据的真值残差 ≈ 0；带噪声时残差与噪声统计相符；
- [ ] `GradientChecker` 通过；
- [ ] 明确按观测还是按分量定义重投影 RMSE；
- [ ] 无噪声可辨识问题从合理初值恢复真值；带噪声时报告估计误差；
- [ ] 去掉视觉，纯 IMU 短时可用；
- [ ] 加外点，Huber 能压住；
- [ ] 边缘化前后，线性问题解一致；
- [ ] 用多次试验评估 ATE/RPE 的统计趋势，不要求单次随机实现随噪声严格单调；
- [ ] 固定不同 gauge，最优代价不变。

---

## 7. 工程清单

**建模**

- 坐标系定义写进注释：谁是 world，谁是 body，谁到谁；
- 四元数顺序写死：Ceres `wxyz`，Eigen `xyzw`；
- 残差维度、参数块维度、顺序，三处必须一致；
- 每个残差都白化；
- 每个鲁棒核阈值都和噪声尺度对齐。

**数值**

- 四元数归一化；
- 避免除零（$z_c$、$\lvert\omega\rvert$、$\sin\theta$）；
- 大角度用完整 Log map，小角度可用 $2\,\mathrm{vec}$；
- 固定 gauge；
- 检查 `Summary::termination_type` 和 `final_cost`。

**性能**

- 大问题用稀疏求解器；
- BA 用 Schur；
- 参数分组；
- 需要时开多线程；
- 先 profile，再优化。

**验证**

- 真值残差；
- 雅可比检查；
- 协方差/NEES；
- 消融实验（去因子、加噪声、换核）。

---

## 8. 练习

1. 用 `GradientChecker` 检查第 4 课的重投影误差雅可比。
2. 用 `Covariance` 估计曲线拟合的参数标准差，和理论值对比。
3. 把 `ITERATIVE_SCHUR` 用到第 4 课 BA，比较耗时。
4. 按第 6 节的结构，写出你自己的 VIO 后端模块图。

答案在 `assignments/07_solution.md`。
