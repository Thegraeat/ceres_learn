# Ceres 建模与接口补充

> 配合第 1～7 课使用。接口语义以本机 Ceres **2.1.0** 官方安装头文件为核验基准，不将它当成“最新版本”。

## 1. CostFunction 返回残差，不是平方代价

给 Ceres 的应是 $r(x)$，而不是 $r(x)^2$；求解器会计算 $\frac{1}{2}\rho(\|r\|^2)$。若在 functor 内先平方，实际优化的是四次误差，鲁棒阈值和噪声解释都会改变。

设有两个独立观测，用两个残差块与一个二维残差块在**无核**时可有相同代价；加入鲁棒核后通常不同：

$$
\rho(r_1^2)+\rho(r_2^2) \ne \rho(r_1^2+r_2^2).
$$

因此残差如何分块也是统计模型的一部分。BA 把同一个像素的 $(u,v)$ 放入同一块，表示对该次二维观测整体降权。

## 2. 一般协方差的白化方向

白化矩阵 $W$ 要满足 $W^\top W = \Sigma^{-1}$，残差为 $W e$。有两种常见计算方式：

- 协方差分解：$\Sigma = L L^\top$，$L$ 为下三角，则 $W e = L^{-1} e$，用三角求解，不显式求逆。
- 信息矩阵分解：$\Omega = \Sigma^{-1} = L L^\top$，则 $W = L^\top$，不是 $L$。

下面假设 `sigma` 为有效的固定 $2\times 2$ 观测协方差：

```cpp
Eigen::LLT<Eigen::Matrix2d> factor(sigma);
if (factor.info() != Eigen::Success) {
    return 1;
}
const Eigen::Matrix2d whitening =
    factor.matrixL().solve(Eigen::Matrix2d::Identity());
```

白化同时作用于残差与雅可比：$r = W e$、$J = W J_e$。如果使用自动微分，functor 中的矩阵乘法保留 `T` 类型，导数会一起传播。协方差若随待优化状态变化，还要决定是否对权重求导，并考虑概率模型中的归一化项，不能沿用“固定协方差”的结论。

## 3. 解析雅可比：ambient 与 tangent 不要混淆

对于 `SizedCostFunction<2, 4, 3, 3>`：

- `Evaluate` 中三个雅可比块的大小分别是 $2\times 4$、$2\times 3$、$2\times 3$，按 **row-major** 填写。
- 即使四元数绑定了 `QuaternionManifold`，第一个块仍要返回 $2\times 4$ 的 ambient 雅可比。
- Ceres 会计算 $J_{local} = J_{ambient}$ `PlusJacobian(q)`，得到 $2\times 3$ 的局部旋转雅可比。
- `jacobians == nullptr` 时只计算残差；某个 `jacobians[index] == nullptr` 时不能写该块。

不能把手推的 $2\times 3$ 李代数雅可比直接填入要求 $2\times 4$ 的缓冲区。还要区分物理小旋转 $\delta\theta$ 与 Ceres 半角增量 $\delta$：$\delta\theta = 2\delta$。

`GradientChecker` 传入的 manifold 数量必须等于参数块数量。对于本课程重投影因子，可以使用：

```cpp
ceres::QuaternionManifold quaternion_manifold;
std::vector<const ceres::Manifold*> manifolds = {
    &quaternion_manifold, nullptr, nullptr};
ceres::NumericDiffOptions numeric_options;
ceres::GradientChecker checker(cost, &manifolds, numeric_options);
double const* parameters[] = {pose.q, pose.t, point.data()};
ceres::GradientChecker::ProbeResults results;
const bool valid = checker.Probe(parameters, 1e-6, &results);
```

这里假设 `cost` 是重投影因子的 `CostFunction*`。检查器不拥有 cost/manifold，它们必须在检查期间有效。Ceres 会在比较时乘流形雅可比，但通过检查不代表物理模型正确。还可以独立用 `Plus(q, ±h e_j)` 做切空间中心差分，核验左/右扰动与因子 2；在零导数附近也要观察绝对误差。

## 4. 鲁棒阈值与噪声维度

白化后 $d$ 维独立高斯残差在真值处满足 $\|r\|^2 \sim \chi^2_d$。因此 Huber 阈值比较的是 $\|r\|$，其典型尺度随维度变化：

- 一维残差的 $\delta=1$ 是按绝对残差判断。
- 二维重投影的 $\delta=1$ 是按二维范数判断，不是“每个坐标都小于一个 sigma”。
- 取二维 $\chi^2$ 的 95% 分位点约 `5.991` 时，范数阈值约为 `2.448`；这只是一个统计尺度参考，并非强制的 Huber 调参规则。

核函数只是降权，不能保证发现所有错误关联，也不等于剔除外点。阈值缩小会同时降低一部分正确观测的权重。不同核、不同白化尺度、不同观测子集的 `final_cost` 一般不可直接横向判优。

若用 IRLS 解释核函数，常见近似权重是 $\sqrt{\rho'(s)}$ 乘在残差/雅可比上；不要把这等同于所有情况下 Ceres 内部完整的二阶修正。

## 5. 生命周期和所有权

`Problem` 默认拥有 cost、loss、manifold，但**不拥有参数数组**。`AutoDiffCostFunction` 默认还拥有其 functor；自行改变 ownership 后必须重新分配好释放责任。

同一个 Problem 内可以复用一个 loss 指针，默认销毁时只释放一次。不要把同一个裸指针交给两个都接管所有权的 Problem；它们会分别试图销毁。使用 `std::unique_ptr` 管理对象时，也不能在未移交/取消接管的情况下又让 Problem 拥有它。

构建 Problem 前先分配好状态容器。之后不要 `push_back` 导致扩容，也不要把参数向量整体替换成另一个内存地址。滑动窗口移位时必须维护参数地址映射及先验的参数顺序，不能只把数组内容移动完就继续复用旧指针。

## 6. 秩、gauge 与协方差

LM 阻尼使某次线性子问题可解，**并不让原始问题可观**。`CONVERGENCE` 也不能证明参数唯一。

| 问题 | 典型全局 gauge / 额外条件 |
|---|---|
| 连通 SE(3) 相对位姿图 | 3 平移 + 3 旋转 |
| 已标定单目 BA | 3 平移 + 3 旋转 + 1 尺度 |
| 重力固定、充分激励的典型 VIO | 3 平移 + 1 yaw；退化运动可能增加不可观测方向 |
| 本课程纯 IMU 链 | 还需给定初始姿态、位置、速度，不能套用完整 VIO 的可观测性结论 |

固定 gauge 后的协方差是在选定规范下的局部/条件性不确定性，不包含被视为精确已知的锚点误差。需要相对位姿不确定性时，应同时考虑交叉协方差，而不只是分别取各个块的对角线。

`GetCovarianceBlock` 给出 ambient 协方差；`GetCovarianceBlockInTangentSpace` 给出局部协方差。对 Ceres 四元数半角切空间，有 $\Sigma_\theta \approx 4\Sigma_\delta$；改变左右扰动后还需相应旋转坐标变换。统计评估如 NEES 必须用同一坐标、同一 gauge 和同一误差维度，单次 NEES 也不应被当成严格的期望值等式。

## 7. 从 Ceres 提取边缘化线性系统

本课程第 6 课手工组装 $H,g$，没有实现通用边缘化模块。扩展时可用 `Problem::Evaluate`，但必须显式设置参数块顺序、残差块顺序和 `apply_loss_function`，不能依赖默认隐式顺序。

- `CostFunction::Evaluate` 的雅可比是 ambient 的；`Problem::Evaluate` 已应用 manifold，相关参数块贡献 tangent 列，二者不能再重复乘 `PlusJacobian`。
- 仅对“要吸收到先验的旧因子集合”组装线性系统。若把某个因子收入先验，再在活动问题里保留它，就重复计数。
- 准备残差时要明确是否应用鲁棒核修正；核权重被冻结后，先验只代表当时的局部近似。
- 信息矩阵来自测量模型，不要把 LM 临时阻尼当成真实观测信息永久加入先验。
- 正定情形使用分解求解；含零空间时用秩揭示分解/截断，并检查梯度是否与信息矩阵的值域一致。

比起构造正规方程，直接对平方根雅可比做 QR 消元可避免条件数平方。对于非线性流形，保存 $A,c,x_{lin}$ 并用一致的 `Minus` 表示局部位移；固定线性先验本身不等于实现 FEJ。

## 8. 版本与官方核对入口

本机使用 Ceres 2.1.0、EigenSparse 和 C++ 线程后端，没有启用 SuiteSparse/OpenMP。这也直接说明“稀疏求解必须有 SuiteSparse”“多线程必须有 OpenMP”都不成立。升级依赖时以目标库导出的编译标准和实际可用后端为准。

本次核验所用的官方安装头文件位于 `/usr/local/include/ceres/`：

| 文件 | 核对内容 |
|---|---|
| `version.h`、`internal/config.h` | 版本、稀疏后端、线程后端 |
| `manifold.h` | 四元数 Plus/Minus、半角约定、Eigen 内存顺序 |
| `cost_function.h` | ambient 雅可比维度、row-major、空指针约定 |
| `problem.h` | ownership、Evaluate 顺序和局部雅可比、移除参数的语义 |
| `covariance.h` | loss 选项、秩亏、ambient/tangent 协方差 |
| `gradient_checker.h`、`solver.h` | 梯度检查、停止条件、结果可用性 |

官方文档入口：[建模](https://ceres-solver.org/nnls_modeling.html)、[求解](https://ceres-solver.org/nnls_solving.html)、[协方差](https://ceres-solver.org/nnls_covariance.html)。这些网页会随版本更新；本次网络读取受限，未声称在线最新版已核验，具体语义以上述本机 2.1.0 头文件为准。
