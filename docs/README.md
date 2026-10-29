# Ceres 在 SLAM 中的应用：课程笔记

本目录是配套讲义，按「基础 → 曲线拟合 → 位姿图 → 视觉 BA → IMU 预积分 → 边缘化 → 工程实践」组织。
每份讲义对应根目录的一个可编译 `.cpp` 文件，公式推导、代码讲解和练习都在讲义里。

环境：**Ceres 2.1.0** / Eigen3 / CMake ≥ 3.10 / C++14。
当前构建：**Release**，GCC 9.4；7 个示例 + 7 个作业答案已全部编译通过。

---

## 课程索引

数学基础不熟时，先读[数学预备：怎样读懂这套教程的公式](00_math_primer.md)。其中用直线拟合解释数据、预测、残差和代价，再介绍矩阵、雅可比、坐标变换与协方差。

| 课 | 讲义 | 代码 | 课后作业 | 主题 |
|---|---|---|---|---|
| 1 | [`01_ceres_basics.md`](01_ceres_basics.md) | `base01.cpp` | [`01_homework`](../assignments/01_homework.md) | 五件套、最小闭环、三种求导、Options、Summary |
| 2 | [`02_curve_fitting.md`](02_curve_fitting.md) | `base02_curve_fitting.cpp` | [`02_homework`](../assignments/02_homework.md) | 白化、Huber/Cauchy、自动微分链式法则 |
| 3 | [`03_pose_graph.md`](03_pose_graph.md) | `base03_pose_graph.cpp` | [`03_homework`](../assignments/03_homework.md) | SE(3)、四元数、Manifold、稀疏 H、gauge freedom |
| 4 | [`04_ba.md`](04_ba.md) | `base04_ba.cpp` | [`04_homework`](../assignments/04_homework.md) | 针孔模型、重投影误差、位姿/点雅可比、Schur |
| 5 | [`05_imu_preintegration.md`](05_imu_preintegration.md) | `base05_imu.cpp` | [`05_homework`](../assignments/05_homework.md) | IMU 测量、预积分、零偏、协方差传播 |
| 6 | [`06_marginalization.md`](06_marginalization.md) | `base06_marginalization.cpp` | [`06_homework`](../assignments/06_homework.md) | 滑动窗口、Schur 补、先验、FEJ |
| 7 | [`07_engineering.md`](07_engineering.md) | `base07_engineering.cpp` | [`07_homework`](../assignments/07_homework.md) | GradientChecker、协方差、调优、综合项目 |

---

## 校验与补充

- [课程校验记录](REVIEW.md)：已修正问题、复现现象、教学简化与验证方式。
- [Ceres 建模与接口补充](08_ceres_modeling_notes.md)：白化矩阵、ambient/tangent 雅可比、鲁棒块阈值、所有权和协方差。

## 讲义覆盖范围 vs 代码实现范围

代码是“可运行的最小实现”，讲义是“完整的方法论”。两者差异如下，读的时候不要混淆：

| 课 | 示例代码实现了什么 | 讲义里讲、但代码没实现的部分 |
|---|---|---|
| 1 | 手写 / 自动 / 数值三种求导的最小闭环 | — |
| 2 | 100 点曲线拟合 + 10% 外点 + Huber | Cauchy、MAD、硬剔除（都在作业里） |
| 3 | 20 节点 SE(3) 位姿图 + 稀疏求解 | 严格 Log map、6×6 信息矩阵、鲁棒核（作业） |
| 4 | 5 相机 / 20 点合成 BA + 重投影误差 | `ITERATIVE_SCHUR` 参数分组、真实数据接入 |
| 5 | 固定零偏的 9 维预积分残差 | 零偏一阶修正、15 维协方差、静止初始化 |
| 6 | 3 变量线性 Schur 补 + 先验 | 非线性边缘化、FEJ 的具体实现 |
| 7 | GradientChecker、Covariance、求解器对比 | 完整 VIO 后端（只给模块设计） |

> 第 5、6、7 课的代码有意简化，是为了让你先把“残差怎么写、先验怎么构造”跑通；完整实现留给作业和综合项目。

---

## 建议的学习方式

1. **先读讲义**，先说清公式在算什么，再跟着中间步骤手推残差和雅可比；
2. **再看代码**，重点看 `operator()` 里残差怎么算、`AddResidualBlock` 怎么连参数；
3. **编译运行**，对照输出理解 `Summary`；
4. **做课后作业**，先自己写，再看 `assignments/` 里的答案。

不要跳课。第 3 课的位姿图是第 4 课 BA 的简化版，第 5 课的 IMU 因子又建立在第 3、4 课的 Manifold 和残差结构上。

### 数学内容怎么查

| 遇到的问题 | 对应位置 | 可以先跟着算的例子 |
|---|---|---|
| 符号、矩阵大小、梯度和雅可比有什么区别？ | [数学预备](00_math_primer.md)第 1～5 节 | 直线预测与 $x^2$ 的小增量 |
| 为什么要解 $H\Delta x=-g$，负号哪里来？ | [第 1 课](01_ceres_basics.md)第 3 节 | 从 $x=5$ 求到 $x=10$ |
| 为什么除以 $\sigma$，鲁棒核怎样减轻外点影响？ | [第 2 课](02_curve_fitting.md)第 2～4 节 | 白化残差 0.5 和 10 的代价比较 |
| 四元数怎样转成矩阵，Exp/Log 是什么？ | [第 3 课](03_pose_graph.md)第 1、2、4 节 | 绕 $z$ 轴转 $90^\circ$ |
| 像素预测和位姿雅可比怎样一步步得到？ | [第 4 课](04_ba.md)第 2～4 节 | 三维点投到 $(420,290)$ |
| 为什么静止也有加计读数，预积分怎样递推？ | [第 5 课](05_imu_preintegration.md)第 1～7 节 | 两步恒定比力积分、重力抵消 |
| 删除旧变量后为什么还能保留约束？ | [第 6 课](06_marginalization.md)第 2～4 节 | 消去 $x_2$ 后手算先验与新解 |
| 协方差、NEES、RMSE 分别说明什么？ | [第 7 课](07_engineering.md)第 2、3、5 节 | 重复测量、二维像素误差 |
| 相关噪声白化方向、接口维度怎么确认？ | [建模补充](08_ceres_modeling_notes.md)第 1～4、7 节 | 解三角方程白化二维误差 |

每个新公式先检查五件事：**目的、已知量与未知量、维度、单位、坐标系**。遇到“近似”还要确认它在什么条件下成立。作业解析继续提供闭式解、MAD 和非最优展开点的计算步骤，不必靠记结论完成练习。

---

## 编译

```bash
cd ~/Desktop/learn/ceres_learn

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 只构建某一课
cmake --build build --target base05_imu -j
```

运行：

```bash
./build/base01
./build/base02_curve_fitting
./build/base03_pose_graph
./build/base04_ba
./build/base05_imu
./build/base06_marginalization
./build/base07_engineering
```

作业答案的 target 是 `hw01_basics` ~ `hw07_engineering`，详见 [`assignments/README.md`](../assignments/README.md)。

---

## 符号约定

- 向量默认是列向量；
- $J$ 是残差对参数的雅可比，$H = J^\top J$，$g = J^\top r$；
- 四元数顺序：Ceres `QuaternionManifold` 用 $[w,x,y,z]$；Eigen 的 `coeffs()`/原始存储为 $[x,y,z,w]$，而 `Eigen::Quaterniond(w,x,y,z)` 的构造参数是标量在前；
- 位姿 $T = [R \mid t]$ 默认表示 body-to-world（除非残差里明确写 $T_{cw}$）；
- 残差默认已经**白化**（除以噪声 `sigma` 或乘 $\Sigma^{-1/2}$）；
- 鲁棒核的阈值作用在白化后的残差上。

---

## 学完之后

1. **读源码**：VINS-Mono 的 `integration_base`、ORB-SLAM3 的 `Optimizer`、GTSAM 的因子图；
2. **大规模 BA**：`ITERATIVE_SCHUR`、参数分组、Schur 补的实现细节；
3. **激光/点云**：ICP、NDT、点到面残差，和第 4 课的重投影误差是同一套结构；
4. **对比 GTSAM**：手写雅可比 vs 自动微分，流形 vs 李群；
5. **一致性**：FEJ、协方差、NEES、滑动窗口的边缘化策略。

---

## 当前进度

| 课 | 讲义 | 示例代码 | 作业题目 | 参考答案 |
|---|---|---|---|---|
| 1 Ceres 基础 | ✅ | ✅ `base01` | ✅ | ✅ `hw01` |
| 2 曲线拟合 | ✅ | ✅ `base02_curve_fitting` | ✅ | ✅ `hw02` |
| 3 位姿图 | ✅ | ✅ `base03_pose_graph` | ✅ | ✅ `hw03` |
| 4 视觉 BA | ✅ | ✅ `base04_ba` | ✅ | ✅ `hw04` |
| 5 IMU 预积分 | ✅ | ✅ `base05_imu` | ✅ | ✅ `hw05` |
| 6 滑动窗口 | ✅ | ✅ `base06_marginalization` | ✅ | ✅ `hw06` |
| 7 工程实践 | ✅ | ✅ `base07_engineering` | ✅ | ✅ `hw07` |

全部 14 个 target 已在 Release 下编译通过。
