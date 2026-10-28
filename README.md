# Ceres 在 SLAM 中的应用

一套从 Ceres 基础到视觉惯性后端的完整课程，共 **7 课**，每课包含：

- 一份带公式推导的讲义（`docs/`）
- 一个可编译的示例（`base0X_*.cpp`）
- 一套课后编程作业 + 参考答案（`assignments/`）

核验环境：**Ceres 2.1.0** / Eigen3 / CMake ≥ 3.10 / C++14。CMake 要求 Ceres ≥ 2.1；升级依赖时以依赖导出的语言标准要求为准。
当前构建：**Release**，GCC 9.4，14 个 target（7 示例 + 7 作业答案）已全部编译通过。

---

## 项目结构

```
ceres_learn/
├── CMakeLists.txt
├── README.md
├── base01.cpp                     # 第 1 课：Ceres 最小闭环
├── base02_curve_fitting.cpp       # 第 2 课：曲线拟合 + 鲁棒核
├── base03_pose_graph.cpp          # 第 3 课：SE(3) 位姿图
├── base04_ba.cpp                  # 第 4 课：视觉 BA
├── base05_imu.cpp                 # 第 5 课：IMU 预积分
├── base06_marginalization.cpp     # 第 6 课：Schur 补与先验
├── base07_engineering.cpp         # 第 7 课：工程工具
├── docs/
│   ├── README.md                  # 讲义总索引
│   ├── 01_ceres_basics.md
│   ├── 02_curve_fitting.md
│   ├── 03_pose_graph.md
│   ├── 04_ba.md
│   ├── 05_imu_preintegration.md
│   ├── 06_marginalization.md
│   └── 07_engineering.md
├── assignments/
│   ├── README.md                  # 作业总览
│   ├── 01_homework.md / 01_solution.md / hw01_basics.cpp
│   ├── ...
│   └── 07_homework.md / 07_solution.md / hw07_engineering.cpp
└── build/                         # CMake 构建目录（可删除重建）
```

---

## 课程地图

| 课 | 主题 | 讲义 | 代码 | 作业 |
|---|---|---|---|---|
| 1 | Ceres 基本使用：五件套、三种求导、Summary | [01](docs/01_ceres_basics.md) | `base01.cpp` | [01](assignments/01_homework.md) |
| 2 | 曲线拟合：白化、Huber/Cauchy、自动微分 | [02](docs/02_curve_fitting.md) | `base02_curve_fitting.cpp` | [02](assignments/02_homework.md) |
| 3 | 位姿图：SE(3)、四元数、Manifold、稀疏 H | [03](docs/03_pose_graph.md) | `base03_pose_graph.cpp` | [03](assignments/03_homework.md) |
| 4 | 视觉 BA：重投影误差、位姿/点雅可比、Schur | [04](docs/04_ba.md) | `base04_ba.cpp` | [04](assignments/04_homework.md) |
| 5 | IMU 预积分：测量模型、递推、零偏、协方差 | [05](docs/05_imu_preintegration.md) | `base05_imu.cpp` | [05](assignments/05_homework.md) |
| 6 | 滑动窗口：Schur 补、先验、FEJ | [06](docs/06_marginalization.md) | `base06_marginalization.cpp` | [06](assignments/06_homework.md) |
| 7 | 工程实践：GradientChecker、协方差、调优、综合项目 | [07](docs/07_engineering.md) | `base07_engineering.cpp` | [07](assignments/07_homework.md) |

> **讲义和代码的范围不完全相同。** 代码为了保证可读、可运行，做了有意的简化：
>
> - 第 4 课用 5 个相机、20 个点的合成数据，不读取真实数据集；
> - 第 5 课代码把零偏固定为 0，只实现 9 维预积分残差；零偏一阶修正、15 维协方差在讲义中，作业让你观察零偏影响；
> - 第 6 课代码是 3 变量线性例子，FEJ 只在讲义中讨论；
> - 第 7 课的综合项目是设计文档，代码实现 `GradientChecker`、`Covariance` 和求解器对比。
>
> 每份讲义都会明确标出“代码实现了什么、哪些是练习”。

完整索引与学习建议见 [`docs/README.md`](docs/README.md)。
作业总览见 [`assignments/README.md`](assignments/README.md)。

课程校验发现与简化边界见 [`docs/REVIEW.md`](docs/REVIEW.md)；额外接口与建模知识见 [`docs/08_ceres_modeling_notes.md`](docs/08_ceres_modeling_notes.md)。

---

## 快速开始

```bash
cd ~/Desktop/learn/ceres_learn

# 配置 + 构建（Release 更快，和现有 build/ 一致）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 只构建某一课
cmake --build build --target base04_ba -j
```

运行示例：

```bash
./build/base01
./build/base02_curve_fitting
./build/base03_pose_graph
./build/base04_ba
./build/base05_imu
./build/base06_marginalization
./build/base07_engineering
```

运行作业答案：

```bash
./build/hw01_basics
./build/hw02_robust
./build/hw03_pose_graph
./build/hw04_ba
./build/hw05_imu
./build/hw06_marginalization
./build/hw07_engineering
```

> 所有程序都用**合成数据**，不依赖外部数据集。随机种子固定，但不同编译器/标准库的数值会有微小差异，验收看趋势和量级，不要逐位对比。

---

## 学习顺序

```
Ceres 基础 (1)
   -> 残差 + 白化 + 鲁棒核 (2)
   -> SE(3) + Manifold + 稀疏求解 (3)
   -> 视觉 BA + Schur (4)
   -> IMU 预积分 + 零偏 (5)
   -> 滑动窗口 + 边缘化 + FEJ (6)
   -> 工程验证 + 综合项目 (7)
```

不要跳课。第 4 课的 BA 是第 3 课位姿图的扩展，第 5 课的 IMU 因子又复用了第 3、4 课的残差结构和 Manifold。

---

## 约定

- 四元数：Ceres `QuaternionManifold` 用 $[w,x,y,z]$，Eigen 用 $[x,y,z,w]$；
- 位姿 $T = [R \mid t]$ 默认 body-to-world，残差里写 $T_{cw}$ 时会明确说明；
- 残差默认白化（除以噪声 sigma）；
- 鲁棒核阈值作用在白化后的残差上。

---

## 当前状态

- [x] 第 1~7 课讲义
- [x] 第 1~7 课示例代码
- [x] 第 1~7 课作业题目
- [x] 第 1~7 课参考答案
- [x] 全部 14 个 CMake target 编译通过（Release）
- [ ] 真实数据集接入（EuRoC / KITTI）
- [ ] 完整 VIO 后端实现（第 7 课只给设计）
