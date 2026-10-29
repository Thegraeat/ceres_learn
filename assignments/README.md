# 课后编程作业

每课一套：**题目**、**参考答案**、**可编译代码**。

先自己做题，再看答案。建议新建 `my_hw0X.cpp`，不要直接改答案文件。已发现问题及参考实现边界见 [课程校验记录](../docs/REVIEW.md)；尤其第 4 课三求解器性能扩展仍需自行完成。

当前状态：7 套作业答案（`hw01_basics` ~ `hw07_engineering`）已在 Release 下编译通过。
输出数值会因编译器、标准库和随机数实现有微小差异，验收看趋势和量级，不要逐位对比。

---

## 目录

| 课 | 题目 | 答案 | 代码 | 对应讲义 | 主题 |
|---|---|---|---|---|---|
| 1 | [`01_homework.md`](01_homework.md) | [`01_solution.md`](01_solution.md) | `hw01_basics.cpp` | [01](../docs/01_ceres_basics.md) | 多参数块、固定参数、三种求导 |
| 2 | [`02_homework.md`](02_homework.md) | [`02_solution.md`](02_solution.md) | `hw02_robust.cpp` | [02](../docs/02_curve_fitting.md) | Huber/Cauchy、MAD、外点剔除 |
| 3 | [`03_homework.md`](03_homework.md) | [`03_solution.md`](03_solution.md) | `hw03_pose_graph.cpp` | [03](../docs/03_pose_graph.md) | 错误回环、Huber、ATE/ARE |
| 4 | [`04_homework.md`](04_homework.md) | [`04_solution.md`](04_solution.md) | `hw04_ba.cpp` | [04](../docs/04_ba.md) | 误匹配、Huber、BA 评估、求解器 |
| 5 | [`05_homework.md`](05_homework.md) | [`05_solution.md`](05_solution.md) | `hw05_imu.cpp` | [05](../docs/05_imu_preintegration.md) | 零偏补偿、预积分残差、误差对比 |
| 6 | [`06_homework.md`](06_homework.md) | [`06_solution.md`](06_solution.md) | `hw06_marginalization.cpp` | [06](../docs/06_marginalization.md) | 非最优线性化点、先验中心、等价性 |
| 7 | [`07_homework.md`](07_homework.md) | [`07_solution.md`](07_solution.md) | `hw07_engineering.cpp` | [07](../docs/07_engineering.md) | 协方差、GradientChecker、性能、设计题 |

---

## 建议练习方式

公式不熟时，先查[数学预备](../docs/00_math_primer.md)和[讲义数学内容索引](../docs/README.md#数学内容怎么查)。各套答案的理论部分补充了计算过程：第 1 套推导直线闭式解，第 2 套解释 MAD 系数，第 3～4 套区分角度与像素评估，第 5 套区分低残差和轨迹准确，第 6 套手算消元梯度，第 7 套逐步核对协方差。

1. **先跑示例**：`./build/base0X_*`，记下正常输出长什么样；
2. **读题目**：打开 `0X_homework.md`，按任务自己写 `my_hw0X.cpp`；
3. **加 target**：在 `CMakeLists.txt` 末尾加自己的可执行文件（见下方）；
4. **对照验收**：题目里的「验收标准」是硬指标，先满足它；
5. **再看答案**：重点对照
   - 残差的写法；
   - 参数块和 `AddResidualBlock` 的顺序；
   - 白化、鲁棒核阈值、协方差；
   - `Summary` 的输出；
6. **做对比**：`diff -u my_hw0X.cpp hw0X_*.cpp`，看自己的实现和答案差在哪。

> 答案不是唯一解。比如第 1 课可以把 $m,b$ 放进一个 2 维参数块，只是固定 $b$ 时要用 `SubsetManifold`。评分看行为，不看写法。

---

## 编译

所有作业答案已经在顶层 `CMakeLists.txt` 里注册：

```bash
cd ~/Desktop/learn/ceres_learn

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 只构建某一套
cmake --build build --target hw04_ba -j
```

运行：

```bash
./build/hw01_basics
./build/hw02_robust
./build/hw03_pose_graph
./build/hw04_ba
./build/hw05_imu
./build/hw06_marginalization
./build/hw07_engineering
```

如果你自己写了 `my_hw01.cpp`，在 `CMakeLists.txt` 末尾加：

```cmake
add_executable(my_hw01 my_hw01.cpp)
target_link_libraries(my_hw01 PRIVATE Ceres::ceres)
```

然后：

```bash
cmake -S . -B build
cmake --build build --target my_hw01 -j
./build/my_hw01
```

---

## 答案代码的定位

- 答案是**参考实现**，优先保证正确、可读，不是性能最优；
- 每个答案文件开头都写了任务、运行命令；
- 答案里的「预期输出」只给形态和量级，不保证精确数值；
- 答案里刻意保留了教学用的简化（标量 sigma、对角信息矩阵、小规模数据），讲义和思考题会指出如何扩展。

---

## 评分标准（自测）

| 等级 | 标准 |
|---|---|
| 及格 | 能编译、能运行，输出符合题目要求 |
| 良好 | 所有「验收标准」都满足，能解释每个参数的含义 |
| 优秀 | 能回答思考题，并能指出答案代码里可以改进的地方 |

---

## 一句话提醒

> 作业的目的不是“跑出正确答案”，而是让你亲手踩一遍坑：维度写错、顺序写反、忘记白化、阈值设错、坐标系混用。
> 这些坑在真实 SLAM 系统里都会以“优化收敛到错解”的形式出现。
