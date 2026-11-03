# homework/ 练习模板

这里是你自己的作业区。`assignments/` 放题目、参考答案和可编译答案，**不要直接改答案文件**；练习代码写在这里。

## 文件与状态

| 文件 | 对应作业 | 状态 |
|---|---|---|
| `my_hw01.cpp` | [第 1 课](../assignments/01_homework.md) | 已完成（直线拟合、自动/数值微分、闭式解对照） |
| `my_hw02.cpp` | [第 2 课](../assignments/02_homework.md) | 模板：数据、流程、输出已给，残差/MAD/3σ 待手写 |
| `my_hw03.cpp` | [第 3 课](../assignments/03_homework.md) | 模板：位姿图数据、初值、错误边、ATE 已给，残差/ARE/求解待手写 |
| `my_hw04.cpp` | [第 4 课](../assignments/04_homework.md) | 模板：BA 数据、外点、评估指标已给，重投影残差/求解待手写 |
| `my_hw05.cpp` | [第 5 课](../assignments/05_homework.md) | 模板：IMU 数据、零偏、预积分递推、指标已给，9 维残差/求解待手写 |
| `my_hw06.cpp` | [第 6 课](../assignments/06_homework.md) | 模板：三个线性残差块、验证循环已给，残差/H/g/Schur/求解待手写 |
| `my_hw07.cpp` | [第 7 课](../assignments/07_homework.md) | 模板：数据、协方差对比/梯度检查流程已给，残差/建模/协方差/梯度检查 cost 待手写 |

## 手写边界

每份模板都在需要你完成的位置标了 `TODO`，主要分三类：

1. **残差 functor**：`operator()` 里只保留占位赋值和公式注释；
2. **Ceres 建模/求解**：`Problem`、`AddParameterBlock`、`Manifold`、`LossFunction`、`AddResidualBlock`、`Options`、`Solve` 都留空；
3. **作业特有的计算**：如第 2 课的 MAD 与 3σ 筛选、第 3 课的 ARE、第 6 课的 Schur 补与先验中心、第 7 课的理论协方差。

数据生成、初值、评估指标、打印格式和 `main` 流程已经写好，避免把时间花在重复的脚手架代码上。

模板在 TODO 未完成时会让求解函数返回 `false`，`main` 打印一行提示后正常退出，因此**能编译、能运行**，不会因为空实现而崩溃或输出看似正确的假结果。

## 编译运行

在仓库根目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target my_hw02 -j
./build/my_hw02
```

一次构建全部练习模板：

```bash
cmake --build build --target my_hw02 my_hw03 my_hw04 my_hw05 my_hw06 my_hw07 -j
```

## 推荐流程

1. 先读 `assignments/0X_homework.md` 的任务和验收标准；
2. 读对应讲义，确认残差的维度、白化方式和参数块顺序；
3. 按 `TODO` 顺序实现，先让程序跑通，再对照验收标准；
4. 最后才看 `assignments/0X_solution.md` 和 `assignments/hw0X_*.cpp`，重点对照残差写法、参数块顺序和 Loss 的设置。

> 验收标准以各课 `0X_homework.md` 为准；随机种子固定，但不同编译器/标准库的数值会有微小差异，看趋势和量级，不要逐位对比。
