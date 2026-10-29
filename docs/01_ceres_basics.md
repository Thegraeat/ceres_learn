# 第 1 课：Ceres 基本使用

> 对应代码：`base01.cpp`
> 目标：独立写出「定义残差 → 构建 Problem → 配置 Solver → 读 Summary」的完整流程。
> 约定：本文所有向量默认是列向量，$J$ 是残差对参数的雅可比。

---

## 1. Ceres 解决什么问题

Ceres 求解的是**非线性最小二乘**问题：

$$
\min_{\mathbf{x}} \; \frac{1}{2}\sum_{i=1}^{m}\rho_i\!\left(\left\|r_i(\mathbf{x})\right\|^2\right)
$$

- $r_i(x) \in \mathbb{R}^n$：第 $i$ 个残差，$n$ 是残差维度；
- $x \in \mathbb{R}^p$：所有待优化参数拼成的向量；
- $\rho_i$：鲁棒核（loss function），默认 $\rho(s) = s$，即普通最小二乘。

### 1.1 把目标函数翻译成一句话

这里 $\min_{\mathbf x}$ 的意思是“调整参数 $\mathbf x$，让后面的数尽量小”。具体做法是：用当前参数算每条观测的预测，把预测与观测的差写成残差，再把残差平方、汇总。

公式中的 $i$ 是残差**块**编号，$m$ 是块数；每块可以包含多个数。例如一个像素观测有横、纵两个误差，所以是一块二维残差。不同残差块的维度也可以不同，更一般应写成 $r_i\in\mathbb R^{n_i}$。

若 $r_i=[r_{i1},r_{i2}]^\top$，那么：

$$
\|r_i\|^2=r_{i1}^2+r_{i2}^2.
$$

平方让正负误差都产生正代价，也会更重地惩罚大误差。前面的 $1/2$ 是为了求导方便：$\frac{d}{dr}(\frac12r^2)=r$；乘一个固定正数不会改变最优参数。

例如本课的残差 $r(x)=10-x$，目标是 $F(x)=\frac12(10-x)^2$。$x=5$ 时 $F=12.5$，$x=9$ 时 $F=0.5$，$x=10$ 时 $F=0$。求解器要做的就是从初值逐步走向 $10$。

### 1.2 白化是在统一“误差有多严重”的尺度

如果观测带有协方差 $\Sigma_i$，残差要先**白化**：

$$
\tilde r_i = \Sigma_i^{-1/2} r_i
$$

这样 $\|\tilde r_i\|^2 = r_i^\top \Sigma_i^{-1} r_i$，得到马氏距离的平方。SLAM 里“重投影误差除以像素噪声”“IMU 残差除以测量噪声”都是这一步。

符号 $\Sigma^{-1/2}$ 在这里表示满足 $W^\top W=\Sigma^{-1}$ 的白化矩阵 $W$；相关噪声不能逐元素开方。

例如两次距离测量都错了 $0.1$ m：第一台传感器标准差 $0.01$ m，第二台为 $0.2$ m。白化残差分别是 $10$ 和 $0.5$。前者非常反常，后者在正常噪声范围内，不能因为原始误差相同就等权处理。

下面为了书写简洁，用 $r$ 表示**已经白化的残差**。

Ceres 不知道什么是位姿、什么是对极几何。它只做一件事：

> 你告诉它每个残差怎么算、雅可比是什么，它帮你迭代求解。

所以 SLAM 后端工程师的工作，80% 在**写残差**，不在调求解器。

---

## 2. 五个核心对象

| 对象 | 作用 | 你必须提供什么 |
|---|---|---|
| `ceres::CostFunction` | 一个残差块 | 残差维度、参数块维度、`operator()` |
| `ceres::LossFunction` | 鲁棒核 | 核函数类型与阈值 |
| `ceres::Problem` | 残差块和参数块的集合 | `AddResidualBlock` / `AddParameterBlock` |
| `ceres::Solver::Options` | 求解器配置 | 线性求解器、迭代上限、容差 |
| `ceres::Solver::Summary` | 求解结果与诊断 | 读它，不要只看“是否收敛” |

一张图记住它们的关系：

```
CostFunction ─┐
LossFunction ─┼──> Problem ──> Solver::Options ──> Solve() ──> Summary
参数块指针   ─┘
```

---

## 3. 求解器在背后做什么

### 3.1 线性化

在当前估计 $x$ 处对残差做一阶泰勒展开：

$$
r_i(\mathbf{x} + \Delta \mathbf{x}) \approx r_i(\mathbf{x}) + J_i \Delta \mathbf{x},
\qquad
J_i = \frac{\partial r_i}{\partial \mathbf{x}}
$$

如果参数有 $P$ 个数，某个残差块有 $n_i$ 个数，$J_i$ 就是 $n_i\times P$ 矩阵。第 $j$ 列表示“只改第 $j$ 个参数时，这块残差如何变化”。

这个近似只适用于足够小的增量。原函数可以是弯曲的，我们只在当前位置用它的切线或切平面代替它；走到新位置后再重新展开。

### 3.2 Gauss-Newton

先考虑无鲁棒核、已白化的残差。把展开式代回目标函数，对 $\Delta x$ 求导并令其为零，得到正规方程：

$$
\left(\sum_i J_i^\top J_i\right)\Delta \mathbf{x}
= -\sum_i J_i^\top r_i
$$

记 $H = \sum_i J_i^\top J_i$、$g = \sum_i J_i^\top r_i$，就是：

$$
H \Delta \mathbf{x} = -g
$$

把中间步骤写出来就容易看出负号从哪里来。将所有残差按行堆叠成 $r$，所有雅可比按行堆叠成 $J$，本轮要解：

$$
\min_{\Delta x}\;Q(\Delta x)=\frac12\|r+J\Delta x\|^2.
$$

展开平方：

$$
Q(\Delta x)=\frac12r^\top r
+\Delta x^\top J^\top r
+\frac12\Delta x^\top J^\top J\Delta x.
$$

对增量求导，第一项不含增量，导数为零：

$$
\frac{\partial Q}{\partial\Delta x}
=J^\top r+J^\top J\Delta x=g+H\Delta x=0.
$$

所以右端是 $-g$。这里 $r\in\mathbb R^M$、$J\in\mathbb R^{M\times P}$，因而 $H\in\mathbb R^{P\times P}$、$g\in\mathbb R^P$。无论有多少观测，正规方程的未知增量个数等于自由参数个数。

用 $r(x)=10-x$ 手算一轮：初值 $x=5$，则 $r=5$、$J=-1$、$H=1$、$g=-5$。解得 $\Delta x=5$，更新后 $x=10$。因为残差本来就是线性的，这一次线性化是精确的；一般非线性问题需要多轮。

然后 $x \leftarrow x \boxplus \Delta x$，重复直到收敛。

普通向量的 $\boxplus$ 就是加法；四元数则要用满足单位约束的更新，第 3 课会讲。$H=J^\top J$ 是 Gauss–Newton 的二阶近似，不是任意非线性目标的精确 Hessian。

### 3.3 Levenberg-Marquardt

GN 在初值差或 $H$ 接近奇异时会发散。LM 加阻尼：

$$
\left(H + \lambda D^\top D\right)\Delta \mathbf{x} = -g
$$

- $\lambda$ 大：接近梯度下降，稳但慢；
- $\lambda$ 小：接近 GN，快但可能发散；
- Ceres 根据每步的实际下降量自动调整 $\lambda$。

$D$ 通常用于调整不同参数方向的尺度。若暂取 $D=I$，LM 等价于给本轮线性化目标加上 $\frac\lambda2\|\Delta x\|^2$，不鼓励一次走太远。它约束的是**本轮步长**，不是新增了一条真实传感器观测。

上面的单变量例子变成 $(1+\lambda)\Delta x=5$。取 $\lambda=4$，本轮只走 $\Delta x=1$，从 $5$ 到 $6$；取 $\lambda\to0$，则接近 GN 的一步到位。一般 $D$ 下，大阻尼对应带尺度调整的梯度方向。

Ceres 默认就是 `TRUST_REGION + LEVENBERG_MARQUARDT`。

### 3.4 稀疏性

$H$ 的结构完全由“哪个残差连哪些参数”决定。SLAM 里一个残差通常只连 $2\sim 4$ 个变量，所以 $H$ 是稀疏的。后面位姿图课会看到，选对线性求解器能差一到两个数量级。

---

## 4. 三种求导方式

### 4.1 自动微分（推荐默认）

Ceres 用**对偶数**（dual number）实现前向自动微分：

$$
x = a + b\epsilon,\qquad \epsilon^2 = 0
$$

对任意光滑函数 $f$：

$$
f(a + b\epsilon) = f(a) + f'(a)\,b\,\epsilon
$$

所以只要让 $b = 1$，算出来的 $\epsilon$ 系数就是导数。四则运算和 $\exp/\sin/\log$ 都满足这套代数：

$$
(a+b\epsilon)(c+d\epsilon) = ac + (ad+bc)\epsilon
$$

$\epsilon^2=0$ 是这套数的运算规则，不是在电脑里选择一个极小浮点数。以 $f(x)=x^2$ 为例：

$$
(3+\epsilon)^2=9+6\epsilon.
$$

普通部分 $9$ 是函数值，$\epsilon$ 的系数 $6$ 是导数。它没有用两个相近数相减，所以和有限差分不同。多参数时把导数部分换成向量，就能同时记录对各参数的偏导。

**一次携带向量导数的前向计算，同时得到函数值和所有偏导**，没有有限差分的截断误差，但仍有浮点舍入误差；运算成本随 Jet 导数维度增加，并不是与参数个数无关。

Ceres 要求你把残差写成**模板**：

```cpp
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];   // f(x) = 10 - x
        return true;
    }
};
```

- `T = double` 时：只算残差；
- `T = ceres::Jet<double, 1>` 时：残差和雅可比一起算。

**关键约束**：依赖待优化参数的中间量必须保留 `T` 类型；观测常量仍可用 `double`。直接把 Jet 表达式赋给 `double` 通常会编译失败，而不是静默得到零导数；手动提取 Jet 的标量部分才会切断导数链。

### 4.2 数值微分

前向差分：

$$
f'(x) \approx \frac{f(x+h)-f(x)}{h} + O(h)
$$

中心差分：

$$
f'(x) \approx \frac{f(x+h)-f(x-h)}{2h} + O(h^2)
$$

成本：$n$ 个参数，前向需要 $n+1$ 次函数求值，中心需要 $2n$ 次。

这里统计的是差分公式需要的求值次数；实际接口还可能单独评估基础残差。$O(h)$、$O(h^2)$ 表示忽略项随步长缩小的量级，不是额外加到计算结果里的修正量。$h$ 太大时近似不准，太小时两个函数值太接近，浮点相减会放大舍入误差，因此不是越小越好。

```cpp
struct CostFunctorNum {
    bool operator()(const double* const x, double* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

auto* cost = new ceres::NumericDiffCostFunction<
    CostFunctorNum, ceres::CENTRAL, 1, 1>(new CostFunctorNum);
```

注意数值微分的 functor **不是模板**，参数是 `double`。

### 4.3 解析微分

自己继承 `SizedCostFunction`，在 `Evaluate` 里同时填残差和雅可比：

```cpp
class SimpleCostFunctor : public ceres::SizedCostFunction<1, 1> {
  public:
    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double x = parameters[0][0];
        residuals[0] = 10.0 - x;
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -1.0;   // df/dx
        }
        return true;
    }
};
```

`SizedCostFunction<1, 1>` 的两个数字：残差维度 1、第一个参数块维度 1。多个参数块就继续写：

```cpp
ceres::SizedCostFunction<2, 3, 4>  // 2 维残差，两个参数块分别 3 维和 4 维
```

### 4.4 三种方式对比

| 方式 | 精度 | 成本 | 代码量 | 适用 |
|---|---|---|---|---|
| 自动微分 | 无差分截断误差 | 一次 Jet 求值，成本随参数维度增加 | 小 | 默认首选 |
| 数值微分 | 有截断误差 | $n+1$ 或 $2n$ 次 | 小 | 无法模板化时 |
| 解析微分 | 取决于推导与实现 | 可利用结构加速，不保证总是最快 | 大、易错 | 性能敏感、残差极简单 |

工程顺序：**自动微分跑对 → 用 `GradientChecker` 验证 → 再考虑解析雅可比加速**。

---

## 5. `base01.cpp` 逐段讲解

完整代码：

```cpp
#include <iostream>
#include <ceres/ceres.h>

struct CostFunctor {
    template<typename T>
    bool operator()(const T *const x, T *residual) const {
        residual[0] = 10.0 - x[0]; // f(x) = 10 - x
        return true;
    }
};

struct CostFunctorNum {
    bool operator()(const double *const x, double *residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

class SimpleCostFunctor : public ceres::SizedCostFunction<1,1> {
public:
    virtual ~SimpleCostFunctor() {};

    virtual bool Evaluate(
      double const* const* parameters, double *residuals, double **jacobians) const {
        const double x = parameters[0][0];
        residuals[0] = 10 - x;

        if(jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = -1;
        }
        return true;
    }
};

int main(int argc, char** argv) {
    double x = 0.5;
    const double initial_x = x;

    ceres::Problem problem;
    ceres::CostFunction *cost_function;

    // 三选一
    // cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    // cost_function = new ceres::NumericDiffCostFunction<CostFunctorNum, ceres::CENTRAL, 1, 1>(new CostFunctorNum);
    cost_function = new SimpleCostFunctor;

    problem.AddResidualBlock(cost_function, NULL, &x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << initial_x << " -> " << x << "\n";
    return 0;
}
```

### 5.1 `AutoDiffCostFunction` 的模板参数

```cpp
ceres::AutoDiffCostFunction<CostFunctor, 1, 1>
//                              ^functor ^残差维度 ^参数块1维度
```

如果有两个参数块：

```cpp
ceres::AutoDiffCostFunction<MyCost, 2, 3, 4>
//                                    ^p1 ^p2
```

**这些数字必须和 `AddResidualBlock` 传入的参数块顺序、维度完全一致。** 写错通常不会编译报错，而是运行时崩溃或结果错误。

### 5.2 `AddResidualBlock`

```cpp
problem.AddResidualBlock(cost_function, loss_function, &x, &y, ...);
```

- 第二个参数是 `LossFunction*`，可以为 `nullptr`；
- 后面的可变参数是**每个参数块的 `double*`**；
- `Problem` **默认接管** cost、loss 和 manifold 的所有权；可通过 `Problem::Options` 改成 `DO_NOT_TAKE_OWNERSHIP`，此时由调用者管理；
- 参数数组由调用者拥有，必须在 `Solve`、`Evaluate`、协方差等所有访问期间有效，且地址稳定。加入 Problem 后不要让 `std::vector` 扩容而移动参数内存。

### 5.3 这个问题的解

$r = 10 - x$，$J = -1$，GN 一步解出：

$$
x = x_0 - \frac{J^\top r}{J^\top J}
= 0.5 - \frac{(-1)(10-0.5)}{1} = 10
$$

所以输出是 `x : 0.5 -> 10`。`base01` 只是用来验证环境，不涉及任何 SLAM 几何。

---

## 6. 参数块（ParameterBlock）

### 6.1 多维参数块

一个残差可以连多个参数块，每个参数块可以是任意维度：

```cpp
double q[4] = {1.0, 0.0, 0.0, 0.0};  // w, x, y, z
double t[3] = {0.0, 0.0, 0.0};

problem.AddParameterBlock(q, 4, new ceres::QuaternionManifold);
problem.AddParameterBlock(t, 3);
problem.AddResidualBlock(cost, nullptr, q, t);
```

### 6.2 固定参数

```cpp
problem.SetParameterBlockConstant(q);
problem.SetParameterBlockConstant(t);
```

用途：

- 消除 gauge freedom（位姿图固定第一个位姿）；
- 固定标定参数；
- 固定已知量。

### 6.3 Manifold 是什么

四元数有 4 个数，但旋转只有 3 个自由度。优化时不能直接 $q + \Delta$（会破坏单位范数），而是：

$$
q \boxplus \delta = \exp(\delta) \otimes q
$$

Ceres 的 `Manifold` 包含 `Plus`、`Minus` 及相应雅可比，其中常用的是：

```cpp
Plus(q, delta)        // R^4 × R^3 -> R^4，单位初值在流形上更新
PlusJacobian(q)       // 4×3，d(q⊞δ)/dδ
```

`QuaternionManifold` 使用 Hamilton 顺序 `[w,x,y,z]`；Eigen 的 `Quaterniond` 内存顺序是 `[x,y,z,w]`，混用时必须换 `EigenQuaternionManifold`。

`Plus` 是左乘单位增量，不负责修复非单位初值。Ceres 的四元数切空间增量 $\delta$ 对应物理旋转向量 $2\delta$，手写雅可比、旋转协方差和其他李群库对接时尤其要注意。

> 完整推导见第 3 课 `03_pose_graph.md`。

---

## 7. 损失函数（鲁棒核）

外点会让普通最小二乘严重偏离。鲁棒核把大残差的权重压下去。

Ceres 的核函数以 $s = \|r\|^2$ 为自变量。

### 7.1 Huber

$$
\rho(s) =
\begin{cases}
s, & s \le \delta^2 \\[4pt]
2\delta\sqrt{s} - \delta^2, & s > \delta^2
\end{cases}
$$

残差小于 $\delta$ 时保持二次代价；超过后变成线性增长，等价于 L1。

### 7.2 Cauchy

$$
\rho(s) = \delta^2 \log\!\left(1 + \frac{s}{\delta^2}\right)
$$

对大残差的抑制更激进，但更容易陷入局部极小。

### 7.3 用法

```cpp
ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
problem.AddResidualBlock(cost, loss, params...);
```

**重要**：本课程的阈值作用在**白化后整个残差块的范数**上。未白化时阈值也有意义，但单位是原始残差单位，不再是“几个 sigma”。多维残差的阈值还应结合块维度选择，不能逐分量理解。

---

## 8. `Solver::Options` 常用配置

```cpp
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_QR;
options.max_num_iterations = 100;
options.minimizer_progress_to_stdout = true;
```

| 选项 | 常用值 | 说明 |
|---|---|---|
| `linear_solver_type` | `DENSE_QR` | 小问题（参数 < 几百） |
| | `DENSE_NORMAL_CHOLESKY` | 稠密、残差多 |
| | `SPARSE_NORMAL_CHOLESKY` | 大位姿图，H 稀疏 |
| | `ITERATIVE_SCHUR` | BA 的经典选择 |
| | `SPARSE_SCHUR` | BA，需要可用稀疏后端，如 SuiteSparse 或 EigenSparse |
| `trust_region_strategy_type` | `LEVENBERG_MARQUARDT` | 默认，稳 |
| | `DOGLEG` | 组合 GN 与最速下降步，需要精确分解型线性求解器 |
| `max_num_iterations` | `50` | 迭代上限 |
| `function_tolerance` | `1e-6` | 代价变化足够小 |
| `gradient_tolerance` | `1e-10` | 梯度足够小 |
| `parameter_tolerance` | `1e-8` | 步长足够小 |
| `minimizer_progress_to_stdout` | `true` | 打印每次迭代 |
| `num_threads` | `1` | 控制残差/雅可比评估线程数；不要求 OpenMP |
| `use_nonmonotonic_steps` | `false` | 允许代价偶尔上升，跳出平台 |

**怎么选线性求解器**：

```
参数规模 < 500，稠密        -> DENSE_QR
位姿图 / 大规模稀疏          -> SPARSE_NORMAL_CHOLESKY
BA（位姿 + 路标，Schur 结构）-> ITERATIVE_SCHUR 或 SPARSE_SCHUR
```

---

## 9. 怎么读 `Summary`

```cpp
std::cout << summary.BriefReport() << "\n";
```

典型输出：

```
Ceres Solver Report: Iterations: 1, Initial cost: 4.512500e+01,
Final cost: 0.000000e+00, Termination: CONVERGENCE
```

关键字段：

| 字段 | 含义 |
|---|---|
| `termination_type` | `CONVERGENCE` / `NO_CONVERGENCE` / `FAILURE` / `USER_SUCCESS` / `USER_FAILURE` |
| `initial_cost` / `final_cost` | 初始与最终代价 |
| `iterations.size()` | 实际迭代次数 |
| `message` | 终止原因的文字说明 |
| `BriefReport()` | 一行摘要 |
| `FullReport()` | 完整诊断，调参时看这个 |

**最容易误解的一点**：

> `CONVERGENCE` 只表示“步长 / 梯度 / 代价变化足够小”，**不表示解是正确的**。

初值差、残差写错、参数化错误、外点太多，都可能让求解器“成功收敛”到一个错误答案。所以永远要：

1. 看 `final_cost` 是否合理；
2. 用真值 / 重投影误差 / ATE 验证；
3. 打印中间量，确认残差几何意义正确。

交付结果前检查 `summary.IsSolutionUsable()`。它与“是否达到收敛容差”不是同一概念：到迭代上限的 `NO_CONVERGENCE` 结果也可能可用；可用仍不代表模型正确。

---

## 10. 编译与运行

```bash
cd ~/Desktop/learn/ceres_learn
cmake -S . -B build
cmake --build build -j
./build/base01
```

`CMakeLists.txt` 的关键两行：

```cmake
find_package(Ceres 2.1 REQUIRED)
target_link_libraries(base01 PRIVATE Ceres::ceres)
```

---

## 11. 新手最常见的 8 个错误

1. **参数相关表达式丢失 `T` 类型**：直接赋给 `double` 通常编译失败；手动提取标量会截断导数。
2. **模板参数维度写错**：`AutoDiffCostFunction<F, 1, 1>` 与实际参数块不符。
3. **`AddResidualBlock` 参数顺序与模板不一致**：不会报错，结果错。
4. **参数数组提前销毁**：`Problem` 只存指针，`Solve` 时访问野指针。
5. **重复 `delete` cost/loss**：所有权已经转给 `Problem`。
6. **四元数不归一化**：初值误差累积，Manifold 也可能被破坏。
7. **忘记固定 gauge**：$H$ 奇异，求解失败或整体漂移。
8. **残差不白化**：各项量级差异巨大，优化被大项主导。

---

## 12. 练习

1. 把 `base01` 的初值改成 `100.0`，观察迭代次数和最终解。
2. 把 `SimpleCostFunctor` 换成 `AutoDiffCostFunction`，确认结果一致。
3. 给问题再加一个残差 $r_2 = 5 - x$（同一个 `x`），重新求解。手算验证解是 $x = 7.5$。
4. 把 `max_num_iterations` 设为 `1`，读 `summary.termination_type` 和 `final_cost`。
5. 把 `x` 改成两个参数 `x[2]`，残差写成 `10 - x[0] - 2*x[1]`，观察解不唯一（$H$ 奇异）时 Ceres 的表现。

下一课：`02_curve_fitting.md`，用一个带外点的曲线拟合，把白化、鲁棒核和自动微分串起来。
