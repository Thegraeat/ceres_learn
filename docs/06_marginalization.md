# 第 6 课：滑动窗口与边缘化

> 对应代码：`base06_marginalization.cpp`
> 目标：理解滑动窗口为什么需要边缘化、Schur 补怎么把旧信息压成先验、FEJ 是什么。
> 前置：第 4 课（BA、Schur 结构）、第 5 课（IMU 因子）。

---

## 1. 为什么需要滑动窗口

BA 和 VIO 的变量随时间无限增长。实时系统只能保留最近一段时间的状态：

$$
\mathcal{W} = \{x_{k-m+1}, \dots, x_k\}
$$

最旧的状态 $x_{k-m}$ 必须移出窗口。但不能直接丢掉它，因为它和窗口内状态之间有约束（视觉、IMU、先验）。

**边缘化（marginalization）** 就是：

> 把要丢掉的变量积分掉，把它携带的信息压缩成一个只作用于剩余变量的先验因子。

例如旧帧 $x_0$ 和当前帧 $x_1$ 之间有 IMU 约束。删掉 $x_0$ 并连带删掉这条约束，会失去旧观测对 $x_1$ 的影响；把 $x_0$ 固定在某个值，又会把一个原本不确定的状态当成精确已知。边缘化要保留的是：允许旧状态在其不确定范围内变化后，旧观测还能给剩余状态提供多少约束。

下面先讨论线性高斯、相关矩阵非退化的情形。这时消元是精确的；非线性问题则先线性化，再保留该次局部近似。

---

## 2. 数学：Schur 补

在当前线性化点 $x_{lin}$，所有残差线性化：

$$
r_i(x) \approx r_i(x_{lin}) + J_i\,\delta x
$$

正规方程：

$$
H\,\delta x = -g,
\qquad
H = \sum_i J_i^\top J_i,
\qquad
g = \sum_i J_i^\top r_i
$$

把变量分成：

- $x_m$：要保留的变量；
- $x_r$：要边缘化的变量。

这里下标 $m$ 表示保留侧，$r$ 表示移除侧，**不要根据英文 marginalization 猜测 $m$ 一定是被删除的一侧**。这是本讲义自己的命名约定。若保留 $a$ 维、删除 $b$ 维，则 $H_{mm}$ 为 $a\times a$，$H_{mr}$ 为 $a\times b$，$H_{rr}$ 为 $b\times b$。

分块：

$$
\begin{bmatrix}
H_{mm} & H_{mr} \\
H_{rm} & H_{rr}
\end{bmatrix}
\begin{bmatrix}
\delta x_m \\ \delta x_r
\end{bmatrix}
=
-\begin{bmatrix}
g_m \\ g_r
\end{bmatrix}
$$

从第二行解出：

$$
\delta x_r = -H_{rr}^{-1}\left(g_r + H_{rm}\delta x_m\right)
$$

代回第一行：

$$
\left(H_{mm} - H_{mr}H_{rr}^{-1}H_{rm}\right)\delta x_m
=
-g_m + H_{mr}H_{rr}^{-1}g_r
$$

定义**先验信息矩阵**和**先验梯度**：

$$
H_{prior} = H_{mm} - H_{mr}H_{rr}^{-1}H_{rm}
$$

$$
g_{prior} = g_m - H_{mr}H_{rr}^{-1}g_r
$$

$H_{prior}$ 就是边缘化后剩余变量获得的信息。

更直白地说，第二行表示“给定一个保留变量增量，旧变量最合适怎样配合它”。代回第一行后，就不必再显式携带旧变量。

$H_{mm}$ 是**删除前**的保留—保留矩阵块，$H_{prior}$ 才是消元后的矩阵，二者不能混称。消掉旧变量后信息通常会减少，因为旧变量有调整空间，不能当成固定常量。

代码不应构造一般大矩阵的逆：可先解 $H_{rr}Y=H_{rm}$、$H_{rr}z=g_r$，再算 $H_{prior}=H_{mm}-H_{mr}Y$、$g_{prior}=g_m-H_{mr}z$。

### 2.1 先验残差

对 $H_{prior}$ 做 Cholesky：

$$
H_{prior} = L L^\top
$$

先验代价：

$$
\frac12 \delta x_m^\top H_{prior}\delta x_m
+ g_{prior}^\top \delta x_m
$$

它的最优点在：

$$
\delta x_m^\star = -H_{prior}^{-1} g_{prior}
$$

所以先验残差可以写成：

$$
r_{prior} = L^\top\left(\delta x_m - \delta x_m^\star\right)
$$

在 Ceres 里就是一个 `2` 维残差块（本例），参数是保留变量。

为什么要乘 $L^\top$？因为 Ceres 会对残差求平方：

$$
\frac12\|L^\top(\delta x_m-\delta x_m^\star)\|^2
=\frac12(\delta x_m-\delta x_m^\star)^\top
H_{prior}(\delta x_m-\delta x_m^\star).
$$

展开后，关于 $\delta x_m$ 的一次项为 $-(\delta x_m^\star)^\top H_{prior}\delta x_m=g_{prior}^\top\delta x_m$，恰好恢复旧信息的梯度；剩余差别只是与当前参数无关的常数。

还要区分“增量的最优值”和“绝对状态的先验中心”：

$$
\delta x_m=x_m-x_{lin,m},\qquad
x_{prior}=x_{lin,m}+\delta x_m^\star
=x_{lin,m}-H_{prior}^{-1}g_{prior}.
$$

所以本例 functor 用 $r_{prior}=L^\top(x_m-x_{prior})$。参考点只是展开位置，不能直接当成旧信息支持的最优位置。

### 2.2 “解方程消元”和“概率边缘化”为什么得到同一个矩阵

线性高斯模型的概率密度与 $\exp[-Q(\delta x_m,\delta x_r)]$ 成正比。对要删除的增量配方后，二次代价可写成：

$$
Q=\frac12\delta x_m^\top H_{prior}\delta x_m
+g_{prior}^\top\delta x_m
+\frac12(\delta x_r-\mu_r)^\top H_{rr}(\delta x_r-\mu_r)
+C,
$$

其中 $\mu_r=-H_{rr}^{-1}(g_r+H_{rm}\delta x_m)$，$C$ 与增量无关。

优化消元是把 $\delta x_r$ 选为 $\mu_r$，让最后那个平方项最小。概率边缘化则把 $\exp(-Q)$ 对所有 $\delta x_r$ 积分；在线性化矩阵固定、$H_{rr}$ 正定时，该高斯积分只产生一个与 $\delta x_m$ 无关的系数。两种操作留下相同的 $H_{prior},g_{prior}$，因此保留变量的最优解一致。

这个等价依赖固定的线性高斯模型。不能据此说任意非线性概率积分都等价于一次 Schur 补。

> **常见错误**：直接用 $x_{lin}$ 当先验中心。只有 $g_{prior} = 0$ 才成立；完整旧问题已达驻点是充分条件，但不是必要条件。上述中心表达要求 $H_{prior}$ 正定；保留 gauge 时应使用截断特征分解或平方根消元构造半正定先验，不能盲目做 LLT/求逆。

---

## 3. 代码里的例子

三个标量变量 $x_0, x_1, x_2$，旧残差：

$$
r_1 = x_0 + x_1 - 3
$$

$$
r_2 = x_1 + x_2 - 5
$$

$$
r_3 = x_2 - 2
$$

在 $x_{lin} = (0,3,2)$ 处，$r = 0$，所以 $g = 0$。原课件写成 $(1,2,3)$ 时其实 $r_3 = 1$，并不是零残差点。

每行残差对三个变量的导数分别是：

$$
J_1=[1,1,0],\qquad J_2=[0,1,1],\qquad J_3=[0,0,1],
$$

$$
J=\begin{bmatrix}1&1&0\\0&1&1\\0&0&1\end{bmatrix},
\qquad H=J^\top J.
$$

例如 $H_{11}=1^2+1^2+0^2=2$（下标从零开始），说明 $x_1$ 出现在两条单位权重残差里；$H_{02}=0$，说明没有旧残差同时直接连接 $x_0,x_2$。

$$
H =
\begin{bmatrix}
1 & 1 & 0 \\
1 & 2 & 1 \\
0 & 1 & 2
\end{bmatrix}
$$

边缘化 $x_2$：

$$
H_{prior} = H_{mm}-H_{mr}H_{rr}^{-1}H_{rm}=
\begin{bmatrix}
1 & 1 \\
1 & 2
\end{bmatrix}
-
\begin{bmatrix}0\\1\end{bmatrix}
\frac{1}{2}
\begin{bmatrix}0 & 1\end{bmatrix}
=
\begin{bmatrix}
1 & 1 \\
1 & 1.5
\end{bmatrix}
$$

Cholesky：

$$
L =
\begin{bmatrix}
1 & 0 \\
1 & \sqrt{0.5}
\end{bmatrix}
$$

先验残差：

$$
r_{prior} =
L^\top
\begin{bmatrix}
x_0 - 0 \\
x_1 - 3
\end{bmatrix}
$$

加入新观测 $x_0 = 1.1$ 后，边缘化问题与全量问题在 $x_0, x_1$ 上的解完全一致（线性问题，精确等价）。

### 3.1 不用矩阵也能手算这次边缘化

给定 $x_0,x_1$ 后，只有 $r_2,r_3$ 涉及 $x_2$。令它们的平方和对 $x_2$ 的导数为零：

$$
(x_1+x_2-5)+(x_2-2)=0
\quad\Longrightarrow\quad x_2^\star=\frac{7-x_1}{2}.
$$

代回旧代价：

$$
F_{old,reduced}=\frac12(x_0+x_1-3)^2+\frac14(x_1-3)^2.
$$

所以等效先验残差正是：

$$
r_{prior}=\begin{bmatrix}
x_0+x_1-3\\\sqrt{0.5}(x_1-3)
\end{bmatrix}
=L^\top\begin{bmatrix}x_0\\x_1-3\end{bmatrix}.
$$

新观测的残差为 $x_0-z$，本例 $z=1.1$，权重为 1。总目标是上式再加 $\frac12(x_0-z)^2$。对 $x_0,x_1$ 分别求导得到：

$$
2x_0+x_1=3+z,\qquad x_0+1.5x_1=4.5.
$$

解得 $x_0=0.75z$、$x_1=3-0.5z$，并由被消去变量的公式得到 $x_2=2+0.25z$。当 $z=1.1$ 时，得到 $(0.825,2.45,2.275)$。这让你能直接验证程序输出，而不只比较两个求解器是否碰巧给出相同答案。

---

## 4. FEJ：First Estimate Jacobian

边缘化先验冻结在历史线性化点，活动因子通常在当前估计处重新线性化。在带 gauge 的非线性系统中，两者对应的不可观测零空间可能不一致，使本应不可观测的方向获得虚假的信息。这是**可观测性/一致性问题**，不等于“同一观测被算了两次”。

FEJ 的核心是在与不可观测性有关的雅可比项上采用一致的首次估计点，使线性化系统尽量保留正确零空间。究竟冻结哪些状态与哪些项，取决于系统模型；不能概括成“无条件冻结所有残差、所有状态的全部雅可比”，也不能保证任意非线性系统从此完全一致。

本课只是线性欧氏例子，固定 $L$ 和先验中心是精确线性模型，**没有实现完整 FEJ**。非线性流形先验通常写成 $r = A (x \boxminus x_{lin}) + c$，必须保存参考状态、扰动约定、$A$ 与 $c$；四元数不可直接逐元素相减。

`EvaluationCallback` 用于评估前准备/共享缓存，不是“自动冻结参数并实现 FEJ”的机制。增加先验强度或 LM 阻尼也不能从根本上修复不可观测零空间不一致。实现相关策略需明确推导并自行提供一致的因子与雅可比。

### 4.1 用零空间理解“虚假信息”

若一个方向 $n$ 本来不可观，应当有 $Jn=0$，于是 $Hn=0$。理想情况下，先验和活动因子应该认同同一个不可观测方向。

举一个抽象的二维示意：旧信息矩阵为 $H_1=\begin{bmatrix}0&0\\0&1\end{bmatrix}$，对第一维完全没有信息。如果新因子因线性化不一致，错误地把同一 gauge 的方向表达成另一条方向，可能得到 $H_2=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$。两个矩阵各有一维零空间，但零空间方向不一致；相加后行列式为 $1$，反而满秩了。

这个例子说明：若理论上应共享的自由方向没有在同一线性化坐标中对齐，系统可能凭空“认为自己知道了”该方向。FEJ 要处理的是这种一致性；不是把真实能提供新信息的观测强行设为退化，也不是简单让求解器多迭代几次。

---

## 5. 边缘化的信息损失

边缘化不是无损的：

1. **线性化误差**：先验是二次近似，离开 $x_{lin}$ 越远越不准；
2. **保留信息的边界**：线性高斯情况下，边缘分布精确保留剩余变量的不确定性与相关性；丢失的是与已移除变量的联合表示/后续重线性化能力，不是笼统地“把它的不确定性丢掉”；
3. **一致性**：FEJ 只能缓解，不能完全消除；
4. **不可逆**：仅靠先验不能恢复旧变量和原始观测；若另外存档了原始数据，可以重建问题再优化，但不能把已吸收的观测叠加回同一先验而重复计数。

所以工程上要谨慎选择窗口大小和边缘化对象。不是“越早边缘化越好”。

---

## 6. 代码讲解

### 6.1 构建 H 和 g

```cpp
Eigen::Matrix3d H = J1 * J1.transpose() + J2 * J2.transpose() +
                    J3 * J3.transpose();
const Eigen::Vector3d g = J1 * r1 + J2 * r2 + J3 * r3;
```

这是“手写正规方程”。真实系统里 $H$ 由 Ceres 或手写 BA 组装。

代码把单条残差的三个导数存成 `Eigen::Vector3d` 列向量，因此 `J1 * J1.transpose()` 等价于数学上行雅可比 $J_1$ 的 $J_1^\top J_1$。变量名相同，存储方向不同；先看类型，就不会误以为代码把正规方程乘反了。

### 6.2 Schur 补

```cpp
const Eigen::Matrix2d Hmm = H.topLeftCorner<2, 2>() -
                            H.topRightCorner<2, 1>() * (1.0 / H(2, 2)) *
                                H.bottomLeftCorner<1, 2>();
const Eigen::Vector2d gm =
    g.head<2>() - H.topRightCorner<2, 1>() * (1.0 / H(2, 2)) * g(2);
```

这里 $H_{rr}$ 是 $1\times 1$，直接求逆。BA 里 $H_{rr}$ 是块对角，逐点求逆。

### 6.3 先验

```cpp
const Eigen::LLT<Eigen::Matrix2d> llt(Hmm);
const Eigen::Matrix2d L = llt.matrixL();

const Eigen::Vector2d x_prior = x_lin.head<2>() - Hmm.ldlt().solve(gm);
```

`PriorCost` 计算：

```cpp
residual[0] = L(0,0)*d0 + L(1,0)*d1;
residual[1] = L(0,1)*d0 + L(1,1)*d1;
```

即 $L^\top (x - x_{prior})$。

### 6.4 全量 vs 边缘化

```cpp
full.AddResidualBlock(...);   // x0,x1,x2 + 旧残差 + 新观测
marg.AddResidualBlock(...);   // x0,x1 + 先验 + 新观测
```

线性问题，两个解在 $x_0,x_1$ 上应该一致。

---

## 7. 运行与预期

```bash
cmake -S . -B build
cmake --build build -j
./build/base06_marginalization
```

预期：

```
[full]         x0=0.825 x1=2.45 x2=2.275
[marginalized] x0=0.825 x1=2.45
diff: |dx0|≈0 |dx1|≈0
```

示例对这个小型线性满秩问题使用 `DENSE_QR + DOGLEG`，关闭代价变化停止条件，并收紧梯度/步长容差至 `1e-14`，检查解与闭式解的差小于 `1e-10`。默认停止条件下两次独立 LM 求解可能相差约 `1e-4`；即使单纯收紧代价容差，也可能在约 `1e-8` 参数误差时因代价变化过小而停止。这不否定线性 Schur 消元的数学等价性，也不能直接宣称任何默认求解都达到机器精度。

---

## 8. 常见坑

1. **先验中心直接用 $x_{lin}$**：忘记减 $H^{-1}g$。
2. **Cholesky 方向错**：$H = L L^\top$，残差是 $L^\top d$，不是 $L d$。
3. **吸收到先验中的旧残差仍被保留**：同一观测被重复计数；若确需新因子连接旧变量，应重新建模，不能仅用先验“恢复”旧变量。
4. **线性化点不一致**：FEJ 问题。
5. **$H_{rr}$ 奇异**：被边缘化变量没有被足够约束，先验会爆炸。
6. **先验尺度错误**：信息矩阵必须和残差白化一致。

---

## 9. 练习

1. 把新观测从 $x_0 = 1.1$ 改成 $x_0 = 1.5$，验证全量与边缘化仍一致。
2. 把 $x_{lin}$ 改成 $(1.2, 1.8, 3.1)$，重新推导 $g$、$g_m$、$x_{prior}$，观察先验中心不再是 `x_lin.head<2>()`。
3. 把残差改成非线性（例如 $r_1 = x_0*x_1 - 2$），观察线性化点对边缘化结果的影响。
4. 思考：边缘化和“直接丢弃旧变量”有什么区别？各适合什么场景？

答案在 `assignments/06_solution.md`。
