# 第 3 课：SE(3) 位姿图优化

> 对应代码：`base03_pose_graph.cpp`
> 目标：把第 2 课的残差结构搬到 SE(3) 上，理解流形、信息矩阵、稀疏性和 gauge freedom。
> 前置：第 1 课、第 2 课。

---

## 1. SE(3) 基础

三维刚体位姿：

$$
T = \begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix}
\in SE(3),\qquad R \in SO(3),\ t \in \mathbb{R}^3
$$

$SO(3)$ 是所有合法三维旋转矩阵的集合；$SE(3)$ 是“旋转加平移”的刚体变换集合。它们不是两种求解器，而是在描述变量允许取什么值。

本课 $T_i=T_{wi}$ 把第 $i$ 个机体系的点变到世界系：$p_w=R_i p_i+t_i$。其中 $t_i$ 是机体原点在世界中的位置。$T$ 是 $4\times4$ 矩阵，对点运算时默认补齐最后一个分量 $1$。

### 1.1 复合与逆

$$
T_1 T_2 =
\begin{bmatrix}
R_1 R_2 & R_1 t_2 + t_1 \\
0^\top & 1
\end{bmatrix},
\qquad
T^{-1} =
\begin{bmatrix}
R^\top & -R^\top t \\
0^\top & 1
\end{bmatrix}
$$

复合公式来自把两次坐标变换接起来：

$$
p'=R_2p+t_2,\qquad
p''=R_1p'+t_1=R_1R_2p+R_1t_2+t_1.
$$

所以 $T_1T_2$ 是先做右边的 $T_2$，再做左边的 $T_1$。平移不能简单写成 $t_1+t_2$，因为 $t_2$ 也要经过第一段旋转。

逆变换则从 $p_w=Rp_b+t$ 解出 $p_b=R^\top(p_w-t)$，于是逆变换的平移是 $-R^\top t$，不是只写 $-t$。

### 1.2 四元数表示

单位四元数

$$
q = [w, x, y, z]^\top,\qquad \|q\| = 1
$$

对应旋转矩阵 $R(q)$。Hamilton 乘法：

$$
\begin{aligned}
w &= w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2 \\
x &= w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2 \\
y &= w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2 \\
z &= w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2
\end{aligned}
$$

单位四元数的逆等于共轭：

$$
q^{-1} = q^\ast = [w, -x, -y, -z]^\top
$$

旋转向量 $v$：

$$
R(q)\,v = v + 2w\,(u \times v) + 2\,u \times (u \times v),
\qquad u = [x,y,z]^\top
$$

这就是代码里 `QuatRotate` 的实现。

#### 1.2.1 四元数里的四个数从哪里来

绕单位轴 $n\in\mathbb R^3$ 按右手定则旋转 $\theta$，单位四元数写成：

$$
q=\begin{bmatrix}\cos(\theta/2)\\n\sin(\theta/2)\end{bmatrix}.
$$

$w$ 是标量部分，$u=(x,y,z)^\top$ 是向量部分；$x,y,z$ 不是三个欧拉角。角度在计算中使用弧度。单位约束来自 $\cos^2(\theta/2)+\sin^2(\theta/2)=1$。

不旋转时 $q=(1,0,0,0)$。绕 $z$ 轴转 $90^\circ$ 时：

$$
q=\left(\frac{\sqrt2}{2},0,0,\frac{\sqrt2}{2}\right).
$$

Hamilton 乘积中的 $q_1,q_2$ 分别是两段旋转；乘积的结果仍是四元数。它满足 $R(q_1\otimes q_2)=R(q_1)R(q_2)$，所以同样是先做右边的旋转，顺序不能随意交换。

#### 1.2.2 从四元数乘法推到旋转矩阵

把向量 $v$ 写成纯四元数 $p=(0,v)$。在本课约定下，旋转通过：

$$
p'=q\otimes p\otimes q^*
$$

实现。用标量和向量表示乘法，比四个分量逐项展开更好理解：

$$
(s,a)\otimes(t,b)
=\left(st-a^\top b,\;sb+ta+a\times b\right).
$$

这里 $a^\top b$ 是点积，$a\times b$ 是叉积。先乘左边两项，再乘共轭：

$$
q\otimes p=(-u^\top v,\;wv+u\times v),
$$

$$
v'=(w^2-u^\top u)v+2u(u^\top v)+2w(u\times v).
$$

要得到上面第二式，可把向量部分先写成
$u(u^\top v)+w^2v+2w(u\times v)-(u\times v)\times u$，再使用
$(u\times v)\times u=(u^\top u)v-(u^\top v)u$。

由于 $w^2+u^\top u=1$，也可以写成代码采用的形式：

$$
v'=v+2w(u\times v)+2u\times(u\times v).
$$

定义叉积矩阵：

$$
[u]_\times=\begin{bmatrix}0&-z&y\\z&0&-x\\-y&x&0\end{bmatrix},
\qquad [u]_\times v=u\times v.
$$

把 $v$ 提到右边，就得到旋转矩阵：

$$
R(q)=I+2w[u]_\times+2[u]_\times^2
=\begin{bmatrix}
1-2(y^2+z^2)&2(xy-wz)&2(xz+wy)\\
2(xy+wz)&1-2(x^2+z^2)&2(yz-wx)\\
2(xz-wy)&2(yz+wx)&1-2(x^2+y^2)
\end{bmatrix}.
$$

这不是另一套旋转规则，只是把 $q p q^*$ 中对 $v$ 的线性运算整理成矩阵。**公式要求 $q$ 是单位四元数**；一般非零四元数要先归一化，零四元数不能表示旋转。

代入绕 $z$ 轴 $90^\circ$ 的例子：

$$
R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix},
\qquad R\begin{bmatrix}1\\0\\0\end{bmatrix}
=\begin{bmatrix}0\\1\\0\end{bmatrix}.
$$

原来的 $x$ 方向转到了 $y$ 方向。$R$ 的三列就是三个单位轴旋转后的结果。把 $q$ 换成 $-q$，每个二次乘积都不变，所以旋转矩阵也不变。

代码中 `tx,ty,tz` 组成的向量就是 $2(u\times v)$；最后三行计算 $v+w\,t+u\times t$，正好对应上述公式，无需先生成完整 $3\times3$ 矩阵。

### 1.3 Exp 和 Log：把三维小增量与旋转联系起来

旋转向量 $\phi=\theta n$ 的方向是旋转轴、长度是旋转角。本文 $\mathrm{Exp}(\phi)$ 表示 $\exp([\phi]_\times)$，结果是旋转矩阵，不是对三个分量分别取指数。

令 $K=[\phi]_\times$、$\theta=\|\phi\|$，Rodrigues 公式为：

$$
\mathrm{Exp}(\phi)
=I+\frac{\sin\theta}{\theta}K
+\frac{1-\cos\theta}{\theta^2}K^2.
$$

当角度很小时，$\sin\theta/\theta\to1$、$(1-\cos\theta)/\theta^2\to1/2$，所以一阶近似是 $\mathrm{Exp}(\phi)\approx I+[\phi]_\times$。不能在 $\theta=0$ 时直接算 $0/0$，实现中要使用极限或级数。

$\mathrm{Log}$ 做反向转换。在选取 $w\ge0$、最短旋转角的单位四元数 $q=(w,u)$ 后，可以写成：

$$
\theta=2\operatorname{atan2}(\|u\|,w),\qquad
\mathrm{Log}(R(q))=\frac{\theta}{\|u\|}u.
$$

$u\to0$ 时用极限 $2u$。接近 $180^\circ$ 时旋转轴的符号存在分支选择，不能认为 Log 在所有旋转上都光滑唯一。

---

## 2. 位姿图残差

节点 $T_i$ 表示 body-to-world，边测量 $T_{ij}$。严格的李代数残差可以定义为

$$
e_{ij} = \mathrm{Log}\!\left(T_{ij}^{-1}\,T_i^{-1}\,T_j\right) \in \mathbb{R}^6
$$

拆成旋转和平移：

$$
R_e = R_{ij}^\top R_i^\top R_j,
\qquad
t_e = R_{ij}^\top\left(R_i^\top(t_j - t_i) - t_{ij}\right)
$$

先从几何上读这两个式子。由当前节点状态预测的相对位姿是：

$$
\hat T_{ij}=T_i^{-1}T_j,
\qquad \hat R_{ij}=R_i^\top R_j,
\qquad \hat t_{ij}=R_i^\top(t_j-t_i).
$$

$t_j-t_i$ 是世界系中的两节点位移；乘 $R_i^\top$ 后变成第 $i$ 帧中的位移。测量 $T_{ij}$ 同样把 $j$ 系坐标变到 $i$ 系，才能与预测比较。

例如 $i$ 帧绕世界 $z$ 轴转了 $90^\circ$，$t_i=(1,2,0)$、$t_j=(1,4,0)$。世界系位移是 $(0,2,0)$，但在 $i$ 系中是 $(2,0,0)$。直接把世界位移与局部测量相减，就会写错残差。

误差变换 $T_e=T_{ij}^{-1}\hat T_{ij}$ 是“先抵消测量，再看预测剩下什么”。两者完全一致时 $R_e=I,t_e=0$。平移误差中额外的 $R_{ij}^\top$ 来自测量逆变换的旋转；它把差值再转到误差变换采用的局部坐标。

- $R_e$ 是三维旋转，取 $\mathrm{Log}(R_e)$ 得到 3 维旋转向量；
- $t_e$ 直接作为 3 维平移残差（局部坐标下的平移误差）。

> 严格的 SE(3) Log 映射平移部分是 $V^{-1}(\omega)t_e$。本文和代码采用 $[2\,\mathrm{vec}(q_e),\, t_e]$ 的分离残差，**并非上式严格的 SE(3) Log**。这是另一种合法建模方式，不保证比 Log 收敛更好；协方差必须与所选残差的坐标系和扰动约定一致。

若要读懂这里的 $V$：取 $\omega=\mathrm{Log}(R_e)$、$\theta=\|\omega\|$，按本课“旋转在前、平移在后”的顺序：

$$
\mathrm{Log}(T_e)=\begin{bmatrix}\omega\\\nu\end{bmatrix},
\quad V(\omega)\nu=t_e,
\quad V=I+\frac{1-\cos\theta}{\theta^2}[\omega]_\times
+\frac{\theta-\sin\theta}{\theta^3}[\omega]_\times^2.
$$

$\nu$ 是李代数中的平移坐标，通常不等于直接读出来的 $t_e$。$\theta\to0$ 时 $V\to I$，二者才接近。初学本课可先掌握代码中的分离残差，避免把两种定义混用。

### 2.1 旋转残差与四元数

若 $q_e$ 是 $R_e$ 对应的单位四元数，则

$$
q_e = q_{ij}^{-1} \otimes q_i^{-1} \otimes q_j
$$

对应的旋转向量 $\theta n$ 与四元数关系：

$$
q_e = \left[\cos\frac{\theta}{2},\; \sin\frac{\theta}{2}\,n\right]
$$

所以

$$
2\,\mathrm{vec}(q_e) = 2\sin\frac{\theta}{2}\,n
\;\approx\; \theta n \quad (\theta \text{ 较小})
$$

代码用 $2\,[x,y,z]$ 作为旋转残差。它是**小角度近似**：

- 优点：计算简单、无需 `acos`，对自动微分友好；
- 缺点：大角度时残差尺度被压缩，收敛路径和严格 Log 不同；
- 不同实现采用不同约定，不能把 $2\,\mathrm{vec}$、SO(3) Log 和 SE(3) Log 当成同一种残差。

例如误差角为 $0.1$ rad，$2\sin(0.05)\approx0.099958$，非常接近 $0.1$；误差角为 $\pi$ 时，近似残差长度只有 $2$，而 Log 的长度是 $\pi$。它们都能在零误差处取零，但远离零时尺度不同。

### 2.2 四元数双覆盖

$q$ 和 $-q$ 表示同一个旋转。如果不处理，$q_e$ 可能在两次迭代间突然反号，残差从 $+\theta$ 跳到 $-\theta$，雅可比不连续。

代码的处理：

```cpp
if (qe[0] < T(0.0)) {
    for (int k = 0; k < 4; ++k) qe[k] = -qe[k];
}
```

把 $q_e$ 统一到 $w \ge 0$ 的半球。

这能统一大多数情况下的符号，但 $w=0$ 对应 $180^\circ$ 的分支边界，不能消除那里固有的轴符号歧义。对当前各向同性平方旋转项，整体反号本身不改变代价；统一符号有助于一致地解释残差和选择短旋转分支。

---

## 3. 信息矩阵与白化

每条边有 6×6 协方差 $\Sigma_{ij}$。残差应白化：

$$
\tilde e_{ij} = \Sigma_{ij}^{-1/2} e_{ij}
$$

如果协方差是对角的：

$$
\Sigma_{ij} = \mathrm{diag}(\sigma_r^2,\sigma_r^2,\sigma_r^2,
\sigma_t^2,\sigma_t^2,\sigma_t^2)
$$

白化就是逐项除以 $\sigma_r$ 或 $\sigma_t$，代码正是这么做的：

```cpp
residual[0] = T(2.0) * qe[1] / T(sigma_rot_);
...
residual[3] = et[0] / T(sigma_trans_);
```

一般情形：

$$
\Sigma^{-1} = L^\top L
\quad\Longrightarrow\quad
\tilde e = L\,e
$$

其中 $L$ 是 $\Sigma^{-1}$ 的 Cholesky 因子。Ceres 的 `Problem` 不直接接收协方差，所以**白化必须你在残差里做**。

这里特意把因子定义为 $\Sigma^{-1}=L^\top L$，所以 $L$ 是上三角平方根信息矩阵。如果库返回的是下三角因子 $L_{low}$，满足 $\Sigma^{-1}=L_{low}L_{low}^\top$，就应该乘 $L_{low}^\top$。

例如旋转误差 $0.02$ rad、平移误差 $0.1$ m，若标准差分别是 $0.01$ rad 和 $0.05$ m，白化后两者都是 $2$，都表示“两倍噪声尺度”。直接把 rad 和 m 混在一起平方相加，则没有这种统计意义。

---

## 4. 流形与左扰动

四元数有 4 个数，但只有 3 个自由度。优化变量不能直接相加：

$$
q+\eta,\qquad \eta\in\mathbb R^4
\quad\text{通常会破坏 }\|q\|=1.
$$

这里 $\eta$ 是对四个存储分量的普通加法；下面的 $\delta\in\mathbb R^3$ 才是三个独立旋转增量，二者的维度和含义不同。

Ceres 的 `Manifold` 定义

$$
q \boxplus \delta
= \left[\cos\|\delta\|,\;
\sin\|\delta\|\,\frac{\delta}{\|\delta\|}\right] \otimes q
$$

这是**左乘单位增量**，在单位初值上保持单位范数，并非每步显式归一化或修复非单位初值。这里 $\delta$ 是半角切空间参数，物理旋转向量为 $2\delta$；Ceres 的 `PlusJacobian` 与此一致。

对残差求导时，链式法则是：

$$
\frac{\partial e}{\partial \delta}
= \frac{\partial e}{\partial q}\,
\frac{\partial (q \boxplus \delta)}{\partial \delta}
$$

自动微分对 $q$ 求的是普通偏导，Ceres 再用 `PlusJacobian` 把它投影到 3 维切空间。**你只需要保证残差对 $q$ 的表达式正确，流形的部分交给 Ceres。**

可以把流形理解为“用 3 个自由增量，安全地修改 4 个有约束的存储分量”。$\delta=0$ 时使用极限，增量四元数是 $(1,0,0,0)$，原姿态不变。

小增量下，$\cos\|\delta\|\approx1$、$\sin\|\delta\|\delta/\|\delta\|\approx\delta$。由 Hamilton 乘法可得：

$$
\left.\frac{\partial(q\boxplus\delta)}{\partial\delta}\right|_{\delta=0}
=\begin{bmatrix}
-x&-y&-z\\
w&z&-y\\
-z&w&x\\
y&-x&w
\end{bmatrix}\in\mathbb R^{4\times3}.
$$

因此，一个六维边残差的 ambient 旋转雅可比是 $6\times4$；再乘这个 $4\times3$ 矩阵，才得到求解时使用的 $6\times3$ 局部雅可比。不要为了“只有 3 个自由度”就给接口少写一列。

> 如果你自己推导解析雅可比，必须确认框架用的是左扰动还是右扰动。左右扰动差一个 $R$，符号写反会导致优化不收敛，而且很难从表面看出来。

---

## 5. 稀疏性

每条边只连两个位姿。$H = J^\top J$ 是 $6N\times 6N$ 的块稀疏矩阵，每个 6×6 块只在边连接的两个节点处非零：

```
        T0      T1      T2      T3
 T0 [  H00     H01      0       0   ]
 T1 [  H10     H11     H12      0   ]
 T2 [   0      H21     H22     H23  ]
 T3 [   0       0      H32     H33  ]
```

N=1000 时 $H$ 有 3600 万个元素，非零不到 1%。所以：

```cpp
options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
```

而不是 `DENSE_QR`。求解器选择错误不会让结果错，但会慢一到两个数量级。

---

## 6. Gauge freedom

把所有位姿同时左乘一个全局变换 $T_g$：

$$
T_i' = T_g T_i
$$

则

$$
T_i'^{-1}T_j' = T_i^{-1}T_g^{-1}T_gT_j = T_i^{-1}T_j
$$

所有残差不变。也就是说，位姿图的解只在“整体位姿”意义下唯一，$H$ 至少有 6 维零空间（3 平移 + 3 旋转）。

处理方式：固定第一个位姿：

```cpp
problem.SetParameterBlockConstant(init[0].q);
problem.SetParameterBlockConstant(init[0].t);
```

对连通、无额外退化的位姿图，这样消掉全局 6 维 gauge。有限协方差的先验则表达软约束，会影响结果和不确定性；它不等于硬固定，也不保证数值上总是更稳。图有多个连通分量时，每个分量都要处理 gauge。

---

## 7. 完整代码

```cpp
// 第 3 课配套代码：SE(3) 位姿图优化
//
// 问题：给定一组位姿节点和它们之间的相对位姿测量（里程计边 + 回环边），
//       求最大似然位姿。
//
// 残差：e_ij = [2 vec(q_e), t_e]，不是严格的 SE(3) Log
// 目标：min Σ || e_ij ||^2_{Σ_ij}
//
// 关键点：
//   1. 四元数是 4 维，但旋转只有 3 个自由度 -> 必须用 Manifold（Ceres 的 ⊞）
//   2. 残差是 6 维（旋转 3 + 平移 3），不是 7 维直接相减
//   3. 固定第一个位姿，消除 gauge freedom（整体平移/旋转不变性）
//   4. H = J^T J 是稀疏的，用 SPARSE_NORMAL_CHOLESKY

#include <ceres/ceres.h>

#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

// ----------------------------- 四元数工具 -----------------------------
// 约定：q = [w, x, y, z]，Hamilton 乘法，对应 Ceres 的 QuaternionManifold。
// 全部写成模板，double 和 ceres::Jet 都能用（自动微分要求）。

template <typename T>
void QuatMul(const T* a, const T* b, T* out) {
    const T w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    const T x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    const T y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    const T z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    out[0] = w;
    out[1] = x;
    out[2] = y;
    out[3] = z;
}

template <typename T>
void QuatConj(const T* q, T* out) {
    out[0] = q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

// out = R(q) * v，用四元数旋转向量（比先转矩阵再乘更省）。
template <typename T>
void QuatRotate(const T* q, const T* v, T* out) {
    const T w = q[0], x = q[1], y = q[2], z = q[3];
    const T tx = T(2.0) * (y * v[2] - z * v[1]);
    const T ty = T(2.0) * (z * v[0] - x * v[2]);
    const T tz = T(2.0) * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * tx + (y * tz - z * ty);
    out[1] = v[1] + w * ty + (z * tx - x * tz);
    out[2] = v[2] + w * tz + (x * ty - y * tx);
}

void NormalizeQuat(double* q) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; ++i) q[i] /= n;
}

// ----------------------------- 数据结构 -----------------------------
struct Pose {
    double q[4] = {1.0, 0.0, 0.0, 0.0};  // w, x, y, z
    double t[3] = {0.0, 0.0, 0.0};
};

struct Edge {
    int i;
    int j;
    Pose meas;  // T_ij = T_i^{-1} * T_j
};

// ----------------------------- 残差块 -----------------------------
struct EdgeCost {
    EdgeCost(const double* q_ij, const double* t_ij, double sigma_rot, double sigma_trans)
        : sigma_rot_(sigma_rot), sigma_trans_(sigma_trans) {
        for (int k = 0; k < 4; ++k) q_ij_[k] = q_ij[k];
        for (int k = 0; k < 3; ++k) t_ij_[k] = t_ij[k];
    }

    template <typename T>
    bool operator()(const T* const q_i, const T* const t_i,
                    const T* const q_j, const T* const t_j,
                    T* residual) const {
        // 1) 旋转残差：q_e = q_ij^{-1} * q_i^{-1} * q_j
        T qij[4] = {T(q_ij_[0]), T(q_ij_[1]), T(q_ij_[2]), T(q_ij_[3])};
        T qij_inv[4];
        QuatConj(qij, qij_inv);

        T qi_inv[4];
        QuatConj(q_i, qi_inv);

        T tmp[4];
        QuatMul(qi_inv, q_j, tmp);
        T qe[4];
        QuatMul(qij_inv, tmp, qe);

        // 四元数双覆盖：q 与 -q 是同一个旋转。统一到 w >= 0 的半球，
        // 否则相邻两次迭代的残差可能突然反号，雅可比会跳变。
        if (qe[0] < T(0.0)) {
            for (int k = 0; k < 4; ++k) qe[k] = -qe[k];
        }

        // 2 * vec(q) 是旋转向量的小角度近似（qe ≈ [1, θ/2] 时正好等于 θ）。
        // 此处使用小角度近似，不等同于完整的 SO(3) Log。
        residual[0] = T(2.0) * qe[1] / T(sigma_rot_);
        residual[1] = T(2.0) * qe[2] / T(sigma_rot_);
        residual[2] = T(2.0) * qe[3] / T(sigma_rot_);

        // 2) 平移残差：e_t = R_ij^T * ( R_i^T * (t_j - t_i) - t_ij )
        T tij[3] = {T(t_ij_[0]), T(t_ij_[1]), T(t_ij_[2])};
        T d[3] = {t_j[0] - t_i[0], t_j[1] - t_i[1], t_j[2] - t_i[2]};

        T ri[3];
        QuatRotate(qi_inv, d, ri);
        ri[0] -= tij[0];
        ri[1] -= tij[1];
        ri[2] -= tij[2];

        T et[3];
        QuatRotate(qij_inv, ri, et);

        residual[3] = et[0] / T(sigma_trans_);
        residual[4] = et[1] / T(sigma_trans_);
        residual[5] = et[2] / T(sigma_trans_);
        return true;
    }

    double q_ij_[4];
    double t_ij_[3];
    double sigma_rot_;
    double sigma_trans_;
};

// ----------------------------- 真值与测量 -----------------------------
const double kPi = 3.14159265358979323846;

// T_ij = T_i^{-1} * T_j
Pose RelativePose(const Pose& a, const Pose& b) {
    Pose out;
    double qa_inv[4];
    QuatConj(a.q, qa_inv);
    QuatMul(qa_inv, b.q, out.q);
    NormalizeQuat(out.q);

    double d[3] = {b.t[0] - a.t[0], b.t[1] - a.t[1], b.t[2] - a.t[2]};
    QuatRotate(qa_inv, d, out.t);
    return out;
}

double AteRmse(const std::vector<Pose>& est, const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        const double dx = est[i].t[0] - gt[i].t[0];
        const double dy = est[i].t[1] - gt[i].t[1];
        const double dz = est[i].t[2] - gt[i].t[2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

int main() {
    const int kNumNodes = 20;
    const double kRadius = 5.0;
    const double kSigmaRot = 0.02;    // rad
    const double kSigmaTrans = 0.1;   // m

    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // 1) 真值轨迹：半径 5 的圆，机头沿切线方向
    std::vector<Pose> gt(kNumNodes);
    for (int i = 0; i < kNumNodes; ++i) {
        const double theta = 2.0 * kPi * i / kNumNodes;
        gt[i].t[0] = kRadius * std::cos(theta);
        gt[i].t[1] = kRadius * std::sin(theta);
        gt[i].t[2] = 0.0;

        const double yaw = theta + kPi / 2.0;
        gt[i].q[0] = std::cos(yaw / 2.0);
        gt[i].q[1] = 0.0;
        gt[i].q[2] = 0.0;
        gt[i].q[3] = std::sin(yaw / 2.0);
    }

    // 2) 边：相邻里程计边 + 每隔 5 个节点的回环边
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < kNumNodes; ++i) {
        pairs.emplace_back(i, (i + 1) % kNumNodes);
    }
    for (int i = 0; i < kNumNodes; i += 5) {
        pairs.emplace_back(i, (i + 5) % kNumNodes);
    }

    // 3) 给每条边生成带噪测量
    std::vector<Edge> edges;
    edges.reserve(pairs.size());
    for (const auto& pr : pairs) {
        Pose meas = RelativePose(gt[pr.first], gt[pr.second]);

        const double yaw_noise = gauss(rng) * kSigmaRot;
        const double dq[4] = {std::cos(yaw_noise / 2.0), 0.0, 0.0,
                              std::sin(yaw_noise / 2.0)};
        double q_tmp[4];
        QuatMul(meas.q, dq, q_tmp);
        for (int k = 0; k < 4; ++k) meas.q[k] = q_tmp[k];
        NormalizeQuat(meas.q);

        for (int k = 0; k < 3; ++k) meas.t[k] += gauss(rng) * kSigmaTrans;

        edges.push_back({pr.first, pr.second, meas});
    }

    // 4) 初值：只用相邻边做航迹推算（deaded reckoning），误差会累积
    std::vector<Pose> init(kNumNodes);
    init[0] = gt[0];
    for (int i = 0; i < kNumNodes - 1; ++i) {
        double q_next[4];
        QuatMul(init[i].q, edges[i].meas.q, q_next);
        NormalizeQuat(q_next);

        double t_local[3];
        QuatRotate(init[i].q, edges[i].meas.t, t_local);

        for (int k = 0; k < 4; ++k) init[i + 1].q[k] = q_next[k];
        for (int k = 0; k < 3; ++k) init[i + 1].t[k] = init[i].t[k] + t_local[k];
    }

    const double ate_before = AteRmse(init, gt);

    // 5) 构建 Ceres 问题
    ceres::Problem problem;
    for (int i = 0; i < kNumNodes; ++i) {
        // 四元数参数块挂 Manifold：单位初值在 3 维切空间上更新。
        problem.AddParameterBlock(init[i].q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(init[i].t, 3);
    }

    // gauge freedom：整体平移 + 旋转不改变任何残差。
    // 固定第一个位姿，否则 H 奇异，线性求解器会失败或给出漂移解。
    problem.SetParameterBlockConstant(init[0].q);
    problem.SetParameterBlockConstant(init[0].t);

    for (const auto& e : edges) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<EdgeCost, 6, 4, 3, 4, 3>(
                new EdgeCost(e.meas.q, e.meas.t, kSigmaRot, kSigmaTrans));

        // 练习：把 nullptr 换成 new ceres::HuberLoss(1.0)，再手动加几条错误回环。
        ceres::LossFunction* loss = nullptr;

        problem.AddResidualBlock(cost, loss, init[e.i].q, init[e.i].t,
                                 init[e.j].q, init[e.j].t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";
    std::cout << "nodes: " << kNumNodes << ", edges: " << edges.size() << "\n";
    std::cout << "ATE before: " << ate_before << " m\n";
    std::cout << "ATE after : " << AteRmse(init, gt) << " m\n";
    return 0;
}
```

---

## 8. 代码逐段讲解

### 8.1 为什么四元数工具要写成模板

```cpp
template <typename T>
void QuatMul(const T* a, const T* b, T* out);
```

`EdgeCost::operator()` 里会以 `T = ceres::Jet<double, 12>` 调用这些函数。如果写成 `double` 版本：

- 要么编译失败（Jet 不能隐式转 double）；
- 要么在内部把 Jet 截断成 double，雅可比全是 0。

`QuatRotate` 先算 `tx,ty,tz` 再写 `out`，所以 `out` 和 `v` 可以指向同一块内存，不会出错。

### 8.2 残差的四个参数块

```cpp
bool operator()(const T* const q_i, const T* const t_i,
                const T* const q_j, const T* const t_j,
                T* residual) const;
```

对应：

```cpp
ceres::AutoDiffCostFunction<EdgeCost, 6, 4, 3, 4, 3>
//                                  ^残差 ^q_i ^t_i ^q_j ^t_j
```

**模板参数必须与 `AddResidualBlock` 的顺序一致**：

```cpp
problem.AddResidualBlock(cost, loss,
                         init[e.i].q, init[e.i].t,
                         init[e.j].q, init[e.j].t);
```

### 8.3 旋转残差

```cpp
QuatMul(qi_inv, q_j, tmp);       // q_i^{-1} q_j
QuatMul(qij_inv, tmp, qe);       // q_ij^{-1} q_i^{-1} q_j
```

注意顺序：**先 `i` 的逆，再 `j`**。反过来就是 $T_j^{-1}T_i$，残差会完全错。

### 8.4 平移残差

```cpp
QuatRotate(qi_inv, d, ri);       // R_i^T (t_j - t_i)
ri[k] -= tij[k];                  // 减去测量平移
QuatRotate(qij_inv, ri, et);     // R_ij^T (...)
```

几何含义：

1. $t_j - t_i$ 是世界系下的位移；
2. 转到 `i` 的机体系：$R_i^\top(t_j - t_i)$；
3. 减去测量值 $t_{ij}$；
4. 转到 `ij` 的机体系：$R_{ij}^\top(\dots)$。

### 8.5 初值

```cpp
init[0] = gt[0];
for (int i = 0; i < kNumNodes - 1; ++i) {
    QuatMul(init[i].q, edges[i].meas.q, q_next);
    QuatRotate(init[i].q, edges[i].meas.t, t_local);
    init[i + 1].t[k] = init[i].t[k] + t_local[k];
}
```

这就是**航迹推算**（dead reckoning）：只用相邻边的测量，把位姿一步步乘起来。误差会累积，20 步后末端偏离真值半米左右。回环边的作用就是把累积误差拉回来。

### 8.6 求解器

```cpp
options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
```

因为 $H$ 是块稀疏的。Ceres 会自动分析稀疏结构，选择消元顺序。

---

## 9. ATE

代码里的 ATE（Absolute Trajectory Error）：

$$
\mathrm{ATE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left\|t_i - t_i^{\mathrm{gt}}\right\|^2}
$$

因为第一个位姿被固定到真值，所以不需要做 Umeyama 对齐，直接比较即可。

预期：

```
nodes: 20, edges: 24
ATE before: 本机基线约 0.422 m
ATE after : 本机约 0.168 m
```

具体值依赖随机数与平台；应验证代价与轨迹误差合理下降，不能承诺降低一个数量级。带噪测量下，真值也不一定使所有边的残差为零。

---

## 10. 运行

```bash
cd ~/Desktop/learn/ceres_learn
cmake -S . -B build
cmake --build build -j
./build/base03_pose_graph
```

---

## 11. 常见坑

1. **测量方向反了**：$T_{ij} = T_i^{-1}T_j$ 还是 $T_j^{-1}T_i$。反了不报错，但会发散或收敛到错解。
2. **四元数顺序**：Ceres 的 `QuaternionManifold` 是 $[w,x,y,z]$；Eigen 的 `coeffs()`/原始存储为 $[x,y,z,w]$，构造函数 `Quaterniond(w,x,y,z)` 则是标量在前。直接把 Eigen 存储交给 Ceres 时使用 `EigenQuaternionManifold`，或先显式重排为 `wxyz`。
3. **参数相关表达式失去 `T` 类型**：编译失败，或手动提取标量后丢失导数。
4. **忘记固定第一个位姿**：$H$ 奇异，求解失败或整体漂移。
5. **sigma 单位不统一**：旋转用弧度、平移用米，量级差太多会让优化被一项主导。
6. **初值太差**：位姿图非凸，角度错 180° 时 $2\,\mathrm{vec}$ 会走错分支。
7. **信息矩阵只取对角**：真实边有 6×6 协方差，旋转-平移耦合不能忽略。
8. **$q$ 不归一化**：初值和测量四元数都要归一化。

---

## 12. 练习

1. **加一条错误回环**：把某条 $i\to i+5$ 的测量平移加 2 米。先用 `loss = nullptr`，再换 `HuberLoss(1.0)`，观察错误边如何拉歪整张图。
2. **扩大规模**：把 `kNumNodes` 改成 500，分别用 `SPARSE_NORMAL_CHOLESKY` 和 `DENSE_QR`，比较耗时。
3. **只固定旋转**：去掉 `SetParameterBlockConstant(init[0].t)`，观察解是否整体漂移、`summary` 是否报奇异。
4. **放大旋转噪声**：把 `kSigmaRot` 改成 `0.2`，观察 ATE 和收敛速度。
5. **换成严格 Log map**：把 $2\,\mathrm{vec}(q_e)$ 换成
   $$
   \theta = 2\arctan_2(\|\mathrm{vec}\|, w),\qquad
   r = \theta\frac{\mathrm{vec}}{\|\mathrm{vec}\|}
   $$
   用 `atan2` 的 Jet 重载实现，比较多组初值下的收敛差异。

---

## 13. 下一站：BA

位姿图只有位姿。视觉 BA 多了一类变量——**路标点**，残差变成重投影误差：

$$
r = \pi\!\left(T_{cw}X_w\right) - \mathbf{uv}
$$

你会遇到三个新问题：

1. 投影函数对位姿和三维点的雅可比（李群 + 链式法则）；
2. $H$ 的 Schur 补结构（消元路标点）；
3. 外点比例远高于位姿图（误匹配）。

那才是 ORB-SLAM / VINS 后端的核心。
