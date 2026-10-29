# 第 5 课：IMU 预积分

> 对应代码：`base05_imu.cpp`
> 目标：理解 IMU 预积分为什么存在、怎么递推、残差怎么写，以及零偏和协方差怎么处理。
> 前置：第 3 课（SE(3)）、第 4 课（BA）。

---

## 1. IMU 测量模型

IMU 输出角速度和比力（specific force）：

$$
\tilde{\boldsymbol\omega}
= \boldsymbol\omega + \mathbf b_g + \mathbf n_g
$$

$$
\tilde{\mathbf a}
= R^\top\left(\mathbf a_w - \mathbf g\right)
+ \mathbf b_a + \mathbf n_a
$$

- $\omega$：body 系角速度；
- $a_w = dv/dt$：世界系运动学加速度；静止时为零，比力与它的关系是 $a_w = R f + g$；
- $g$：世界系重力，通常取 $(0,0,-9.81)$；
- $b_g, b_a$：陀螺仪/加速度计零偏；
- $n_g, n_a$：白噪声。

### 1.1 先把坐标系和单位对齐

本课 $R=R_{wb}$ 把机体系向量转到世界系，$p,v,g,a_w$ 在世界系；IMU 输出、零偏和测量噪声在机体系。两条测量方程中的各项必须处于同一坐标系，不能直接把机体系读数和世界系重力相加。

| 量 | 直白含义 | 单位 |
|---|---|---|
| $\tilde\omega$ | 陀螺仪实际读到的三轴角速度 | rad/s |
| $\tilde a$ | 加速度计实际读到的三轴比力 | m/s² |
| $b_g,b_a$ | 传感器长期偏离零点的部分 | 分别为 rad/s、m/s² |
| $n_g,n_a$ | 每次测量上下波动的部分 | 与对应读数相同 |
| $p,v,a_w$ | 世界系位置、速度、速度变化率 | m、m/s、m/s² |

### 1.2 静止为什么还能读到 $9.81$

加速度计测的是比力，不是直接输出 $a_w$。先忽略零偏与噪声，记比力为 $f$：

$$
f=R^\top(a_w-g),\qquad a_w=Rf+g.
$$

水平静止且 $R=I$ 时，$a_w=0$，所以 $f=-g=(0,0,9.81)$。把这个读数转回世界系，再加 $g$，才得到真实运动学加速度为零。

理想自由落体时 $a_w=g$，因此 $f=0$。这两个极端例子可以帮助检查重力符号。恢复运动学加速度的完整计算是：

$$
\hat\omega=\tilde\omega-b_g,\qquad
\hat a_w=R(\tilde a-b_a)+g.
$$

先在传感器所在的机体系减零偏，再旋转，再加世界系重力。

注意 $a_m = R^\top(a_w - g)$ 里的**减号**。很多实现把重力定义成 $(0,0,9.81)$ 再改符号，两种约定不能混。

---

## 2. 运动学

忽略零偏和噪声：

$$
\dot R = R\left[\boldsymbol\omega\right]_\times,
\qquad
\dot{\mathbf v} = \mathbf a_w = R\tilde{\mathbf a} + \mathbf g,
\qquad
\dot{\mathbf p} = \mathbf v
$$

直接积分需要知道每一时刻的 $R$，而 $R$ 又是待优化的状态。如果每次优化迭代都重新积分，代价太高。

$\dot p$ 表示位置对时间的导数；积分则是把每个微小时段的变化累加。暂时认为一个采样间隔 $h$ 内角速度和比力不变：

$$
R_{k+1}\approx R_k\mathrm{Exp}(\hat\omega_k h),
$$

$$
v_{k+1}\approx v_k+(R_k\hat a_k+g)h,
\qquad
p_{k+1}\approx p_k+v_kh+\frac12(R_k\hat a_k+g)h^2.
$$

这里 $\hat a_k=\tilde a_k-b_a$ 仍是机体系比力。$R$ 的更新右乘增量，是因为输入角速度在机体系；这与 Ceres 参数更新采用的半角左增量是不同操作，不能混用。

单位也能检查公式：角速度乘秒得到弧度，加速度乘秒得到速度，加速度乘秒平方得到位置。$\mathrm{Exp}$ 的含义见[第 3 课第 1.3 节](03_pose_graph.md#13-exp-和-log把三维小增量与旋转联系起来)。

---

## 3. 预积分的核心思想

把两帧之间的 IMU 测量**预先积分**成一个与起始状态无关的量：

$$
\Delta R_{ij} = R_i^\top R_j
$$

$$
\Delta v_{ij} = R_i^\top\left(v_j - v_i - g\Delta t\right)
$$

$$
\Delta p_{ij} = R_i^\top\left(p_j - p_i - v_i\Delta t - \frac12 g\Delta t^2\right)
$$

这些量只依赖 IMU 测量和零偏，不依赖 $R_i, p_i, v_i$。所以：

- 前端可以在两帧之间**只积分一次**；
- 后端每次迭代直接用 $\Delta R, \Delta v, \Delta p$ 构造残差；
- 零偏变化时，用一阶修正更新预积分量，不必重新积分。

这就是 VINS-Mono、ORB-SLAM3 IMU 模式的基础。

### 3.1 为什么换到第 $i$ 帧就能预先计算

要区分两种写法：上面右端用状态表达的是“状态预测的相对变化”；真正缓存的测量预积分量，则从 IMU 数据在参考零偏处计算出来。有噪声时，这两者不会严格相等，它们的差就是后面要优化的残差。

令两帧总时长 $T=t_j-t_i$，并记 $\Delta R_i(t)=R_i^\top R(t)$，它把时刻 $t$ 的机体系向量转到第 $i$ 帧机体系。在理想无噪声模型下：

$$
\Delta v_{ij}=\int_{t_i}^{t_j}\Delta R_i(t)\bigl(\tilde a(t)-b_a\bigr)\,dt,
$$

$$
\Delta p_{ij}=\int_{t_i}^{t_j}(t_j-t)\Delta R_i(t)\bigl(\tilde a(t)-b_a\bigr)\,dt.
$$

第二个式子是加速度积分两次的结果：越早出现的一小段加速度，对最终位置作用的时间越长，所以带有 $(t_j-t)$。

从世界系速度积分式减去 $gT$、再乘 $R_i^\top$，就得到第一个积分；对位置同样扣掉 $p_i+v_iT+\frac12gT^2$，就得到第二个。这样，起始姿态、速度、位置和已知重力的影响都留在状态预测侧，IMU 累加只在局部参考帧中进行。

因此，缓存的 $\Delta R,\Delta v,\Delta p$ 可以由 IMU 序列和参考零偏计算，不需要知道第 $i$ 帧在世界中的绝对位姿。后端更新 $R_i,p_i,v_i$ 时无需重复积分；参考零偏改变较大时才需要重积分。

---

## 4. 离散递推

### 4.1 欧拉积分

$$
\Delta R_{k+1}
= \Delta R_k \exp\left((\tilde{\boldsymbol\omega}_k - \mathbf b_g)\Delta t\right)
$$

$$
\Delta v_{k+1}
= \Delta v_k + \Delta R_k\left(\tilde{\mathbf a}_k - \mathbf b_a\right)\Delta t
$$

$$
\Delta p_{k+1}
= \Delta p_k + \Delta v_k\Delta t
+ \frac12\Delta R_k\left(\tilde{\mathbf a}_k - \mathbf b_a\right)\Delta t^2
$$

这里递推下标 $k$ 遍历两帧之间的高频 IMU 采样。为了避免与两帧总时长混淆，可把本节每步的 $\Delta t$ 记为 $h$，第 3、5 节两端状态之间的 $\Delta t$ 记为 $T=Nh$（等间隔采样）。

初始化时，从第 $i$ 帧到自身没有变化：

$$
\Delta R_0=I,\qquad \Delta v_0=0,\qquad\Delta p_0=0.
$$

每步先取 $a_i^{(k)}=\Delta R_k(\tilde a_k-b_a)$：它表示当前测到的比力在第 $i$ 帧中的坐标。用它增加速度、位置，再更新旋转。右端全部使用旧的 $\Delta R_k,\Delta v_k$，不能更新一半后又把新值代回旧公式。

### 4.1.1 手算两步：看清为什么预积分里保留了重力对应的读数

假设机体不旋转、零偏为零，读数恒为 $\tilde a=(1,0,9.81)$ m/s²，每步 $h=0.1$ s：

| 已积分时长 | $\Delta v$（m/s） | $\Delta p$（m） |
|---|---|---|
| $0$ | $(0,0,0)$ | $(0,0,0)$ |
| $0.1$ s | $(0.1,0,0.981)$ | $(0.005,0,0.04905)$ |
| $0.2$ s | $(0.2,0,1.962)$ | $(0.02,0,0.1962)$ |

第二步位置使用旧速度，因此横向增量是 $0.1\times0.1+\frac12\times1\times0.1^2=0.015$ m，加上旧位置 $0.005$ m 得到 $0.02$ m。

这还不是完整的世界系运动。若初始 $R_i=I,v_i=0,p_i=0$，恢复两帧运动时：

$$
v_j=v_i+gT+R_i\Delta v_{ij}=(0.2,0,0),
$$

$$
p_j=p_i+v_iT+\frac12gT^2+R_i\Delta p_{ij}=(0.02,0,0).
$$

竖直项恰好抵消，实际只沿水平方向加速。预积分里不能再提前减一遍世界系重力，否则会重复扣除。

### 4.2 中值积分

本节 $\omega_k=\tilde\omega_k-b_g$、$a_k=\tilde a_k-b_a$，均已经减过参考零偏。中值形式同时利用区间两端读数，减少只使用左端测量造成的离散化偏差。

$$
\Delta R_{k+1}
= \Delta R_k
\exp\left(\frac12(\boldsymbol\omega_k + \boldsymbol\omega_{k+1})\Delta t\right)
$$

$$
\Delta v_{k+1}
= \Delta v_k
+ \frac12\left(\Delta R_k\mathbf a_k + \Delta R_{k+1}\mathbf a_{k+1}\right)\Delta t
$$

$$
\Delta p_{k+1}
= \Delta p_k + \Delta v_k\Delta t
+ \frac14\left(\Delta R_k\mathbf a_k + \Delta R_{k+1}\mathbf a_{k+1}\right)\Delta t^2
$$

本课代码用欧拉积分，`dt = 0.001`，误差足够小。真实系统通常用中值或更高阶积分。

---

## 5. 残差

状态变量：

- $q_i, p_i, v_i$：第 i 帧姿态、位置、速度；
- $q_j, p_j, v_j$：第 j 帧。

残差 9 维：

$$
\mathbf r_R
= \mathrm{Log}\left(\Delta R_{ij}^\top R_i^\top R_j\right)
$$

$$
\mathbf r_v
= R_i^\top\left(v_j - v_i - g\Delta t\right) - \Delta v_{ij}
$$

$$
\mathbf r_p
= R_i^\top\left(p_j - p_i - v_i\Delta t - \frac12 g\Delta t^2\right) - \Delta p_{ij}
$$

白化后除以各自的噪声标准差。Ceres 的残差块：

```cpp
AutoDiffCostFunction<ImuError, 9, 4, 3, 3, 4, 3, 3>
```

对应参数块：

```
q_i(4), p_i(3), v_i(3), q_j(4), p_j(3), v_j(3)
```

### 5.1 旋转残差

$\Delta R^\top R_i^\top R_j$ 是 $\mathrm{SO}(3)$ 元素，取 $\mathrm{Log}$ 得到 3 维旋转向量。代码里用 `2 * vec(q)` 的小角度近似，和第 3 课位姿图一致。

### 5.2 把九维残差按“预测减测量”读一遍

例如速度部分：先由当前两端状态算 $v_j-v_i$，减去已知重力在时长 $T$ 内产生的变化 $gT$，转到第 $i$ 帧，再减去 IMU 缓存的 $\Delta v_{ij}$。如果两端状态与测量完全一致，这一项为零。

位置部分还要扣掉初速度带来的 $v_iT$，因为 $\Delta p$ 只代表比力累积造成的位置变化。可以反向检查：由 $r_v=r_p=0$ 解出
$v_j=v_i+gT+R_i\Delta v_{ij}$、$p_j=p_i+v_iT+\frac12gT^2+R_i\Delta p_{ij}$，应当正好恢复上节的运动预测。

九维是 $3$ 维旋转、$3$ 维速度、$3$ 维位置的总和。六个输入参数块存储了 $4+3+3+4+3+3=20$ 个数；应用两端四元数流形后，有 $18$ 个局部自由增量。输入维度与残差维度不必相同。

本课将三组误差分别除以 `sigma_r,sigma_v,sigma_p`，相当于假设九维协方差为分组三轴等方差的对角矩阵。真实预积分误差通常有关联，应整体白化，不能把三个教学权重误认为完整传播结果。

---

## 6. 零偏

### 6.1 为什么零偏重要

陀螺仪零偏 $0.01\ \mathrm{rad/s}$ 在 10 秒内会累积 $0.1\ \mathrm{rad}$ 的姿态误差；加速度计零偏 $0.1\ \mathrm{m/s^2}$ 在 10 秒内会累积 $5\ \mathrm{m}$ 的位置误差。

零偏不是常数，而是缓慢游走：

$$
\dot{\mathbf b}_g = \mathbf n_{bg},
\qquad
\dot{\mathbf b}_a = \mathbf n_{ba}
$$

若连续时间零偏随机游走的噪声谱密度为 $Q_{bg}$，离散模型为：

$$
\mathbf b_{g,j} = \mathbf b_{g,i} + \mathbf w_{bg},
\qquad \mathbf w_{bg}\sim\mathcal N(0,Q_{bg}\Delta t)
$$

增量标准差随 $\sqrt{\Delta t}$ 缩放，不是把连续白噪声简单乘 $\Delta t$。加速度计零偏同理；若使用离散采样噪声而非连续谱密度，必须另行说明单位及离散化方式。

常值零偏和独立随机噪声的累积规律也不同。若一维加速度零偏始终为 $b$，忽略转动，有 $\delta v=bT$、$\delta p=\frac12bT^2$。误差每步方向一致，会持续积累。

若每次采样的加速度噪声独立、标准差是固定的 $\sigma_a$，则 $N$ 步速度噪声为 $h\sum_{k=1}^N n_k$，所以：

$$
\operatorname{Var}(\delta v)=N\sigma_a^2h^2,
\qquad \operatorname{Std}(\delta v)=\sigma_a\sqrt{Th}.
$$

这里假设固定的是**每个样本的标准差**。若固定的是连续噪声谱密度 $q_a$，采用等效区间平均噪声时 $\sigma_a^2=q_a/h$，于是速度方差变成 $q_aT$。这解释了为什么“采样噪声”和“噪声密度”的公式不能直接互换。

### 6.2 一阶修正

如果零偏变化很小，不需要重新积分，只需一阶修正：

$$
\Delta R(\mathbf b_g)
\approx \Delta R(\bar{\mathbf b}_g)
\exp\left(
J_{R,b_g}
\left(\mathbf b_g - \bar{\mathbf b}_g\right)
\right)
$$

$$
\Delta v(\mathbf b)
\approx \Delta v(\bar{\mathbf b})
+ \frac{\partial \Delta v}{\partial \mathbf b_g}\delta\mathbf b_g
+ \frac{\partial \Delta v}{\partial \mathbf b_a}\delta\mathbf b_a
$$

$$
\Delta p(\mathbf b)
\approx \Delta p(\bar{\mathbf b})
+ \frac{\partial \Delta p}{\partial \mathbf b_g}\delta\mathbf b_g
+ \frac{\partial \Delta p}{\partial \mathbf b_a}\delta\mathbf b_a
$$

$J_{R,b_g}$ 是把旋转误差映射到 3 维切空间后、对零偏的 $3\times 3$ 雅可比，不是对 $3\times 3$ 旋转矩阵直接求导。修正使用右乘 SO(3) 增量，与 Ceres 半角左扰动的参数更新不是同一个约定。这些雅可比在预积分时一起递推；零偏变化过大时仍需重新预积分。

这本质上仍是一阶泰勒展开。$\bar b$ 是原先积分时保存的参考零偏，$\delta b=b-\bar b$ 是本次估计相对参考的变化；雅可比表示“参考零偏变化一点，预积分量变化多少”。

例如无旋转、区间长度 $T$、固定参考坐标系时：

$$
\Delta v(b_a)=\int_0^T(\tilde a-b_a)dt
\quad\Longrightarrow\quad J_{v,b_a}=-TI,
\qquad J_{p,b_a}=-\frac12T^2I.
$$

增加估计零偏，会从读数中减掉更多，所以导数为负。一般运动下还要考虑 $\Delta R$ 的变化，不能把这两个简单矩阵直接用于所有轨迹。每次修正都从保存的参考预积分量出发，不能把同一个 $b-\bar b$ 反复叠加到上次修正结果。

> 本课代码把零偏固定为 0，目的是先讲清楚预积分本身。作业会让你亲手看零偏如何破坏结果。

### 6.3 初始化

静止时：

$$
\mathbf b_g = \mathrm{mean}\left(\tilde{\boldsymbol\omega}\right),
\qquad
\mathbf b_a = \mathrm{mean}\left(\tilde{\mathbf a}\right) - R^\top(-\mathbf g)
$$

如果 $R = I$、$g = (0,0,-9.81)$，则 $R^\top(-g) = (0,0,9.81)$，加速度计均值应该接近 $(0,0,9.81)$。这个“静止初始化”是 VIO 启动的第一步。

此处加速度计零偏公式以姿态/重力已知为前提；单一静止姿态不能同时独立估计任意三轴加速度计零偏与重力方向。不能把比力均值直接当成二者的唯一分解。

---

## 7. 协方差传播

预积分的噪声也要传播，用于：

- IMU 因子的信息矩阵；
- 零偏随机游走；
- 边缘化先验。

线性化：

$$
\Sigma_{k+1}
= A_k\Sigma_k A_k^\top
+ B_k\Sigma_n B_k^\top
$$

$A_k$ 是误差状态转移，$B_k$ 是噪声雅可比。误差状态通常取：

$$
\delta\mathbf x =
\begin{bmatrix}
\delta\boldsymbol\theta\\\delta\mathbf v\\\delta\mathbf p\\
\delta\mathbf b_g\\\delta\mathbf b_a
\end{bmatrix}
\in \mathbb{R}^{15}
$$

上式将五个三维向量**竖直堆叠成列向量**。$A_k\in\mathbb R^{15\times15}$ 说明“原有误差经过一步运动如何变化”；$B_k$ 说明“这一时段的新噪声如何进入状态”。传播公式来自：

$$
\delta x_{k+1}\approx A_k\delta x_k+B_kn_k.
$$

在旧误差和新噪声不相关、噪声零均值的假设下，计算上式的协方差，两个交叉项为零，就得到 $A_k\Sigma_kA_k^\top+B_k\Sigma_nB_k^\top$。若噪声有时间相关性，需要增加相应的相关模型，不能照抄这个独立噪声公式。

### 7.1 一个二维例子：为什么速度误差和位置误差相关

只考虑一步、同一个一维加速度噪声 $n$，其方差为 $\sigma_a^2$。它同时造成 $\delta v=hn$、$\delta p=\frac12h^2n$，因此：

$$
\begin{bmatrix}\delta v\\\delta p\end{bmatrix}
=\underbrace{\begin{bmatrix}h\\\frac12h^2\end{bmatrix}}_B n,
\qquad
\Sigma_{vp}=\sigma_a^2
\begin{bmatrix}
h^2&\frac12h^3\\
\frac12h^3&\frac14h^4
\end{bmatrix}.
$$

非对角项不为零，因为两种误差由同一份噪声造成。这是预积分不能只看各轴标准差的直观原因。该单步、单噪声例子本身是秩亏的，用于说明相关性；不是要把这个 $2\times2$ 矩阵直接拿去构造完整 IMU 因子。

### 7.2 十五维误差转移矩阵怎样逐行读

必须先固定误差定义。下面给出**右乘局部旋转误差** $R_{\mathrm{true}} = R_{\mathrm{est}}\,\mathrm{Exp}(\delta\theta)$、世界系加性速度/位置误差的一阶时间离散示意。记 $\hat\omega = \omega_m - b_g$、$\hat a = a_m - b_a$，各项保留到 $O(\Delta t)$：

$$
A_k =
\begin{bmatrix}
I-[\hat\omega_k]_\times\Delta t & 0 & 0 & -I\Delta t & 0 \\
-R_k[\hat a_k]_\times\Delta t & I & 0 & 0 & -R_k\Delta t \\
0 & I\Delta t & I & 0 & 0 \\
0 & 0 & 0 & I & 0 \\
0 & 0 & 0 & 0 & I
\end{bmatrix}
$$

矩阵中每一个 $I$ 或 $0$ 都是 $3\times3$ 块。按状态顺序逐行看：

- 第一行：姿态误差随转动变化，同时受陀螺零偏误差影响，时间越长，零偏累积越大。
- 第二行：倾角误差使比力转到世界系时方向有偏差，产生 $-R[\hat a]_\times h$；加计零偏产生 $-Rh$ 项。
- 第三行：位置误差累积旧速度误差，所以有 $Ih$。
- 最后两行：零偏的旧误差继续保留；新的随机游走不确定性通过噪声项加入。

这里展示的是导航误差状态，而第 3 节的测量预积分在参考帧中表达；采用不同误差坐标，矩阵中的旋转项也会变化。

这是导航误差状态的一阶示意，不是可直接照抄到任意预积分误差定义中的完整递推。若位置递推保留 $\Delta t^2$，还需相应的 $-\frac{1}{2} R[\hat a]_\times \Delta t^2$ 和 $-\frac{1}{2} R\,\Delta t^2$ 等块，并匹配噪声雅可比和离散噪声协方差；局部/全局误差不可混用。

课件代码中的 `0.001`、`0.01` 是每次采样的噪声标准差，不是噪声密度。因子里的 `sigma_r, sigma_v, sigma_p` 是手工教学权重，不是上述采样噪声本身，也没有实现协方差传播。9 维运动残差应匹配传播得到的完整 $9\times 9$ 协方差；若把两端零偏与随机游走残差一起加入，则常形成 $15\times 15$ 联合协方差。

真实系统里，协方差传播和零偏雅可比是一起递推的，代码量不小。本课不展开，但你必须知道：**IMU 因子的权重不是拍脑袋定的，来自这个协方差**。

---

## 8. 代码讲解

### 8.1 真值轨迹

代码用一条解析轨迹：

$$
p(t) = (3\cos t,\ 3\sin t,\ 0.2t),
\qquad
R(t) = R_z(t)
$$

$$
v(t) = (-3\sin t,\ 3\cos t,\ 0.2),
\qquad
a_w(t) = (-3\cos t,\ -3\sin t,\ 0)
$$

由 $a_m = R^\top(a_w - g)$ 生成加速度计测量，由 $\omega = (0,0,1)$ 生成陀螺仪测量。

### 8.2 预积分

```cpp
const Eigen::Vector3d accel_i = delta_rotation * accel;  // ΔR * a

delta_position += delta_velocity * kDt + 0.5 * accel_i * kDt * kDt;
delta_velocity += accel_i * kDt;

delta_rotation = (delta_rotation * rotation_step).normalized();
```

`delta_position` 和 `delta_velocity` 是 `pre.dp`、`pre.dv` 的 Eigen Map 视图，更新它们就会写入对应数组。`delta_rotation` 是 Eigen 四元数，积分结束后按 `wxyz` 写回 `pre.dq`。`rotation_step` 用 `AngleAxisd(angle, omega / angular_speed)` 表达单步旋转，并在角度过小时使用单位四元数，避免除零。

注意顺序：位置用旧速度，速度用旧的 $\Delta R$，最后更新 $\Delta R$。Eigen 简化了逐分量计算，递推关系仍然是第 4 节的公式。

### 8.3 残差

```cpp
const Vector3 velocity_error =
    rotation_i.conjugate() * (velocity_j - velocity_i - gravity * dt)
    - delta_velocity;

Eigen::Map<Eigen::Matrix<T, 9, 1>> error(residual);
error.template segment<3>(3) = velocity_error / T(sigma_v_);
```

`Vector3` 是 `Eigen::Matrix<T,3,1>`，`rotation_i` 是由 `wxyz` 参数构造的 `Eigen::Quaternion<T>`；共轭对应单位旋转的逆。九维残差依次放旋转、速度和位置，`segment<3>(3)` 表示从下标 3 开始的三个速度残差分量。对预积分常量使用 `cast<T>()`，将常量转成自动微分可参与运算的标量类型。

`ImuError` 的类、构造函数、参数块顺序和外部创建方式均沿用原来的组织，仅用 Eigen 表达内部计算。

### 8.4 固定第一帧

本例是**纯 IMU 链**，没有视觉或位置观测。代码把第一帧的姿态、位置、速度当作已知初始条件固定下来，后续状态可由预积分逐帧传播；这不只是选择坐标系。第一帧速度不固定时，可同时令 $v_i \leftarrow v_i + c$、$p_i \leftarrow p_i + c(t_i - t_0)$，运动残差仍不变。

不能据此宣称一般 VIO 有 6 维全局位姿 gauge：当重力方向在世界系中固定、尺度与零偏在充分激励下可观时，典型 VIO 剩下 3 维全局平移与 1 维绕重力的 yaw gauge，roll/pitch 不可任意全局旋转。本例固定完整首帧和速度提供了更多已知信息。

纯链的各段都有自由末端状态，所以即使测量有噪声、甚至使用了错误零偏，也能让因子残差接近零；`final_cost` $\approx 0$ **不表示轨迹准确或零偏被估计出来了**。本例只演示积分约束与初始条件，不是可观测性完备的 VIO。

---

## 9. 运行与预期

```bash
cmake -S . -B build
cmake --build build -j
./build/base05_imu
```

预期：

```
position RMSE: 本机约 0.144 m -> 9.8e-5 m
velocity RMSE: 本机约 0.130 m/s -> 5.9e-4 m/s
rotation RMSE: 本机约 0.029 rad -> 1.7e-5 rad
```

如果误差不降，优先检查：

1. 重力符号；
2. $a_m = R^\top(a_w - g)$ 还是 $R^\top(a_w + g)$；
3. 预积分更新顺序；
4. 四元数乘法的左右顺序。

---

## 10. 常见坑

1. **重力符号**：$g = (0,0,-9.81)$，测量里是 $a_w - g$。
2. **世界系/机体系混淆**：$\Delta v, \Delta p$ 都表达在**第 i 帧机体系**，残差最后要乘 $R_i^\top$。
3. **零偏单位**：陀螺仪 rad/s，加速度计 m/s²。
4. **预积分量与起始状态无关**：如果实现里偷偷用了 $R_i$，就失去预积分的意义。
5. **离散化误差**：`dt=0.01` 时欧拉积分误差明显，真实系统用中值/龙格库塔。
6. **四元数不归一化**：递推 100 步后会漂。
7. **固定量不足**：只固定位姿不固定速度，$H$ 仍可能奇异。

---

## 11. 练习

1. 把 `kDt` 改成 `0.01`，同时把 `kStepsPerFrame` 改成 `10`，保持关键帧间隔和轨迹时长不变，再观察离散化误差；还需说明每采样噪声与噪声密度的区别，否则不是公平对比。
2. 把欧拉积分改成中值积分，比较结果。
3. 给 IMU 加常值零偏，观察残差如何被破坏（作业）。
4. 思考：如果两帧之间没有 IMU 测量，预积分残差还能用吗？
5. 思考：为什么 IMU 因子通常要和视觉因子一起做联合优化，而不是分开？

答案在 `assignments/05_solution.md`。
