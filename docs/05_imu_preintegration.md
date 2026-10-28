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

### 4.2 中值积分

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
\left[\delta\boldsymbol\theta,\ \delta\mathbf v,\ \delta\mathbf p,
\delta\mathbf b_g,\ \delta\mathbf b_a\right]
\in \mathbb{R}^{15}
$$

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
double a_i[3];
QuatRotate(pre.dq, a, a_i);   // ΔR * a

pre.dp[j] += pre.dv[j] * dt + 0.5 * a_i[j] * dt * dt;
pre.dv[j] += a_i[j] * dt;

QuatMul(pre.dq, dq_step, dq_new);   // ΔR <- ΔR * Exp(ω dt)
```

注意顺序：`dp` 用旧的 `dv`，`dv` 用旧的 $\Delta R$，最后更新 $\Delta R$。

### 8.3 残差

```cpp
QuatRotateInv(q_i, dv_world, dv_body);   // R_i^T(...)
residual[3] = (dv_body[0] - pre_.dv[0]) / sigma_v_;
```

`QuatRotateInv` 用共轭四元数实现 $R^\top v$。

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
