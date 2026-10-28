# 第 4 课 课后编程作业

> 对应讲义：`docs/04_ba.md`
> 对应答案：`assignments/04_solution.md`（做完再看）
> 可编译答案：`assignments/hw04_ba.cpp`

---

## 任务：带外点的 Bundle Adjustment

以 `base04_ba.cpp` 为起点。

### 1. 注入误匹配

随机选 10% 的观测，把像素坐标各加一个较大的偏差：

```cpp
std::uniform_real_distribution<double> outlier(-40.0, 40.0);
std::shuffle(obs.begin(), obs.end(), rng);
if (i % 10 == 0) {
    obs[i].u += outlier(rng);
    obs[i].v += outlier(rng);
}
```

要求：

- 记录哪些观测被污染；
- 先求解一次 `loss = nullptr`；
- 再求解一次 `HuberLoss(1.0)`；
- 输出两次的重投影 RMSE、点 RMSE、位姿平移/旋转 RMSE。

### 2. 评估

实现或复用：

$$
\mathrm{ReprojectionRMSE} =
\sqrt{\frac{1}{|\mathcal{O}|}\sum_{(c,k)}
\left\|\pi(T_c^{-1}X_k) - \mathbf{uv}_{ck}\right\|^2}
$$

注意：评估时用**原始观测**（包括被污染的点），还是只用内点？两种都算一遍，解释差异。

### 3. 比较线性求解器

在 `kNumCams = 20`、`kNumPoints = 500` 的规模下，比较：

| 求解器 | 说明 |
|---|---|
| `DENSE_QR` | 小问题 |
| `SPARSE_NORMAL_CHOLESKY` | 稀疏 H |
| `ITERATIVE_SCHUR` | 利用 Schur 结构 |

记录耗时和最终代价。

> 规模小的时候 `DENSE_QR` 可能不慢；规模上去后差距才明显。

增大相机数量时保持相机覆盖范围不变，例如 `x = -2 + 4*i/(kNumCams-1)`，不能只让相机不断远离场景；检查每个点有多视角观测。使用讲义第 7 课给出的显式 Schur ordering，点放 group 0，旋转和平移都放 group 1。先打乱观测再按比例污染，避免每个相机都污染同一批路标。

### 4. 思考题

1. 为什么单目 BA 固定一个相机仍不足以消除 gauge？额外尺度约束来自哪里？
2. 为什么点可以直接在 $\mathbb{R}^3$ 优化，而位姿需要 Manifold？
3. 一个只被一个相机看到的点，在 BA 里可解吗？为什么？
4. 外点比例从 10% 升到 30%，Huber 还够用吗？

---

## 验收标准

| 检查项 | 标准 |
|---|---|
| 无核 + 10% 外点 | 重投影 RMSE 明显高于 0.5 px |
| Huber + 10% 外点 | 内点 RMSE 明显低于无核；本机约 1.69 px 对 5.41 px，不承诺回到纯噪声水平 |
| 位姿/点误差 | Huber 优于无核 |
| 三种求解器 | 都能收敛，输出耗时 |
| 代码 | 能编译、能运行 |

---

## 提示

1. 外点要在**拷贝出来的观测**上注入，不要破坏真值观测，否则评估没有基准。
2. 评估“内点 RMSE”时，只统计未被污染的观测。
3. `ITERATIVE_SCHUR` 可能需要参数分组，Ceres 会自动推断；如果报错，先用 `SPARSE_NORMAL_CHOLESKY`。
4. 固定首帧和尺度基准，并确保固定值与评估坐标系一致；本课程示例额外给定第二帧一个平移分量。

---

## 提交

把三次运行的输出贴回来，重点说明：

- 外点让无核的解偏了多少；
- Huber 把误差压回了多少；
- 哪种求解器最快。
