# 第 3 课 课后作业参考答案

> 对应题目：`assignments/03_homework.md`
> 可编译代码：`assignments/hw03_pose_graph.cpp`

---

## 1. 解题思路

### 1.1 错误回环为什么可怕

位姿图的每条带噪边是由协方差加权的**软约束**，并非必须严格满足的等式。3 米的平移错误，相对于 `sigma_trans = 0.1` 是 30 个 sigma；普通最小二乘可能为了降低这条边的大代价，把整段轨迹拉弯。

只考虑这一个平移误差分量，白化残差为 $r=3/0.1=30$。普通代价为 $\frac12r^2=450$；取 Huber 阈值 $\delta=1$ 时，代价为 $\frac12(2|r|-1)=29.5$。它仍在影响优化，但影响明显减弱；不能保证错误边从此不会主导解，结果还取决于正确边的数量和几何分布。

### 1.2 ATE 与 ARE

ATE（平移）：

$$
\mathrm{ATE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left\|t_i - t_i^{gt}\right\|^2}
$$

ARE（旋转）：

$$
\mathrm{ARE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\theta_i^2},
\qquad
\theta_i = 2\arccos\!\left(|w_i|\right)
$$

其中

$$
q_i = \left(q_i^{est}\right)^{-1} \otimes q_i^{gt}
= [w_i, x_i, y_i, z_i]^\top
$$

取 $|w|$ 是为了处理四元数双覆盖，`clamp` 到 $[0,1]$ 是为了避免 `acos` 的数值问题。

这个公式来自单位四元数的 $w=\cos(\theta/2)$。旋转误差要取两个姿态之间的**相对旋转角**，不能直接计算四个四元数分量之差的长度。

例如两帧的角度误差为 $0.1,0.2$ rad，ARE 为 $\sqrt{(0.1^2+0.2^2)/2}\approx0.1581$ rad，而不是平均角度 $0.15$ rad。一个姿态的四元数若仅整体反号，几何旋转误差仍为零。

计算 ATE 前还要统一全局坐标系；计算 ARE 前要统一全局朝向。一个全局坐标旋转会同时改变位置与姿态的表示，不能只对平移做对齐、却拿未对齐姿态计算旋转误差。

### 1.3 固定节点的作用

固定第 0 个节点，等价于把解锚定在 $T_0 = T_0^{gt}$。位姿图的 6 维零空间（整体平移 + 整体旋转）被消掉。

固定最后一个节点也能去掉全局 gauge。对于对应于同一解的不同全局刚体变换，**最优代价不变**，但不对齐时的 ATE 可以不同。带噪声及错误回环时，真值并不满足所有边；若比较不同锚点的结果，应先统一 gauge 或按同一 SE(3) 对齐协议评估，不能保证原始 ATE 差异很小。

---

## 2. 完整答案代码

```cpp
// 第 3 课课后作业参考答案
//
// 任务：在位姿图里加入两条错误回环，比较普通最小二乘与 Huber 鲁棒核。
//   1) 构造 20 个节点的圆轨迹，相邻边 + 回环边
//   2) 把 (0,5) 和 (5,10) 两条回环的平移测量加 3 米
//   3) 分别用 loss=nullptr 和 HuberLoss(1.0) 求解
//   4) 输出 ATE（平移）和 ARE（旋转）
//
// 编译后运行：./build/hw03_pose_graph

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

// ----------------------------- 四元数工具 -----------------------------
template <typename T>
void QuatMul(const T* a, const T* b, T* out) {
    // 构造参数为 wxyz；Eigen 负责 Hamilton 乘法。
    const Eigen::Quaternion<T> qa(a[0], a[1], a[2], a[3]);
    const Eigen::Quaternion<T> qb(b[0], b[1], b[2], b[3]);
    const Eigen::Quaternion<T> product = qa * qb;
    out[0] = product.w();
    out[1] = product.x();
    out[2] = product.y();
    out[3] = product.z();
}

template <typename T>
void QuatConj(const T* q, T* out) {
    const Eigen::Quaternion<T> rotation(q[0], q[1], q[2], q[3]);
    const Eigen::Quaternion<T> inverse = rotation.conjugate();
    out[0] = inverse.w();
    out[1] = inverse.x();
    out[2] = inverse.y();
    out[3] = inverse.z();
}

template <typename T>
void QuatRotate(const T* q, const T* v, T* out) {
    const Eigen::Quaternion<T> rotation(q[0], q[1], q[2], q[3]);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> vector(v);
    // 先算出结果，再写输出；输入输出可以使用同一数组。
    const Eigen::Matrix<T, 3, 1> rotated = rotation * vector;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> result(out);
    result = rotated;
}

void NormalizeQuat(double* q) {
    Eigen::Map<Eigen::Vector4d> quaternion(q);
    quaternion.normalize();
}

// ----------------------------- 数据结构 -----------------------------
struct Pose {
    double q[4] = {1.0, 0.0, 0.0, 0.0};
    double t[3] = {0.0, 0.0, 0.0};
};

struct Edge {
    int i;
    int j;
    Pose meas;
};

const double kPi = 3.14159265358979323846;
const double kSigmaRot = 0.02;
const double kSigmaTrans = 0.1;

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
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        const Eigen::Quaternion<T> rotation_i(q_i[0], q_i[1], q_i[2], q_i[3]);
        const Eigen::Quaternion<T> rotation_j(q_j[0], q_j[1], q_j[2], q_j[3]);
        const Eigen::Quaternion<T> measured_rotation{
            T(q_ij_[0]), T(q_ij_[1]), T(q_ij_[2]), T(q_ij_[3])};
        const Eigen::Map<const Vector3> position_i(t_i);
        const Eigen::Map<const Vector3> position_j(t_j);
        const Vector3 measured_translation =
            Eigen::Map<const Eigen::Vector3d>(t_ij_).cast<T>();

        // q_e = q_ij^{-1} * q_i^{-1} * q_j，统一到 w >= 0 的半球。
        Eigen::Quaternion<T> rotation_error =
            measured_rotation.conjugate() * rotation_i.conjugate() * rotation_j;
        if (rotation_error.w() < T(0.0)) {
            rotation_error.coeffs() = -rotation_error.coeffs();
        }

        // e_t = R_ij^T * (R_i^T * (t_j - t_i) - t_ij)。
        const Vector3 translation_error = measured_rotation.conjugate() *
            (rotation_i.conjugate() * (position_j - position_i) - measured_translation);

        Eigen::Map<Eigen::Matrix<T, 6, 1>> error(residual);
        error.template head<3>() = T(2.0) * rotation_error.vec() / T(sigma_rot_);
        error.template tail<3>() = translation_error / T(sigma_trans_);
        return true;
    }

    double q_ij_[4];
    double t_ij_[3];
    double sigma_rot_;
    double sigma_trans_;
};

// ----------------------------- 工具函数 -----------------------------
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

double RotationError(const Pose& est, const Pose& gt) {
    double qe_inv[4];
    QuatConj(est.q, qe_inv);

    double qe[4];
    QuatMul(qe_inv, gt.q, qe);

    double w = std::fabs(qe[0]);
    w = std::min(1.0, std::max(-1.0, w));
    return 2.0 * std::acos(w);
}

double AreRmse(const std::vector<Pose>& est, const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        const double a = RotationError(est[i], gt[i]);
        sum += a * a;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

void Solve(const std::vector<Edge>& edges,
           const std::vector<Pose>& init,
           bool use_huber,
           std::vector<Pose>& out,
           ceres::Solver::Summary* summary) {
    out = init;
    const int n = static_cast<int>(out.size());

    ceres::Problem problem;
    for (int i = 0; i < n; ++i) {
        problem.AddParameterBlock(out[i].q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(out[i].t, 3);
    }

    problem.SetParameterBlockConstant(out[0].q);
    problem.SetParameterBlockConstant(out[0].t);

    for (const auto& e : edges) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<EdgeCost, 6, 4, 3, 4, 3>(
                new EdgeCost(e.meas.q, e.meas.t, kSigmaRot, kSigmaTrans));

        ceres::LossFunction* loss = use_huber ? new ceres::HuberLoss(1.0) : nullptr;

        problem.AddResidualBlock(cost, loss, out[e.i].q, out[e.i].t,
                                 out[e.j].q, out[e.j].t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;

    ceres::Solve(options, &problem, summary);
}

int main() {
    const int n = 20;
    const double radius = 5.0;

    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // 真值
    std::vector<Pose> gt(n);
    for (int i = 0; i < n; ++i) {
        const double theta = 2.0 * kPi * i / n;
        gt[i].t[0] = radius * std::cos(theta);
        gt[i].t[1] = radius * std::sin(theta);

        const double yaw = theta + kPi / 2.0;
        gt[i].q[0] = std::cos(yaw / 2.0);
        gt[i].q[3] = std::sin(yaw / 2.0);
    }

    // 边：相邻 + 跨 5 个节点
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i) {
        pairs.emplace_back(i, (i + 1) % n);
    }
    for (int i = 0; i < n; i += 5) {
        pairs.emplace_back(i, (i + 5) % n);
    }

    // 带噪测量
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

    // 初值：航迹推算
    std::vector<Pose> init(n);
    init[0] = gt[0];
    for (int i = 0; i < n - 1; ++i) {
        double q_next[4];
        QuatMul(init[i].q, edges[i].meas.q, q_next);
        NormalizeQuat(q_next);

        double t_local[3];
        QuatRotate(init[i].q, edges[i].meas.t, t_local);

        for (int k = 0; k < 4; ++k) init[i + 1].q[k] = q_next[k];
        for (int k = 0; k < 3; ++k) init[i + 1].t[k] = init[i].t[k] + t_local[k];
    }

    // 注入两条错误回环
    std::vector<Edge> bad_edges = edges;
    for (auto& e : bad_edges) {
        if ((e.i == 0 && e.j == 5) || (e.i == 5 && e.j == 10)) {
            e.meas.t[0] += 3.0;
        }
    }

    std::cout << "bad loop closures: (0,5), (5,10)  +3.0 m\n";
    std::cout << "ATE before: " << AteRmse(init, gt) << " m\n\n";

    std::vector<Pose> est_no;
    std::vector<Pose> est_huber;
    ceres::Solver::Summary summary_no;
    ceres::Solver::Summary summary_huber;

    Solve(bad_edges, init, false, est_no, &summary_no);
    Solve(bad_edges, init, true, est_huber, &summary_huber);

    std::cout << "[no loss] " << summary_no.BriefReport() << "\n";
    std::cout << "          ATE=" << AteRmse(est_no, gt)
              << " m, ARE=" << AreRmse(est_no, gt) << " rad\n\n";

    std::cout << "[huber ]  " << summary_huber.BriefReport() << "\n";
    std::cout << "          ATE=" << AteRmse(est_huber, gt)
              << " m, ARE=" << AreRmse(est_huber, gt) << " rad\n";

    return 0;
}
```

---

## 3. 预期输出形态

```
bad loop closures: (0,5), (5,10)  +3.0 m
ATE before: 0.5 ~ 0.8 m

[no loss] Ceres Solver Report: ...
          ATE=1.0 ~ 3.0 m, ARE=0.05 ~ 0.2 rad

[huber ]  Ceres Solver Report: ...
          ATE=0.05 ~ 0.3 m, ARE=0.01 ~ 0.05 rad
```

关键判据：

- 无核的 ATE 明显比优化前还差（错误边把图拉歪）；
- Huber 的 ATE 接近 `base03` 的正常水平；
- 两种配置都能“收敛”，但收敛到不同的解。这再次说明 `CONVERGENCE` 不等于正确。

---

## 4. 代码讲解

### 4.1 为什么先拷贝 `edges`

```cpp
std::vector<Edge> bad_edges = edges;
```

航迹推算初值用的是 `edges`。如果直接改 `edges`，初值也会被 3 米错误污染，实验就不干净了。真实系统里，初值来自前端，回环测量来自另一条链路，二者独立。

### 4.2 求解函数复用

```cpp
void Solve(const std::vector<Edge>& edges, ..., bool use_huber, ...);
```

把 `use_huber` 作为参数，避免复制两份建图代码。这是写实验脚本的标准做法。

### 4.3 旋转误差

```cpp
QuatConj(est.q, qe_inv);
QuatMul(qe_inv, gt.q, qe);
double w = std::fabs(qe[0]);
w = std::min(1.0, std::max(-1.0, w));
return 2.0 * std::acos(w);
```

- $q_{est}^{-1} q_{gt}$ 是相对旋转；
- 单位四元数 $[\cos(\theta/2), \sin(\theta/2)n]$，所以 $\theta = 2\arccos(|w|)$；
- $|w|$ 处理双覆盖；
- `clamp` 处理浮点误差导致 $|w| > 1$。

---

## 5. 思考题答案

### 5.1 只加平移错误，为什么旋转也变差？

位姿图的所有节点是耦合的。错误平移约束会通过相邻边把邻近节点的平移拉开，而平移和旋转在 SE(3) 中耦合（$t_e$ 的计算依赖 $R_i$、$R_{ij}$）。优化器会同时调整旋转来分摊误差，所以 ARE 也会上升。

### 5.2 Huber 阈值改成 10 或 0.1 会怎样？

- $\delta=10$：对白化残差 $30$，Huber 局部权重为 $10/30=1/3$，明显大于 $\delta=1$ 时的 $1/30$，错误边保留了更多影响，可能使轨迹更偏。
- $\delta=0.1$：许多正确观测也进入线性段，目标更接近按误差范数求和，会改变拟合的折中方式和局部曲率。降权不等于“没有约束”，更不能保证解会停留在初值；效果需要用相同数据与初值比较。

阈值不是越大或越小越好，要匹配噪声尺度。

### 5.3 固定第 0 个 vs 固定最后一个

两种方式消掉的都是 6 维全局 gauge。对于相同局部最优解对应的不同全局坐标表示，代价不变；实际非凸求解仍可能落入不同局部极小。带噪真值不满足全部边，不对齐的 ATE/ARE 也不必相近，比较时必须统一锚定或对齐协议。

### 5.4 两条互相矛盾的错误边

Huber 会把两条都降权，但如果它们都很强、又没有足够多的正确边，优化可能陷入局部极小。真实系统需要：

- 回环几何一致性检验（RANSAC）；
- 卡方检验；
- switchable constraints；
- 多假设。

鲁棒核不是万能的。

### 5.5 真实回环误检比例与保护机制

误检比例与可承受程度取决于场景、边的分布和验证机制，没有本题可以推出的统一百分比阈值。常见保护：

- 前端：RANSAC PnP / ICP，几何验证；
- 后端：卡方检验、Huber/Cauchy、switchable constraints；
- 系统层：GPS/IMU 先验、回环延迟接受、人工确认。

---

## 6. 常见错误

1. **在原 `edges` 上改错误边**：初值被污染，实验对比失效。
2. **忘记固定节点**：$H$ 奇异，`SPARSE_NORMAL_CHOLESKY` 报数值错误。
3. **ARE 里四元数没归一化**：`acos` 参数越界，输出 `NaN`。
4. **Huber 阈值作用在未白化残差上**：3 米和 1σ 的关系判断错误。
5. **固定节点后仍期望 ATE 完全为 0**：测量有噪声，优化只能让残差最小，不能消除噪声。
