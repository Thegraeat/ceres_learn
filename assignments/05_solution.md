# 第 5 课 课后作业参考答案

> 对应题目：`assignments/05_homework.md`
> 可编译代码：`assignments/hw05_imu.cpp`

---

## 1. 关键结论

### 1.1 零偏如何进入预积分

测量：

$$
\tilde{\boldsymbol\omega} = \boldsymbol\omega + \mathbf b_g + \mathbf n_g
$$

$$
\tilde{\mathbf a} = R^\top(\mathbf a_w - \mathbf g) + \mathbf b_a + \mathbf n_a
$$

预积分时如果直接用 $\tilde\omega$、$\tilde a$（不减零偏），相当于把 $b_g$、$b_a$ 当成真实运动：

- $b_g$ 让 $\Delta R$ 每步多转 $b_g \Delta t$；
- $b_a$ 让 $\Delta v$ 每步多加速 $b_a \Delta t$，$\Delta p$ 再积分一次。

$0.01\ \mathrm{rad/s}$ 在 $0.1\ \mathrm{s}$ 内造成 $0.001\ \mathrm{rad}$ 的姿态误差；$0.1\ \mathrm{m/s^2}$ 在 $0.1\ \mathrm{s}$ 内造成 $0.01\ \mathrm{m/s}$ 的速度误差和 $0.0005\ \mathrm{m}$ 的位置误差。单段看起来小，但四段累积、并且误差会通过姿态耦合放大。

### 1.2 残差范数

在真值状态处：

- 模式 A：残差包含零偏项，范数明显大；
- 模式 B：残差只剩噪声和离散化误差。如果用正确完整协方差白化，真值处平方范数的期望约为 9（不计离散化误差）；本例使用手工权重，不能按这个统计量验收。

> 代码里的 `sigma_r=0.01, sigma_v=0.05, sigma_p=0.05` 是手工设的，不是从协方差传播来的。真实系统必须用 $9\times 9$ 协方差白化。

### 1.3 优化结果

模式 A 的优化会试图用状态补偿零偏：

- 姿态可以部分吸收陀螺仪零偏；
- 速度可以部分吸收加速度计零偏；
- 但零偏是**常值**，而状态是逐帧独立的，补偿只能分段折中，四段累积后仍有明显误差。

模式 B 直接用正确零偏预积分，误差回到噪声水平。

---

## 2. 完整答案代码

```cpp
// 第 5 课课后作业参考答案
//
// 任务：零偏对预积分的影响
//   模式 A：预积分不减零偏（bg=0, ba=0）
//   模式 B：预积分减去真值零偏
//   比较两种模式在真值处的残差范数，以及优化后的位姿/速度误差。
//
// 编译后运行：./build/hw05_imu

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

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

template <typename T>
void QuatRotateInv(const T* q, const T* v, T* out) {
    T qc[4];
    QuatConj(q, qc);
    QuatRotate(qc, v, out);
}

void NormalizeQuat(double* q) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; ++i) q[i] /= n;
}

struct Pose {
    double q[4] = {1.0, 0.0, 0.0, 0.0};
    double t[3] = {0.0, 0.0, 0.0};
};

struct State {
    Pose pose;
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
};

struct ImuSample {
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
};

struct Preintegration {
    double dq[4] = {1.0, 0.0, 0.0, 0.0};
    double dv[3] = {0.0, 0.0, 0.0};
    double dp[3] = {0.0, 0.0, 0.0};
    double dt = 0.0;
};

const double kGravity[3] = {0.0, 0.0, -9.81};
const double kDt = 0.001;
const int kStepsPerFrame = 100;
const int kNumFrames = 5;

const double kBgTrue[3] = {0.01, 0.01, 0.01};
const double kBaTrue[3] = {0.1, 0.1, 0.1};

Preintegration Integrate(const std::vector<ImuSample>& samples,
                         int start,
                         int end,
                         const double bg[3],
                         const double ba[3]) {
    Preintegration pre;
    pre.dt = (end - start) * kDt;

    for (int k = start; k < end; ++k) {
        const double omega[3] = {samples[k].gyro.x() - bg[0],
                                 samples[k].gyro.y() - bg[1],
                                 samples[k].gyro.z() - bg[2]};
        const double a[3] = {samples[k].accel.x() - ba[0],
                             samples[k].accel.y() - ba[1],
                             samples[k].accel.z() - ba[2]};

        const double angle = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1] +
                                       omega[2] * omega[2]) *
                             kDt;
        double dq_step[4] = {1.0, 0.0, 0.0, 0.0};
        if (angle > 1e-12) {
            const double s = std::sin(angle / 2.0) / (angle / kDt);
            dq_step[0] = std::cos(angle / 2.0);
            dq_step[1] = s * omega[0];
            dq_step[2] = s * omega[1];
            dq_step[3] = s * omega[2];
        }

        double a_i[3];
        QuatRotate(pre.dq, a, a_i);

        for (int j = 0; j < 3; ++j) {
            pre.dp[j] += pre.dv[j] * kDt + 0.5 * a_i[j] * kDt * kDt;
            pre.dv[j] += a_i[j] * kDt;
        }

        double dq_new[4];
        QuatMul(pre.dq, dq_step, dq_new);
        NormalizeQuat(dq_new);
        for (int j = 0; j < 4; ++j) pre.dq[j] = dq_new[j];
    }

    return pre;
}

struct ImuError {
    ImuError(const Preintegration& pre, double sigma_r, double sigma_v, double sigma_p)
        : pre_(pre), sigma_r_(sigma_r), sigma_v_(sigma_v), sigma_p_(sigma_p) {}

    template <typename T>
    bool operator()(const T* const q_i, const T* const p_i, const T* const v_i,
                    const T* const q_j, const T* const p_j, const T* const v_j,
                    T* residual) const {
        T dq[4] = {T(pre_.dq[0]), T(pre_.dq[1]), T(pre_.dq[2]), T(pre_.dq[3])};
        T dq_inv[4];
        QuatConj(dq, dq_inv);

        T qi_inv[4];
        QuatConj(q_i, qi_inv);

        T tmp[4];
        QuatMul(qi_inv, q_j, tmp);
        T qe[4];
        QuatMul(dq_inv, tmp, qe);

        if (qe[0] < T(0.0)) {
            for (int k = 0; k < 4; ++k) qe[k] = -qe[k];
        }

        residual[0] = T(2.0) * qe[1] / T(sigma_r_);
        residual[1] = T(2.0) * qe[2] / T(sigma_r_);
        residual[2] = T(2.0) * qe[3] / T(sigma_r_);

        const T dt = T(pre_.dt);

        T dv_world[3] = {
            v_j[0] - v_i[0] - T(kGravity[0]) * dt,
            v_j[1] - v_i[1] - T(kGravity[1]) * dt,
            v_j[2] - v_i[2] - T(kGravity[2]) * dt,
        };
        T dv_body[3];
        QuatRotateInv(q_i, dv_world, dv_body);
        residual[3] = (dv_body[0] - T(pre_.dv[0])) / T(sigma_v_);
        residual[4] = (dv_body[1] - T(pre_.dv[1])) / T(sigma_v_);
        residual[5] = (dv_body[2] - T(pre_.dv[2])) / T(sigma_v_);

        T dp_world[3] = {
            p_j[0] - p_i[0] - v_i[0] * dt - T(0.5 * kGravity[0]) * dt * dt,
            p_j[1] - p_i[1] - v_i[1] * dt - T(0.5 * kGravity[1]) * dt * dt,
            p_j[2] - p_i[2] - v_i[2] * dt - T(0.5 * kGravity[2]) * dt * dt,
        };
        T dp_body[3];
        QuatRotateInv(q_i, dp_world, dp_body);
        residual[6] = (dp_body[0] - T(pre_.dp[0])) / T(sigma_p_);
        residual[7] = (dp_body[1] - T(pre_.dp[1])) / T(sigma_p_);
        residual[8] = (dp_body[2] - T(pre_.dp[2])) / T(sigma_p_);

        return true;
    }

    Preintegration pre_;
    double sigma_r_;
    double sigma_v_;
    double sigma_p_;
};

void Truth(double t, Pose* pose, Eigen::Vector3d* v, Eigen::Vector3d* a_world) {
    const double c = std::cos(t);
    const double s = std::sin(t);

    *v = Eigen::Vector3d(-3.0 * s, 3.0 * c, 0.2);
    *a_world = Eigen::Vector3d(-3.0 * c, -3.0 * s, 0.0);

    Eigen::Matrix3d R;
    R << c, -s, 0.0,
         s, c, 0.0,
         0.0, 0.0, 1.0;
    Eigen::Quaterniond q(R);
    pose->q[0] = q.w();
    pose->q[1] = q.x();
    pose->q[2] = q.y();
    pose->q[3] = q.z();

    pose->t[0] = 3.0 * c;
    pose->t[1] = 3.0 * s;
    pose->t[2] = 0.2 * t;
}

double PositionRmse(const std::vector<State>& est, const std::vector<State>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        const double dx = est[i].pose.t[0] - gt[i].pose.t[0];
        const double dy = est[i].pose.t[1] - gt[i].pose.t[1];
        const double dz = est[i].pose.t[2] - gt[i].pose.t[2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

double VelocityRmse(const std::vector<State>& est, const std::vector<State>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        sum += (est[i].v - gt[i].v).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

double RotationRmse(const std::vector<State>& est, const std::vector<State>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        double qe_inv[4];
        QuatConj(est[i].pose.q, qe_inv);
        double qe[4];
        QuatMul(qe_inv, gt[i].pose.q, qe);
        double w = std::fabs(qe[0]);
        w = std::min(1.0, std::max(-1.0, w));
        const double angle = 2.0 * std::acos(w);
        sum += angle * angle;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

double ResidualNorm(const ImuError& error, const State& a, const State& b) {
    double r[9];
    error(a.pose.q, a.pose.t, a.v.data(), b.pose.q, b.pose.t, b.v.data(), r);
    double sum = 0.0;
    for (int k = 0; k < 9; ++k) sum += r[k] * r[k];
    return std::sqrt(sum);
}

void Solve(const std::vector<Preintegration>& pre,
           std::vector<State>& est,
           ceres::Solver::Summary* summary) {
    ceres::Problem problem;
    for (size_t i = 0; i < est.size(); ++i) {
        problem.AddParameterBlock(est[i].pose.q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(est[i].pose.t, 3);
        problem.AddParameterBlock(est[i].v.data(), 3);
    }

    problem.SetParameterBlockConstant(est[0].pose.q);
    problem.SetParameterBlockConstant(est[0].pose.t);
    problem.SetParameterBlockConstant(est[0].v.data());

    for (size_t i = 0; i + 1 < est.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ImuError, 9, 4, 3, 3, 4, 3, 3>(
                new ImuError(pre[i], 0.01, 0.05, 0.05));

        problem.AddResidualBlock(cost, nullptr,
                                 est[i].pose.q, est[i].pose.t, est[i].v.data(),
                                 est[i + 1].pose.q, est[i + 1].pose.t,
                                 est[i + 1].v.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;

    ceres::Solve(options, &problem, summary);
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<double> gauss(0.0, 1.0);

    const int total_steps = (kNumFrames - 1) * kStepsPerFrame;

    std::vector<State> gt(kNumFrames);
    for (int i = 0; i < kNumFrames; ++i) {
        Pose pose;
        Eigen::Vector3d v, a_world;
        Truth(i * kStepsPerFrame * kDt, &pose, &v, &a_world);
        gt[i].pose = pose;
        gt[i].v = v;
    }

    std::vector<ImuSample> samples(total_steps);
    for (int k = 0; k < total_steps; ++k) {
        const double t = k * kDt;
        Pose pose;
        Eigen::Vector3d v, a_world;
        Truth(t, &pose, &v, &a_world);

        const Eigen::Quaterniond q(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
        const Eigen::Vector3d a_body =
            q.conjugate() * (a_world - Eigen::Vector3d(0, 0, -9.81));

        samples[k].gyro =
            Eigen::Vector3d(0.0, 0.0, 1.0) + Eigen::Vector3d(kBgTrue[0], kBgTrue[1], kBgTrue[2]) +
            Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)) * 0.001;

        samples[k].accel =
            a_body + Eigen::Vector3d(kBaTrue[0], kBaTrue[1], kBaTrue[2]) +
            Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)) * 0.01;
    }

    const double zero_bias[3] = {0.0, 0.0, 0.0};
    std::vector<Preintegration> pre_a(kNumFrames - 1);
    std::vector<Preintegration> pre_b(kNumFrames - 1);
    for (int i = 0; i < kNumFrames - 1; ++i) {
        pre_a[i] = Integrate(samples, i * kStepsPerFrame, (i + 1) * kStepsPerFrame,
                             zero_bias, zero_bias);
        pre_b[i] = Integrate(samples, i * kStepsPerFrame, (i + 1) * kStepsPerFrame,
                             kBgTrue, kBaTrue);
    }

    // 真值处第一段残差
    const ImuError err_a(pre_a[0], 0.01, 0.05, 0.05);
    const ImuError err_b(pre_b[0], 0.01, 0.05, 0.05);
    std::cout << "residual norm at truth (segment 0):\n";
    std::cout << "  A (no bias compensation): " << ResidualNorm(err_a, gt[0], gt[1]) << "\n";
    std::cout << "  B (bias compensated):     " << ResidualNorm(err_b, gt[0], gt[1]) << "\n\n";

    // 同一个初值
    std::vector<State> init = gt;
    for (int i = 1; i < kNumFrames; ++i) {
        for (int k = 0; k < 3; ++k) {
            init[i].pose.t[k] += 0.1 * gauss(rng);
            init[i].v[k] += 0.1 * gauss(rng);
        }
        const double angle = 0.03 * gauss(rng);
        double dq[4] = {std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)};
        double q_tmp[4];
        QuatMul(init[i].pose.q, dq, q_tmp);
        for (int k = 0; k < 4; ++k) init[i].pose.q[k] = q_tmp[k];
        NormalizeQuat(init[i].pose.q);
    }

    std::cout << "initial: pos=" << PositionRmse(init, gt)
              << " vel=" << VelocityRmse(init, gt)
              << " rot=" << RotationRmse(init, gt) << "\n\n";

    std::vector<State> est_a = init;
    std::vector<State> est_b = init;
    ceres::Solver::Summary summary_a;
    ceres::Solver::Summary summary_b;

    Solve(pre_a, est_a, &summary_a);
    Solve(pre_b, est_b, &summary_b);

    std::cout << "[A no bias] pos=" << PositionRmse(est_a, gt)
              << " vel=" << VelocityRmse(est_a, gt)
              << " rot=" << RotationRmse(est_a, gt) << "\n";
    std::cout << "[B bias   ] pos=" << PositionRmse(est_b, gt)
              << " vel=" << VelocityRmse(est_b, gt)
              << " rot=" << RotationRmse(est_b, gt) << "\n\n";

    std::cout << summary_a.BriefReport() << "\n";
    std::cout << summary_b.BriefReport() << "\n";
    return 0;
}
```

---

## 3. 预期输出形态

```
residual norm at truth (segment 0):
  A (no bias compensation): 本机约 0.390
  B (bias compensated):     本机约 0.00467

initial: pos=0.1~0.2 vel=0.1~0.2 rot=0.02~0.05

[A no bias] 本机约 pos=0.00735 vel=0.0429 rot=0.00423
[B bias   ] 本机约 pos=9.8e-5 vel=5.9e-4 rot=1.7e-5
```

判据：

- A 的残差范数明显大于 B；
- A 优化后误差仍明显大于 B；
- B 的误差接近初值噪声以下，说明预积分与轨迹一致。

---

## 4. 思考题答案

### 4.1 为什么通常不每次优化都重新积分？

预积分的主要目的就是避免重复积分。一次优化有几十次迭代，每次迭代零偏都可能变；如果每次都从原始 IMU 测量重新积分：

- 每条 IMU 因子都要 $O(N)$ 次运算；
- 滑动窗口里有几十个因子，代价爆炸；
- 增加前后端共享原始数据与同步的复杂度。

不是数学上不能重积分，而是计算成本通常不划算。预积分在固定参考零偏附近支持一阶修正；零偏变化超过线性近似范围时，应重新积分，而不是无限沿用旧量。

### 4.2 零偏变化时怎么更新？

预积分时同时递推零偏雅可比：

$$
J_{bg} = \frac{\partial \Delta R}{\partial b_g},\quad
J_{vg} = \frac{\partial \Delta v}{\partial b_g},\quad
J_{va} = \frac{\partial \Delta v}{\partial b_a},\quad
J_{pg} = \frac{\partial \Delta p}{\partial b_g},\quad
J_{pa} = \frac{\partial \Delta p}{\partial b_a}
$$

零偏更新后：

$$
\Delta R \leftarrow \Delta R\exp(J_{bg}\delta b_g)
$$

$$
\Delta v \leftarrow \Delta v + J_{vg}\delta b_g + J_{va}\delta b_a
$$

$$
\Delta p \leftarrow \Delta p + J_{pg}\delta b_g + J_{pa}\delta b_a
$$

这些雅可比在预积分递推中一起更新，VINS-Mono 的实现是标准参考。

这里 $J_{bg}$ 严格说是右乘局部旋转误差对零偏的 $3\times 3$ 雅可比；所有修正都应相对于保存的参考零偏 $b_{bar}$ 与参考预积分量计算，不能每次把完整 $b-b_{bar}$ 再累加到上次修正结果，否则会重复补偿。

### 4.3 陀螺仪零偏还是加速度计零偏影响大？

短期看，加速度计零偏直接影响位置（二次积分），更直接。

在小角度、无外部修正、水平陀螺仪常值零偏的典型情形下，倾角误差约随 $t$ 增长，重力泄漏的加速度误差约随 $t$ 增长，位置误差因而可含 $g b_g t^3/6$ 项，而不是一概按 $t^2$ 增长。绕重力的纯 yaw 零偏又不同；哪种零偏影响更大还取决于运动、时间跨度和视觉约束。

### 4.4 能不能用一个标量 sigma 白化 9 维残差？

不能。9 维残差的三组量纲不同：

- 旋转：rad；
- 速度：m/s；
- 位置：m。

而且它们之间有相关性（零偏、姿态误差会耦合）。正确做法是用预积分协方差 $\Sigma$ 白化：

$$
\tilde{\mathbf r} = L^{-1}\mathbf r,
\qquad
\Sigma = LL^\top
$$

代码里用三个标量 `sigma_r, sigma_v, sigma_p` 是教学简化，真实系统必须用完整协方差。

---

## 5. 常见错误

1. **预积分时不减零偏**：本作业的模式 A 就是故意犯这个错。
2. **生成测量时加零偏，预积分时又加一次**：符号搞反。
3. **两种模式共用同一个 `est`**：第二次从第一次的解出发，对比不公平。
4. **固定量不足**：只固定位姿不固定速度，$H$ 仍可能奇异。
5. **残差范数用白化后的值比较**：可以，但要保证两种模式的 sigma 相同。
