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
- 这种补偿会改变轨迹，使状态越来越偏离真值。

本题只有 IMU 链，没有独立视觉或位置约束。在固定第一帧状态后，每一段末端的姿态、速度、位置仍能按错误预积分递推，所以模式 A 也可以把各段残差降得接近零。这里不一定发生“多个传感器之间的折中”；低残差只是说明轨迹符合所用预积分，不代表预积分正确。

模式 B 使用生成数据时已知的真值零偏重新积分，通常能明显减小轨迹误差。它演示的是**已知零偏补偿**，没有把零偏加进参数块联合估计。

### 1.4 用一维例子分清“残差小”和“轨迹准”

忽略重力和旋转，真实加速度为零，初始 $p_0=v_0=0$。如果加计读数中有 $b=0.1$ m/s² 的常值零偏但未扣除，积分 $T=1$ s 后得到：

$$
\Delta v=0.1\ \mathrm{m/s},\qquad\Delta p=0.05\ \mathrm m.
$$

在真值末端状态 $p_1=v_1=0$ 处，原始运动残差是 $r_v=-0.1$ m/s、$r_p=-0.05$ m。若让优化器自由调整末端为 $v_1=0.1$、$p_1=0.05$，两个残差就都为零，但位置已经错了 $5$ cm。

因此要分别报告：**真值处的测量一致性残差、优化后的因子代价、优化后相对真值的轨迹误差**。三者回答不同问题。若再加入可靠视觉观测，错误 IMU 与视觉之间才会发生折中，也才有机会在充分激励下从多种约束中估计零偏。

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

template <typename T>
void QuatRotateInv(const T* q, const T* v, T* out) {
    const Eigen::Quaternion<T> rotation(q[0], q[1], q[2], q[3]);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> vector(v);
    const Eigen::Matrix<T, 3, 1> rotated = rotation.conjugate() * vector;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> result(out);
    result = rotated;
}

void NormalizeQuat(double* q) {
    Eigen::Map<Eigen::Vector4d> quaternion(q);
    quaternion.normalize();
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
    const Eigen::Map<const Eigen::Vector3d> gyro_bias(bg), accel_bias(ba);
    Eigen::Map<Eigen::Vector3d> delta_velocity(pre.dv), delta_position(pre.dp);
    Eigen::Quaterniond delta_rotation = Eigen::Quaterniond::Identity();

    for (int k = start; k < end; ++k) {
        const Eigen::Vector3d omega = samples[k].gyro - gyro_bias;
        const Eigen::Vector3d accel = samples[k].accel - accel_bias;
        const double angular_speed = omega.norm();
        const double angle = angular_speed * kDt;
        Eigen::Quaterniond rotation_step = Eigen::Quaterniond::Identity();
        if (angle > 1e-12) {
            rotation_step = Eigen::AngleAxisd(angle, omega / angular_speed);
        }

        // 都用本步更新前的旋转和速度：先位置，再速度，最后旋转。
        const Eigen::Vector3d accel_i = delta_rotation * accel;
        delta_position += delta_velocity * kDt + 0.5 * accel_i * kDt * kDt;
        delta_velocity += accel_i * kDt;
        delta_rotation = (delta_rotation * rotation_step).normalized();
    }

    // 预积分结构仍保存 wxyz 数组，避免改变调用方的数据约定。
    pre.dq[0] = delta_rotation.w();
    pre.dq[1] = delta_rotation.x();
    pre.dq[2] = delta_rotation.y();
    pre.dq[3] = delta_rotation.z();
    return pre;
}

struct ImuError {
    ImuError(const Preintegration& pre, double sigma_r, double sigma_v, double sigma_p)
        : pre_(pre), sigma_r_(sigma_r), sigma_v_(sigma_v), sigma_p_(sigma_p) {}

    template <typename T>
    bool operator()(const T* const q_i, const T* const p_i, const T* const v_i,
                    const T* const q_j, const T* const p_j, const T* const v_j,
                    T* residual) const {
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        const Eigen::Quaternion<T> rotation_i(q_i[0], q_i[1], q_i[2], q_i[3]);
        const Eigen::Quaternion<T> rotation_j(q_j[0], q_j[1], q_j[2], q_j[3]);
        const Eigen::Quaternion<T> delta_rotation{
            T(pre_.dq[0]), T(pre_.dq[1]), T(pre_.dq[2]), T(pre_.dq[3])};
        const Eigen::Map<const Vector3> position_i(p_i), position_j(p_j);
        const Eigen::Map<const Vector3> velocity_i(v_i), velocity_j(v_j);
        const Vector3 gravity = Eigen::Map<const Eigen::Vector3d>(kGravity).cast<T>();
        const Vector3 delta_velocity = Eigen::Map<const Eigen::Vector3d>(pre_.dv).cast<T>();
        const Vector3 delta_position = Eigen::Map<const Eigen::Vector3d>(pre_.dp).cast<T>();
        const T dt = T(pre_.dt);

        Eigen::Quaternion<T> rotation_error =
            delta_rotation.conjugate() * rotation_i.conjugate() * rotation_j;
        if (rotation_error.w() < T(0.0)) {
            rotation_error.coeffs() = -rotation_error.coeffs();
        }

        const Vector3 velocity_error =
            rotation_i.conjugate() * (velocity_j - velocity_i - gravity * dt)
            - delta_velocity;
        const Vector3 position_error = rotation_i.conjugate() *
            (position_j - position_i - velocity_i * dt - T(0.5) * gravity * dt * dt)
            - delta_position;

        Eigen::Map<Eigen::Matrix<T, 9, 1>> error(residual);
        error.template segment<3>(0) = T(2.0) * rotation_error.vec() / T(sigma_r_);
        error.template segment<3>(3) = velocity_error / T(sigma_v_);
        error.template segment<3>(6) = position_error / T(sigma_p_);
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
    return Eigen::Map<const Eigen::Matrix<double, 9, 1>>(r).norm();
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
