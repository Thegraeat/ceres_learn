// 第 5 课 课后作业模板（my_hw05.cpp）
//
// 任务：零偏对 IMU 预积分的影响
//   模式 A：预积分不减零偏（bg = 0, ba = 0）
//   模式 B：预积分减去真值零偏
//   比较真值处的 9 维残差，以及优化后的位置/速度/姿态 RMSE
//
// 这个文件已经准备好：IMU 采样生成（含常值零偏）、预积分递推、真值轨迹、
// 统一的带噪初值、RMSE 指标和两种模式的对比流程。需要你手写的地方：
//   TODO 1  ImuError::operator() —— 9 维白化预积分残差
//   TODO 2  Solve()              —— Manifold、参数块、残差块、求解
//
// 编译运行：
//   cmake --build build --target my_hw05 -j && ./build/my_hw05
//
// 验收标准见 assignments/05_homework.md。

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
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

void NormalizeQuat(double* q) {
    Eigen::Map<Eigen::Vector4d> quaternion(q);
    quaternion.normalize();
}

// ----------------------------- 数据结构 -----------------------------
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

const double kBgTrue[3] = {0.01, 0.01, 0.01};  // rad/s
const double kBaTrue[3] = {0.1, 0.1, 0.1};     // m/s^2

// ---------------------------------------------------------------------------
// 预积分递推（已给出，作业重点是零偏如何进入预积分和残差）
// ---------------------------------------------------------------------------
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
        // 生成测量时加了零偏，预积分时减掉。
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

// ----------------------------- 残差块 -----------------------------
struct ImuError {
    ImuError(const Preintegration& pre, double sigma_r, double sigma_v, double sigma_p)
        : pre_(pre), sigma_r_(sigma_r), sigma_v_(sigma_v), sigma_p_(sigma_p) {}

    template <typename T>
    bool operator()(const T* const q_i, const T* const p_i, const T* const v_i,
                    const T* const q_j, const T* const p_j, const T* const v_j,
                    T* residual) const {
        // TODO 1: 计算 9 维白化预积分残差。
        //   旋转残差：q_e = dq^{-1} * q_i^{-1} * q_j，统一到 w >= 0
        //             residual[0..2] = 2 * q_e.vec() / sigma_r
        //   速度残差：R_i^T * (v_j - v_i - g*dt) - dv
        //             residual[3..5] = ... / sigma_v
        //   位置残差：R_i^T * (p_j - p_i - v_i*dt - 0.5*g*dt^2) - dp
        //             residual[6..8] = ... / sigma_p
        //   注意 g 是 kGravity 向量，dt = pre_.dt。
        //   完成后删除下面的占位语句。
        (void)q_i;
        (void)p_i;
        (void)v_i;
        (void)q_j;
        (void)p_j;
        (void)v_j;
        for (int k = 0; k < 9; ++k) residual[k] = T(0.0);
        return true;
    }

    Preintegration pre_;
    double sigma_r_;
    double sigma_v_;
    double sigma_p_;
};

// ----------------------------- 真值与指标 -----------------------------
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

// ---------------------------------------------------------------------------
// 求解
// ---------------------------------------------------------------------------
bool Solve(const std::vector<Preintegration>& pre,
           std::vector<State>& est,
           ceres::Solver::Summary* summary) {
    // TODO 2: 构建并求解 IMU 预积分问题。
    //   1. 每个状态注册三个参数块：
    //        AddParameterBlock(est[i].pose.q, 4, new QuaternionManifold)
    //        AddParameterBlock(est[i].pose.t, 3)
    //        AddParameterBlock(est[i].v.data(), 3)
    //   2. 固定第 0 帧的 q、t、v
    //   3. 对每对相邻帧：
    //        AutoDiffCostFunction<ImuError, 9, 4, 3, 3, 4, 3, 3>
    //        new ImuError(pre[i], 0.01, 0.05, 0.05)
    //        AddResidualBlock(cost, nullptr, ...6 个参数块...)
    //   4. Options: DENSE_QR，max_num_iterations = 50
    //   5. Solve，并返回 summary->IsSolutionUsable()
    (void)pre;
    (void)est;
    (void)summary;
    return false;  // TODO 2 完成后替换
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<double> gauss(0.0, 1.0);

    const int total_steps = (kNumFrames - 1) * kStepsPerFrame;

    // 真值状态
    std::vector<State> gt(kNumFrames);
    for (int i = 0; i < kNumFrames; ++i) {
        Pose pose;
        Eigen::Vector3d v, a_world;
        Truth(i * kStepsPerFrame * kDt, &pose, &v, &a_world);
        gt[i].pose = pose;
        gt[i].v = v;
    }

    // 含零偏和噪声的 IMU 测量
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
            Eigen::Vector3d(0.0, 0.0, 1.0) +
            Eigen::Vector3d(kBgTrue[0], kBgTrue[1], kBgTrue[2]) +
            Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)) * 0.001;

        samples[k].accel =
            a_body + Eigen::Vector3d(kBaTrue[0], kBaTrue[1], kBaTrue[2]) +
            Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)) * 0.01;
    }

    // 两种预积分模式
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

    // 同一个带噪初值
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

    if (!Solve(pre_a, est_a, &summary_a) || !Solve(pre_b, est_b, &summary_b)) {
        std::cout << "TODO: 先完成 TODO 1/2，IMU 预积分问题才能求解。\n";
        return 0;
    }

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
