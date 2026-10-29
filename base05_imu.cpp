// 第 5 课配套代码：IMU 预积分
//
// 状态：每帧位姿 T_wb = [R(q) | p]，速度 v（世界系）
// 预积分：在两个关键帧之间对 IMU 测量做积分，得到与起始状态无关的
//         ΔR、Δv、Δp，再构造 9 维残差。
//
// 本课为了聚焦预积分本身，零偏固定为 0；零偏估计与一阶修正见讲义和作业。
//
// 编译后运行：./build/base05_imu

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
const int kStepsPerFrame = 100;  // 0.1 s
const int kNumFrames = 5;

// ----------------------------- 预积分 -----------------------------
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

// ----------------------------- 残差块 -----------------------------
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

// ----------------------------- 真值轨迹 -----------------------------
// p(t) = (3 cos t, 3 sin t, 0.2 t)
// R(t) = Rz(t),  v(t) = (-3 sin t, 3 cos t, 0.2)
// a_w(t) = (-3 cos t, -3 sin t, 0)
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

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // 1) 真值状态（5 帧）
    const int total_steps = (kNumFrames - 1) * kStepsPerFrame;
    std::vector<State> gt(kNumFrames);
    for (int i = 0; i < kNumFrames; ++i) {
        Pose pose;
        Eigen::Vector3d v, a_world;
        Truth(i * kStepsPerFrame * kDt, &pose, &v, &a_world);
        gt[i].pose = pose;
        gt[i].v = v;
    }

    // 2) IMU 测量
    std::vector<ImuSample> samples(total_steps);
    for (int k = 0; k < total_steps; ++k) {
        const double t = k * kDt;
        Pose pose;
        Eigen::Vector3d v, a_world;
        Truth(t, &pose, &v, &a_world);

        const Eigen::Quaterniond q(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
        const Eigen::Vector3d a_body =
            q.conjugate() * (a_world - Eigen::Vector3d(0, 0, -9.81));

        samples[k].gyro = Eigen::Vector3d(0.0, 0.0, 1.0) +
                          Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)) * 0.001;
        samples[k].accel =
            a_body + Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)) * 0.01;
    }

    // 3) 预积分（零偏固定为 0）
    const double zero_bias[3] = {0.0, 0.0, 0.0};
    std::vector<Preintegration> pre(kNumFrames - 1);
    for (int i = 0; i < kNumFrames - 1; ++i) {
        pre[i] = Integrate(samples, i * kStepsPerFrame, (i + 1) * kStepsPerFrame,
                           zero_bias, zero_bias);
    }

    // 4) 初值：在真值上加噪声
    std::vector<State> est = gt;
    for (int i = 1; i < kNumFrames; ++i) {
        for (int k = 0; k < 3; ++k) {
            est[i].pose.t[k] += 0.1 * gauss(rng);
            est[i].v[k] += 0.1 * gauss(rng);
        }
        const double angle = 0.03 * gauss(rng);
        double dq[4] = {std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)};
        double q_tmp[4];
        QuatMul(est[i].pose.q, dq, q_tmp);
        for (int k = 0; k < 4; ++k) est[i].pose.q[k] = q_tmp[k];
        NormalizeQuat(est[i].pose.q);
    }

    const double pos_before = PositionRmse(est, gt);
    const double vel_before = VelocityRmse(est, gt);
    const double rot_before = RotationRmse(est, gt);

    // 4) 构建问题
    ceres::Problem problem;
    for (int i = 0; i < kNumFrames; ++i) {
        problem.AddParameterBlock(est[i].pose.q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(est[i].pose.t, 3);
        problem.AddParameterBlock(est[i].v.data(), 3);
    }

    // 固定第 0 帧（位姿 + 速度），消除 gauge
    problem.SetParameterBlockConstant(est[0].pose.q);
    problem.SetParameterBlockConstant(est[0].pose.t);
    problem.SetParameterBlockConstant(est[0].v.data());

    for (int i = 0; i < kNumFrames - 1; ++i) {
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
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";
    std::cout << "position RMSE: " << pos_before << " -> " << PositionRmse(est, gt)
              << " m\n";
    std::cout << "velocity RMSE: " << vel_before << " -> " << VelocityRmse(est, gt)
              << " m/s\n";
    std::cout << "rotation RMSE: " << rot_before << " -> " << RotationRmse(est, gt)
              << " rad\n";
    return 0;
}
