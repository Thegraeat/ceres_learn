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

// ----------------------------- 残差块 -----------------------------
struct ImuError {
    ImuError(const Preintegration& pre, double sigma_r, double sigma_v, double sigma_p)
        : pre_(pre), sigma_r_(sigma_r), sigma_v_(sigma_v), sigma_p_(sigma_p) {}

    template <typename T>
    bool operator()(const T* const q_i, const T* const p_i, const T* const v_i,
                    const T* const q_j, const T* const p_j, const T* const v_j,
                    T* residual) const {
        // r_R = Log( dq^T * q_i^{-1} * q_j )
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

        // r_v = R_i^T (v_j - v_i - g dt) - dv
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

        // r_p = R_i^T (p_j - p_i - v_i dt - 0.5 g dt^2) - dp
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
