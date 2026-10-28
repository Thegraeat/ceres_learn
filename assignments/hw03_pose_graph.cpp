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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
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

void NormalizeQuat(double* q) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; ++i) q[i] /= n;
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
        T qij[4] = {T(q_ij_[0]), T(q_ij_[1]), T(q_ij_[2]), T(q_ij_[3])};
        T qij_inv[4];
        QuatConj(qij, qij_inv);

        T qi_inv[4];
        QuatConj(q_i, qi_inv);

        T tmp[4];
        QuatMul(qi_inv, q_j, tmp);
        T qe[4];
        QuatMul(qij_inv, tmp, qe);

        if (qe[0] < T(0.0)) {
            for (int k = 0; k < 4; ++k) qe[k] = -qe[k];
        }

        residual[0] = T(2.0) * qe[1] / T(sigma_rot_);
        residual[1] = T(2.0) * qe[2] / T(sigma_rot_);
        residual[2] = T(2.0) * qe[3] / T(sigma_rot_);

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
