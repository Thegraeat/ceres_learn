// 第 3 课配套代码：SE(3) 位姿图优化
//
// 问题：给定一组位姿节点和它们之间的相对位姿测量（里程计边 + 回环边），
//       求最大似然位姿。
//
// 残差：e_ij = [2 vec(q_e), t_e]，不是严格的 SE(3) Log
// 目标：min Σ || e_ij ||^2_{Σ_ij}
//
// 关键点：
//   1. 四元数是 4 维，但旋转只有 3 个自由度 -> 必须用 Manifold（Ceres 的 ⊞）
//   2. 残差是 6 维（旋转 3 + 平移 3），不是 7 维直接相减
//   3. 固定第一个位姿，消除 gauge freedom（整体平移/旋转不变性）
//   4. H = J^T J 是稀疏的，用 SPARSE_NORMAL_CHOLESKY

#include <ceres/ceres.h>

#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

// ----------------------------- 四元数工具 -----------------------------
// 约定：q = [w, x, y, z]，Hamilton 乘法，对应 Ceres 的 QuaternionManifold。
// 全部写成模板，double 和 ceres::Jet 都能用（自动微分要求）。

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

// out = R(q) * v，用四元数旋转向量（比先转矩阵再乘更省）。
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
    double q[4] = {1.0, 0.0, 0.0, 0.0};  // w, x, y, z
    double t[3] = {0.0, 0.0, 0.0};
};

struct Edge {
    int i;
    int j;
    Pose meas;  // T_ij = T_i^{-1} * T_j
};

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
        // 1) 旋转残差：q_e = q_ij^{-1} * q_i^{-1} * q_j
        T qij[4] = {T(q_ij_[0]), T(q_ij_[1]), T(q_ij_[2]), T(q_ij_[3])};
        T qij_inv[4];
        QuatConj(qij, qij_inv);

        T qi_inv[4];
        QuatConj(q_i, qi_inv);

        T tmp[4];
        QuatMul(qi_inv, q_j, tmp);
        T qe[4];
        QuatMul(qij_inv, tmp, qe);

        // 四元数双覆盖：q 与 -q 是同一个旋转。统一到 w >= 0 的半球，
        // 否则相邻两次迭代的残差可能突然反号，雅可比会跳变。
        if (qe[0] < T(0.0)) {
            for (int k = 0; k < 4; ++k) qe[k] = -qe[k];
        }

        // 2 * vec(q) 是旋转向量的小角度近似（qe ≈ [1, θ/2] 时正好等于 θ）。
        // 此处使用小角度近似，不等同于完整的 SO(3) Log。
        residual[0] = T(2.0) * qe[1] / T(sigma_rot_);
        residual[1] = T(2.0) * qe[2] / T(sigma_rot_);
        residual[2] = T(2.0) * qe[3] / T(sigma_rot_);

        // 2) 平移残差：e_t = R_ij^T * ( R_i^T * (t_j - t_i) - t_ij )
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

// ----------------------------- 真值与测量 -----------------------------
const double kPi = 3.14159265358979323846;

// T_ij = T_i^{-1} * T_j
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

int main() {
    const int kNumNodes = 20;
    const double kRadius = 5.0;
    const double kSigmaRot = 0.02;    // rad
    const double kSigmaTrans = 0.1;   // m

    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // 1) 真值轨迹：半径 5 的圆，机头沿切线方向
    std::vector<Pose> gt(kNumNodes);
    for (int i = 0; i < kNumNodes; ++i) {
        const double theta = 2.0 * kPi * i / kNumNodes;
        gt[i].t[0] = kRadius * std::cos(theta);
        gt[i].t[1] = kRadius * std::sin(theta);
        gt[i].t[2] = 0.0;

        const double yaw = theta + kPi / 2.0;
        gt[i].q[0] = std::cos(yaw / 2.0);
        gt[i].q[1] = 0.0;
        gt[i].q[2] = 0.0;
        gt[i].q[3] = std::sin(yaw / 2.0);
    }

    // 2) 边：相邻里程计边 + 每隔 5 个节点的回环边
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < kNumNodes; ++i) {
        pairs.emplace_back(i, (i + 1) % kNumNodes);
    }
    for (int i = 0; i < kNumNodes; i += 5) {
        pairs.emplace_back(i, (i + 5) % kNumNodes);
    }

    // 3) 给每条边生成带噪测量
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

    // 4) 初值：只用相邻边做航迹推算（deaded reckoning），误差会累积
    std::vector<Pose> init(kNumNodes);
    init[0] = gt[0];
    for (int i = 0; i < kNumNodes - 1; ++i) {
        double q_next[4];
        QuatMul(init[i].q, edges[i].meas.q, q_next);
        NormalizeQuat(q_next);

        double t_local[3];
        QuatRotate(init[i].q, edges[i].meas.t, t_local);

        for (int k = 0; k < 4; ++k) init[i + 1].q[k] = q_next[k];
        for (int k = 0; k < 3; ++k) init[i + 1].t[k] = init[i].t[k] + t_local[k];
    }

    const double ate_before = AteRmse(init, gt);

    // 5) 构建 Ceres 问题
    ceres::Problem problem;
    for (int i = 0; i < kNumNodes; ++i) {
        // 四元数参数块挂 Manifold：单位初值在 3 维切空间上更新。
        problem.AddParameterBlock(init[i].q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(init[i].t, 3);
    }

    // gauge freedom：整体平移 + 旋转不改变任何残差。
    // 固定第一个位姿，否则 H 奇异，线性求解器会失败或给出漂移解。
    problem.SetParameterBlockConstant(init[0].q);
    problem.SetParameterBlockConstant(init[0].t);

    for (const auto& e : edges) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<EdgeCost, 6, 4, 3, 4, 3>(
                new EdgeCost(e.meas.q, e.meas.t, kSigmaRot, kSigmaTrans));

        // 练习：把 nullptr 换成 new ceres::HuberLoss(1.0)，再手动加几条错误回环。
        ceres::LossFunction* loss = nullptr;

        problem.AddResidualBlock(cost, loss, init[e.i].q, init[e.i].t,
                                 init[e.j].q, init[e.j].t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";
    std::cout << "nodes: " << kNumNodes << ", edges: " << edges.size() << "\n";
    std::cout << "ATE before: " << ate_before << " m\n";
    std::cout << "ATE after : " << AteRmse(init, gt) << " m\n";
    return 0;
}
