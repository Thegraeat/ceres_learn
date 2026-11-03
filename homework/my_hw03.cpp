// 第 3 课 课后作业模板（my_hw03.cpp）
//
// 任务：错误回环下的鲁棒位姿图
//   1) 在 (0,5)、(5,10) 两条回环的平移测量上各加 3 米
//   2) 分别用 loss = nullptr 和 HuberLoss(1.0) 求解
//   3) 实现旋转误差 ARE
//   4) 把固定节点从第 0 个换到最后一个
//
// 这个文件已经准备好：四元数工具、轨迹/边/噪声/航迹推算初值、错误回环注入、
// ATE 与结果打印。需要你手写的地方都标了 TODO：
//   TODO 1  EdgeCost::operator()  —— 6 维白化残差
//   TODO 2  RotationError()/AreRmse() —— 旋转误差
//   TODO 3  Solve()               —— Manifold、参数块、残差块、求解
//   TODO 4  main() 末尾的换固定节点实验
//
// 编译运行：
//   cmake --build build --target my_hw03 -j && ./build/my_hw03
//
// 验收标准见 assignments/03_homework.md。

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
        // TODO 1: 计算 6 维白化残差 residual[0..5]。
        //   旋转误差：q_e = q_ij^{-1} * q_i^{-1} * q_j，统一到 w >= 0 的半球
        //   平移误差：e_t = R_ij^T * (R_i^T * (t_j - t_i) - t_ij)
        //   白化：
        //     residual[0..2] = 2 * q_e.vec() / sigma_rot
        //     residual[3..5] = e_t / sigma_trans
        //   可用 Eigen::Quaternion<T> 和 Eigen::Map 完成。
        //   完成后删除下面的占位语句。
        (void)q_i;
        (void)t_i;
        (void)q_j;
        (void)t_j;
        for (int k = 0; k < 6; ++k) residual[k] = T(0.0);
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

// TODO 2a: 实现单帧旋转误差，返回弧度。
//   q_e = q_est^{-1} * q_gt，theta = 2 * acos(|w_e|)
//   提示：四元数先归一化，|w| 截断到 [0,1]，避免 acos 产生 NaN。
double RotationError(const Pose& est, const Pose& gt) {
    (void)est;
    (void)gt;
    return 0.0;
}

// TODO 2b: 把所有单帧旋转误差聚合成 ARE（弧度）。
double AreRmse(const std::vector<Pose>& est, const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        const double a = RotationError(est[i], gt[i]);
        sum += a * a;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

// ---------------------------------------------------------------------------
// 求解位姿图
// ---------------------------------------------------------------------------
bool Solve(const std::vector<Edge>& edges,
           const std::vector<Pose>& init,
           bool use_huber,
           std::vector<Pose>& out,
           ceres::Solver::Summary* summary) {
    out = init;
    const int n = static_cast<int>(out.size());

    // TODO 3: 构建并求解位姿图。
    //   1. 每个位姿注册两个参数块：
    //        AddParameterBlock(out[i].q, 4, new ceres::QuaternionManifold)
    //        AddParameterBlock(out[i].t, 3)
    //   2. 固定 out[0].q 和 out[0].t
    //   3. 对每条边：
    //        AutoDiffCostFunction<EdgeCost, 6, 4, 3, 4, 3>
    //        loss = use_huber ? new HuberLoss(1.0) : nullptr
    //        AddResidualBlock(cost, loss,
    //                         out[e.i].q, out[e.i].t,
    //                         out[e.j].q, out[e.j].t)
    //   4. Options: SPARSE_NORMAL_CHOLESKY，max_num_iterations = 100
    //   5. Solve，并返回 summary->IsSolutionUsable()
    (void)edges;
    (void)use_huber;
    (void)n;
    (void)summary;
    return false;  // TODO 3 完成后替换
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

    // 注入两条错误回环：在拷贝出来的边上改
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

    if (!Solve(bad_edges, init, false, est_no, &summary_no) ||
        !Solve(bad_edges, init, true, est_huber, &summary_huber)) {
        std::cout << "TODO: 先完成 TODO 1/3，位姿图才能求解。\n";
        return 0;
    }

    std::cout << "[no loss] " << summary_no.BriefReport() << "\n";
    std::cout << "          ATE=" << AteRmse(est_no, gt)
              << " m, ARE=" << AreRmse(est_no, gt) << " rad\n\n";

    std::cout << "[huber ]  " << summary_huber.BriefReport() << "\n";
    std::cout << "          ATE=" << AteRmse(est_huber, gt)
              << " m, ARE=" << AreRmse(est_huber, gt) << " rad\n";

    // TODO 4: 把 Solve() 里的 SetParameterBlockConstant 从 out[0]
    //   改到 out[n-1]，重新运行并回答：
    //   - final_cost 变了吗？
    //   - ATE / ARE 变了吗？
    //   - 为什么？

    return 0;
}
