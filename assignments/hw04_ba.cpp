// 第 4 课课后作业参考答案
//
// 任务：带外点的 Bundle Adjustment
//   1) 注入 10% 误匹配（像素偏移 ±40）
//   2) 比较 loss=nullptr 与 HuberLoss(1.0)
//   3) 输出全量 / 内点重投影 RMSE、点误差、位姿误差
//
// 编译后运行：./build/hw04_ba

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

struct Observation {
    int cam;
    int pt;
    double u;
    double v;
    bool outlier;
};

const double kFx = 500.0;
const double kFy = 500.0;
const double kCx = 320.0;
const double kCy = 240.0;
const double kSigmaPx = 0.5;

// ----------------------------- 残差块 -----------------------------
struct ReprojectionError {
    ReprojectionError(double u, double v, double sigma)
        : u_(u), v_(v), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t, const T* const p,
                    T* residual) const {
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        // 参数数组沿用 wxyz，显式构造四元数；向量通过 Map 读取。
        const Eigen::Quaternion<T> rotation_wc(q[0], q[1], q[2], q[3]);
        const Eigen::Map<const Vector3> translation_wc(t);
        const Eigen::Map<const Vector3> point_world(p);
        const Vector3 point_camera =
            rotation_wc.conjugate() * (point_world - translation_wc);
        if (point_camera.z() <= T(1e-8)) return false;

        const T u_pred = T(kFx) * point_camera.x() / point_camera.z() + T(kCx);
        const T v_pred = T(kFy) * point_camera.y() / point_camera.z() + T(kCy);
        residual[0] = (u_pred - T(u_)) / T(sigma_);
        residual[1] = (v_pred - T(v_)) / T(sigma_);
        return true;
    }

    double u_, v_, sigma_;
};

// ----------------------------- 工具函数 -----------------------------
Pose MakePose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Pose out;
    Eigen::Quaterniond q(R);
    q.normalize();
    out.q[0] = q.w();
    out.q[1] = q.x();
    out.q[2] = q.y();
    out.q[3] = q.z();
    out.t[0] = t.x();
    out.t[1] = t.y();
    out.t[2] = t.z();
    return out;
}

Pose LookAt(const Eigen::Vector3d& eye, const Eigen::Vector3d& target) {
    const Eigen::Vector3d z = (target - eye).normalized();
    const Eigen::Vector3d up = std::abs(z.y()) < 0.9
                                  ? Eigen::Vector3d(0.0, 1.0, 0.0)
                                  : Eigen::Vector3d(1.0, 0.0, 0.0);
    const Eigen::Vector3d x = up.cross(z).normalized();
    const Eigen::Vector3d y = z.cross(x);

    Eigen::Matrix3d R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return MakePose(R, eye);
}

Eigen::Vector3d RotateInv(const Pose& pose, const Eigen::Vector3d& v) {
    const Eigen::Quaterniond q(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
    return q.conjugate() * v;
}

Eigen::Vector2d Project(const Pose& pose, const Eigen::Vector3d& X) {
    const Eigen::Vector3d t(pose.t[0], pose.t[1], pose.t[2]);
    const Eigen::Vector3d pc = RotateInv(pose, X - t);
    return Eigen::Vector2d(kFx * pc.x() / pc.z() + kCx,
                           kFy * pc.y() / pc.z() + kCy);
}

double ReprojectionRmse(const std::vector<Observation>& obs,
                        const std::vector<Pose>& poses,
                        const std::vector<Eigen::Vector3d>& points,
                        bool inliers_only) {
    double sum = 0.0;
    int count = 0;
    for (const auto& o : obs) {
        if (inliers_only && o.outlier) continue;

        const Eigen::Vector2d uv = Project(poses[o.cam], points[o.pt]);
        const double du = uv.x() - o.u;
        const double dv = uv.y() - o.v;
        sum += du * du + dv * dv;
        ++count;
    }
    return std::sqrt(sum / static_cast<double>(count));
}

double PointRmse(const std::vector<Eigen::Vector3d>& est,
                 const std::vector<Eigen::Vector3d>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        sum += (est[i] - gt[i]).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

double PoseTranslationRmse(const std::vector<Pose>& est,
                           const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (size_t i = 0; i < est.size(); ++i) {
        const double dx = est[i].t[0] - gt[i].t[0];
        const double dy = est[i].t[1] - gt[i].t[1];
        const double dz = est[i].t[2] - gt[i].t[2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

double PoseRotationRmse(const std::vector<Pose>& est,
                        const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (size_t index = 0; index < est.size(); ++index) {
        const Eigen::Quaterniond estimated(est[index].q[0], est[index].q[1],
                                            est[index].q[2], est[index].q[3]);
        const Eigen::Quaterniond truth(gt[index].q[0], gt[index].q[1],
                                       gt[index].q[2], gt[index].q[3]);
        const double angle = estimated.angularDistance(truth);
        sum += angle * angle;
    }
    return std::sqrt(sum / static_cast<double>(est.size()));
}

void BuildAndSolve(bool use_huber,
                   const std::vector<Observation>& obs,
                   std::vector<Pose>& poses,
                   std::vector<Eigen::Vector3d>& points,
                   ceres::Solver::Summary* summary) {
    const int num_cams = static_cast<int>(poses.size());
    const int num_points = static_cast<int>(points.size());

    ceres::Problem problem;
    for (int i = 0; i < num_cams; ++i) {
        problem.AddParameterBlock(poses[i].q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(poses[i].t, 3);
    }
    for (int i = 0; i < num_points; ++i) {
        problem.AddParameterBlock(points[i].data(), 3);
    }

    problem.SetParameterBlockConstant(poses[0].q);
    problem.SetParameterBlockConstant(poses[0].t);
    problem.SetManifold(poses[1].t, new ceres::SubsetManifold(3, {0}));

    for (const auto& o : obs) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
                new ReprojectionError(o.u, o.v, kSigmaPx));

        ceres::LossFunction* loss = use_huber ? new ceres::HuberLoss(1.0) : nullptr;

        problem.AddResidualBlock(cost, loss, poses[o.cam].q, poses[o.cam].t,
                                 points[o.pt].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;

    ceres::Solve(options, &problem, summary);
}

void PrintRow(const char* name,
              const std::vector<Observation>& obs,
              const std::vector<Pose>& poses,
              const std::vector<Eigen::Vector3d>& points,
              const std::vector<Pose>& gt_poses,
              const std::vector<Eigen::Vector3d>& gt_points) {
    std::cout << name
              << " full RMSE=" << ReprojectionRmse(obs, poses, points, false)
              << " px, inlier RMSE=" << ReprojectionRmse(obs, poses, points, true)
              << " px, point=" << PointRmse(points, gt_points)
              << " m, trans=" << PoseTranslationRmse(poses, gt_poses)
              << " m, rot=" << PoseRotationRmse(poses, gt_poses) << " rad\n";
}

int main() {
    const int kNumCams = 5;
    const int kNumPoints = 20;

    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> outlier(-40.0, 40.0);

    // 真值
    std::vector<Pose> gt_poses(kNumCams);
    for (int i = 0; i < kNumCams; ++i) {
        const double x = -2.0 + 4.0 * i / (kNumCams - 1);
        gt_poses[i] = LookAt(Eigen::Vector3d(x, 0.0, 0.0),
                             Eigen::Vector3d(0.0, 0.0, 6.0));
    }

    std::vector<Eigen::Vector3d> gt_points(kNumPoints);
    for (int i = 0; i < kNumPoints; ++i) {
        gt_points[i] = Eigen::Vector3d(2.5 * uniform(rng), 1.2 * uniform(rng),
                                       4.0 + 4.0 * std::abs(uniform(rng)));
    }

    // 观测（先不加外点）
    std::vector<Observation> obs;
    for (int c = 0; c < kNumCams; ++c) {
        for (int p = 0; p < kNumPoints; ++p) {
            const Eigen::Vector3d t(gt_poses[c].t[0], gt_poses[c].t[1],
                                    gt_poses[c].t[2]);
            const Eigen::Vector3d pc = RotateInv(gt_poses[c], gt_points[p] - t);
            if (pc.z() < 0.5) continue;

            const Eigen::Vector2d uv = Project(gt_poses[c], gt_points[p]);
            if (uv.x() < 0.0 || uv.x() > 640.0 || uv.y() < 0.0 || uv.y() > 480.0) {
                continue;
            }

            obs.push_back({c, p, uv.x() + gauss(rng) * kSigmaPx,
                           uv.y() + gauss(rng) * kSigmaPx, false});
        }
    }

    // 注入 10% 外点
    std::shuffle(obs.begin(), obs.end(), rng);
    for (size_t i = 0; i < obs.size(); ++i) {
        if (i % 10 == 0) {
            obs[i].u += outlier(rng);
            obs[i].v += outlier(rng);
            obs[i].outlier = true;
        }
    }

    // 初值
    std::vector<Pose> init_poses = gt_poses;
    for (int i = 0; i < kNumCams; ++i) {
        for (int k = 0; k < 3; ++k) {
            init_poses[i].t[k] += 0.1 * gauss(rng);
        }
        const double angle = 0.02 * gauss(rng);
        double dq[4] = {std::cos(angle / 2.0), 0.0, 0.0, std::sin(angle / 2.0)};
        double q_tmp[4];
        QuatMul(init_poses[i].q, dq, q_tmp);
        for (int k = 0; k < 4; ++k) init_poses[i].q[k] = q_tmp[k];
        NormalizeQuat(init_poses[i].q);
    }

    std::vector<Eigen::Vector3d> init_points = gt_points;
    for (int i = 0; i < kNumPoints; ++i) {
        for (int k = 0; k < 3; ++k) {
            init_points[i][k] += 0.2 * gauss(rng);
        }
    }

    init_poses[0] = gt_poses[0];
    init_poses[1].t[0] = gt_poses[1].t[0];

    std::cout << "observations: " << obs.size() << ", outliers: ";
    int outlier_count = 0;
    for (const auto& o : obs) outlier_count += o.outlier ? 1 : 0;
    std::cout << outlier_count << "\n\n";

    std::vector<Pose> poses_no = init_poses;
    std::vector<Eigen::Vector3d> points_no = init_points;
    ceres::Solver::Summary summary_no;
    BuildAndSolve(false, obs, poses_no, points_no, &summary_no);
    PrintRow("[no loss]", obs, poses_no, points_no, gt_poses, gt_points);

    std::vector<Pose> poses_huber = init_poses;
    std::vector<Eigen::Vector3d> points_huber = init_points;
    ceres::Solver::Summary summary_huber;
    BuildAndSolve(true, obs, poses_huber, points_huber, &summary_huber);
    PrintRow("[huber  ]", obs, poses_huber, points_huber, gt_poses, gt_points);

    std::cout << "\n" << summary_no.BriefReport() << "\n";
    std::cout << summary_huber.BriefReport() << "\n";
    return summary_no.IsSolutionUsable() && summary_huber.IsSolutionUsable() ? 0 : 1;
}
