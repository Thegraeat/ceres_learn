# 第 4 课 课后作业参考答案

> 对应题目：`assignments/04_homework.md`
> 可编译代码：`assignments/hw04_ba.cpp`

---

## 1. 关键结论

### 1.1 外点对 BA 的影响

10% 的观测偏移 ±40 像素，相对于 $\sigma = 0.5$ 是 80 个 sigma。普通最小二乘会让这些残差主导优化：

- 位姿被拉偏；
- 点被拉向错误射线；
- 内点重投影误差也被抬高。

以单个坐标偏移 $40$ px、另一坐标误差为零的简化情况为例，白化残差范数是 $40/0.5=80$。普通代价为 $\frac12\times80^2=3200$；Huber 阈值为 1 时，代价为 $80-0.5=79.5$。若横纵坐标都偏移 $40$ px，应该先计算二维范数 $80\sqrt2$，再代入核，不能只按一个坐标计算。

### 1.2 两种 RMSE 的意义

全量 RMSE：

$$
\mathrm{RMSE}_{all} =
\sqrt{\frac{1}{|\mathcal{O}|}\sum_{(c,k)}
\left\|\pi(T_c^{-1}X_k)-\mathbf{uv}_{ck}\right\|^2}
$$

内点 RMSE：

$$
\mathrm{RMSE}_{in} =
\sqrt{\frac{1}{|\mathcal{O}_{in}|}\sum_{(c,k)\in\mathcal{O}_{in}}
\left\|\pi(T_c^{-1}X_k)-\mathbf{uv}_{ck}\right\|^2}
$$

- $\mathrm{RMSE}_{all}$ 会被外点拉高，即使内点拟合很好；
- $\mathrm{RMSE}_{in}$ 才反映“模型对内点的解释能力”。

工业评估通常同时报告两者，或者用鲁棒统计量（中位数、分位数）。

把它们联系起来，令观测总数为 $N$、内点数为 $N_{in}$、外点数为 $N_{out}$，则：

$$
\mathrm{RMSE}_{all}^2
=\frac{N_{in}}N\mathrm{RMSE}_{in}^2
+\frac{N_{out}}N\mathrm{RMSE}_{out}^2.
$$

例如 90% 内点的二维 RMSE 为 1 px，10% 外点的二维 RMSE 为 40 px，全量 RMSE 仍约为 $\sqrt{0.9\times1^2+0.1\times40^2}\approx12.69$ px。这不与内点拟合良好矛盾。

这里报告的是**原始像素误差**，不带白化，也不带核函数。鲁棒优化改变参数的估计方式，不会让评估时的原始外点误差自动消失。合成实验可以用注入污染时保存的标签定义内点集；真实数据中不能把未经验证的算法自选内点集当成真值标签。

### 1.3 Huber 之后的预期

- $\mathrm{RMSE}_{all}$ 仍然高于 0.5（外点残差还在）；
- $\mathrm{RMSE}_{in}$ 应明显低于无核，但 Huber 只是降权、不是彻底剔除外点，小数据集和弱几何下不保证回到纯噪声水平；当前本机约为 `1.69 px`，无核约 `5.41 px`；
- 点误差和位姿误差明显优于无核。

---

## 2. 完整答案代码

```cpp
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
        const T d[3] = {p[0] - t[0], p[1] - t[1], p[2] - t[2]};
        T pc[3];
        QuatRotateInv(q, d, pc);
        if (pc[2] <= T(1e-8)) return false;

        const T u_pred = T(kFx) * pc[0] / pc[2] + T(kCx);
        const T v_pred = T(kFy) * pc[1] / pc[2] + T(kCy);

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
```

---

## 3. 预期输出形态

```
observations: ~100, outliers: ~10

[no loss] full RMSE=... px, inlier RMSE=... px, point=... m, trans=... m
[huber  ] full RMSE=... px, inlier RMSE=... px, point=... m, trans=... m
```

判据：

- `no loss` 的内点 RMSE 明显高于 0.5 px；
- `huber` 的内点 RMSE 明显下降，不强制等于 `0.5 px`；二维 RMSE 的尺度解释见第 4 课；
- `huber` 的 point/trans 误差更小；
- 两者的 `full RMSE` 都可能高于 0.5，因为外点还在统计里。

---

## 4. 思考题答案

### 4.1 为什么必须固定一个相机？

BA 的残差只依赖相对位姿。把所有相机和点同时乘一个全局变换：

$$
T_c' = T_g T_c,\qquad X_k' = T_g X_k
$$

重投影误差完全不变，所以存在全局平移和旋转的 6 维 gauge。对于本题单目 BA，还可以以首帧中心为中心缩放全部相机和点，投影同样不变，合计 **7 维**。固定首帧只去掉 6 维；代码还把第二帧的 `t[0]` 设为已知真值并通过 `SubsetManifold` 固定，去掉尺度自由度。这是教学提供的尺度基准，不是从图像恢复了米制尺度。不处理 gauge 时，即使 LM 能给出某个解，绝对位姿/尺度与协方差也不能直接解释。

### 4.2 点为什么不需要 Manifold？

点就是 $\mathbb{R}^3$ 中的普通向量，没有约束，直接 $X + \Delta X$ 即可。位姿的旋转部分在 $\mathrm{SO}(3)$ 上，四元数必须保持单位范数，所以需要 Manifold 在切空间里做加法。平移部分同样是 $\mathbb{R}^3$，不需要 Manifold。

### 4.3 只被一个相机看到的点可解吗？

不可解。一个点有 3 个未知数，一个相机只提供 2 个观测方程，欠定。几何上，点只能被约束在一条射线上，深度不可知。BA 中这种点会让 $H$ 出现零空间，表现为点跑到无穷远或数值失败。

更明确地，把测量像素写成归一化射线方向：

$$
d_c=\begin{bmatrix}(u-c_x)/f_x\\(v-c_y)/f_y\\1\end{bmatrix},
\qquad X_w=t_c+\lambda R_cd_c,\quad\lambda>0.
$$

任意正深度 $\lambda$ 都投影到同一个像素。即使相机位姿已经知道，也缺少确定 $\lambda$ 的信息；再多迭代不会产生新信息。程序不一定报错，也可能返回其中某个深度，因此“求解成功”仍不代表深度唯一。

解决方案：

- 要求至少 2 个具有足够视差的视角；
- 加深度先验；
- 直接把这个点边缘化/剔除。

### 4.4 外点到 30% 会怎样？

Huber 的效果取决于外点位置、杠杆效应、初值和可用几何约束，不能给非线性 BA 套用统一的“接近 50% 崩溃点”。外点增多时，可结合：

- 前端 RANSAC 先筛；
- 更鲁棒的核（Cauchy、Tukey）；
- 多次重启 / 增量式 BA；
- 一致性检查（卡方、几何验证）。

---

## 5. 常见错误

1. **评估时把外点也算进“内点 RMSE”**：指标会虚高。
2. **外点注入到真值观测上**：没有干净基准可比。
3. **两次求解共用同一组 `poses/points`**：第二次从第一次的解出发，不是同一个初值。
4. **忘记固定相机**：`SPARSE_NORMAL_CHOLESKY` 报数值错误。
5. **Huber 阈值设得太大/太小**：`1.0` 是相对白化残差，别设成 `0.01`。
