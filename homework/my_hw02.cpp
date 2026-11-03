// 第 2 课 课后作业模板（my_hw02.cpp）
//
// 任务：带外点的曲线拟合  y = exp(a*x^2 + b*x + c)
//   1) 比较 No loss / Huber / Cauchy 三种损失
//   2) 用 MAD 从残差估计 sigma，再重新白化
//   3) 用 3 sigma 阈值剔除外点，重新做普通最小二乘
//
// 这个文件已经准备好：数据生成、三种 loss 的调用流程、输出格式。
// 需要你手写的地方都标了 TODO：
//   TODO 1  CurveResidual::operator()  —— 残差、白化
//   TODO 2  Solve()                    —— Ceres 建模与求解
//   TODO 3  EstimateSigma()            —— MAD 尺度估计
//   TODO 4  SolveInliers()             —— 3 sigma 内点筛选
//
// 编译运行：
//   cmake --build build --target my_hw02 -j && ./build/my_hw02
//
// 验收标准见 assignments/02_homework.md。

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// 残差块
// ---------------------------------------------------------------------------
struct CurveResidual {
    CurveResidual(double x, double y, double sigma)
        : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        // TODO 1: 计算白化残差。
        //   原始残差：r = y - exp(a*x^2 + b*x + c)
        //   白化：   residual[0] = r / sigma
        //   注意用 ceres::exp 或 T 版本的 std::exp，别对 T 调 double 的 exp。
        //   完成后删除下面的占位语句。
        (void)abc;
        residual[0] = T(0.0);
        return true;
    }

    const double x_, y_, sigma_;
};

enum class LossType { NONE, HUBER, CAUCHY };

struct Result {
    double abc[3];
    double cost;
    bool ok;
};

// ---------------------------------------------------------------------------
// 求解：同一套流程跑三种 loss
// ---------------------------------------------------------------------------
Result Solve(LossType type,
             double sigma,
             const std::vector<double>& xs,
             const std::vector<double>& ys,
             const double init[3]) {
    if (xs.size() != ys.size() || xs.size() < 3 ||
        !std::isfinite(sigma) || sigma <= 0.0) {
        return {{0.0, 0.0, 0.0}, -1.0, false};
    }

    double abc[3] = {init[0], init[1], init[2]};

    // TODO 2: 在这里写 Ceres 建模与求解。
    //   1. ceres::Problem problem;
    //   2. 对每个观测 new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
    //          new CurveResidual(xs[i], ys[i], sigma));
    //   3. 按 type 选 loss：
    //        NONE   -> nullptr
    //        HUBER  -> new ceres::HuberLoss(1.0)
    //        CAUCHY -> new ceres::CauchyLoss(1.0)
    //   4. problem.AddResidualBlock(cost, loss, abc);
    //   5. Options: DENSE_QR、max_num_iterations = 100
    //   6. Solve 后检查 summary.IsSolutionUsable()
    //   7. 成功时返回 {abc[0], abc[1], abc[2], summary.final_cost, true}
    (void)type;
    (void)sigma;
    (void)xs;
    (void)ys;
    (void)abc;

    return {{0.0, 0.0, 0.0}, -1.0, false};  // TODO 2 完成后替换
}

// ---------------------------------------------------------------------------
// 工具：中位数（偶数个元素取中间两个的平均）
// ---------------------------------------------------------------------------
double Median(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of empty data");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------
// 用 MAD 估计噪声 sigma
// ---------------------------------------------------------------------------
double EstimateSigma(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     const double abc[3]) {
    // TODO 3: 用未白化残差估计 sigma。
    //   1. 对每个点算 pred = exp(a*x^2 + b*x + c)，残差 r_i = y_i - pred
    //   2. med = Median(r)
    //   3. 把 r_i 换成 |r_i - med|
    //   4. 返回 1.4826 * Median(|r - med|)
    // 提示：MAD 只能估计尺度，不能修正被外点拉偏的模型。
    (void)xs;
    (void)ys;
    (void)abc;
    return 1.0;  // TODO 3 完成后替换；这里返回 1 只是为了避免除零
}

// ---------------------------------------------------------------------------
// 先鲁棒拟合，再用 3 sigma 剔除外点，最后在内点集上做普通最小二乘
// ---------------------------------------------------------------------------
Result SolveInliers(double sigma,
                    double threshold,
                    const std::vector<double>& xs,
                    const std::vector<double>& ys,
                    const double init[3]) {
    const Result rough = Solve(LossType::HUBER, sigma, xs, ys, init);
    if (!rough.ok) return rough;

    std::vector<double> xi, yi;

    // TODO 4: 用 rough.abc 计算每个点的原始残差，只保留
    //   |y_i - pred_i| < threshold * sigma
    // 的点，存进 xi / yi。
    //
    // 提示：先鲁棒拟合再剔除，避免被外点拉偏的模型误删内点。
    (void)threshold;

    std::cout << "[inlier] kept " << xi.size() << " / " << xs.size() << " points\n";
    if (xi.size() < 3) return {{0.0, 0.0, 0.0}, -1.0, false};

    return Solve(LossType::NONE, sigma, xi, yi, rough.abc);
}

void PrintRow(const char* name, const Result& r, const double truth[3]) {
    if (!r.ok) {
        std::cout << name << " <not solved: TODO 尚未完成>\n";
        return;
    }
    double err2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double d = r.abc[k] - truth[k];
        err2 += d * d;
    }
    std::cout << name << " abc=(" << r.abc[0] << ", " << r.abc[1] << ", "
              << r.abc[2] << ") |err|=" << std::sqrt(err2)
              << " cost=" << r.cost << "\n";
}

int main() {
    const double truth[3] = {1.0, 2.0, 0.5};
    const double init[3] = {0.8, 1.8, 0.2};

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);

    std::vector<double> xs, ys;
    for (int i = 0; i < 100; ++i) {
        const double x = -1.0 + 2.0 * i / 99.0;
        double y = std::exp(truth[0] * x * x + truth[1] * x + truth[2]) + noise(rng);
        if (i % 10 == 0) {
            y += 0.4 * y;  // 外点
        }
        xs.push_back(x);
        ys.push_back(y);
    }

    std::cout << "truth abc=(" << truth[0] << ", " << truth[1] << ", "
              << truth[2] << ")\n\n";

    const Result none = Solve(LossType::NONE, 0.02, xs, ys, init);
    const Result huber = Solve(LossType::HUBER, 0.02, xs, ys, init);
    const Result cauchy = Solve(LossType::CAUCHY, 0.02, xs, ys, init);

    PrintRow("[none  ]", none, truth);
    PrintRow("[huber ]", huber, truth);
    PrintRow("[cauchy]", cauchy, truth);

    if (!none.ok || !huber.ok || !cauchy.ok) {
        std::cout << "\nTODO: 先完成 TODO 1/2，任务 1 才能跑出结果。\n";
        return 0;
    }

    // 假设 sigma 未知：粗尺度 -> 鲁棒拟合 -> MAD -> 重新白化。
    const double initial_scale = std::max(1e-8, EstimateSigma(xs, ys, none.abc));
    const Result preliminary =
        Solve(LossType::CAUCHY, initial_scale, xs, ys, none.abc);
    if (!preliminary.ok) {
        std::cout << "\nTODO: 任务 2 的鲁棒预拟合没有完成。\n";
        return 0;
    }

    const double sigma_hat =
        std::max(1e-8, EstimateSigma(xs, ys, preliminary.abc));
    std::cout << "\nMAD sigma_hat = " << sigma_hat << " (true 0.02)\n";

    const Result mad = Solve(LossType::HUBER, sigma_hat, xs, ys, preliminary.abc);
    PrintRow("[huber+MAD]", mad, truth);

    // 用 3 sigma 阈值剔除外点后，再做普通最小二乘
    const Result inlier = SolveInliers(sigma_hat, 3.0, xs, ys, mad.abc);
    PrintRow("[3sigma LS]", inlier, truth);

    return 0;
}
