// 第 2 课课后作业参考答案
//
// 任务：带外点的曲线拟合 y = exp(a x^2 + b x + c)
//   1) 比较 No loss / Huber / Cauchy
//   2) 用 MAD 从残差估计 sigma，再重新白化
//   3) 用 3 sigma 阈值剔除外点，重新做普通最小二乘
//
// 编译后运行：./build/hw02_robust

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

struct CurveResidual {
    CurveResidual(double x, double y, double sigma)
        : x_(x), y_(y), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] =
            (T(y_) - ceres::exp(abc[0] * T(x_ * x_) + abc[1] * T(x_) + abc[2])) /
            T(sigma_);
        return true;
    }

    const double x_, y_, sigma_;
};

enum class LossType { NONE, HUBER, CAUCHY };

struct Result {
    double abc[3];
    double cost;
};

Result Solve(LossType type,
             double sigma,
             const std::vector<double>& xs,
             const std::vector<double>& ys,
             const double init[3]) {
    double abc[3] = {init[0], init[1], init[2]};
    if (xs.size() != ys.size() || xs.size() < 3 ||
        !std::isfinite(sigma) || sigma <= 0.0) {
        throw std::invalid_argument("need matching data, at least 3 points and positive sigma");
    }

    ceres::Problem problem;
    for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
                new CurveResidual(xs[i], ys[i], sigma));

        ceres::LossFunction* loss = nullptr;
        if (type == LossType::HUBER) {
            loss = new ceres::HuberLoss(1.0);
        } else if (type == LossType::CAUCHY) {
            loss = new ceres::CauchyLoss(1.0);
        }

        problem.AddResidualBlock(cost, loss, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) throw std::runtime_error(summary.BriefReport());

    return {abc[0], abc[1], abc[2], summary.final_cost};
}

double Median(std::vector<double> v) {
    if (v.empty()) throw std::invalid_argument("median of empty data");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 用残差的中位数绝对偏差（MAD）估计噪声 sigma。
// 1.4826 是正态分布下 MAD 与标准差的换算系数。
double EstimateSigma(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     const double abc[3]) {
    std::vector<double> residuals;
    residuals.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        const double pred =
            std::exp(abc[0] * xs[i] * xs[i] + abc[1] * xs[i] + abc[2]);
        residuals.push_back(ys[i] - pred);
    }

    const double med = Median(residuals);
    for (double& r : residuals) {
        r = std::abs(r - med);
    }
    return 1.4826 * Median(residuals);
}

Result SolveInliers(double sigma,
                    double threshold,
                    const std::vector<double>& xs,
                    const std::vector<double>& ys,
                    const double init[3]) {
    // 先做鲁棒拟合，避免被外点拉偏的模型误删内点。
    const Result rough = Solve(LossType::HUBER, sigma, xs, ys, init);

    std::vector<double> xi, yi;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double pred = std::exp(rough.abc[0] * xs[i] * xs[i] +
                                     rough.abc[1] * xs[i] + rough.abc[2]);
        if (std::abs(ys[i] - pred) < threshold * sigma) {
            xi.push_back(xs[i]);
            yi.push_back(ys[i]);
        }
    }

    std::cout << "[inlier] kept " << xi.size() << " / " << xs.size() << " points\n";
    return Solve(LossType::NONE, sigma, xi, yi, rough.abc);
}

void PrintRow(const char* name, const Result& r, const double truth[3]) {
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

    // 假设 sigma 未知：粗尺度 -> 鲁棒拟合 -> MAD -> 重新白化。
    const double initial_scale = std::max(1e-8, EstimateSigma(xs, ys, none.abc));
    const Result preliminary = Solve(LossType::CAUCHY, initial_scale, xs, ys, none.abc);
    const double sigma_hat = std::max(1e-8, EstimateSigma(xs, ys, preliminary.abc));
    std::cout << "\nMAD sigma_hat = " << sigma_hat << " (true 0.02)\n";

    const Result mad = Solve(LossType::HUBER, sigma_hat, xs, ys, preliminary.abc);
    PrintRow("[huber+MAD]", mad, truth);

    // 用 3 sigma 阈值剔除外点后，再做普通最小二乘
    const Result inlier = SolveInliers(sigma_hat, 3.0, xs, ys, mad.abc);
    PrintRow("[3sigma LS]", inlier, truth);

    return 0;
}
