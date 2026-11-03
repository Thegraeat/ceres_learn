#include <iostream>
#include <random>
#include <vector>
#include <ceres/ceres.h>

struct CostFunctor{

    CostFunctor(double x, double y) : x_(x), y_(y) {}

    template<typename T>
    bool operator()(const T *const m, const T *const b, T *residual) const{
        residual[0] = T(y_) - (m[0] * T(x_) + b[0]);
        return true;
    }

    const double x_, y_;
};

struct CostFunctorNum{
    CostFunctorNum(double x, double y) : x_(x), y_(y) {}

    bool operator()(const double *const m, const double *const b, double *residual) const{
        residual[0] = y_ - (m[0] * x_ + b[0]);
        return true;
    }

    const double x_, y_;

};



int main(int argc, char const *argv[]){

    double m = 0.0, b = 0.0;
    const double m_true = 2.0, b_true = -0.5;

    const int n = 200;
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.05);

    std::vector<double> data_x;
    std::vector<double> data_y;

    for (int i = 0; i < n; ++i) {
        double x = -1.0 + 2.0 * i / (n - 1);
        double y = m_true * x + b_true + noise(rng);
        data_x.push_back(x);
        data_y.push_back(y);
    }

    double x_sum = 0.0, y_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x_sum += data_x[i];
        y_sum += data_y[i];
    }
    const double x_bar = x_sum / n;
    const double y_bar = y_sum / n;

    double sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        sxx += (data_x[i] - x_bar) * (data_x[i] - x_bar);
        sxy += (data_x[i] - x_bar) * (data_y[i] - y_bar);
    }
    const double m_ls = sxy / sxx;
    const double b_ls = y_bar - m_ls * x_bar;

    ceres::Problem problem;

    // 自动微分
    for (size_t i = 0; i < data_x.size(); i++ ){
        ceres::CostFunction *cost_function = 
         new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1>(new CostFunctor(data_x[i],data_y[i]));
    
        problem.AddResidualBlock(cost_function, NULL, &m, &b);
    }

    // 数值微分
    // for (size_t i = 0; i < data_x.size(); i++){
    //     ceres::CostFunction *cost_function = 
    //     new ceres::NumericDiffCostFunction<CostFunctorNum, ceres::CENTRAL, 1, 1, 1>(new CostFunctorNum(data_x[i], data_y[i]));

    //     problem.AddResidualBlock(cost_function, NULL, &m, &b);
    // }
    

    if (0){
        problem.SetParameterBlockConstant(&b);
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "m = " << m << ",b = " << b << std::endl;
    std::cout << "m_ls = " << m_ls << ", b_ls = " << b_ls << std::endl;


    return 0;
}
