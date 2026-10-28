#include <iostream>
#include <ceres/ceres.h>

struct CostFunctor {
    template<typename T>
    bool operator()(const T *const x, T *residual) const {
        residual[0] = 10.0 - x[0] - 2.0 * x[1]; 
        return true;
    }
};

struct CostFunctorNum {
    bool operator()(const double *const x, double *residual) const {
        residual[0] = 10.0 - x[0]; // f(x) = 10 - x
        return true;
    }
};

class SimpleCostFunctor : public ceres::SizedCostFunction<1,1> {
public:
    virtual ~SimpleCostFunctor() {};

    virtual bool Evaluate(
      double const* const* parameters, double *residuals, double **jacobians) const {
        const double x = parameters[0][0];

        residuals[0] = 10 - x; // f(x) = 10 - x

        if(jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = -1; // f'(x) = -1
        }

        return true;
    }
};

int main(int argc, char** argv) {

    double x[2] = {100.0, 100.0};
    const double initial_x[2] = {x[0], x[1]};

    ceres::Problem problem;

    // Set up the only cost function (also known as residual)
    ceres::CostFunction *cost_function;

    // auto-differentiation
    cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 2>(new CostFunctor);

     //numeric differentiation
//    cost_function =
//    new ceres::NumericDiffCostFunction<CostFunctorNum, ceres::CENTRAL, 1, 1>(
//      new CostFunctorNum);

    // cost_function = new SimpleCostFunctor;

    // 添加代价函数cost_function和损失函数NULL，其中x为状态量
    problem.AddResidualBlock(cost_function, NULL, x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    // options.max_num_iterations = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : [" << initial_x[0] << ", " << initial_x[1] << "] -> ["
              << x[0] << ", " << x[1] << "]\n";

    return 0;
}
