#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <cmath>
#include "regression_metrics.hpp"
#include <omp.h>
#include <Eigen/Sparse>
using namespace std;
using namespace Eigen;

VectorXd LinearRegression(const MatrixXd& X,const VectorXd& y)
{
    
    return (X.transpose()*X).ldlt().solve(X.transpose()*y);
}
