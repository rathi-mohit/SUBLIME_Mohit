#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "./lib/utils.hpp"
#include "./lib/regression_models.hpp"

#include "iboss.hpp"

using namespace Eigen;
using namespace std;

int main() {

    auto data = clean_numerical_load_csv("your_filename_here"); 
    // edit the filename above

    MatrixXd X(data.rows(), data.cols());
    X.col(0) = VectorXd::Constant(data.rows(), 1);
    X.rightCols(data.cols() - 1) = data.leftCols(data.cols() - 1);
    VectorXd y = data.rightCols(1);
    // Builds X and y such that the last column of the data is taken to be the response
    // and the rest as the predictors

    auto start = chrono::high_resolution_clock::now();
    auto result = k_selection(X, y, 10000);
    auto end = chrono::high_resolution_clock::now();
    // timed k-selection

    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << duration.count() << '\n';
    return 0;    
}   