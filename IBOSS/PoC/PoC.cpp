#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "./lib/utils.hpp"
#include "./lib/regression_models.hpp"

#include "iboss.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[]) {

    auto data = clean_numerical_load_csv(argv[1], true, ','); 
    ofstream out(argv[2]);

    MatrixXd X(data.rows(), data.cols() + 1);
    X.leftCols(1) = VectorXd::Constant(data.rows(), 1);
    X.rightCols(data.cols()) = data.leftCols(data.cols());
    VectorXd y = VectorXd::Constant(data.rows(), 1);
    for (int i = 0; i < data.rows(); ++i) {
        y(i) = i;
    }
    
    auto start = chrono::high_resolution_clock::now();
    auto result = k_selection(X, y, stoll(argv[3]));
    auto betaIBOSS = betaOLS(result.first, result.second, true);
    auto end_iboss = chrono::high_resolution_clock::now();

    auto beta = betaOLS(X, y, true);
    auto end_OLS = chrono::high_resolution_clock::now();

    out << "X1,X2" << '\n';
    out << fixed << setprecision(12);
    for (int i = 0; i < result.first.rows(); ++i) {
        out << result.first(i, 1) << "," << result.first(i, 2);
        out << '\n';
    }

    chrono::duration<double> duration = end_iboss - start;
    cout << "Time taken for IBOSS (selection + OLS): " << duration.count() << '\n';
    
                            duration = end_OLS - end_iboss;
    cout << "Time taken for OLS on full data: " << duration.count() << '\n';

    cout << "R2 score for IBOSS trained: " << r_squared_score(y, X * betaIBOSS) << '\n';
    cout << "R2 score for OLS trained on full data: " << r_squared_score(y, X * beta) << '\n';
    return 0;    
}   