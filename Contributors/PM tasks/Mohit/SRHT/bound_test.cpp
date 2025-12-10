#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "./lib/utils.hpp"
#include "./lib/regression_models.hpp"
#include "iboss.hpp"

#define num long double

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[]) {
    auto data = clean_numerical_load_csv(argv[1]);
    ofstream out(argv[2]);

    MatrixXd X(data.rows(), data.cols());
    X.col(0) = VectorXd::Constant(data.rows(), 1.0);
    X.rightCols(data.cols() - 1) = data.leftCols(data.cols() - 1);
    VectorXd y = data.rightCols(1);

    
    auto start = chrono::high_resolution_clock::now();

    int n = X.rows();
    int d = X.cols();   
    int p = d - 1;         
    num rhsp = 1.0L;

    for (int j = 1; j < d; ++j) {
        VectorXd X_j = X.col(j);
        auto [z_min_it, z_max_it] = minmax_element(X_j.data(), X_j.data() + X_j.size());
        num range = ((num) *z_max_it - *z_min_it);
        rhsp *= (range * range);     
    }

    VectorXd betaols = betaOLS(X, y);
    VectorXd residuals = y - X * betaols;

    num sigma2 = (num) (residuals.squaredNorm()) / (num) (n - d);
    num sigma2p = pow(sigma2, d);
    map<int, num> mp, mP;
    for (int i = 1; i < 101; ++i) {
        num k = (num) (i * 100);   
        auto result = k_selection(X, y, k);
        MatrixXd Xs = result.first;     
        MatrixXd XtXs = Xs.transpose() * Xs;
        num det_XtXs = (num) (XtXs.determinant());
        num lhs = fabs(det_XtXs) / sigma2p;
        mp[(int)k] = lhs;
        num con = k * pow(k / 4.0L, p);
        num rhs = (con / sigma2p) * rhsp;
        mP[(int)k] = rhs;
    }

    auto end = chrono::high_resolution_clock::now();
    auto m = --mP.end(); auto ms = --mp.end();
    for (auto it = mp.begin(), it1 = mP.begin(); it != mp.end(); ++it, ++it1) {
        out << it->first << ',' << it -> second / ms -> second << ',' << it1->second / m -> second << '\n';
    }
    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << duration.count() << '\n';
    return 0;
}
