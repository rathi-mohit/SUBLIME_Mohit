#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <utility>
#include <omp.h>
#include <chrono>
#include <Eigen/Dense>

#include "fast_float/fast_float.h"
#include "regression_metrics.hpp"
#include "regression_models.hpp"
#include "utils.hpp"

// #include <random>
// #include <map>

using namespace Eigen;
using namespace std;

pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X_given, const VectorXd &y_given, const int k);

int main() {

    // for compilation: g++ zselk.cpp -fopenmp -o2 -march=native 
    // link to fast_float: https://github.com/fastfloat/fast_float (20x faster than stod)

    auto start = chrono::high_resolution_clock::now();
    string filename; cin >> filename;

    auto numerical_data = clean_numerical_load_csv(filename); 

    
    MatrixXd X(numerical_data.rows(), numerical_data.cols());
    VectorXd y(numerical_data.rows());
    X.col(0) = VectorXd::Constant(X.rows(), 1.0);   
    X.rightCols(X.cols() - 1) = numerical_data.rightCols(X.cols() - 1);
    y = numerical_data.col(0);  

    
    auto X_and_y = k_selection(X, y, 50000);
    VectorXd beta_ols = betaOLS(X, y);
    VectorXd beta_k_selected = betaOLS(X_and_y.first, X_and_y.second);

    cout << "Normal OLS:" << '\n';
    cout << beta_ols << '\n';
    cout << "R2 Score for OLS: " << R2_score(X, y, beta_ols) << "\n\n\n";

    cout << "K-selected:" << '\n';
    cout << beta_k_selected << '\n';
    cout << "R2 Score for OLS: " << R2_score(X, y, beta_k_selected) << "\n\n\n"; 


    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time Taken: " << duration.count() << '\n';
    return 0;
}
pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X, const VectorXd &y, int k) {
    const int p = X.cols();
    const int N = X.rows();
    const int r = k / (2 * (p - 1));  
    vector<bool> selected(N, false);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 1; j < p; ++j) {
        priority_queue<pair<double, int>, vector<pair<double, int>>, less<pair<double, int>>> minimals;
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> maximals;
        
        for (int i = 0; i < N; ++i) {
            // if (selected[i]) continue;
            double x_ij = X(i, j);
            
            if (minimals.size() < r) {
                minimals.emplace(x_ij, i);
            } 
            else if (x_ij < minimals.top().first) {
                minimals.pop();
                minimals.emplace(x_ij, i);
            }
            
            if (maximals.size() < r) {
                maximals.emplace(x_ij, i);
            } 
            else if (x_ij > maximals.top().first) {
                maximals.pop();
                maximals.emplace(x_ij, i);
            }
        }

        #pragma omp critical
        {
            while (!minimals.empty()) {
                selected[minimals.top().second] = 1;
                minimals.pop();
            }
            
            while (!maximals.empty()) {
                selected[maximals.top().second] = 1;
                maximals.pop();
            }
        }
    }

    k = r * 2 * (p - 1);

    MatrixXd X_iboss(k, p);
    VectorXd y_iboss(k);
    size_t row = 0;

    for (size_t i = 0; i < N; ++i) {
        if (selected[i]) {
            X_iboss.row(row) = X.row(i);
            y_iboss(row) = y(i);
            ++row;
        }
    }

    return make_pair(X_iboss, y_iboss);
}

// **Initial idea**

// pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X_given, const VectorXd &y_given, const int k) {
//     int p = X_given.cols();
//     int N = X_given.rows();
//     int r = k / (2 * (p - 1));
//     vector<bool> selected(N, false);

//     for (int i = 1; i < p; ++i) {
//         map<double, size_t> ordered_z;
//         VectorXd z_i = X_given.col(i);
//         for (size_t j = 0; j < z_i.rows(); ++j) ordered_z[z_i(j)] = j;

//         int count = 0;
//         for (auto it = ordered_z.begin(); it != ordered_z.end(); ++it) {
//             if (!selected[it -> second]) {
//                 selected[it -> second] = 1;
//                 if (++count == r) break;
//             }
//         }

//         count = 0;
//         for (auto it = --ordered_z.end(); it != ordered_z.begin(); --it) {
//             if (!selected[it -> second]) {
//                 selected[it -> second] = 1;
//                 if (++count == r) break;
//             }
//         }
//     }
    
//     MatrixXd X_ans(selected.size(), X_given.cols());
//     VectorXd y_ans(selected.size());
//     size_t row_idx = 0;
//     for (size_t i = 0; i < N; ++i) {
//         if (selected[i]) {
//             X_ans.row(row_idx) = X_given.row(i);
//             y_ans(row_idx) = y_given(i);
//             ++row_idx;
//         }
//     }
//     return make_pair(X_ans, y_ans);
// }