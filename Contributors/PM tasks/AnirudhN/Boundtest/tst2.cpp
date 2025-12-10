#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <set>
#include <Eigen/Dense>

// Use high precision for log calculations
using namespace Eigen;
using namespace std;

// Function to calculate the Log of the Upper Bound from Theorem 2 
// Formula: log( (k^(p+1)) / (4^p) * Product(Range_j^2) )
double calculate_log_theoretical_bound(int k, int p, const VectorXd& ranges) {
    // Term 1: (p+1) * ln(k)
    double term1 = (p + 1) * std::log(k);

    // Term 2: p * ln(4)
    double term2 = p * std::log(4.0);

    // Term 3: Sum of 2 * ln(Range_j)
    double term3 = 0.0;
    for (int i = 0; i < ranges.size(); ++i) {
        term3 += 2.0 * std::log(ranges[i]);
    }

    return term1 - term2 + term3;
}

// Function to calculate Log Determinant of a symmetric positive definite matrix
// using Cholesky decomposition (LLT) to prevent overflow.
double calculate_log_det(const MatrixXd& M) {
    LLT<MatrixXd> llt(M);
    if (llt.info() == Eigen::NumericalIssue) {
        throw runtime_error("Matrix is not positive definite!");
    }
    // Fix: Use matrixLLT() to access the underlying matrix diagonal
    return 2.0 * llt.matrixLLT().diagonal().array().log().sum();
}

// Algorithm 1: IBOSS Selection 
// Selects k indices from the full dataset Z
vector<int> iboss_selection(const MatrixXd& Z, int k) {
    int n = Z.rows();
    int p = Z.cols();
    int r = k / (2 * p); // r data points from each tail [cite: 184]

    vector<int> available_indices(n);
    // Initialize indices 0 to n-1
    iota(available_indices.begin(), available_indices.end(), 0);

    // Boolean mask to track selected status for O(1) checking
    vector<bool> is_selected(n, false);
    vector<int> selected_indices;
    selected_indices.reserve(k);

    for (int j = 0; j < p; ++j) {
        // Filter available indices (remove those already selected)
        // Note: In a highly optimized version, we would partition in place.
        // Here we gather remaining indices for clarity.
        vector<int> current_pool;
        current_pool.reserve(n - selected_indices.size());
        for(int idx : available_indices) {
             if(!is_selected[idx]) current_pool.push_back(idx);
        }

        if (current_pool.size() < 2 * r) {
            // Fallback if insufficient data
            for(int idx : current_pool) {
                if(!is_selected[idx]) {
                    selected_indices.push_back(idx);
                    is_selected[idx] = true;
                }
            }
            break;
        }

        // Sort current pool based on values in column j of Z
        // Using std::sort (O(N log N)) for simplicity. 
        // For strict O(N) performance per paper, use std::nth_element[cite: 189].
        std::sort(current_pool.begin(), current_pool.end(), 
            [&Z, j](int a, int b) {
                return Z(a, j) < Z(b, j);
            }
        );

        // Select r smallest
        for (int i = 0; i < r; ++i) {
            int idx = current_pool[i];
            selected_indices.push_back(idx);
            is_selected[idx] = true;
        }

        // Select r largest
        for (int i = 0; i < r; ++i) {
            int idx = current_pool[current_pool.size() - 1 - i];
            selected_indices.push_back(idx);
            is_selected[idx] = true;
        }
    }
    
    return selected_indices;
}

int main() {
    // Simulation Parameters
    const int p = 5;          // Number of covariates
    const int k = 1000;       // Subdata size (fixed)
    vector<int> n_sizes = {10000, 50000, 100000, 500000}; // Different Full Data Set sizes

    // Random Number Generation (Standard Normal Case 1) [cite: 316]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    cout << left << setw(15) << "N (Full Data)" 
         << setw(20) << "Log Det (IBOSS)" 
         << setw(20) << "Log Bound (Thm 2)" 
         << "Ratio (M / Bound)" << endl;
    cout << string(85, '-') << endl;

    for (int n : n_sizes) {
        // 1. Generate Synthetic Data
        MatrixXd Z(n, p);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) {
                Z(i, j) = dist(gen);
            }
        }

        // 2. Calculate Ranges for Theorem 2 Bound (Max - Min per column)
        VectorXd ranges(p);
        for (int j = 0; j < p; ++j) {
            double min_val = Z.col(j).minCoeff();
            double max_val = Z.col(j).maxCoeff();
            ranges(j) = max_val - min_val;
        }

        // 3. Calculate Theoretical Upper Bound (Log Scale)
        double log_bound = calculate_log_theoretical_bound(k, p, ranges);

        // 4. Run IBOSS Algorithm
        vector<int> idx_iboss = iboss_selection(Z, k);

        // 5. Construct Subdata Design Matrix X* (Add intercept column)
        // X* has dimensions k x (p+1)
        MatrixXd X_star(idx_iboss.size(), p + 1);
        for (size_t i = 0; i < idx_iboss.size(); ++i) {
            X_star(i, 0) = 1.0; // Intercept term
            for (int j = 0; j < p; ++j) {
                X_star(i, j + 1) = Z(idx_iboss[i], j);
            }
        }

        // 6. Calculate Information Matrix: M = (X*)^T * X*
        MatrixXd M = X_star.transpose() * X_star;

        // 7. Calculate Log Determinant
        double log_det_iboss = calculate_log_det(M);

        // 8. Calculate Ratio: exp(log_det - log_bound)
        double ratio = std::exp(log_det_iboss - log_bound);

        cout << left << setw(15) << n 
             << setw(20) << fixed << setprecision(4) << log_det_iboss 
             << setw(20) << log_bound 
             << setprecision(6) << ratio << endl;
    }

    return 0;
}
