#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

/**
 * Most recent changes:
 * vector<char> instead of vector<bool> on l25 to get rid of the critical section and hence cause speed-up
 * use vectors isntead of heaps and then make_heap, push_heap, pop_heap 
 * loop unrolling for initial filling of maximals and minimals to avoid N*p comparisions
 * Parallelized the outer loop over features
 * Parallelized the final copying of selected rows
 * 
 * minor speed-up occurs (0.43s instead of 0.67s on normal_data for example)
 *  */ 


pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X, const VectorXd &y, size_t k) {

    const size_t p = X.cols();
    const size_t N = X.rows();
    
    if (k > N) k = N;
    const size_t r = (p > 1) ? (k / (2 * (p - 1))) : (k / 2);
    vector<char> selected(N, 0);

    #pragma omp parallel 
    {
        vector<pair<double, size_t>> maximals; maximals.reserve(r + 1);
        vector<pair<double, size_t>> minimals; minimals.reserve(r + 1);

        #pragma omp for schedule(dynamic)
        for (size_t j = 1; j < p; ++j) {
            
            maximals.clear();
            minimals.clear();

            for (size_t i = 0; i < r; ++i) {
                double val = X(i, j);
                maximals.emplace_back(val, i);
                minimals.emplace_back(val, i);
            }

            make_heap(maximals.begin(), maximals.end(), greater<pair<double, size_t>>());
            make_heap(minimals.begin(), minimals.end());

            for (size_t i = r; i < N; ++i) {
                double val = X(i, j);

                if (val > maximals.front().first) {
                    pop_heap(maximals.begin(), maximals.end(), greater<pair<double, size_t>>());
                    maximals.pop_back(); 
                    maximals.emplace_back(val, i);
                    push_heap(maximals.begin(), maximals.end(), greater<pair<double, size_t>>());
                }

                if (val < minimals.front().first) {
                    pop_heap(minimals.begin(), minimals.end());
                    minimals.pop_back(); 
                    minimals.emplace_back(val, i);
                    push_heap(minimals.begin(), minimals.end());
                }
            }

            for (const auto& p : maximals) selected[p.second] = 1;
            for (const auto& p : minimals) selected[p.second] = 1;
        }
    }
    
    vector<size_t> write_index(N);
    size_t actual_k = 0;
    
    for(size_t i = 0; i < N; ++i) {
        if(selected[i]) {
            write_index[i] = actual_k++;
        }
    }

    MatrixXd X_iboss(actual_k, p);
    VectorXd y_iboss(actual_k);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        if (selected[i]) {
            size_t row_idx = write_index[i];
            X_iboss.row(row_idx) = X.row(i);
            y_iboss(row_idx) = y(i);
        }
    }

    return make_pair(move(X_iboss), move(y_iboss));
}
