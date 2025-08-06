#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <chrono>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class DOptimalitySelector
{
private:
    const MatrixXd& X;
    const VectorXd& y;
    int k;
    int p, r, N;
    bool has_intercept;

public:
    DOptimalitySelector(const MatrixXd& X_input, const VectorXd& y_input, int k_input, bool has_intercept)
        : X(X_input), y(y_input), k(k_input), has_intercept(has_intercept)
        {
        p = X.cols();
        N = X.rows();
        if (has_intercept)
        {
            r = k / (2 * (p - 1)); // Remove 1 for column containing 1
        }
        else {r = k / (2 * p);}
    }

    pair<MatrixXd, VectorXd> select()
    {
        vector<bool> selected(N, false);
        
        #pragma omp parallel for schedule(dynamic)
        for (int j = has_intercept ? 1 : 0; j < p; ++j) {
            vector<pair<double, int>> column_values;
            column_values.reserve(N);
            
            for (int i = 0; i < N; ++i) {
                column_values.emplace_back(X(i, j), i);
            }
            
            // Bottom r
            nth_element(column_values.begin(), column_values.begin() + r - 1, column_values.end());
                       
            // Top r
            nth_element(column_values.begin() + r, column_values.end() - r, column_values.end());

            #pragma omp critical
            {
                for (int i = 0; i < r; ++i)
                {
                    selected[column_values[i].second] = true;
                }
                
                for (int i = N - r; i < N; ++i)
                {
                    selected[column_values[i].second] = true;
                }
            }
        }

        int k_ = r * 2 * (p - 1);
        MatrixXd X_selected(k_, p);
        VectorXd y_selected(k_);
        size_t row = 0;

        for (size_t i = 0; i < N; ++i) {
            if (selected[i]) {
                X_selected.row(row) = X.row(i);
                y_selected(row) = y(i);
                ++row;
            }
        }

        return make_pair(X_selected, y_selected);
    }
};