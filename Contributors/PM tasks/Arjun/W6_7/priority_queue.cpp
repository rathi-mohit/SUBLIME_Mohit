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
        : X(X_input), y(y_input), k(k_input)
    {
        p = X.cols();
        N = X.rows();
        if (has_intercept)
        {
            r = k / (2*(p-1)); // Remove 1 for column containing 1
        }
        else { r = k / (2*p); }
    }

    pair<MatrixXd, VectorXd> select()
    {
        vector<bool> selected(N, false);

        #pragma omp parallel for schedule(dynamic)
        for (int j = 1; j < p; ++j)
        {
            priority_queue<pair<double, int>, vector<pair<double, int>>, less<pair<double, int>>> minimals;
            priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> maximals;
            
            for (int i = 0; i < N; ++i) {
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
                    selected[minimals.top().second] = true;
                    minimals.pop();
                }
                
                while (!maximals.empty()) {
                    selected[maximals.top().second] = true;
                    maximals.pop();
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