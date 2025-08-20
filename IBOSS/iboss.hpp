#include <string>
#include <vector>
#include <queue>
#include <utility>
#include <omp.h>
#include <chrono>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X, const VectorXd &y, int k) {

    const int p = X.cols();
    const int N = X.rows();
    const int r = k / (2 * (p - 1));  
    vector<bool> selected(N, false);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 1; j < p; ++j) {
        priority_queue<pair<double, int>, vector<pair<double, int>>, less<pair<double, int>>> maximals;
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> minimals;
        
        for (int i = 0; i < N; ++i) {
            // if (selected[i]) continue;
            double x_ij = X(i, j);
            
            if (maximals.size() < r) {
                maximals.emplace(x_ij, i);
            } 
            else if (x_ij < maximals.top().first) {
                maximals.pop();
                maximals.emplace(x_ij, i);
            }
            
            if (minimals.size() < r) {
                minimals.emplace(x_ij, i);
            } 
            else if (x_ij > minimals.top().first) {
                minimals.pop();
                minimals.emplace(x_ij, i);
            }
        }

        #pragma omp critical
        {
            while (!maximals.empty()) {
                selected[maximals.top().second] = 1;
                maximals.pop();
            }
            
            while (!minimals.empty()) {
                selected[minimals.top().second] = 1;
                minimals.pop();
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

