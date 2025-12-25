#include <RcppEigen.h>
#include <vector>
#include <queue>
#include <utility>
#include <omp.h>
#include <algorithm>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

//' Selects Data Points Based on Extreme Feature Values
//' @param X A numeric matrix of predictors.
//' @param y A numeric vector of the response variable.
//' @param k An integer representing the target number of points.
//' @return A list containing the selected X matrix and y vector.
// [[Rcpp::export]]
Rcpp::List k_selection(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int k) {

    const int p = X.cols();
    const int N = X.rows();

    if (p <= 1) {
        Eigen::MatrixXd X_empty(0, p);
        Eigen::VectorXd y_empty(0);
        return Rcpp::List::create(Rcpp::Named("X_selected") = X_empty,
                                  Rcpp::Named("y_selected") = y_empty);
    }

    const int r = k / (2 * (p - 1));
    
    if (r == 0) {
        Eigen::MatrixXd X_empty(0, p);
        Eigen::VectorXd y_empty(0);
        return Rcpp::List::create(Rcpp::Named("X_selected") = X_empty,
                                  Rcpp::Named("y_selected") = y_empty);
    }
    
    std::vector<bool> selected(N, false);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 1; j < p; ++j) {
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::less<std::pair<double, int>>>  maximals;
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> minimals;
        
        for (int i = 0; i < N; ++i) {
            double x_ij = X(i, j);
            
            if (maximals.size() < r) {
                maximals.emplace(x_ij, i);
            } else if (x_ij < maximals.top().first) {
                maximals.pop();
                maximals.emplace(x_ij, i);
            }
            
            if (minimals.size() < r) {
                minimals.emplace(x_ij, i);
            } else if (x_ij > minimals.top().first) {
                minimals.pop();
                minimals.emplace(x_ij, i);
            }
        }

        #pragma omp critical
        {
            while (!maximals.empty()) {
                selected[maximals.top().second] = true;
                maximals.pop();
            }
            
            while (!minimals.empty()) {
                selected[minimals.top().second] = true;
                minimals.pop();
            }
        }
    }

    const int actual_k = std::count(selected.begin(), selected.end(), true);

    Eigen::MatrixXd X_iboss(actual_k, p);
    Eigen::VectorXd y_iboss(actual_k);
    int current_row = 0;

    for (int i = 0; i < N; ++i) {
        if (selected[i]) {
            X_iboss.row(current_row) = X.row(i);
            y_iboss(current_row) = y(i);
            ++current_row;
        }
    }

    return Rcpp::List::create(Rcpp::Named("X_selected") = X_iboss,
                              Rcpp::Named("y_selected") = y_iboss);
}
