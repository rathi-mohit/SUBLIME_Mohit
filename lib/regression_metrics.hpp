#ifndef REG_METRICS_HPP
#define REG_METRICS_HPP

#include <Eigen/Dense>
using namespace Eigen;

/**
 * @brief Returns the total sum of squares (TSS) for the response variable 'y'.
 * @param y Eigen::VectorXd - Vector of observed values of response variable (n × 1).
 */
inline double total_sum_of_squares(const VectorXd &y) {
    VectorXd y_mean_vec = VectorXd::Constant(y.rows(), y.mean());
    return (y - y_mean_vec).squaredNorm();
}

/**
 * @brief Returns the sum of squares of residuals (RSS) between the actual and predcited values of response variable 'y'.
 * @param y_actual Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param y_pred Eigen::VectorXd - Vector of predicted values of response variable (n × 1).
 */
inline double residual_sum_of_squares(const VectorXd &y_actual, const VectorXd &y_pred) {
    VectorXd residuals = y_actual - y_pred;
    return residuals.squaredNorm();
}

/**
 * @brief Returns the mean squared error (MSE) between the actual and predicted values of response variable 'y'.
 * @param y_actual Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param y_pred Eigen::VectorXd - Vector of predicted values of response variable (n × 1).
 */
inline double mean_squared_error(const VectorXd &y_actual, const VectorXd &y_pred) {
    double sum_sq_error = 0.0;
    int n = y_actual.rows();

    #pragma omp parallel for reduction(+:sum_sq_error)
    for (int i = 0; i < n; ++i) {
        double err = y_actual(i) - y_pred(i);
        sum_sq_error += err * err;
    }
    return sum_sq_error / n;
}

/**
 * @brief Returns the coefficient of determination (R²) for the predicted values of 'y'.
 * @param y Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param y_pred Eigen::VectorXd - Vector of predicted values of response variable (n × 1).
 */
inline double r_squared_score(const VectorXd &y, const VectorXd &y_pred) {
    double rss = residual_sum_of_squares(y, y_pred);
    double tss = total_sum_of_squares(y);
    return 1 - (rss / tss);
}

/**
 * @brief Returns the adjusted R² score for the linear model with given predictors and coefficients.
 * @param X Eigen::MatrixXd - Design matrix (n × p).
 * @param y Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param beta Eigen::VectorXd - Estimated coefficient vector (p × 1).
 */
inline double adjusted_r_squared_score(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows();
    int p = X.cols();

    double rss = residual_sum_of_squares(y, X * beta);
    double tss = total_sum_of_squares(y);
    return 1 - ((rss / (N - p)) / (tss / (N - 1)));
}

/**
 * @brief Returns the residual standard error (RSE) of the fitted linear model.
 * @param X Eigen::MatrixXd - Design matrix (n × (p + 1)), including intercept if present.
 * @param y Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param beta Eigen::VectorXd - Estimated coefficient vector ((p + 1) × 1).
 */
inline double residual_standard_error(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows();
    int p = X.cols() - 1;

    double rss = residual_sum_of_squares(y, X * beta);
    return sqrt(rss / (N - p - 1));
}

/**
 * @brief Returns the t-statistics for each coefficient in the fitted linear model.
 * @param X Eigen::MatrixXd - Design matrix (n × (p + 1)), including intercept.
 * @param y_actual Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param beta Eigen::VectorXd - Estimated coefficient vector ((p + 1) × 1).
 */
VectorXd t_statistics(const MatrixXd &X, const VectorXd &y_actual, const VectorXd &beta) {
    int N = X.rows();
    size_t p = X.cols() - 1;

    double sigma2_hat = residual_sum_of_squares(y_actual, X * beta) / (N - p - 1);
    MatrixXd cov_beta = sigma2_hat * (X.transpose() * X).inverse();

    VectorXd t_stats(p + 1);
    for (int j = 0; j <= p; ++j) {
        double se = sqrt(cov_beta(j, j));
        t_stats(j) = beta(j) / se;
    }
    return t_stats;
}

double f_statistic(const MatrixXd &X, const VectorXd &y_actual, const VectorXd &beta) {
    int N = X.rows();
    int p = X.cols() - 1;

    double RSS = residual_sum_of_squares(y_actual, X * beta);
    double numerator = (total_sum_of_squares(y_actual) - RSS) / p;
    double denominator = RSS / (N - p - 1);

    return numerator / denominator;
}

double mallows_cp(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows();
    double RSS = residual_sum_of_squares(y, X * beta);

    VectorXd beta_OLS = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    double sigma_square_full_model = residual_standard_error(X, y, beta_OLS);
    int d = beta.rows() - 1;

    return (RSS + 2*d*sigma_square_full_model) / N;
}
/**
 * @brief Returns the Akaike Information Criterion (AIC) for the fitted linear model.
 * @param X Eigen::MatrixXd - Design matrix (n × p).
 * @param y Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 * @param beta Eigen::VectorXd - Estimated coefficient vector (p × 1).
 */
inline double aic_score(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows();
    int p = X.cols();
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + 2 * p;
}

/**
 * @brief Returns the Bayesian Information Criterion (BIC) for the fitted linear model.
 * @param X Eigen::MatrixXd - Design matrix (n × p).
 * @param y Eigen::VectorXd - Vector of actual values of response variable (n × 1).
 */
inline double bic_score(const MatrixXd &X, const VectorXd &y) {
    int N = X.rows();
    int p = X.cols();
    VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + p * log(N);
}

#endif
