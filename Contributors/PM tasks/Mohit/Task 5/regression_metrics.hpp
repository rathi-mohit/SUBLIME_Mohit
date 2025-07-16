#include <Eigen/Dense>
using namespace Eigen;

double RSS(const MatrixXd &X, const VectorXd &y) {
    VectorXd beta = (X.transpose() * X).inverse() * (X.transpose() * y);
    VectorXd residuals = y - X * beta;
    return residuals.squaredNorm();
}
double RSS(const VectorXd &y, const VectorXd &y_pred) {
    VectorXd residuals = y - y_pred;
    return residuals.squaredNorm();
}
double MSE(const VectorXd &y_true, const VectorXd &y_pred) {
    double sum_sq_error = 0.0;
    int n = y_true.rows();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double err = y_true(i) - y_pred(i);
        sum_sq_error += err * err;
    }
    return sum_sq_error / n;
}
double TSS(const VectorXd &y) {
    VectorXd y_mean_vec = VectorXd::Constant(y.rows(), y.mean());
    return (y - y_mean_vec).squaredNorm();
}
double R2_score(const VectorXd &y, const VectorXd &y_pred) {
    double rss = RSS(y, y_pred);
    double tss = TSS(y);
    return 1 - (rss / tss);
}
double R2_score(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    VectorXd y_pred = X * beta;
    return R2_score(y, y_pred);
}
double adj_R2_score(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows(), p = X.cols();
    double rss = RSS(X, y);
    double tss = TSS(y);
    return 1 - ((rss / (N - p)) / (tss / (N - 1)));
}
double RSE(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    int N = X.rows(), p = X.cols() - 1;
    VectorXd y_pred = X * beta;
    VectorXd e = y - y_pred;
    double rss = e.squaredNorm();
    return sqrt(rss / (N - p - 1));
}
VectorXd t_stat(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    VectorXd y_pred = X * beta;
    VectorXd residuals = y - y_pred;
    double rss = residuals.squaredNorm();
    int N = X.rows(), p = X.cols() - 1;
    double sigma2_hat = rss / (N - p - 1);
    MatrixXd XtX_inv = (X.transpose() * X).inverse();
    MatrixXd cov_beta = sigma2_hat * XtX_inv;
    VectorXd t_stats(p + 1);
    for (int j = 0; j <= p; ++j) {
        double se = sqrt(cov_beta(j, j));
        t_stats(j) = beta(j) / se;
    }
    return t_stats;
}
double AIC(const MatrixXd &X, const VectorXd &y) {
    int N = X.rows();
    int p = X.cols();
    VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + 2 * p;
}
double AIC(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows();
    int p = X.cols();
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + 2 * p;
}
double BIC(const MatrixXd &X, const VectorXd &y) {
    int N = X.rows();
    int p = X.cols();
    VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + p * log(N);
}