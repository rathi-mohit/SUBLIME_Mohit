#pragma once

#include <Eigen/Dense>
using namespace Eigen;

VectorXd betaOLS(MatrixXd &X, VectorXd &y, bool intercept = true) {
    if (intercept) {
        return (X.transpose() * X).ldlt().solve(X.transpose() * y);
    }
    else {
        MatrixXd X_with_intercept(X.rows(), X.cols()+1);
        X_with_intercept.col(0) = VectorXd::Constant(X.rows(), 1);
        X_with_intercept.rightCols(X.cols()) = X;
        return betaOLS(X_with_intercept, y, true);
    }
}
VectorXd betaRidge(MatrixXd &X, VectorXd &y, double lambda, bool intercept = true) {
    if (intercept) {
        MatrixXd I = MatrixXd::Identity(X.cols(), X.cols()); I(0, 0) = 0;
        return (X.transpose() * X + lambda*I).ldlt().solve(X.transpose() * y);
    } 
    else {
        MatrixXd X_with_intercept(X.rows(), X.cols()+1);
        X_with_intercept.col(0) = VectorXd::Constant(X.rows(), 1);
        X_with_intercept.rightCols(X.cols()) = X;
        return betaRidge(X_with_intercept, y, lambda, true);
    }
}
VectorXd betaLasso(MatrixXd &X, VectorXd &y, double lambda, double res = 1e-4, int max_iter = 1000, bool intercept = true) {
    int N = X.rows();
    int p = X.cols();

    VectorXd beta = VectorXd::Constant(p, 0.0);
    VectorXd beta_prev = beta;

    for (int i = 0; i < max_iter; ++i) {
        beta_prev = beta;
        for (size_t j = 0; j < p; ++j) {
            VectorXd r_j = y - X * beta + (X.col(j) * beta(j));
            double z_j = X.col(j).squaredNorm() / N;
            double rho_j = (X.col(j).transpose() * r_j)(0) / N;
            double s = 0;
            if (j == 0) beta(j) = rho_j / z_j;
            else {
                if (rho_j > lambda) {
                    beta(j) = (rho_j - lambda) / z_j;
                } 
                else if (rho_j < -1.0*lambda) {
                    beta(j) = (rho_j + lambda) / z_j;
                } 
                else {
                    beta(j) = 0.0;
                }
            }
        }
        if ((beta - beta_prev).squaredNorm() < res) break;
    }

    return beta;
}