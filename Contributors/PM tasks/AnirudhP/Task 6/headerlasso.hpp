#ifndef HEADERLASSO_HPP
#define HEADERLASSO_HPP
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void dividing_data_for_cv(const MatrixXd& X, const VectorXd& y, int k, vector<MatrixXd>& X_tra, vector<MatrixXd>& X_test, vector<VectorXd>& Y_tra, vector<VectorXd>& Y_test);

void coord_descent(const MatrixXd&X, double lambda, VectorXd& b, int k,const VectorXd& r, double ss);

VectorXd lasso(const MatrixXd&X, const VectorXd&y, double lambda, int maxiter, double tol, int miniter);

VectorXd lasso(const MatrixXd&X, const VectorXd&y, double lambda);

void kfold_cv_lasso(const MatrixXd& X_inp, const VectorXd& y, int k, double lb, double ub, double stepsize, int nochange, int maxiter, double tol, int miniter);

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k);

VectorXd kfold_cv_lassomain(const MatrixXd& X_inp, const VectorXd& y, int k, double lb, double ub, double stepsize, int nochange, int maxiter, double tol, int miniter);

VectorXd kfold_cvlasso(const MatrixXd& X, const VectorXd& y, int k);

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, double stepsize, int nochange, int maxiter, double tol, int miniter);

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, int n, int nochange, int maxiter, double tol, int miniter);

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, int n, double ub, int nochange, int maxiter, double tol, int miniter);

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, double stepsize, int n, int nochange, int maxiter, double tol, int miniter);



#endif