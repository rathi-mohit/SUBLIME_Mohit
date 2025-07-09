#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include <unordered_map>
#include <functional>

#include <fstream>
#include <sstream>

#include <ctime>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
using namespace Eigen;

using namespace std;

VectorXd ridge(const MatrixXd & x, const VectorXd & y, double lambda)
{
    int p = x.cols(), n = x.rows();
    auto X = x.normalized();
    MatrixXd XT = X.transpose();
    // Identity matrix of size p*p
    MatrixXd I = MatrixXd::Identity(p, p);

    // Ridge regression using closed form
    VectorXd beta = (XT*X + lambda*I).inverse()*XT*y;

    // Intercept term will just be y_mean because we have standardized x
    double beta_0 = y.mean();

    // prediction vector
    auto yhat = X*beta + beta_0*VectorXd::Ones(n);

    // Metrics
    cout << "R2:" << calc_r2(y, yhat) << endl;
    cout << "Adjusted R2:" << calc_adj_r2(y, yhat, p) << endl;
    cout << "Cp:" << calc_cp(y, yhat, p) << endl;
    cout << "AIC:" << calc_aic(y, yhat, p) << endl;
    cout << "BIC:" << calc_bic(y, yhat, p) << endl;

    return beta;
}

VectorXd sgn(const VectorXd & v)
{
    VectorXd sgn_v = VectorXd::Ones(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        if (v[i] > 0)
        {
            sgn_v[i] = 1;
        }
        else if (v[i] < 0)
        {
            sgn_v[i] = -1;
        }
        else
        {
            sgn_v[i] = 0;
        }
    }
    return sgn_v;
}

VectorXd lasso(const MatrixXd & x, const VectorXd & y, double lambda, int iter, double soft_threshold)
{
    auto X = x.normalized();
    auto XT = X.transpose();

    int n = x.rows(), p = x.cols();
    // initialize a beta vector all zeroes
    VectorXd beta = VectorXd::Ones(p);
    for (int s = 0; s < iter; s++)
    {
        // Gradient matrix given by 1/p (X^T (y - X*beta)) + lambda*sgn(beta)
        VectorXd grad_beta = -1/(p + 0.0)*(XT*(y - X*beta)) + lambda*sgn(beta);
        // Set grad_beta to 0 for those features which have been set to 0
        for (int i = 0; i < p; i++)
        {
            if (beta[i] == 0)
            {
                grad_beta[i] = 0;
            }
        }
        beta -= grad_beta;

        // Soft thresholding
        for (int i = 0; i < p; i++)
        {
            if (abs(beta[i]) < soft_threshold)
            {
                beta[i] = 0;
            }
            else
            {
                beta[i] = (beta[i] > 0 ? 1:-1) * (abs(beta[i]) - soft_threshold);
            }
        }

        // Check if MSE < 100
        VectorXd yhat = X*beta;
        double mse = (y - yhat).squaredNorm()/(n + 0.0);
        if (mse < 100)
        {
            cout << "Stopped at iter " << s << " with MSE: " << mse << endl;
            break;
        }
    }
    // Intercept term will just be y_mean because we have standardized x
    double beta_0 = y.mean();
    // prediction vector
    auto yhat = X*beta + beta_0*VectorXd::Ones(n);
    // Metrics
    cout << "R2:" << calc_r2(y, yhat) << endl;
    cout << "Adjusted R2:" << calc_adj_r2(y, yhat, p) << endl;
    cout << "Cp:" << calc_cp(y, yhat, p) << endl;
    cout << "AIC:" << calc_aic(y, yhat, p) << endl;
    cout << "BIC:" << calc_bic(y, yhat, p) << endl;

    return beta;
}

double calc_adj_r2(const VectorXd & y, const VectorXd & yhat, const double & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double tss = (y.array() - y.mean()).square().sum();
    double adj_r2 = 1 - rss/(n-p-1)/(tss/(n-1));
    return adj_r2;
}

double calc_r2(const VectorXd& y, const VectorXd & yhat)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double tss = (y.array() - y.mean()).square().sum();
    double r2 = 1 - rss/tss;
    return r2;
}

double calc_cp(const VectorXd & y, const VectorXd & yhat, const int & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double varhat = (y.array() - y.mean()).square().sum()/(n - p - 1);
    double cp = (rss + 2*p*varhat)/n;
    return cp;
}

double calc_aic(const VectorXd & y, const VectorXd & yhat, const int & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double varhat = (y.array() - y.mean()).square().sum()/(n - p - 1);
    double aic = 2.0*p/n + rss/(n*varhat);
    return aic;
}

double calc_bic(const VectorXd & y, const VectorXd & yhat, const int & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double varhat = (y.array() - y.mean()).square().sum()/(n - p - 1);
    double bic = rss/n + log(n)*p*varhat;
    return bic;
}