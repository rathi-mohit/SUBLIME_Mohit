#include <iostream>

#include <vector>
#include <string>
#include <iomanip>
#include <typeinfo>

#include <fstream>
#include <sstream>

#include <ctime>
#include <cmath>

#include <algorithm>
#include <stdexcept>
#include <set>

#include <Eigen/Dense>
using namespace Eigen;

using namespace std;

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

struct LinearRegressionModel()