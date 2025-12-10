#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "LinearRegression.hpp"
#include <cmath>
#include <random>
#include <bit>

using namespace std;
using namespace Eigen;

void multiply(Ref<VectorXd> y)
{
    int n=y.rows();
    for(int h=1; h<n; h=h*2)
    {
        for(int i=0; i<n; i+=2*h)
        {
            for(int j=i; j<i+h; j++)
            {
                double s=y(j);
                double t=y(j+h);
                y(j)=s+t;
                y(j+h)=s-t;
            }
        }
    }
}

void padwithzeroes(MatrixXd& X, VectorXd& y)
{
    int n=X.rows();
    int q=(int)pow(2,ceil(log2(n)));

    if(q==n)
    {
        return;
    }
    
    else
    {
        X.conservativeResize(q,X.cols());
        y.conservativeResize(q);

        X.bottomRows(q-n).setZero();
        y.tail(q-n).setZero();
    }
}

VectorXd SRHT(const MatrixXd& A, const VectorXd& b)
{
    //I do not multiply by the constant factor sqrt(n/r) since it gets cancelled in the end.
    int n=A.rows();
    int d=A.cols();
    unsigned int n_padded=bit_ceil((unsigned)n);
    MatrixXd X=MatrixXd::Zero(n_padded,d); //requires c++20
    VectorXd y=VectorXd::Zero(n_padded);
    y.head(n)=b;
    X.block(0,0,n,d)=A;
    //int r = ceil(max(48*48*d*log(40*n*d)*log(100*100*d*log(40*n*d)),40*d*log(40*n*d)/e));
    int r=(int)(n/10);

    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<> dist(0, X.rows()-1);
    vector<int> S(r);

    for (int i=0; i<r; ++i)
    {
        S[i]=dist(mt);
    }

    uniform_int_distribution<> diag(0, 1);

    for (int i=0; i<n_padded; ++i)
    {
        auto temp=(2*diag(mt)-1);
        X.row(i)=X.row(i)*temp;
        y.row(i)=y.row(i)*temp;
    }
    
    #pragma omp parallel for
    for(int i=0; i<d; i++)
    {
        multiply(X.col(i));
    }
    multiply(y);

    MatrixXd X_f(r,d);
    VectorXd y_f(r);

    for(int i=0; i<r; i++)
    {
        X_f.row(i)=X.row(S[i]);
        y_f(i)=y(S[i]);
    }
    
    return LinearRegression(X_f,y_f);
}
