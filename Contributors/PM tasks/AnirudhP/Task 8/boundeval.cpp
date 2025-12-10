#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "iboss.hpp"
#include "LinearRegression.hpp"
#include "regression_metrics.hpp"

using namespace std;
using namespace Eigen;

double bound(MatrixXd X, int k)
{
    int p=X.cols()-1;
    double sum=(p+1)*log(k)-p*log(4.0);

    for(int i=1; i<p+1; i++)
    {
        auto temp=(X.col(i).maxCoeff()-X.col(i).minCoeff());
        sum+=log(temp*temp);
    }

    return sum;
}


int main()
{
    //things that could affect bound:
    //size of the data: number of rows, number of columns
    //number of rows subsampled
    //variance of error, variance in data.


    random_device rd;
    mt19937 mt(rd());
    
    double mean_data=0;
    double stddev_data=2;

    normal_distribution<double> gen_data(mean_data, stddev_data);

    double mean_par=0;
    double stddev_par=5;

    normal_distribution<double> gen_par(mean_par, stddev_par);

    double mean_err=0;
    double stddev_err=1;

    normal_distribution<double> gen_err(mean_err, stddev_err);

    int rows=10000;
    int cols=11;
    MatrixXd X(rows, cols);
    VectorXd b(cols);
    VectorXd err(rows);
    double s=0;
    int k=(int)(pow(2,10));

    for(int h=0; h<100; h++)
    {
    for(int i=0; i<rows; i++)
    {
        X(i,0)=1;
        for(int j=1; j<cols; j++)
        {
            X(i,j)=gen_data(mt);
        }

        err(i)=gen_err(mt);
    }

    for(int j=0; j<cols; j++)
    {
        b(j)=gen_par(mt);
    }

    VectorXd y=X*b+err;

    pair<MatrixXd, VectorXd> IBOSS_result=k_selection(X,y,k);
    MatrixXd X_IBOSS=IBOSS_result.first;
    VectorXd y_IBOSS=IBOSS_result.second;

    LLT<MatrixXd> llt(X_IBOSS.transpose()*X_IBOSS);
    int u=llt.matrixL().rows();
    double sum=0;
    for(int i=0; i<u; i++)
    {
        sum+=log(llt.matrixL()(i,i));
    }
    double det=sum*2;
    s+=exp(det-bound(X,k));
   }
    s=s/100;
    cout<<s;
}