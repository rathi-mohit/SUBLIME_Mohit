#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "iboss.hpp"
#include "LinearRegression.hpp"
#include "regression_metrics.hpp"

using namespace std;
using namespace Eigen;


int main()
{
    random_device rd;
    mt19937 mt(rd());
    
    double mean_data=0;
    double stddev_data=2;

    normal_distribution<double> gen_data(mean_data, stddev_data);

    double mean_par=0;
    double stddev_par=10;

    normal_distribution<double> gen_par(mean_par, stddev_par);

    double mean_err=0;
    double stddev_err=20;

    normal_distribution<double> gen_err(mean_err, stddev_err);

    int rows=10000;
    int cols=11;
    MatrixXd X(rows, cols);
    VectorXd b(cols);
    VectorXd err(rows);
    double s=0;
    double r=0;
    double t=0;
    double l=0;
    int k=100;

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

    LLT<MatrixXd> llt(X.transpose()*X);
    LLT<MatrixXd> llt_IBOSS(X_IBOSS.transpose()*X_IBOSS);
    int u=llt_IBOSS.matrixL().rows();
    double sum=0;
    double sum1=0;
    for(int i=0; i<u; i++)
    {
        sum+=log(llt_IBOSS.matrixL()(i,i));
        sum1+=log(llt.matrixL()(i,i));
    }
    double det_IBOSS=sum*2;
    double det=sum1*2;
    s+=exp(det_IBOSS);
    t+=exp(det);
    l+=r_squared_score(y,X*LinearRegression(X,y));
    r+=r_squared_score(y,X*LinearRegression(X_IBOSS, y_IBOSS));
   }
    s=(s/100);
    t=(t/100);
    r=r/100;
    l=l/100;
    cout<<log(t)-2*(cols)*log(stddev_data)<<endl;
    cout<<log(s)-2*(cols)*log(stddev_data)<<endl;
    cout<<l<<endl;
    cout<<r<<endl;
}