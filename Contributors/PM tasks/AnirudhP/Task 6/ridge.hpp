#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <ctime> 
#include <iomanip>
#include <cmath>
#include <fstream>
#include "regression_metrics.hpp"
#include <omp.h>
using namespace std;
using namespace Eigen;



//for internal use
namespace{

void dividing_data_for_cv(const MatrixXd& X, const VectorXd& y, int k, vector<MatrixXd>& X_tra, vector<MatrixXd>& X_test, vector<VectorXd>& Y_tra, vector<VectorXd>& Y_test)
{
    int si=X.rows()/k;

    for(int i=0; i<k; i++)
    {
        X_tra[i].resize((k-1)*si, X.cols());
        X_test[i].resize(si, X.cols());

        Y_tra[i].resize((k-1)*si);
        Y_test[i].resize(si);
        

        X_tra[i]<<X.middleRows(0,i*si), X.middleRows((i+1)*si,(k-1-i)*si);
        Y_tra[i]<<y.middleRows(0,i*si), y.middleRows((i+1)*si,(k-1-i)*si);

        X_test[i]=X.middleRows(si*i, si);
        Y_test[i]=y.middleRows(si*i, si);
    }

}

}

//for use by user

VectorXd ridge(const MatrixXd&X, const VectorXd&y, double lambda)
{
    if (X.rows() != y.size()) throw invalid_argument("Number of rows in X must match size of y.");
    if (lambda<0) throw invalid_argument("Lambda should be greater than 0.");
    VectorXd b(X.cols());
    MatrixXd A=MatrixXd::Identity(X.cols(), X.cols());
    A(0,0)=0; //this is to prevent regularizing the constant term b0
    b.noalias()=(X.transpose()*X+lambda*A).ldlt().solve((X.transpose())*y);
    return b;
}

void kfold_cv_ridge(const MatrixXd& X_inp, const VectorXd& y, int k, double lb, double ub, double stepsize, int nochange)
{
    if (X_inp.rows() != y.size()) throw invalid_argument("Number of rows in X must match size of y.");
    if (X_inp.rows() < k) throw invalid_argument("Number of rows must be at least equal to the number of folds k.");
    if (k == 0) throw std::invalid_argument("Number of folds should not be equal to 0.");
    if (lb >= ub) throw std::invalid_argument("Lower bound for lambda must be less than upper bound."); 
    if (stepsize <= 0) throw std::invalid_argument("Step size must be positive.");

    MatrixXd X=X_inp;
    VectorXd stddev_forrescaling(X.cols());
    for(int i=1; i<X.cols(); i++)
    {
        double x_mean=X.col(i).mean();
        X.col(i)=(X.col(i).array()-x_mean).matrix();
        double x_stddev=sqrt(X.col(i).dot(X.col(i))/(X.rows()));
        X.col(i)=X.col(i)/x_stddev;
        stddev_forrescaling(i)=x_stddev; //we store the stddev values for rescaling
    }

    double lambda=0;
    int s=(X.rows()/k); 
    double MSE;
    double MSE_min=numeric_limits<double>::max();;
    float ind;
    double l=pow(10, lb);
    double mult=pow(10,stepsize);
    vector<MatrixXd> X_tra(k);
    vector<VectorXd> Y_tra(k);
    vector<VectorXd> Y_test(k);
    vector<MatrixXd> X_test(k);

    dividing_data_for_cv(X,y,k,X_tra,X_test,Y_tra,Y_test);

    for(float i=lb; i<=ub; i=i+stepsize) //log scale for lambda
    {
       MSE=0;

       #pragma omp parallel for reduction(+:MSE)
       for(int j=0; j<k; j++)
       {
          VectorXd b(X.cols());
          b=ridge(X_tra[j], Y_tra[j], l); //training the model
          MSE+=(Y_test[j]-X_test[j]*b).squaredNorm()/s; //testing the model
       }
       MSE=MSE/k; //cross-validation error.

       //minimum cross validation error

       if(MSE<MSE_min)
       {
        MSE_min=MSE;
        ind=i;
        lambda=l;
       }
       //if the minimum does not change for a few consecutive cases,
       //we assume that our current minimum is the global minimum
       
       if(i-ind>=stepsize*nochange)
       {
         break;
       }
       l=l*mult;
    }

    cout<<"Lambda: "<<lambda<<endl;
    cout<<left<<setw(17)<<"S.No"<<setw(17)<<"Coefficient"<<setw(17)<<"t-statistic"<<endl;
    VectorXd b=ridge(X, y, lambda);

    for(int i=1; i<X.cols(); i++) //rescaling
    {
        b(i)=b(i)/stddev_forrescaling(i);
        b(0)-=X_inp.col(i).mean()*b(i);
    }


    double RSS=residual_sum_of_squares(y,X_inp*b);
    VectorXd tstat=t_statistics(X_inp,y,b);
    for(int i=0; i<b.rows(); i++)
    {
        cout<<left<<setw(17)<<(i+1)<<setw(17)<<b(i)<<setw(17)<<tstat(i)<<endl;
    }
    double fstat=f_statistic(X_inp,y,b);
    double Rsq=r_squared_score(y,X_inp*b);
    double RSE=residual_standard_error(X_inp,y,b);
    cout<<"F-statistic "<<fstat<<endl;
    cout<<"RSS "<<RSS<<endl;
    cout<<"RSE "<<RSE<<endl;
    cout<<"R^2 "<<Rsq<<endl;
}

void kfold_cv_ridge(const MatrixXd& X, const VectorXd& y, int k)
{
    kfold_cv_ridge(X,y,k,-4,10,0.01,7);
}

void kfold_cv_ridge(const MatrixXd& X, const VectorXd& y, int k, double lb, double ub, int n, int nochange)
{
    double stepsize=(ub-lb)/(n-1);
    kfold_cv_ridge(X,y,k,lb,ub,stepsize,nochange);
}

void kfold_cv_ridge(const MatrixXd& X, const VectorXd& y, int k, double lb, double stepsize, int n, int nochange)
{
    double ub=lb+stepsize*(n-1);
    kfold_cv_ridge(X,y,k,lb,ub,stepsize,nochange);
}
