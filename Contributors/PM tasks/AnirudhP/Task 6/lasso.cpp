#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <ctime> 
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <omp.h>
#include "regression_metrics.hpp"
#include "headerlasso.hpp"
using namespace std;
using namespace Eigen;


void dividing_data_for_cv(const MatrixXd& X, const VectorXd& y, int k, vector<MatrixXd>& X_tra, vector<MatrixXd>& X_test, vector<VectorXd>& Y_tra, vector<VectorXd>& Y_test)
{
    int si=X.rows()/k;
    vector<int> nums(X.rows());
    iota(nums.begin(), nums.end(), 0);

    random_device rd;
    mt19937 mt(rd());
    shuffle(nums.begin(), nums.end(),mt);

    int count=0;
    for(int i=0; i<k; i++)
    {
        X_tra[i].resize((k-1)*si, X.cols());
        X_test[i].resize(si, X.cols());

        Y_tra[i].resize((k-1)*si);
        Y_test[i].resize(si);

        count=i*si;
        for(int j=0; j<(k-1)*si; j++)
        {
            X_tra[i].row(j)=X.row(nums[count%X.rows()]);
            Y_tra[i](j)=y(nums[count%X.rows()]);
            count++;
        }

        for(int j=0; j<si; j++)
        {
            X_test[i].row(j)=X.row(nums[count%X.rows()]);
            Y_test[i](j)=y(nums[count%X.rows()]);
            count++;
        }
    }

    for(int i=0; i<k; i++)
    {
        for(int j=1; j<X.cols(); j++)
        {
            X_tra[i].col(j)=(X_tra[i].col(j).array()-X_tra[i].col(j).mean()).matrix();
            X_tra[i].col(j)=X_tra[i].col(j)/X_tra[i].col(j).norm();
        }

        for(int j=1; j<X.cols(); j++)
        {
            X_test[i].col(j)=(X_test[i].col(j).array()-X_test[i].col(j).mean()).matrix();
            X_test[i].col(j)=X_test[i].col(j)/X_test[i].col(j).norm();
        } 
    }
}

void coord_descent(const MatrixXd&X, double lambda, VectorXd& b, int k,const VectorXd& r, double ss)
{
    double m=X.col(k).dot(r)+ss*b(k);

    if(m>lambda)
    {
       b(k)=(m-lambda)/ss;
    }
    else if(m<-lambda)
    {
       b(k)=(m+lambda)/ss;
    }
    else
    {
        b(k)=0;
    }
}


VectorXd lasso(const MatrixXd&X, const VectorXd&y, double lambda, int maxiter, double tol, int miniter)
{
    if (X.rows() != y.size()) throw invalid_argument("Number of rows in X must match size of y.");
    if (lambda<0) throw invalid_argument("Lambda should be greater than 0.");
    
    VectorXd b=VectorXd::Zero(X.cols());
    VectorXd b_old=VectorXd::Ones(X.cols());
    b(0)=y.mean();
    b_old(0)=b(0);
    int f=0;

    VectorXd r=y-X*b;

    //storing all column norms (squared) in a vector so that we don't calculate it each time.
    VectorXd ss=X.colwise().squaredNorm(); 
    while(((b-b_old).lpNorm<Infinity>()/max(b.lpNorm<Infinity>(),1.0)>tol&&f<maxiter)||(f<miniter))
    {
        b_old=b;
        for(int i=1; i<X.cols(); i++)
        {
            coord_descent(X, lambda, b, i, r, ss(i)); 
            //r=y-Xb, b has changed, so we update r to reflect the change
            r-=(b(i)-b_old(i))*X.col(i);
        }
        f=f+1;
    }
    return b;
}

VectorXd lasso(const MatrixXd&X, const VectorXd&y, double lambda)
{
    return lasso(X,y,lambda,200,0.0001,30);
}


VectorXd kfold_cv_lassomain(const MatrixXd& X_inp, const VectorXd& y, int k, double lb, double ub, double stepsize, int nochange, int maxiter, double tol, int miniter)
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
        double x_stddev=sqrt(X.col(i).dot(X.col(i))/(X.rows()-1));
        if (x_stddev != 0) 
        {
            X.col(i)=X.col(i)/x_stddev;
        }
        stddev_forrescaling(i)=x_stddev; //we store the stddev values for rescaling
    }

    double lambda=0;
    int s=(X.rows()/k); 
    double MSE;
    double MSE_min=numeric_limits<double>::max();
    //float ind;
    double l=pow(10, lb);
    double mult=pow(10,stepsize);
    vector<MatrixXd> X_tra(k);
    vector<VectorXd> Y_tra(k);
    vector<VectorXd> Y_test(k);
    vector<MatrixXd> X_test(k);

    VectorXd b(X.cols());
    dividing_data_for_cv(X,y,k,X_tra,X_test,Y_tra,Y_test);

    for(float i=lb; i<=ub; i=i+stepsize) //log scale for lambda
    {
       MSE=0;
       #pragma omp parallel for reduction(+:MSE)
       for(int j=0; j<k; j++)
       {
          b=lasso(X_tra[j], Y_tra[j], l, maxiter, tol, miniter); //training the model
          MSE+=(Y_test[j]-X_test[j]*b).squaredNorm()/s; //testing the model
       }
       MSE=MSE/k; //cross-validation error.

       //minimum cross validation error

       if(MSE<MSE_min)
       {
        MSE_min=MSE;
        //ind=i;
        lambda=l;
       }
       //if the minimum does not change for a few consecutive cases,
       //we assume that our current minimum is the global minimum
       
       /*
       if(i-ind>=stepsize*nochange)
       {
         break;
       }
       */
       l=l*mult;
    }

    for(int i=1; i<X.cols(); i++) //rescaling
    {
        if(stddev_forrescaling(i)!=0)
        {
           b(i)=b(i)/stddev_forrescaling(i);
        }
        b(0)-=X_inp.col(i).mean()*b(i);
    }

    b=lasso(X, y, lambda, maxiter, tol, miniter);
    return b;
}

VectorXd kfold_cvlasso(const MatrixXd& X, const VectorXd& y, int k)
{
    double ub=log10(((X.transpose()*y).lpNorm<Infinity>())); //optimal ub.
    double lb=0.001*ub;
    double stepsize=ub*(1-0.001)/100;
    int nochange=20;
    double maxiter=75;
    double tol=0.002;
    double miniter=30;
    return kfold_cv_lassomain(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
}

/*

void kfold_cv_lasso(const MatrixXd& X_inp, const VectorXd& y, int k, double lb, double ub, double stepsize, int nochange, int maxiter, double tol, int miniter)
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
        double x_stddev=sqrt(X.col(i).dot(X.col(i))/(X.rows()-1));
        X.col(i)=X.col(i)/x_stddev;
        stddev_forrescaling(i)=x_stddev; //we store the stddev values for rescaling
    }

    double lambda=0;
    int s=(X.rows()/k); 
    double MSE;
    double MSE_min=numeric_limits<double>::max();
    float ind;
    double l=pow(10, lb);
    double mult=pow(10,stepsize);
    vector<MatrixXd> X_tra(k);
    vector<VectorXd> Y_tra(k);
    vector<VectorXd> Y_test(k);
    vector<MatrixXd> X_test(k);

    dividing_data_for_cv(X,y,k,X_tra,X_test,Y_tra,Y_test);

    for(double i=lb; i<=ub; i=i+stepsize) //log scale for lambda
    {
       MSE=0;

       #pragma omp parallel for reduction(+:MSE)
       for(int j=0; j<k; j++)
       {
          VectorXd b(X.cols());
          b=lasso(X_tra[j], Y_tra[j], l, maxiter, tol, miniter); //training the model
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
    VectorXd b=lasso(X_inp, y, lambda, maxiter, tol, miniter);

    for(int i=1; i<X.cols(); i++) //rescaling
    {
        if(stddev_forrescaling(i)!=0)
        {
           b(i)=b(i)/stddev_forrescaling(i);
        }
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



void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k)
{
    double ub=log10(2*((X.transpose()*y).lpNorm<Infinity>())); //optimal ub.
    double lb=0.001*ub;
    double stepsize=0.01;
    int nochange=7;
    double maxiter=200;
    double tol=0.0001;
    double miniter=10;
    kfold_cv_lasso(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
} 

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, double stepsize, int nochange, int maxiter, double tol, int miniter)
{
    double ub=log10(2*((X.transpose()*y).lpNorm<Infinity>()));
    kfold_cv_lasso(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
}

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, int n, int nochange, int maxiter, double tol, int miniter)
{
    double ub=log10(2*((X.transpose()*y).lpNorm<Infinity>()));
    double stepsize=(ub-lb)/(n-1);
    kfold_cv_lasso(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
}

void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, int n, double ub, int nochange, int maxiter, double tol, int miniter)
{
    double stepsize=(ub-lb)/(n-1);
    kfold_cv_lasso(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
}


void kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k, double lb, double stepsize, int n, int nochange, int maxiter, double tol, int miniter)
{
    double ub=lb+stepsize*(n-1);
    kfold_cv_lasso(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
}

*/