#include <iostream>
#include <Eigen/Dense>
#include <random>
using namespace std;
using namespace Eigen;

VectorXd LEV(MatrixXd X, VectorXd y, int k)
{
    
    vector<double> pi(X.rows());
    VectorXd wt=VectorXd::Zero(X.rows());
    MatrixXd calc_temp=(X.transpose()*X).ldlt().solve(MatrixXd::Identity(X.cols(), X.cols()));
    for(int i=0; i<X.rows(); i++)
    {
        pi[i] = (X.row(i)*(calc_temp)*X.row(i).transpose())(0,0); 
        //not dividing by p+1 since it doesn't matter, pi is used as a weight
    }

    random_device rd;
    mt19937 mt(rd());
    std::discrete_distribution<> dist(pi.begin(), pi.end());

    for (int i=0; i<k; ++i)
    {
        auto temp=dist(mt);
        wt(temp)+=1/pi[temp];
    }

    return (X.transpose()*wt.asDiagonal()*X).ldlt().solve((X.transpose()*wt.asDiagonal()*y));
}