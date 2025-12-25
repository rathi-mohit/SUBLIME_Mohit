#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>
#include <utility>
#include <omp.h>
#include <cstdlib>   
#include <ctime>
#include <chrono>
#include <fstream>
#include <set>
#include <map>
#include <iomanip>
#include "regression_metrics.hpp"
#include "headerlasso.hpp"
#include "lasso.cpp"
#include "iboss.hpp"
using namespace std;
using namespace Eigen;


VectorXd LinearRegression(const MatrixXd& X,const VectorXd& y)
{
    
    return (X.transpose()*X).ldlt().solve(X.transpose()*y);
}

void sampling(const MatrixXd& X, const VectorXd& y, int nsample, MatrixXd& P, VectorXd& Q)
{
    random_device rd;
    mt19937 mt(rd());
    vector<int> nums(X.rows());
    iota(nums.begin(), nums.end(), 0);

    vector<int> out;
    sample(nums.begin(), nums.end(), back_inserter(out), nsample, mt);
    for(int i=0; i<nsample; i++)
    {
        P.row(i)=X.row(out[i]);
        Q(i)=y(out[i]);
    }
}

double CLASS_algorithm(MatrixXd X, VectorXd y, int nsample, int ntimes, int k)
{
    //0th column is the intercept column
    vector<int> c(X.cols()-1);
    int q=X.cols()-1;
    
    #pragma omp parallel for
    for(int i=0; i<ntimes; i++)
    {
        MatrixXd P(nsample, X.cols());
        VectorXd Q(nsample);
        sampling(X,y,nsample,P,Q);
        VectorXd b=kfold_cvlasso(P,Q,10);
        for(int j=0; j<q; j++)
        {
            if(b(j+1)!=0)
            {
            #pragma omp atomic
            c[j]++;
            }
        }
    }

    random_device rd;
    mt19937 mt(rd());
    vector<int> nums(q);
    iota(nums.begin(), nums.end(), 0);
    vector<int> out;
    sample(nums.begin(), nums.end(), back_inserter(out), 2, mt);
    double mean1=c[out[0]];
    double mean2=c[out[1]];
    double nm1=mean1+1;
    double nm2=mean2+2;
    vector<int> c1f;
    vector<int> c2f;
    //k_means
    while(abs((nm1-mean1)/max(mean2,1.0))>0.001&&abs(((nm2-mean2)/max(mean2,1.0))>0.001))
    {
        vector<int> c1;
        vector<int> c2;

        for(int i=0; i<q;i++)
        {
        if(abs(mean1-c[i])<abs(mean2-c[i]))
        {
           c1.push_back(i);
        }
        else
        {
           c2.push_back(i);
        }
        }

        nm1=mean1;
        mean1=0;
        for(auto temp:c1)
        {
            mean1=mean1+c[temp];
        }
        mean1=mean1/c1.size();

        nm2=mean2;
        mean2=0;
        for(auto temp:c2)
        {
            mean2=mean2+c[temp];
        }
        mean2=mean2/c2.size();
        c1f=c1;
        c2f=c2;

    }
    
    vector<int> cf;
    if(mean1>mean2)
    {
       cf=c1f;
    }
    else
    {
        cf=c2f;
    }

    MatrixXd X_reduced(X.rows(),cf.size()+1); 
    X_reduced.col(0)=X.col(0);   

    for(int i=0; i<cf.size();i++)
    {
       X_reduced.col(i+1)=X.col(cf[i]+1);
    }

    pair<MatrixXd, VectorXd> result = k_selection(X_reduced,y,k);
    MatrixXd X_class = result.first;
    VectorXd y_class = result.second;

    VectorXd b_class=LinearRegression(X_class, y_class);
    VectorXd pred_y_class=X_reduced*b_class;
    return r_squared_score(y, pred_y_class);
}

int main()
{
    ifstream file("chem_data.csv");
    string a;
    MatrixXd X(13910,128);
    VectorXd y(13910);
    for(int i=0; i<13910; i++)
    {
        X(i,0)=1;
        getline(file, a, ',');
        y(i)=stod(a);
        for(int j=1; j<127; j++)
        {
            getline(file, a, ',');
            X(i,j)=stod(a);
        }
        getline(file, a, '\n');
        X(i,127)=stod(a);
    }
    auto start = chrono::high_resolution_clock::now();
    cout<<CLASS_algorithm(X,y,100,1000,5000)<<endl;
    auto end = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout<<duration1.count()<<endl;
    auto start1 = chrono::high_resolution_clock::now();
    LinearRegression(X,y);
    auto end1 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    cout<<duration2.count()<<endl;

}