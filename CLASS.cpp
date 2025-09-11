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
using namespace std;
using namespace Eigen;

double residual_sum_of_squares(const VectorXd &y_actual, const VectorXd &y_pred) {
    VectorXd residuals = y_actual - y_pred;
    return residuals.squaredNorm();
}

double total_sum_of_squares(const VectorXd &y) {
    VectorXd y_mean_vec = VectorXd::Constant(y.rows(), y.mean());
    return (y - y_mean_vec).squaredNorm();
}

double r_squared_score(const VectorXd &y, const VectorXd &y_pred) {
    double rss = residual_sum_of_squares(y, y_pred);
    double tss = total_sum_of_squares(y);
    return 1 - (rss / tss);
}

pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X, const VectorXd &y, int k) {

    const int p = X.cols();
    const int N = X.rows();
    const int r = k / (2 * (p - 1));  
    vector<bool> selected(N, false);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 1; j < p; ++j) {
        priority_queue<pair<double, int>, vector<pair<double, int>>, less<pair<double, int>>> maximals;
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> minimals;

        for (int i = 0; i < N; ++i) {
            // if (selected[i]) continue;
            double x_ij = X(i, j);
            
            if (maximals.size() < r) {
                maximals.emplace(x_ij, i);
            } 
            else if (x_ij < maximals.top().first) {
                maximals.pop();
                maximals.emplace(x_ij, i);
            }
            
            if (minimals.size() < r) {
                minimals.emplace(x_ij, i);
            } 
            else if (x_ij > minimals.top().first) {
                minimals.pop();
                minimals.emplace(x_ij, i);
            }
        }

        #pragma omp critical
        {
            while (!maximals.empty()) {
                selected[maximals.top().second] = 1;
                maximals.pop();
            }
            
            while (!minimals.empty()) {
                selected[minimals.top().second] = 1;
                minimals.pop();
            }
        }
    }

    k = selected.size();

    MatrixXd X_iboss(k, p);
    VectorXd y_iboss(k);
    size_t row = 0;

    for (size_t i = 0; i < N; ++i) {
        if (selected[i]) {
            X_iboss.row(row) = X.row(i);
            y_iboss(row) = y(i);
            ++row;
        }
    }

    return make_pair(X_iboss, y_iboss);
}


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


void dividing_data_for_cv(const MatrixXd& X, const VectorXd& y, int k, vector<MatrixXd>& X_tra, vector<MatrixXd>& X_test, vector<VectorXd>& Y_tra, vector<VectorXd>& Y_test)
{
    int si=X.rows()/k;

    #pragma omp parallel for
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

void coord_descent(const MatrixXd&X, double lambda, VectorXd& b, int k,const VectorXd& r, double ss)
{
    double m=X.col(k).dot(r)+ss*b(k);

    if(m>lambda/2)
    {
       b(k)=(m-lambda/2)/ss;
    }
    else if(m<-lambda/2)
    {
       b(k)=(m+lambda/2)/ss;
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
    b_old(0)=y.mean();
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
    VectorXd b=lasso(X_inp, y, lambda, maxiter, tol, miniter);

    for(int i=1; i<X.cols(); i++) //rescaling
    {
        b(i)=b(i)/stddev_forrescaling(i);
        b(0)-=X_inp.col(i).mean()*b(i);
    }

    return b;
}



VectorXd kfold_cv_lasso(const MatrixXd& X, const VectorXd& y, int k)
{
    double ub=log10(2*((X.transpose()*y).lpNorm<Infinity>())); //optimal ub.
    double lb=0.001*ub;
    double stepsize=0.01;
    int nochange=7;
    double maxiter=75;
    double tol=0.0015;
    double miniter=10;
    return kfold_cv_lassomain(X, y, k, lb, ub, stepsize, nochange, maxiter, tol, miniter);
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
        VectorXd b=kfold_cv_lasso(P,Q,10);
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
    cout<<CLASS_algorithm(X,y,300,10,5000)<<endl;
    auto end = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout<<duration1.count()<<endl;
    auto start1 = chrono::high_resolution_clock::now();
    LinearRegression(X,y);
    auto end1 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    cout<<duration2.count()<<endl;

}