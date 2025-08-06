#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <ctime> 
#include <iomanip>
#include <cmath>
#include "regression_metrics.hpp"
using namespace std;
using namespace Eigen;

namespace
{
    void display(string a, double para_min, int para_minpos,const vector<VectorXd>& M,const vector<vector<int>>& M_list)
    {
        cout<<a<<para_min<<" No of parameters: "<<(para_minpos+1)<<endl;
        cout<<"Estimators: \n"<<M[para_minpos]<<endl;

        cout<<"Parameters chosen: "<<endl;
        for(int i=0; i<M_list[para_minpos].size()-1; i++)
        {
        cout<<M_list[para_minpos][i]<<", ";
        }
        cout<<M_list[para_minpos][M_list[para_minpos].size()-1]<<endl;   
    }
}

void forwardsubsetselection(const MatrixXd& X,const VectorXd& y)
{
    int p=X.cols()-1;
    vector<int> v(p);
    VectorXd AIC(p);
    VectorXd BIC(p);
    VectorXd Cp(p);
    VectorXd AdjRsq(p);
    vector<Eigen::VectorXd> M(p); //stores the value of the estimators of each model, from size to 1 to p.
    vector<int> predictors;  //temporarily stores the index of each estimator used in a model
    vector<vector<int>> M_list(p); //stores the index of each estimator used in each model
    Eigen::VectorXd b;
    Eigen::VectorXd bmax;
    double rss;
    double rss_max;
    double tss=total_sum_of_squares(y);
    double Rsq;
    double Rsq_max;
    //finding variance
    Eigen::VectorXd r1=(X.transpose()*X).ldlt().solve(X.transpose()*y);
    double r2=(y-X*r1).squaredNorm();
    double var=r2/(X.rows()-(p+1));
    int k;
    //pool of estimators which can be added:
    for(int i=0; i<p; i++)
    {
        v[i]=i+1;
    }

    for(int i=0; i<p; i++)
    {
        Rsq_max=0;
        for(int j=0; j<p-i; j++)
        {
            MatrixXd W(X.rows(),i+2); //matrix with observations corresponding to i+1 predictors.
            W.col(0)=X.col(0); //Column of ones (intercept term)
            for(int g=0; g<i; g++)
            {
                W.col(g+1)=X.col(M_list[i-1][g]); //previously chosen predictors
            }
            W.col(i+1)=X.col(v[j]); //adding a new estimator to see if it improves the model
            b=(W.transpose()*W).ldlt().solve(W.transpose()*y); //least squares solution
            rss=(y-W*b).squaredNorm();
            Rsq=1-rss/tss;
            //finding model with maximum R_sq
            if(Rsq>Rsq_max)
            {
                Rsq_max=Rsq;
                rss_max=rss;
                bmax=b;
                k=v[j];
            }
        }
        //AIC(i)=(rss_max+2*(i+1)*var)/(var*X.rows()); //find AIC, BIC, Cp and adjusted R^2 for the models
        AIC(i)=X.rows()*log(rss_max/X.rows())+2*(i+2);
        //BIC(i)=(rss_max+2*log(X.rows())*(i+1)*var)/X.rows();
        BIC(i)=X.rows()*log(rss_max/X.rows())+(i+2)*log(X.rows());
        Cp(i)=(rss_max+2*(i+1)*var)/X.rows(); 
        AdjRsq(i)=1-(rss_max*(X.rows()-1))/(tss*(X.rows()-i-2));
        predictors.push_back(k); //adding k to the list of predictors, of the model with i+1 estimators
        M_list[i]=predictors; //adding the list of predictors to the M_list array
        v.erase(remove(v.begin(), v.end(), k), v.end()); //removing the estimator from the pool of estimators which can be added
        M[i]=bmax;
    }


    int AIC_minpos;
    int BIC_minpos;
    int Cp_minpos;
    int AdjR_maxpos;
    double AIC_min=AIC.minCoeff(&AIC_minpos);
    double Cp_min=Cp.minCoeff(&Cp_minpos);
    double BIC_min=BIC.minCoeff(&BIC_minpos);
    double AdjR_max=AdjRsq.maxCoeff(&AdjR_maxpos);

    
    display("Min Cp: ", Cp_min, Cp_minpos, M, M_list);
    display("Min AIC: ", AIC_min, AIC_minpos, M, M_list);
    display("Min BIC: ", BIC_min, BIC_minpos, M, M_list);
    display("Max adjusted R^2: ", AdjR_max, AdjR_maxpos, M, M_list);
}