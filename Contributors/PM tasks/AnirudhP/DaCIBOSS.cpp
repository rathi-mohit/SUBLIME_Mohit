#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <iostream>
#include <Eigen/Dense>
#include "iboss.hpp"
#include <random>
#include <vector>
#include <fstream>
#include "regression_metrics.hpp"

using namespace std;
using namespace Eigen;

VectorXd LinearRegression(const MatrixXd& X,const VectorXd& y)
{
    
    return (X.transpose()*X).ldlt().solve(X.transpose()*y);
}

pair<MatrixXd, VectorXd> DCIBOSS(ifstream& file, int k, int B)
{
    string a;
    getline(file,a);
    //int k=5000;
    //int B=10;
    int ncols=count(a.begin(),a.end(),',')+1;
    int nrows=1;
    while(getline(file,a))
    {
        nrows++;
    }

    file.close();
    file.open("chem_data.csv");
    int ext=nrows%B;
    int part_size=(int)nrows/B;


    int k_B=ceil(k/B);

    MatrixXd X_f(B*k_B,ncols);
    VectorXd y_f(B*k_B);
    int pos=0;

    for(int l=0; l<ext; l++)
    {
        MatrixXd X_i(part_size+1,ncols);
        VectorXd y_i(part_size+1);
        for(int i=0; i<part_size+1; i++)
        {
            X_i(i,0)=1;
            getline(file,a,',');
            y_i(i)=stod(a);
            for(int j=1; j<ncols-1; j++)
            {
                getline(file,a,',');
                X_i(i,j)=stod(a);
            }
            getline(file,a,'\n');
            X_i(i,ncols-1)=stod(a);
        }

        pair<MatrixXd, VectorXd> result = k_selection(X_i,y_i,k_B);
        int s=result.first.rows();
        X_f.middleRows(pos,s)=result.first;
        y_f.middleRows(pos,s)=result.second;
        pos=pos+s;
    }

    for(int l=ext; l<B; l++)
    {
        MatrixXd X_i(part_size,ncols);
        VectorXd y_i(part_size);

        for(int i=0; i<part_size; i++)
        {
            X_i(i,0)=1;
            getline(file,a,',');
            y_i(i)=stod(a);
            for(int j=1; j<ncols-1; j++)
            {
                getline(file,a,',');
                X_i(i,j)=stod(a);
            }
            getline(file,a,'\n');
            X_i(i,ncols-1)=stod(a);
        }

        pair<MatrixXd, VectorXd> result = k_selection(X_i,y_i,k_B);
        int s=result.first.rows();
        X_f.middleRows(pos,s)=result.first;
        y_f.middleRows(pos,s)=result.second;
        pos=pos+s;
    }

    return make_pair(X_f, y_f);
}

int main()
{
    ifstream file("chem_data.csv"); 
    pair<MatrixXd, MatrixXd> result=DCIBOSS(file, 5000, 10);
    MatrixXd X=result.first;
    VectorXd y=result.second;
}

