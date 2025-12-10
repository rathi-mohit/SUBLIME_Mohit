#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "SRHT.cpp"
#include <chrono>
#include "regression_metrics.hpp"

using namespace std;
using namespace Eigen;

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
    VectorXd b=LinearRegression(X,y);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    auto start1 = chrono::high_resolution_clock::now();
    VectorXd b_SRHT=SRHT(X,y);
    auto end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);

    cout<<duration.count()<<endl;
    cout<<duration1.count()<<endl;

    double r2=r_squared_score(y,X*b);
    double r2_SRHT=r_squared_score(y,X*b_SRHT);

    cout<<r2<<endl;
    cout<<r2_SRHT<<endl;

}
