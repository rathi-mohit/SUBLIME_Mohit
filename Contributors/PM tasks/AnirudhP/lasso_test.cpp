#include <iostream>
#include "lasso.cpp"
#include <fstream>
#include <ctime>
#include <chrono>
#include "LinearRegression.hpp"
#include "regression_metrics.hpp"
#include "headerlasso.hpp"
#include "headerridge.hpp"

int main()
{
    ifstream file("blogData_train.csv");
    string a;
    MatrixXd X(52397,281);
    VectorXd y(52397);
    for(int i=0; i<52397; i++)
    {
        X(i,0)=1;
        for(int j=1; j<281; j++)
        {
            getline(file,a,',');
            X(i,j)=stod(a);
        }
        getline(file,a,'\n');
        y(i)=stod(a);
    }
        
    /*
    auto start=chrono::high_resolution_clock::now();
    VectorXd betaLASSO=kfold_cvlasso(X,y,10);
    auto end=chrono::high_resolution_clock::now();
    auto duration=chrono::duration_cast<chrono::milliseconds>(end-start);
    cout<<duration.count()<<endl;
    */

    /*
    auto start=chrono::high_resolution_clock::now();
    */
    VectorXd betaLASSO=kfold_cvlasso(X,y,10);

    /*
    auto end=chrono::high_resolution_clock::now();
    auto duration=chrono::duration_cast<chrono::milliseconds>(end-start);
    cout<<duration.count()<<endl;
    */

    /*
    std::clock_t c_start = std::clock();
    VectorXd betaLASSO=kfold_cvlasso(X,y,10);
    std::clock_t c_end = std::clock();

    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " <<time_elapsed_ms<< " ms\n";
    */

    //VectorXd y_predL=X*betaLASSO;
    //cout<<r_squared_score(y, y_predL)<<endl;
    //35.4355
    //783206
    //13.053 mins
}
