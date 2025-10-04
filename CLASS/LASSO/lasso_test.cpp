#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#include "regression_metrics.hpp"
#include "headerlasso.hpp"
#include "lasso.cpp"

int main()
{
    ifstream file("chem_data.csv");
    string a;
    // Use number of rows and columns of your dataset here.
    MatrixXd X(13910,281);
    VectorXd y(13910);

    for(int i=0; i<13910; i++)
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

    std::clock_t c_start = std::clock();
    VectorXd b_lasso=kfold_cvlasso(X,y,10);
    std::clock_t c_end = std::clock();
    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " <<time_elapsed_ms<< " ms\n";
    VectorXd y_pred=X*b_lasso;

    cout << r_squared_score(y, y_pred) << endl;

    std::ofstream outFile("cpp_lasso_chem.txt");
    if (outFile.is_open()) 
    {
        for (int i = 0; i < b_lasso.size(); ++i) {
            outFile << b_lasso(i) << std::endl;
        }
        outFile.close();
        std::cout << "VectorXd successfully written to output_vector.txt" << std::endl;
    } else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }

    return 0;
}
