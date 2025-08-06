#include <iostream>

#include <vector>
#include <numeric>
#include <set>
#include <string>
#include <iomanip>
#include <typeinfo>

#include <fstream>
#include <sstream>

#include <ctime>
#include <cmath>

#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct DOptSelector
{
    private:
    MatrixXd X;
    VectorXd y; // response vector
    int k;
    int p, r, n;

    public:
    DOptSelection(const MatrixXd & X, const VectorXd & y, const int & k) : X (X), y (y), k (k)
    {
        int p = X.cols();
        int r = k/(2*p);
        int n = X.rows();
    }

    static MatrixXd top_rows_nth_element(const MatrixXd & X, const VectorXd & y; int c, int k)
    {
        int n = X.rows();
        if (k >= n)
        {
            return make_pair(X, y);
        }

        vector<int> selected_rows(n);
        nth_element(selected_rows.begin() + n - k, selected_rows.begin(), [&](int a, int b) { return X(a, c) < X(b, c)});
        
    }
}