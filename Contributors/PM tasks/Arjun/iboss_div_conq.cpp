#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace Eigen;

void iboss_select(const MatrixXd& X, const VectorXd& y, int n, MatrixXd& X_iboss, VectorXd& y_iboss) {
    vector<int> selected_indices;
    int rows = X.rows();
    int cols = X.cols();

    for (int j = 0; j < cols; ++j) {
        vector<pair<double, int>> col_vals;
        for (int i = 0; i < rows; ++i) {
            col_vals.emplace_back(X(i, j), i);
        }
        sort(col_vals.begin(), col_vals.end());
        for (int k = 0; k < n && k < rows; ++k) {
            selected_indices.push_back(col_vals[k].second);
            selected_indices.push_back(col_vals[rows - 1 - k].second);
        }
    }
    sort(selected_indices.begin(), selected_indices.end());
    selected_indices.erase(unique(selected_indices.begin(), selected_indices.end()), selected_indices.end());

    X_iboss.resize(selected_indices.size(), cols);
    y_iboss.resize(selected_indices.size());
    for (size_t i = 0; i < selected_indices.size(); ++i) {
        X_iboss.row(i) = X.row(selected_indices[i]);
        y_iboss(i) = y(selected_indices[i]);
    }
}

int main() {
    ifstream file("chem_data.csv");
    string a;
    const int total_rows = 13910;
    const int cols = 281;
    const int n_i = 2000; 
    const int k = 1000;

    const int iboss_n = k / (2 * cols);

    MatrixXd X_all(total_rows, cols);
    VectorXd y_all(total_rows);

    int row = 0;
    while (row < total_rows)
    {
        int read_rows = min(n_i, total_rows - row);
        vector<string> lines(read_rows);

        for (int i = 0; i < read_rows; ++i)
        {
            getline(file, lines[i]);
        }

        #pragma omp parallel for
        for (int i = 0; i < read_rows; ++i) {
            int cur_row = row + i;
            stringstream ss(lines[i]);
            string val;
            X_all(cur_row, 0) = 1;
            for (int j = 1; j < cols; ++j) {
            getline(ss, val, ',');
            X_all(cur_row, j) = stod(val);
            }
            getline(ss, val, ',');
            y_all(cur_row) = stod(val);
        }
        row += read_rows;
    }

    MatrixXd X_iboss;
    VectorXd y_iboss;
    iboss_select(X_all, y_all, iboss_n, X_iboss, y_iboss);

    VectorXd b_iboss = (X_iboss.transpose() * X_iboss).ldlt().solve(X_iboss.transpose() * y_iboss);
    VectorXd y_pred = X_all * b_iboss;

    double ss_res = (y_all - y_pred).squaredNorm();
    double ss_tot = (y_all.array() - y_all.mean()).matrix().squaredNorm();
    double r2 = 1 - ss_res / ss_tot;

    cout << "R^2: " << r2 << endl;

    ofstream outFile("cpp_iboss_chem.txt");
    if (outFile.is_open()) {
        for (int i = 0; i < b_iboss.size(); ++i) {
            outFile << b_iboss(i) << endl;
        }
        outFile.close();
        cout << "VectorXd successfully written to cpp_iboss_chem.txt" << endl;
    } else {
        cerr << "Error: Unable to open file for writing." << endl;
    }

    return 0;
}
