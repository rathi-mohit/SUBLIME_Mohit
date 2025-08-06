#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <csv.hpp>

using namespace Eigen;
using namespace std;

pair<MatrixXd, VectorXd> readCSVToMatrix(const string& filename, int response_col = -1) {
    csv::CSVReader reader(filename);
    
    vector<vector<double>> data;
    vector<double> response;
    
    for (csv::CSVRow& row : reader) {
        vector<double> row_data;
        for (int i = 0; i < row.size(); ++i) {
            try {
                double val = row[i].get<double>();
                if (i == response_col) {
                    response.push_back(val);
                } else {
                    row_data.push_back(val);
                }
            } catch (...) {
                if (i == response_col) {
                    response.push_back(0.0);
                } else {
                    row_data.push_back(0.0);
                }
            }
        }
        if (!row_data.empty()) {
            data.push_back(row_data);
        }
    }
    
    if (data.empty()) {
        return {MatrixXd(), VectorXd()};
    }
    
    int n = data.size();
    int p = data[0].size();
    
    MatrixXd X(n, p);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            X(i, j) = data[i][j];
        }
    }
    
    VectorXd y;
    if (response_col >= 0 && !response.empty()) {
        y = Map<VectorXd>(response.data(), response.size());
    } else {
        y = VectorXd::Zero(n);
    }
    
    return {X, y};
}

class TopKSelector {
public:
    static vector<int> getTopKIndices(const MatrixXd& matrix, int column, int k) {
        if (k >= matrix.rows()) {
            vector<int> allIndices(matrix.rows());
            iota(allIndices.begin(), allIndices.end(), 0);
            return allIndices;
        }
        
        vector<int> indices(matrix.rows());
        iota(indices.begin(), indices.end(), 0);
        
        nth_element(indices.begin(), 
                   indices.begin() + (matrix.rows() - k),
                   indices.end(),
                   [&](int a, int b) { return matrix(a, column) < matrix(b, column); });
        
        return vector<int>(indices.begin() + (matrix.rows() - k), indices.end());
    }
    
    static vector<int> getBottomKIndices(const MatrixXd& matrix, int column, int k) {
        if (k >= matrix.rows()) {
            vector<int> allIndices(matrix.rows());
            iota(allIndices.begin(), allIndices.end(), 0);
            return allIndices;
        }
        
        vector<int> indices(matrix.rows());
        iota(indices.begin(), indices.end(), 0);
        
        nth_element(indices.begin(), 
                   indices.begin() + k,
                   indices.end(),
                   [&](int a, int b) { return matrix(a, column) < matrix(b, column); });
        
        return vector<int>(indices.begin(), indices.begin() + k);
    }
};

struct DOptSelection {
private:
    MatrixXd X;
    VectorXd y;
    int k;
    int p, r, n;

public:
    DOptSelection(const MatrixXd& X, const VectorXd& y, int k) : X(X), y(y), k(k) {
        p = X.cols();
        r = k / (2 * p);
        n = X.rows();
    }

    pair<MatrixXd, VectorXd> get_dopt_selection() {
        vector<int> selected_indices;
        
        for (int feature = 0; feature < p; ++feature) {
            auto top_indices = TopKSelector::getTopKIndices(X, feature, r);
            auto bottom_indices = TopKSelector::getBottomKIndices(X, feature, r);
            
            selected_indices.insert(selected_indices.end(), top_indices.begin(), top_indices.end());
            selected_indices.insert(selected_indices.end(), bottom_indices.begin(), bottom_indices.end());
        }
        
        sort(selected_indices.begin(), selected_indices.end());
        selected_indices.erase(unique(selected_indices.begin(), selected_indices.end()), selected_indices.end());
        
        int s = selected_indices.size();
        MatrixXd sel_X(s, p);
        VectorXd sel_y(s);
        
        for (int i = 0; i < s; ++i) {
            sel_X.row(i) = X.row(selected_indices[i]);
            sel_y(i) = y(selected_indices[i]);
        }
        
        return {sel_X, sel_y};
    }
};

VectorXd ols_reg(const MatrixXd& X, const VectorXd& y) {
    MatrixXd XT = X.transpose();
    VectorXd beta = (XT * X).ldlt().solve(XT * y);
    return beta;
}

int main() {
    string filename = "txt";
    
    auto start_read = chrono::high_resolution_clock::now();
    auto xy = readCSVToMatrix(filename, 2);
    auto end_read = chrono::high_resolution_clock::now();
    auto read_duration = chrono::duration_cast<chrono::milliseconds>(end_read - start_read);
    cout << "CSV time = " << read_duration.count() << " ms" << endl;
    
    MatrixXd X = xy.first;
    VectorXd y = xy.second;
    
    auto start_dopt = chrono::high_resolution_clock::now();
    DOptSelection dopt(X, y, 100000);
    auto sel = dopt.get_dopt_selection();
    MatrixXd sel_X = sel.first;
    VectorXd sel_y = sel.second;
    auto end_dopt = chrono::high_resolution_clock::now();
    auto dopt_duration = chrono::duration_cast<chrono::microseconds>(end_dopt - start_dopt);
    cout << "D-Opt time = " << dopt_duration.count() << " μs" << endl;
    
    auto start_ols = chrono::high_resolution_clock::now();
    VectorXd beta = ols_reg(sel_X, sel_y);
        
    return 0;
}