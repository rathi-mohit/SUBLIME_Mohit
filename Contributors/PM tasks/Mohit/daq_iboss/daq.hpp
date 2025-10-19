#include <iostream>
#include <fstream>
#include <utility>
#include <queue>
#include <Eigen/Dense>

#include "fast_float/fast_float.h"

using namespace Eigen;
using namespace std;

MatrixXd cvt2MatrixXd(ifstream &file, size_t n_lines, const bool header = true, const char separator = ',') {
    vector<string> lines;

    string line;

    if (header) getline(file, line);

    for (size_t i = 0; i < n_lines; ++i) {
        getline(file, line);
        lines.push_back(line);
    }

    size_t rows = lines.size();
    size_t cols = 0;

    stringstream ss(lines[0]);
    string field;
    while (getline(ss, field, separator)) {
        ++cols;
    }
    
    MatrixXd data(rows, cols);

    // #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < lines.size(); ++i) {
        stringstream ss(lines[i]);
        string field;
        for (size_t j = 0; j < cols; ++j) {
            getline(ss, field, separator);
            const string &s = field;
            auto res = fast_float::from_chars(s.data(), s.data() + s.size(), data(i, j));
        }
    }

    return data;
}

pair<MatrixXd, VectorXd> k_selection(const MatrixXd &X, const VectorXd &y, int k) {

    const int p = X.cols();
    const int N = X.rows();
    const int r = k / (2 * (p - 1));  
    vector<bool> selected(N, false);

    // #pragma omp parallel for schedule(dynamic)
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

        // #pragma omp critical
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

    k = count(selected.begin(), selected.end(), true);

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

pair<MatrixXd, VectorXd> daq_iboss(const string filename, size_t y_colIndex, size_t B, size_t k, size_t fullDataSize, const bool header = true, const char separator = ',') {

    vector<size_t> sample_indices;
    size_t nB = fullDataSize / B;
    size_t kB = k / B;

    for (size_t i = 0; i < fullDataSize; i += nB) {
        sample_indices.push_back(i);
    }
    sample_indices.push_back(fullDataSize);
    vector<pair<MatrixXd, VectorXd>> selected(B);
    size_t daq_r = 0, daq_c;

    #pragma omp parallel for schedule(dynamic) reduction(+:daq_r)
    for (size_t i = 0; i < B; ++i) {

        ifstream file(filename);
        size_t skip = sample_indices[i];
        for (int j = 0; j < skip; ++j) 
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        
        auto B_i = cvt2MatrixXd(file, sample_indices[i+1] - sample_indices[i]);

        MatrixXd X_i(B_i.rows(), B_i.cols());
        VectorXd y_i(B_i.rows());

        X_i.col(0) = VectorXd::Constant(X_i.rows(), 1.0);
        X_i.middleCols(1, y_colIndex) = B_i.leftCols(y_colIndex);
        X_i.rightCols(X_i.cols() - y_colIndex - 1) = B_i.rightCols(X_i.cols() - y_colIndex - 1);
        y_i = B_i.col(y_colIndex);

        selected[i] = k_selection(X_i, y_i, kB);
        daq_r += selected[i].first.rows();
    }
    daq_c = selected[0].first.cols();

    MatrixXd X_daq(daq_r, daq_c);
    VectorXd y_daq(daq_r);

    size_t current_row_offset = 0;
    for (size_t i = 0; i < B; ++i) {
        size_t rows_in_block = selected[i].first.rows();
        X_daq.middleRows(current_row_offset, rows_in_block) = selected[i].first;
        y_daq.middleRows(current_row_offset, rows_in_block) = selected[i].second;
        current_row_offset += rows_in_block;
    }

    return {X_daq, y_daq};
}