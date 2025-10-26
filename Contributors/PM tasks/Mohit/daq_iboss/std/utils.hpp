#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include "fast_float/fast_float.h"
#include "regression_metrics.hpp"
#include "regression_models.hpp"

using namespace std;
using namespace Eigen;

vector<vector<string>> load_csv(const string &filename, const bool &header = true, const char separator = ',') {
    vector<string> lines;

    ifstream file(filename);
    string line;

    if (header) getline(file, line);

    while (getline(file, line)) {
        lines.push_back(line);
    }

    vector<vector<string>> data(lines.size());

    #pragma omp parallel for
    for (int i = 0; i < lines.size(); ++i) {
        stringstream ss(lines[i]);
        string field;
        vector<string> row;
        while (getline(ss, field, ',')) {
            row.push_back(field);
        }
        data[i] = row;
    }

    return data;
}
vector<size_t> numerical_indices(vector<vector<string>> &data) {
    vector<size_t> indices;
    for (size_t i = 0; i < data[0].size(); ++i) {
        try {
            double field = stod(data[0][i]);
            indices.push_back(i);
        } catch(...) {
            continue;
        }
    }
    return indices;
}
MatrixXd convert_to_matrix(vector<vector<string>> &data, vector<size_t> &indices) {
    vector<vector<double>> numerical_data;
    
    for (size_t i = 0; i < data.size(); ++i) {
        vector<double> dataRow;
        for (size_t j = 0; j < indices.size(); ++j) {
            try {
                dataRow.push_back(stod(data[i][indices[j]]));
            } catch (...) {
                continue;
            }
        }
        if (dataRow.size() == indices.size()) numerical_data.push_back(dataRow);
    }
    
    MatrixXd X(numerical_data.size(), indices.size());
    for (size_t i = 0; i < numerical_data.size(); ++i) {
        X.row(i) = Map<VectorXd>(numerical_data[i].data(), numerical_data[i].size());
    }

    return X;
}   

MatrixXd cleaned_convert_to_matrix(const vector<vector<string>> &data, const vector<size_t> &indices) {
    size_t rows = data.size();
    size_t cols = indices.size();
    MatrixXd X(rows, cols);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const string &s = data[i][indices[j]];
            auto res = fast_float::from_chars(s.data(), s.data() + s.size(), X(i, j));
        }
    }

    return X;
}

MatrixXd clean_numerical_load_csv(const string filename, const bool header = true, const char separator = ',') {
    vector<string> lines;

    ifstream file(filename);
    string line;

    if (header) getline(file, line);

    while (getline(file, line)) {
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

    #pragma omp parallel for schedule(static)
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

    #pragma omp parallel for schedule(dynamic)
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

void standardize(MatrixXd &X, bool intercept = true) {
    
    if (intercept) {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 1; i < X.cols(); ++i) {
            double column_mean = X.col(i).mean();
            VectorXd column_mean_vec = VectorXd::Constant(X.rows(), column_mean);
            double denom = (X.col(i) - column_mean_vec).squaredNorm() / X.rows();
            for (size_t j = 0; j < X.rows(); ++j) {
                X(j, i) = (X(j, i) - column_mean) / sqrt(denom);
            }
        }
    }

    else {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < X.cols(); ++i) {
            double column_mean = X.col(i).mean();
            VectorXd column_mean_vec = VectorXd::Constant(X.rows(), column_mean);
            double denom = (X.col(i) - column_mean_vec).squaredNorm() / X.rows();
            for (size_t j = 0; j < X.rows(); ++j) {
                X(j, i) = (X(j, i) - column_mean) / sqrt(denom);
            }
        }
    }

    // vector<double> col_denoms(X.cols(), 1);

    // #pragma omp parallel for
    // for (; i < X.cols(); ++i) {
    //     double column_mean = X.col(i).mean();
    //     VectorXd column_mean_vec = VectorXd::Constant(X.rows(), column_mean);
    //     double denom = (X.col(i) - column_mean_vec).squaredNorm() / X.rows();
    //     col_denoms[i] = denom;
    // }

    // #pragma omp parallel for
    // for (size_t j = 0; j < X.rows(); ++j) {
    //     for (size_t k = 1; k < X.cols(); ++k) {
    //         X(j, k) = X(j, k) / col_denoms[k];
    //     }
    // }
}

double cross_validation(const MatrixXd &X_given, const VectorXd &y_given, int k) {
    size_t n = X_given.rows();
    size_t fold_size = n / k;
    n = k * fold_size;
    size_t curr = 0;

    vector<MatrixXd> X_test(k), X_train(k);
    vector<VectorXd> y_test(k), y_train(k);

    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        size_t start = i * fold_size;
        size_t size = fold_size;

        X_test[i] = X_given.middleRows(start, size);
        y_test[i] = y_given.middleRows(start, size);

        MatrixXd X_temp(n - size, X_given.cols());
        VectorXd y_temp(n - size);

        int curr = 0;
        for (int j = 0; j < k; ++j) {
            if (j == i) continue;
            size_t j_start = j * fold_size;
            size_t j_size = fold_size;

            X_temp.middleRows(curr, j_size) = X_given.middleRows(j_start, j_size);
            y_temp.middleRows(curr, j_size) = y_given.middleRows(j_start, j_size);

            curr += j_size;
        }

        X_train[i] = X_temp;
        y_train[i] = y_temp;
    }

    vector<double> R2_score_vec(k, 0);
    vector<double> MSE_vec(k, 0);

    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        VectorXd beta = betaLasso(X_train[i], y_train[i], 10);
        VectorXd y_pred = X_test[i] * beta;

        double mse = mean_squared_error(y_test[i], y_pred);
        double r2 = r_squared_score(y_test[i], y_pred);

        R2_score_vec[i] = r2;
        MSE_vec[i] = mse;
    }

    double avg_R2 = accumulate(R2_score_vec.begin(), R2_score_vec.end(), 0.0) / k;
    double avg_MSE = accumulate(MSE_vec.begin(), MSE_vec.end(), 0.0) / k;
    // cout << "Average R2: " << avg_R2 << endl;
    // cout << "Average MSE: " << avg_MSE << endl;
    return avg_R2;
}
