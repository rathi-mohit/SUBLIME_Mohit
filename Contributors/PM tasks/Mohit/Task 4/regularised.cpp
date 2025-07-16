#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <utility>
#include <omp.h>
#include <set>
#include <map>

using namespace Eigen;
using namespace std;

vector<vector<string>> load_csv(const string &filename, const bool &header = true, const char separator = ',');
vector<string> extract_headers(const string &filename, const char separator = ',');
vector<size_t> numerical_indices(vector<vector<string>> &data);
MatrixXd convert_to_matrix(vector<vector<string>> &data, vector<size_t> &indices);
void standardize(MatrixXd &X, bool intercept_ = true);

VectorXd betaOLS(MatrixXd &X, VectorXd &y, bool intercept = true);
VectorXd betaRidge(MatrixXd &X, VectorXd &y, double lambda = 1.0, bool intercept = true);
VectorXd betaLasso(MatrixXd &X, VectorXd &y, double lambda = 1.0, double res = 1e-4, int iter = 1000, bool intercepted = true);
  
double RSS(const MatrixXd &X, const VectorXd &y);
double RSS(const VectorXd &y, const VectorXd &y_pred);
double TSS(const VectorXd &y);
double R2_score(MatrixXd &X, VectorXd &y, VectorXd &beta);
double R2_score(const VectorXd &y, const VectorXd &y_pred);
double adj_R2_score(const MatrixXd &X, const VectorXd &y, const VectorXd &beta);
double RSE(MatrixXd &X, VectorXd &y, VectorXd &beta);
double MSE(const VectorXd &y_true, const VectorXd &y_pred);
double AIC(const MatrixXd &X, const VectorXd &y);
double AIC(const MatrixXd &X, const VectorXd &y, const VectorXd &beta);
double BIC(const MatrixXd &X, const VectorXd &y);
VectorXd t_stat(MatrixXd &X, VectorXd &y, VectorXd &beta);

vector<size_t> k_selection(const int k, const MatrixXd &X_given);
vector<size_t> k_selection(const vector<vector<string>> &data, const int k);
double cross_validation(const MatrixXd &X, const VectorXd &y, int k);

int main() {
    auto start = chrono::high_resolution_clock::now();

    const string lasso_filename = "lasso_data.dat";
    const string housing_filename = "housing.csv";
    const string filename = housing_filename;
    // vector<string> headers = extract_headers(filename);
    vector<vector<string>> data = load_csv(filename);
    vector<size_t> numerical_cols = numerical_indices(data);
    shuffle(data.begin(), data.end(), default_random_engine(42));

    MatrixXd numerical_data = convert_to_matrix(data, numerical_cols);

    MatrixXd X(numerical_data.rows(), numerical_data.cols());
    X.col(0) = VectorXd::Constant(numerical_data.rows(), 1.0);
    X.rightCols(numerical_data.cols() - 1) = numerical_data.leftCols(numerical_data.cols() - 1);
    VectorXd y = numerical_data.rightCols(1);

    standardize(X);

    // // train-test-split if necessary
    // int N8 = 0.8*N;
    // int N2 = N - N8;

    // MatrixXd X_train = X.topRows(N8);
    // MatrixXd X_test = X.bottomRows(N2);
    // VectorXd y_train = y.topRows(X_train.rows());
    // VectorXd y_test = y.bottomRows(X_test.rows());

    VectorXd beta_OLS = betaOLS(X, y);
    // VectorXd beta_Lasso = betaLasso(X, y, 1);
    // cout << t_stat(X, y, beta_OLS) << "\n\n\n";
    // cout << beta_Lasso << "\n\n\n";
    cout << beta_OLS << "\n\n\n";

    cout << "R2_Score: " << R2_score(X, y, beta_OLS) << '\n';
    // cout << R2_score(X, y, beta_Lasso) << '\n';

    // cout << adj_R2_score(X, y, beta_OLS) << '\n';
    // cout << adj_R2_score(X, y, beta_Lasso) << '\n';

    vector<size_t> selected_data = k_selection(8000, X);
    // Good k == 7200 or 8000 for housing.csv (sized 20k approx)
    // Good k == 200 for lasso_data.dat (sized 5k approx)

    MatrixXd k_selected_X(selected_data.size(), X.cols());
    VectorXd k_selected_y(selected_data.size());
    for (size_t i = 0; i < selected_data.size(); ++i) {
        k_selected_X.row(i) = X.row(i);
        k_selected_y(i) = y(i);
    }

    VectorXd beta_new = betaOLS(k_selected_X, k_selected_y);

    cout << beta_new << "\n\n\n";
    cout << "R2_score of the k-selected based on D-optimality: " 
            << R2_score(X, y, beta_new) << '\n';

    cout << "Relative difference in betas from full-data and k-selected data: " <<
            (beta_OLS - beta_new).squaredNorm() / beta_OLS.squaredNorm() << '\n';
    
    cout << t_stat(X, y, beta_OLS) << "\n\n\n";
    cout << betaLasso(X, y, 10) << "\n\n\n";

    cout << cross_validation(X, y, 10) << "\n\n\n"; 
    // This cross_val single-handedly increases time to like one minute or two unless omp'd
    // hence we should compile like "g++ regularised.cpp -fopenmp" 
    // the -fopenmp flag is for parallelization, which causes the time to come down to 15s approx
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time Taken: " << duration.count() << '\n';
    return 0;
}
vector<vector<string>> load_csv(const string &filename, const bool &header, const char separator) {
    vector<vector<string>> data;

    ifstream file(filename);
    string line;

    if (header) getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string field;

        try {
            vector<string> row;
            while (getline(ss, field, separator)) {
                row.push_back(field);
            }
            data.push_back(row);
        } catch(...) {
            continue;
        }
    }
    file.close();
    return data;
}
vector<string> extract_headers(const string &filename, const char separator) {
    ifstream file(filename);
    string line;
    vector<string> cols;

    getline(file, line);
    stringstream ss(line);
    string field;

    while (getline(ss, field, separator)) {
        cols.push_back(field);
    }
    file.close();
    return cols;
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
void standardize(MatrixXd &X, bool intercept) {
    
    if (intercept) {
        #pragma omp parallel for
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
        #pragma omp parallel for
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
VectorXd betaOLS(MatrixXd &X, VectorXd &y, bool intercept) {
    if (intercept) {
        return (X.transpose() * X).ldlt().solve(X.transpose() * y);
    }
    else {
        MatrixXd X_with_intercept(X.rows(), X.cols()+1);
        X_with_intercept.col(0) = VectorXd::Constant(X.rows(), 1);
        X_with_intercept.rightCols(X.cols()) = X;
        return betaOLS(X_with_intercept, y, true);
    }
}
VectorXd betaRidge(MatrixXd &X, VectorXd &y, double lambda, bool intercept) {
    if (intercept) {
        MatrixXd I = MatrixXd::Identity(X.cols(), X.cols()); I(0, 0) = 0;
        return (X.transpose() * X + lambda*I).ldlt().solve(X.transpose() * y);
    } 
    else {
        MatrixXd X_with_intercept(X.rows(), X.cols()+1);
        X_with_intercept.col(0) = VectorXd::Constant(X.rows(), 1);
        X_with_intercept.rightCols(X.cols()) = X;
        return betaRidge(X_with_intercept, y, lambda, true);
    }
}
VectorXd betaLasso(MatrixXd &X, VectorXd &y, double lambda, double res, int max_iter, bool intercept) {
    int N = X.rows();
    int p = X.cols();

    VectorXd beta = VectorXd::Constant(p, 0.0);
    VectorXd beta_prev = beta;

    for (int i = 0; i < max_iter; ++i) {
        beta_prev = beta;
        for (size_t j = 0; j < p; ++j) {
            VectorXd r_j = y - X * beta + (X.col(j) * beta(j));
            double z_j = X.col(j).squaredNorm() / N;
            double rho_j = (X.col(j).transpose() * r_j)(0) / N;
            double s = 0;
            if (j == 0) beta(j) = rho_j / z_j;
            else {
                if (rho_j > lambda) {
                    beta(j) = (rho_j - lambda) / z_j;
                } 
                else if (rho_j < -1.0*lambda) {
                    beta(j) = (rho_j + lambda) / z_j;
                } 
                else {
                    beta(j) = 0.0;
                }
            }
        }
        if ((beta - beta_prev).squaredNorm() < res) break;
    }

    return beta;
}
double RSS(const MatrixXd &X, const VectorXd &y) {
    VectorXd beta = (X.transpose() * X).inverse() * (X.transpose() * y);
    VectorXd residuals = y - X * beta;
    return residuals.squaredNorm();
}
double MSE(const VectorXd &y_true, const VectorXd &y_pred) {
    double sum_sq_error = 0.0;
    int n = y_true.rows();

    for (int i = 0; i < n; ++i) {
        double err = y_true(i) - y_pred(i);
        sum_sq_error += err * err;
    }
    return sum_sq_error / n;
}
double RSS(const VectorXd &y, const VectorXd &y_pred) {
    VectorXd residuals = y - y_pred;
    return residuals.squaredNorm();
}
double TSS(const VectorXd &y) {
    VectorXd y_mean_vec = VectorXd::Constant(y.rows(), y.mean());
    return (y - y_mean_vec).squaredNorm();
}
double R2_score(const VectorXd &y, const VectorXd &y_pred) {
    double rss = RSS(y, y_pred);
    double tss = TSS(y);
    return 1 - (rss / tss);
}
double R2_score(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    VectorXd y_pred = X * beta;
    return R2_score(y, y_pred);
}
double adj_R2_score(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows(), p = X.cols();
    double rss = RSS(X, y);
    double tss = TSS(y);
    return 1 - ((rss / (N - p)) / (tss / (N - 1)));
}
double RSE(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    int N = X.rows(), p = X.cols() - 1;
    VectorXd y_pred = X * beta;
    VectorXd e = y - y_pred;
    double rss = e.squaredNorm();
    return sqrt(rss / (N - p - 1));
}
VectorXd t_stat(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    VectorXd y_pred = X * beta;
    VectorXd residuals = y - y_pred;
    double rss = residuals.squaredNorm();
    int N = X.rows(), p = X.cols() - 1;
    double sigma2_hat = rss / (N - p - 1);
    MatrixXd XtX_inv = (X.transpose() * X).inverse();
    MatrixXd cov_beta = sigma2_hat * XtX_inv;
    VectorXd t_stats(p + 1);
    for (int j = 0; j <= p; ++j) {
        double se = sqrt(cov_beta(j, j));
        t_stats(j) = beta(j) / se;
    }
    return t_stats;
}
double AIC(const MatrixXd &X, const VectorXd &y) {
    int N = X.rows();
    int p = X.cols();
    VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + 2 * p;
}
double AIC(const MatrixXd &X, const VectorXd &y, const VectorXd &beta) {
    int N = X.rows();
    int p = X.cols();
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + 2 * p;
}
double BIC(const MatrixXd &X, const VectorXd &y) {
    int N = X.rows();
    int p = X.cols();
    VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    VectorXd residuals = y - X * beta;
    double rss = residuals.squaredNorm();
    return N * log(rss / N) + p * log(N);
}
vector<size_t> k_selection(const int k, const MatrixXd &X_given) {
    MatrixXd X = X_given.rightCols(X_given.cols() - 1);
    int p = X.cols();
    int r = k / (2 * p);
    set<size_t> selected;
    for (int i = 0; i < p; ++i) {
        map<double, size_t> ordered_z;
        VectorXd z_i = X.col(i);
        for (size_t j = 0; j < z_i.rows(); ++j) ordered_z[z_i(j)] = j;

        int count = 0;
        for (auto it = ordered_z.begin(); it != ordered_z.end(); ++it) {
            if (selected.find(it -> second) == selected.end()) {
                ++count;
                selected.insert(it -> second);
            }
            if (count == r) break;
        }

        count = 0;
        for (auto it = --ordered_z.end(); it != ordered_z.begin(); --it) {
            if (selected.find(it -> second) == selected.end()) {
                ++count;
                selected.insert(it -> second);
            }
            if (count == r) break;
        }
    }
    vector<size_t> selected_rows(selected.begin(), selected.end());
    return selected_rows;
}
vector<size_t> k_selection(vector<vector<string>> &data, const int k) {
    vector<size_t> numerical_cols = numerical_indices(data);
    MatrixXd numerical_data = convert_to_matrix(data, numerical_cols);

    MatrixXd X(data.size(), numerical_cols.size());
    X.col(0) = VectorXd::Constant(data.size(), 1.0);
    X.rightCols(numerical_cols.size() - 1) = numerical_data.leftCols(numerical_cols.size() - 1);

    standardize(X);
    return k_selection(k, X);
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

        double mse = MSE(y_test[i], y_pred);
        double r2 = R2_score(y_test[i], y_pred);

        R2_score_vec[i] = r2;
        MSE_vec[i] = mse;
    }

    double avg_R2 = accumulate(R2_score_vec.begin(), R2_score_vec.end(), 0.0) / k;
    double avg_MSE = accumulate(MSE_vec.begin(), MSE_vec.end(), 0.0) / k;
    // cout << "Average R2: " << avg_R2 << endl;
    // cout << "Average MSE: " << avg_MSE << endl;
    return avg_R2;
}
