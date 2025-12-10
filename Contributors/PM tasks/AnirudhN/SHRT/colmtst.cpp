/*
 * Improved Subsampled Randomized Hadamard Transform (ISRHT)
 * EIGEN LIBRARY IMPLEMENTATION
 * * Features:
 * - CSV Data Loading (Hardcoded to C:/testEigen/Lasso.csv)
 * - Linear Regression R^2 Evaluation
 * - 3 Sampling Methods: Uniform, Top-r, Supervised (Binned)
 * * Dependencies: Eigen3 (http://eigen.tuxfamily.org/)
 * Compile: g++ -I C:/toolbox/eigen-3.4.0 isrht_eigen.cpp -o Hard_col
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <map>
#include <string>
#include <Eigen/Dense>

using MatrixRowMaj = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// --- Helper Functions ---

// 1. Next Power of 2
int nextPowerOfTwo(int n) {
    if ((n > 0) && ((n & (n - 1)) == 0)) return n;
    return std::pow(2, std::ceil(std::log2(n)));
}

// 2. Recursive Fast Walsh-Hadamard Transform
void fwht_recursive(double* a, int n) {
    if (n == 1) return;
    int half = n / 2;
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                double u = a[i + j];
                double v = a[i + len + j];
                a[i + j] = u + v;
                a[i + len + j] = u - v;
            }
        }
    }
}

// 3. Simple CSV Reader
// Assumes LAST column is the target variable (y)
// Returns pair: {Feature Matrix X, Target Vector y}
std::pair<Eigen::MatrixXd, Eigen::VectorXd> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string val_str;
        std::vector<double> row;
        
        while (std::getline(ss, val_str, ',')) {
            try {
                row.push_back(std::stod(val_str));
            } catch (...) {
                // Ignore non-numeric parsing errors (e.g., headers)
                continue;
            }
        }
        if (!row.empty()) data.push_back(row);
    }

    if (data.empty()) {
        std::cerr << "Error: No valid data found in CSV." << std::endl;
        exit(1);
    }

    int rows = data.size();
    int cols = data[0].size();
    
    // Safety check for single column files
    if (cols < 2) {
         std::cerr << "Error: CSV must have at least 1 feature column and 1 target column." << std::endl;
         exit(1);
    }

    int feature_cols = cols - 1; // Last col is target

    Eigen::MatrixXd X(rows, feature_cols);
    Eigen::VectorXd y(rows);

    for (int i = 0; i < rows; i++) {
        if (data[i].size() != cols) continue; // Skip inconsistent rows
        for (int j = 0; j < feature_cols; j++) {
            X(i, j) = data[i][j];
        }
        y(i) = data[i][feature_cols];
    }

    return {X, y};
}

// 4. Bin Continuous Targets for Supervised Method
std::vector<int> bin_continuous_targets(const Eigen::VectorXd& y, int n_bins) {
    int n = y.size();
    std::vector<int> labels(n);
    double min_y = y.minCoeff();
    double max_y = y.maxCoeff();
    double range = max_y - min_y;

    if (range < 1e-9) return std::vector<int>(n, 0);

    for(int i = 0; i < n; i++) {
        int bin = (int)((y[i] - min_y) / range * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        labels[i] = bin;
    }
    return labels;
}

// 5. Calculate Linear Regression R^2
double calculate_ols_r2(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int n = X.rows();
    int k = X.cols();

    // Add intercept term (column of 1s)
    Eigen::MatrixXd X_bias(n, k + 1);
    X_bias.col(0) = Eigen::VectorXd::Ones(n);
    X_bias.block(0, 1, n, k) = X;

    // Solve OLS: w = (X^T X)^-1 X^T y
    // Using robust ColPivHouseholderQR decomposition
    Eigen::VectorXd w = X_bias.colPivHouseholderQr().solve(y);

    // Predictions
    Eigen::VectorXd y_pred = X_bias * w;

    // R^2 Calculation
    double y_mean = y.mean();
    double ss_tot = (y.array() - y_mean).square().sum();
    double ss_res = (y.array() - y_pred.array()).square().sum();

    if (ss_tot < 1e-9) return 0.0; // Avoid NaN for constant target
    return 1.0 - (ss_res / ss_tot);
}


// --- Main Class ---

class ISRHT {
private:
    std::mt19937 rng;
    // Eigen::VectorXd random_signs; // Removed: Generating fresh signs per row now

public:
    ISRHT(int seed = 42) {
        rng.seed(seed);
    }

    // Rotate Data
    MatrixRowMaj rotateData(const Eigen::MatrixXd& X) {
        int n = X.rows();
        int d = X.cols();
        int padded_d = nextPowerOfTwo(d);

        // Prepare Distribution
        std::uniform_int_distribution<> dist(0, 1);

        MatrixRowMaj X_rotated = MatrixRowMaj::Zero(n, padded_d);
        X_rotated.block(0, 0, n, d) = X;
        
        // No broadcasting of fixed signs here.
        // We iterate per row, apply fresh signs, then transform.

        double scale = 1.0 / std::sqrt((double)padded_d);
        
        for (int i = 0; i < n; i++) {
            // Generate and apply NEW random signs for this specific row
            // We only need to flip signs for the actual data columns (d)
            for (int j = 0; j < d; j++) {
                double sign = (dist(rng) == 0) ? 1.0 : -1.0;
                X_rotated(i, j) *= sign;
            }

            fwht_recursive(X_rotated.row(i).data(), padded_d);
        }
        X_rotated *= scale;
        return X_rotated;
    }

    // 1. Uniform
    Eigen::MatrixXd fit_transform_uniform(const MatrixRowMaj& X_rotated, int r) {
        int n = X_rotated.rows();
        int d = X_rotated.cols();
        std::vector<int> indices(d);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        Eigen::MatrixXd X_new(n, r);
        double scale_factor = std::sqrt((double)d / r);

        for (int j = 0; j < r; j++) {
            X_new.col(j) = X_rotated.col(indices[j]) * scale_factor;
        }
        return X_new;
    }

    // 2. Top-r
    Eigen::MatrixXd fit_transform_top_r(const MatrixRowMaj& X_rotated, int r) {
        int d = X_rotated.cols();
        Eigen::VectorXd norms = X_rotated.colwise().squaredNorm();
        std::vector<std::pair<double, int>> col_norms(d);
        for (int i = 0; i < d; i++) {
            col_norms[i] = {norms[i], i};
        }
        std::sort(col_norms.rbegin(), col_norms.rend());

        Eigen::MatrixXd X_new(X_rotated.rows(), r);
        for (int j = 0; j < r; j++) {
            X_new.col(j) = X_rotated.col(col_norms[j].second);
        }
        return X_new;
    }

    // 3. Supervised
    Eigen::MatrixXd fit_transform_supervised(const MatrixRowMaj& X_rotated, const std::vector<int>& labels, int r, double a_param = 1.0) {
        int n = X_rotated.rows();
        int d = X_rotated.cols();
        
        // Find unique labels
        std::vector<int> unique_labels = labels;
        std::sort(unique_labels.begin(), unique_labels.end());
        unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
        
        Eigen::MatrixXd class_sums(unique_labels.size(), d);
        Eigen::MatrixXd class_sq_sums(unique_labels.size(), d);
        std::vector<int> class_counts(unique_labels.size());

        for (size_t c = 0; c < unique_labels.size(); ++c) {
            int label = unique_labels[c];
            std::vector<int> indices;
            for(int i=0; i<n; ++i) if(labels[i] == label) indices.push_back(i);
            class_counts[c] = indices.size();

            Eigen::MatrixXd X_class(indices.size(), d);
            for(size_t k=0; k<indices.size(); ++k) {
                X_class.row(k) = X_rotated.row(indices[k]);
            }
            class_sums.row(c) = X_class.colwise().sum();
            class_sq_sums.row(c) = X_class.array().square().colwise().sum();
        }

        Eigen::VectorXd total_sums = X_rotated.colwise().sum();
        Eigen::VectorXd term_Av = Eigen::VectorXd::Zero(d);
        Eigen::VectorXd term_Dv = Eigen::VectorXd::Zero(d);

        for (size_t c = 0; c < unique_labels.size(); ++c) {
            term_Av += (class_sums.row(c).array().square()).matrix();
        }
        term_Av *= (1.0 + a_param);
        term_Av -= a_param * (total_sums.array().square()).matrix();

        for (size_t c = 0; c < unique_labels.size(); ++c) {
            double coeff = (double)class_counts[c] - a_param * (n - (double)class_counts[c]);
            term_Dv += coeff * class_sq_sums.row(c);
        }

        Eigen::VectorXd b_scores = term_Dv - term_Av;
        std::vector<std::pair<double, int>> sorted_scores(d);
        for(int j=0; j<d; ++j) sorted_scores[j] = {b_scores[j], j};
        std::sort(sorted_scores.begin(), sorted_scores.end());

        Eigen::MatrixXd X_new(n, r);
        for (int j = 0; j < r; j++) {
            X_new.col(j) = X_rotated.col(sorted_scores[j].second);
        }
        return X_new;
    }
};

int main() {
    // --- Configuration ---
    // Using forward slashes is standard C++ and usually works on Windows.
    // If it fails, try "C:\\testEigen\\Lasso.csv"
    std::string filename = "C:/testEigen/normal_data.csv"; 
    
    int r = 200;                         // Target reduced dimension
    int supervised_bins = 5000;           // Bins for regression -> classification mapping
    // ---------------------

    std::cout << "1. Loading Data from " << filename << "..." << std::endl;
    
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    
    // --- CSV Loading Block ---
    // Reads directly. If file fails, readCSV will print error and exit(1)
    auto data_pair = readCSV(filename); 
    X = data_pair.first; 
    y = data_pair.second;

    std::cout << "   Shape: " << X.rows() << " samples x " << X.cols() << " features" << std::endl;
    std::cout << "   Target Reduced Dimension (r): " << r << std::endl;

    // Run Rotation
    ISRHT algo(123);
    std::cout << "2. Rotating Data..." << std::endl;
    MatrixRowMaj X_rot = algo.rotateData(X);

    // --- Method 1: Uniform ---
    std::cout << "\n--- Method 1: Uniform Sampling ---" << std::endl;
    Eigen::MatrixXd X_unif = algo.fit_transform_uniform(X_rot, r);
    double r2_unif = calculate_ols_r2(X_unif, y);
    std::cout << "R^2 Score: " << r2_unif << std::endl;

    // --- Method 2: Top-r ---
    std::cout << "\n--- Method 2: Deterministic Top-r ---" << std::endl;
    Eigen::MatrixXd X_top = algo.fit_transform_top_r(X_rot, r);
    double r2_top = calculate_ols_r2(X_top, y);
    std::cout << "R^2 Score: " << r2_top << std::endl;

    // --- Method 3: Supervised (Binned) ---
    std::cout << "\n--- Method 3: Supervised (Binned Metric Learning) ---" << std::endl;
    // Bin continuous y into discrete labels
    std::vector<int> labels = bin_continuous_targets(y, supervised_bins);
    Eigen::MatrixXd X_sup = algo.fit_transform_supervised(X_rot, labels, r, 1.0);
    double r2_sup = calculate_ols_r2(X_sup, y);
    std::cout << "R^2 Score: " << r2_sup << std::endl;

    return 0;
}Colum   
