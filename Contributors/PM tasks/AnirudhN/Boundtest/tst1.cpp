#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// --- CSV Reader Function ---
// Reads a CSV file into an Eigen Matrix. 
// Assumes the file contains only numeric data (covariates).
// Set skip_header to true if the first row contains labels.
MatrixXd load_csv(const string& path, bool skip_header = false) {
    ifstream indata(path);
    if (!indata.is_open()) {
        throw runtime_error("Could not open file: " + path);
    }

    string line;
    vector<double> values;
    int rows = 0;
    
    // Skip header if requested
    if (skip_header) {
        getline(indata, line);
    }

    while (getline(indata, line)) {
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            // Remove any potential whitespace or newline chars
            cell.erase(remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
            if (!cell.empty()) {
                try {
                    values.push_back(stod(cell));
                } catch (const invalid_argument& e) {
                    cerr << "Warning: Skipping non-numeric value '" << cell << "' at row " << rows + 1 << endl;
                }
            }
        }
        ++rows;
    }

    if (rows == 0) return MatrixXd(0, 0);

    // Calculate columns (assuming consistent row lengths)
    int cols = values.size() / rows;
    
    // Map vector to Matrix (RowMajor is standard for row-by-row reading)
    return Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), rows, cols);
}

// --- Theoretical Bound Calculation (Theorem 2) ---
double calculate_log_theoretical_bound(int k, int p, const VectorXd& ranges) {
    double term1 = (p + 1) * std::log(k);
    double term2 = p * std::log(4.0);
    double term3 = 0.0;
    for (int i = 0; i < ranges.size(); ++i) {
        term3 += 2.0 * std::log(ranges[i]);
    }
    return term1 - term2 + term3;
}

// --- Log Determinant Calculation ---
double calculate_log_det(const MatrixXd& M) {
    LLT<MatrixXd> llt(M);
    if (llt.info() == Eigen::NumericalIssue) {
        throw runtime_error("Matrix is not positive definite!");
    }
    // Fix: Use matrixLLT() to access the underlying matrix diagonal
    return 2.0 * llt.matrixLLT().diagonal().array().log().sum();
}

// --- IBOSS Algorithm ---
vector<int> iboss_selection(const MatrixXd& Z, int k) {
    int n = Z.rows();
    int p = Z.cols();
    
    // Safety check
    if (k > n) {
        cerr << "Error: Subdata size k (" << k << ") cannot be larger than full data n (" << n << ")." << endl;
        return {};
    }

    int r = k / (2 * p); 
    if (r == 0) {
        cerr << "Warning: k is too small relative to p. r calculated as 0." << endl;
        r = 1; // Minimal fallback
    }

    vector<int> available_indices(n);
    iota(available_indices.begin(), available_indices.end(), 0);

    vector<bool> is_selected(n, false);
    vector<int> selected_indices;
    selected_indices.reserve(k);

    for (int j = 0; j < p; ++j) {
        vector<int> current_pool;
        current_pool.reserve(n - selected_indices.size());
        for(int idx : available_indices) {
             if(!is_selected[idx]) current_pool.push_back(idx);
        }

        if (current_pool.size() < 2 * r) {
            for(int idx : current_pool) {
                if(!is_selected[idx]) {
                    selected_indices.push_back(idx);
                    is_selected[idx] = true;
                }
            }
            break;
        }

        // Sort based on column j values
        std::sort(current_pool.begin(), current_pool.end(), 
            [&Z, j](int a, int b) {
                return Z(a, j) < Z(b, j);
            }
        );

        // Select r smallest
        for (int i = 0; i < r; ++i) {
            int idx = current_pool[i];
            selected_indices.push_back(idx);
            is_selected[idx] = true;
        }

        // Select r largest
        for (int i = 0; i < r; ++i) {
            int idx = current_pool[current_pool.size() - 1 - i];
            selected_indices.push_back(idx);
            is_selected[idx] = true;
        }
    }
    
    return selected_indices;
}

int main() {
    // --- Configuration ---
    const string filename = "C:/testEigen/chem_data.csv"; // Change this to your CSV file path
    const int k = 1000;                 // Desired Subdata size
    const bool has_header = false;      // Set to true if CSV has a header row

    try {
        cout << "Loading data from " << filename << "..." << endl;
        
        // 1. Load Data
        // Assumes file exists and contains only the Covariate Matrix Z
        MatrixXd Z = load_csv(filename, has_header);
        
        int n = Z.rows();
        int p = Z.cols();

        cout << "Data Loaded. Dimensions: n=" << n << ", p=" << p << endl;

        if (n == 0) {
            cerr << "Error: Dataset is empty." << endl;
            return 1;
        }

        // 2. Calculate Ranges (Max - Min per column) for Theorem 2
        VectorXd ranges(p);
        for (int j = 0; j < p; ++j) {
            double min_val = Z.col(j).minCoeff();
            double max_val = Z.col(j).maxCoeff();
            ranges(j) = max_val - min_val;
        }

        // 3. Calculate Theoretical Upper Bound
        double log_bound = calculate_log_theoretical_bound(k, p, ranges);

        // 4. Run IBOSS Selection
        cout << "Running IBOSS selection (k=" << k << ")..." << endl;
        vector<int> idx_iboss = iboss_selection(Z, k);

        if (idx_iboss.empty()) return 1;

        // 5. Construct Subdata Design Matrix X* (Add intercept)
        MatrixXd X_star(idx_iboss.size(), p + 1);
        for (size_t i = 0; i < idx_iboss.size(); ++i) {
            X_star(i, 0) = 1.0; 
            for (int j = 0; j < p; ++j) {
                X_star(i, j + 1) = Z(idx_iboss[i], j);
            }
        }

        // 6. Calculate Information Matrix: M = (X*)^T * X*
        MatrixXd M = X_star.transpose() * X_star;

        // 7. Calculate Results
        double log_det_iboss = calculate_log_det(M);
        double ratio = std::exp(log_det_iboss - log_bound);

        // 8. Output
        cout << string(60, '-') << endl;
        cout << left << setw(20) << "Metric" << "Value" << endl;
        cout << string(60, '-') << endl;
        cout << left << setw(20) << "Log Det (IBOSS)" << fixed << setprecision(4) << log_det_iboss << endl;
        cout << left << setw(20) << "Log Bound (Thm 2)" << log_bound << endl;
        cout << left << setw(20) << "Ratio (M / Bound)" << scientific << setprecision(6) << ratio << endl;
        cout << string(60, '-') << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
