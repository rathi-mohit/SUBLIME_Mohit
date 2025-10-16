#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>   // Required for file I/O
#include <sstream>   // Required for string parsing
#include <vector>    // Required to temporarily hold data
#include <string>    // Required for string manipulation
#include <stdexcept> // Required for error handling

// This is the header for the divide-and-conquer i-BOSS implementation
#include "divconib.hpp"

using namespace Eigen;
using namespace std;

/**
 * @brief Reads a CSV file into an Eigen Matrix and Vector.
 *
 * This function assumes:
 * 1. The CSV file contains only numeric data without headers.
 * 2. The last column is the response variable (y).
 * 3. All preceding columns are features.
 * An intercept column (a column of ones) is automatically added to the feature matrix X.
 *
 * @param filepath The path to the CSV file.
 * @return A pair containing the design matrix X (with intercept) and the response vector y.
 */
pair<MatrixXd, VectorXd> read_csv(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        throw runtime_error("Error: Could not open file " + filepath);
    }

    vector<vector<double>> data;
    string line;

    // Read file line by line
    while (getline(file, line)) {
        if (line.empty()) continue; // Skip empty lines
        stringstream lineStream(line);
        string cell;
        vector<double> row;
        while (getline(lineStream, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const invalid_argument& e) {
                throw runtime_error("Error: Non-numeric value '" + cell + "' found in CSV.");
            }
        }
        data.push_back(row);
    }
    file.close();

    if (data.empty() || data[0].size() < 2) {
        throw runtime_error("Error: CSV file must have at least two columns (one feature, one response).");
    }

    size_t rows = data.size();
    size_t feature_cols = data[0].size() - 1;

    // Create matrix X with an added intercept column, and vector y
    MatrixXd X(rows, feature_cols + 1);
    VectorXd y(rows);

    for (size_t i = 0; i < rows; ++i) {
        if (data[i].size() != feature_cols + 1) {
             throw runtime_error("Error: Inconsistent number of columns in CSV file at row " + to_string(i + 1));
        }
        X(i, 0) = 1.0; // Set the intercept term to 1
        for (size_t j = 0; j < feature_cols; ++j) {
            X(i, j + 1) = data[i][j]; // Features start from column 1 of X
        }
        y(i) = data[i][feature_cols]; // The last column of the data is the response
    }

    return make_pair(X, y);
}


int main(int argc, char* argv[]) {
    // --- 1. Check for Command-Line Arguments ---
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <path_to_csv> [k] [num_chunks]" << endl;
        cerr << "  <path_to_csv>: Required. Path to the input data file." << endl;
        cerr << "  [k]: Optional. Desired final subset size (default: 2000)." << endl;
        cerr << "  [num_chunks]: Optional. Number of parallel chunks (default: 8)." << endl;
        return 1;
    }
    string filepath = argv[1];

    // --- 2. Load Data from CSV ---
    MatrixXd X;
    VectorXd y;
    try {
        pair<MatrixXd, VectorXd> data = read_csv(filepath);
        X = data.first;
        y = data.second;
        cout << "Successfully loaded data from " << filepath << "." << endl;
        cout << "Dimensions: N=" << X.rows() << ", p=" << X.cols() << " (including intercept)" << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }

    // --- 3. Set Parameters for i-BOSS ---
    int k = 2000;
    int num_chunks = 8;
    
    // Allow user to override default k and num_chunks from command line
    if (argc > 2) k = stoi(argv[2]);
    if (argc > 3) num_chunks = stoi(argv[3]);

    cout << "\nRunning Divide and Conquer i-BOSS..." << endl;
    cout << "  - Final subset size (k): " << k << endl;
    cout << "  - Number of parallel chunks: " << num_chunks << endl;
    
    // --- 4. Run the Algorithm and Time it ---
    auto start_time = chrono::high_resolution_clock::now();

    pair<MatrixXd, VectorXd> result;
    try {
        result = divide_and_conquer_iboss(X, y, k, num_chunks);
    } catch (const exception& e) {
        cerr << "\nAn error occurred during i-BOSS execution: " << e.what() << endl;
        return 1;
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    MatrixXd X_subset = result.first;
    VectorXd y_subset = result.second;

    // --- 5. Calculate and Print BLUE Parameters ---
    cout << "\n--- Calculating BLUE Parameters (OLS Coefficients) ---" << endl;
    VectorXd beta;
    if (X_subset.rows() > 0) {
        // Solve for beta using the subset via a stable Cholesky decomposition
        beta = (X_subset.transpose() * X_subset).ldlt().solve(X_subset.transpose() * y_subset);
        
        cout << "Calculated Beta Coefficients:" << endl;
        cout << "---------------------------" << endl;
        for (int i = 0; i < beta.size(); ++i) {
            cout << "beta(" << i << "): \t" << beta(i) << (i == 0 ? " (Intercept)" : "") << endl;
        }
        cout << "---------------------------" << endl;
    } else {
        cout << "Warning: The resulting subset is empty, cannot calculate parameters." << endl;
    }


    // --- 6. Print i-BOSS Summary ---
    cout << "\n--- i-BOSS Summary ---" << endl;
    cout << "Algorithm completed in: " << duration.count() << " ms" << endl;
    cout << "Original dimensions: (" << X.rows() << ", " << X.cols() << ")" << endl;
    cout << "Subset dimensions:   (" << X_subset.rows() << ", " << X_subset.cols() << ")" << endl;

    if (X_subset.rows() > 0) {
        cout << "Success: The subset was generated with " << X_subset.rows() << " rows." << endl;
    } else {
        cout << "Warning: The resulting subset is empty." << endl;
    }

    return 0;
}
