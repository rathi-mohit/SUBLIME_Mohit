#ifndef DC_IBOSS_HPP
#define DC_IBOSS_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <omp.h>
#include "iboss.hpp" // Using the provided k_selection implementation

using namespace Eigen;
using namespace std;

/**
 * @brief Performs i-BOSS subset selection using a divide and conquer strategy.
 * * The dataset is split into 'num_chunks'. The i-BOSS algorithm is run on each chunk in parallel
 * to select an intermediate, oversampled subset. These intermediate subsets are then merged,
 * and the i-BOSS algorithm is run a final time to produce the final subset of size k.
 *
 * @param X The full design matrix (N x p).
 * @param y The full response vector (N x 1).
 * @param k The final desired subset size.
 * @param num_chunks The number of chunks to divide the data into for parallel processing.
 * @return A pair containing the selected subset matrix X_iboss (k x p) and vector y_iboss (k x 1).
 */
pair<MatrixXd, VectorXd> divide_and_conquer_iboss(const MatrixXd &X, const VectorXd &y, int k, int num_chunks) {
    const int N = X.rows();
    const int p = X.cols();

    // --- Input Validation ---
    if (num_chunks <= 0) {
        throw invalid_argument("Number of chunks must be positive.");
    }
    if (k <= 0) {
        throw invalid_argument("Desired subset size k must be positive.");
    }
    if (N < k) {
        throw invalid_argument("Total number of rows N cannot be smaller than the desired subset size k.");
    }
    if (N < num_chunks) {
        throw invalid_argument("Number of rows N cannot be smaller than the number of chunks.");
    }

    // Base case: If only one chunk, run standard i-BOSS directly.
    if (num_chunks == 1) {
        return k_selection(X, y, k);
    }

    // --- 1. Divide Step ---
    vector<int> chunk_sizes(num_chunks);
    vector<int> start_indices(num_chunks + 1, 0);
    int chunk_size_base = N / num_chunks;
    int remainder = N % num_chunks;

    for (int i = 0; i < num_chunks; ++i) {
        chunk_sizes[i] = chunk_size_base + (i < remainder ? 1 : 0);
        start_indices[i+1] = start_indices[i] + chunk_sizes[i];
    }

    // --- 2. Conquer Step (in Parallel) ---
    // This vector will hold the results from running i-BOSS on each chunk.
    vector<pair<MatrixXd, VectorXd>> intermediate_results(num_chunks);
    
    // Oversampling factor ensures we have a rich pool for the final selection
    const double oversampling_factor = 2.0; 

    #pragma omp parallel for
    for (int i = 0; i < num_chunks; ++i) {
        // Create a view or copy of the data chunk
        MatrixXd X_chunk = X.block(start_indices[i], 0, chunk_sizes[i], p);
        VectorXd y_chunk = y.segment(start_indices[i], chunk_sizes[i]);

        // Determine the subset size for this chunk
        int k_sub = static_cast<int>(oversampling_factor * k / num_chunks);
        if (k_sub < 2 * p) k_sub = 2 * p; // Ensure k_sub is meaningful
        if (k_sub > chunk_sizes[i]) k_sub = chunk_sizes[i]; // Cannot select more than available

        intermediate_results[i] = k_selection(X_chunk, y_chunk, k_sub);
    }

    // --- 3. Combine Step ---
    // Calculate total size of the combined intermediate dataset
    int total_intermediate_rows = 0;
    for (int i = 0; i < num_chunks; ++i) {
        total_intermediate_rows += intermediate_results[i].first.rows();
    }

    // Create a new matrix and vector to hold the merged data
    MatrixXd X_combined(total_intermediate_rows, p);
    VectorXd y_combined(total_intermediate_rows);

    // Copy data from intermediate results into the combined containers
    int current_row = 0;
    for (int i = 0; i < num_chunks; ++i) {
        int rows_in_chunk = intermediate_results[i].first.rows();
        X_combined.block(current_row, 0, rows_in_chunk, p) = intermediate_results[i].first;
        y_combined.segment(current_row, rows_in_chunk) = intermediate_results[i].second;
        current_row += rows_in_chunk;
    }

    // Final reduction step: run i-BOSS on the combined data to get the final subset
    return k_selection(X_combined, y_combined, k);
}

#endif // DC_IBOSS_HPP
