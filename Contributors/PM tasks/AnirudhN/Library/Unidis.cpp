#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>

using namespace Eigen;

// Function to create indicator vector eta_L for sampled indices
VectorXi sampleIndicator(int n, int sample_size, bool withReplacement = true) {
    VectorXi eta = VectorXi::Zero(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    if (withReplacement) {
        for (int i = 0; i < sample_size; ++i) {
            int idx = dis(gen);
            eta(idx) += 1;   // mark as selected
        }
    } else {
        // Without replacement
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int i = 0; i < sample_size; ++i) {
            eta(indices[i]) = 1;
        }
    }

    return eta;
}

// Subsampling-based BLUE estimator
VectorXf subsampleEstimator(const MatrixXf &X, const VectorXf &y, 
                            const VectorXi &eta) {
    int n = X.rows();
    int p = X.cols();

    MatrixXf A = MatrixXf::Zero(p, p); // ∑ w_i η_i x_i x_i^T
    VectorXf b = VectorXf::Zero(p);    // ∑ w_i η_i x_i y_i

    float w = 1.0;   // uniform weight

    for (int i = 0; i < n; ++i) {
        for(long long int j=0;j<eta(i);j++) {
            VectorXf xi = X.row(i);
            A += w * (xi * xi.transpose());
            b += w * xi * y(i);
        }
    }

    return A.ldlt().solve(b);
}

int main() {
    int n = 100;  // number of samples
    int p = 5;    // number of parameters
    int m = 30;   // number of subsamples

    // Generate synthetic data
    MatrixXf X = MatrixXf::Random(n, p);
    VectorXf beta_true(p);
    beta_true << 1, -2, 0.5, 3, -1;  // ground truth
    VectorXf y = X * beta_true + 0.1 * VectorXf::Random(n);

    // Sample indicator (uniform subsampling, without replacement)
    VectorXi eta = sampleIndicator(n, m, false);

    // Compute estimator
    VectorXf beta_hat = subsampleEstimator(X, y, eta);

    std::cout << "Estimated beta (subsample BLUE):\n" 
              << beta_hat.transpose() << std::endl;
    std::cout << "True beta:\n" << beta_true.transpose() << std::endl;

    return 0;
}
