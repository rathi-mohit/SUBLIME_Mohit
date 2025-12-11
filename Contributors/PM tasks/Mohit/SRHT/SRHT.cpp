#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>
#include <Eigen/Dense>
#include <chrono>

#include "./lib/utils.hpp"
#include "./lib/regression_models.hpp"

using namespace Eigen;
using namespace std;

size_t ceilLog2(size_t n) {
    // https://stackoverflow.com/questions/1322510/given-an-integer-how-do-i-find-the-next-largest-power-of-two-using-bit-twiddlin/1322548#1322548
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

void fwht_parallel(Ref<VectorXd> v) {
    size_t n = v.size();

    // https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

    for (size_t h = 1; h < n; h <<= 1) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i += (h << 1)) {
            for (size_t j = i; j < i + h; ++j) {
                double x = v[j];
                double y = v[j + h];
                v[j]     = x + y;
                v[j + h] = x - y;
            }
        }
    }
}

MatrixXd srht_row(MatrixXd& X_in, int r) {
    size_t n = X_in.rows();
    size_t p = X_in.cols();

    size_t N = ceilLog2(n);
    MatrixXd X_ceil = MatrixXd::Zero(N, p);
    X_ceil.topRows(n) = X_in;

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1);

    VectorXd signs(N);
    for (size_t i = 0; i < N; ++i)
        signs[i] = (dist(gen) == 0 ? 1 : -1);

    X_ceil = X_ceil.array().colwise() * signs.array();
    for (size_t j = 0; j < p; ++j) {
        fwht_parallel(X_ceil.col(j));
    }

    vector<int> idxs(N); 
    iota(idxs.begin(), idxs.end(), 0);
    shuffle(idxs.begin(), idxs.end(), gen);

    double sf = 1.0 / sqrt(double(r));
    MatrixXd X_r(r, p);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < r; ++i) {
        X_r.row(i) = X_ceil.row(idxs[i]) * sf;
    }

    return X_r;
}

MatrixXd srht_col(MatrixXd& X_in, int r) {
    MatrixXd X_T = X_in.transpose();
    MatrixXd X_T_r = srht_row(X_T, r);
    return X_T_r.transpose();
}

int main() {
    auto data = clean_numerical_load_csv("data.csv");
    MatrixXd X(data.rows(), data.cols());
    X.col(0) = VectorXd::Constant(data.rows(), 1);
    X.rightCols(data.cols()-1) = data.leftCols(data.cols()-1);
    VectorXd y = data.rightCols(1);

    int r = 500;
    auto start = chrono::high_resolution_clock::now();
    MatrixXd X_r = srht_row(X, r);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << duration.count() << " seconds\n";
    return 0;
}
