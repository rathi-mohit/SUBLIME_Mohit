#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <random>
using namespace std;
using namespace Eigen;

VectorXd LEV(MatrixXd X, VectorXd y, int k)
{
    
    vector<double> pi(X.rows());
    VectorXd wt=VectorXd::Zero(X.rows());
    MatrixXd calc_temp=(X.transpose()*X).ldlt().solve(MatrixXd::Identity(X.cols(), X.cols()));
    for(int i=0; i<X.rows(); i++)
    {
        pi[i] = (X.row(i)*(calc_temp)*X.row(i).transpose())(0,0); 
        //not dividing by p+1 since it doesn't matter, pi is used as a weight
    }

    random_device rd;
    mt19937 mt(rd());
    std::discrete_distribution<> dist(pi.begin(), pi.end());

    for (int i=0; i<k; ++i)
    {
        auto temp=dist(mt);
        wt(temp)+=1/pi[temp];
    }

    return (X.transpose()*wt.asDiagonal()*X).ldlt().solve((X.transpose()*wt.asDiagonal()*y));

}

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
