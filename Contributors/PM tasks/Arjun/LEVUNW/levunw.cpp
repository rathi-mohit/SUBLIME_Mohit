#include <iostream>

#include <vector>
#include <string>
#include <iomanip>
#include <typeinfo>

#include <fstream>
#include <sstream>

#include <ctime>

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <set>

#include <random>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class LenUnwSelector
{
private:
    const MatrixXd& X;
    const VectorXd& y;
    int k;
    int p, N;
    bool has_intercept;

    // Random sampling of k values using Bob Floyd algorithm
    std::unordered_set<size_t> floyd_sample(const size_t& k, const size_t& N)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::unordered_set<size_t> elems;
        for (size_t r = N - k; r < N; ++r)
        {
            size_t v = std::uniform_int_distribution<size_t>(0, r)(gen);
            if (!elems.insert(v).second) elems.insert(r);
        }
        return elems;
    }

    pair<MatrixXd, VectorXd> leveraged_sample()
    {
        auto selected = floyd_sample(k, N);

        MatrixXd X_(k, p);
        VectorXd y_(k);
        int i = 0;
        for (int idx : selected)
        {
            X_.row(i) = X.row(idx);
            y_(i) = y(idx);
            i++;
        }

        return make_pair(X_, y_);
    }

    std::vector<int> leverage_weighted_sample()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<int> selected_indices;
        std::unordered_set<int> selected_set;
        
        // Create cumulative distribution
        std::vector<double> cumulative_probs(N);
        cumulative_probs[0] = leverage_scores(0);
        for (int i = 1; i < N; ++i) {
            cumulative_probs[i] = cumulative_probs[i-1] + leverage_scores(i);
        }
        
        // Sample k unique indices
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        while (selected_indices.size() < k) {
            double rand_val = dis(gen);
            
            // Find the index corresponding to this random value
            auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), rand_val);
            int idx = std::distance(cumulative_probs.begin(), it);
            
            // Ensure we don't go out of bounds
            if (idx >= N) idx = N - 1;
            
            // Add to selection if not already selected
            if (selected_set.insert(idx).second) {
                selected_indices.push_back(idx);
            }
        }
        
        return selected_indices;
    }

public:
    LenUnwSelector(const MatrixXd & X, const VectorXd & y, int k) : X(X), y(y), k(k)
        {
            p = X.cols();
            N = X.rows();
        }

    MatrixXd hii_matrix = X*(X.transpose()*X).inverse()*X.transpose();
    VectorXd hii_diag = hii_matrix.diagonal();
    VectorXd hii = hii_diag.sum();

    pair<MatrixXd, VectorXd> get_sample()
    {
        return leveraged_sample();
    }

};

int main()
{
    string filename = "household_power_consumption.txt";
    df dframe = readCSV(filename);

    auto xy = convert(dframe);
    MatrixXd X = xy.first;
    VectorXd y = xy.second;

    LevUnw = LevUnwSelector(X, y, 50000)
    auto [X_, y_] = LevUnw.get_sample();

    // Run tests
    // Note that for LevUnw w is constant
    VectorXd beta = (X_sampled.transpose() * X_sampled).inverse() * X_sampled.transpose() * y_sampled;
    VectorXd beta_levunw = (X_.transpose()*X_).inverse() * X_.transpose() * y_;

    auto pred = X * beta;
    auto pred_levunw = X_ * beta_levunw;

    auto delta = y - pred;
    auto delta_levunw = y_ - pred_levunw;

    cout << delta.norm() << endl;
    cout << delta_levunw.norm() << endl;

    cin.get();
    return 0;
}
