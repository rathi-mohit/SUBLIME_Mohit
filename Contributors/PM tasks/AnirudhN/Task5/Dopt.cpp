#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <Eigen/Dense>
#include<numeric>
using namespace std;
using namespace Eigen;

vector<string> parse(const string& lin) {
    stringstream ss(lin);
    vector<string> Fields;
    string field;
    while (getline(ss, field, ',')) {
        field.erase(0, field.find_first_not_of("\t"));
        field.erase(field.find_last_not_of("\t") + 1);
        Fields.push_back(field);
    }
    return Fields;
}

// Helper: get sorted indices (ascending or descending)
vector<int> get_sorted_indices(const VectorXf& col, bool descending) {
    vector<int> indices(col.size());
    iota(indices.begin(), indices.end(), 0);
    if (descending)
        sort(indices.begin(), indices.end(), [&](int a, int b){ return col(a) > col(b); });
    else
        sort(indices.begin(), indices.end(), [&](int a, int b){ return col(a) < col(b); });
    return indices;
}

int main() {
    MatrixXf mat(199999, 6);
    string file;
    cout << "Enter filename: ";
    cin >> file;
    ifstream fl(file);
    if (!fl.is_open()) {
        cout << "file not open" << endl;
        return 0;
    }
    long long int ct = 1, len = 0;
    string line;
    getline(fl, line); // skip header

    while (getline(fl, line)) {
        vector<string> ln = parse(line);
        if (ln.size() < 9) { ct++; continue; }
        try {
            for (int i = 4; i < ln.size(); i++)
                mat(len, i - 4) = stof(ln[i]);
            mat(len, 5) = stof(ln[2]);
            len++;
        } catch (const std::exception& e) {
            cerr << "Conversion error at line " << ct << ": " << e.what() << " for value: " << ln[1] << endl;
        }
        ct++;
    }

    long long int k;
    cout << "Enter K value: ";
    cin >> k;

    int p = 5; // number of features
    if (k < 2 * p) {
        cout << "K too small for selection rule";
        return 1;
    }
    if (k > len) {
        cout << "K too large for available data";
        return 1;
    }

    int num_each = k / (2 * p); // number to select per feature per side
    set<int> selected_indices;
    vector<int> low_ptr(p, 0), high_ptr(p, 0);

    // Precompute sorted indices for each feature
    vector<vector<int>> sorted_low(p), sorted_high(p);
    for (int feat = 0; feat < p; ++feat) {
        VectorXf col = mat.topRows(len).col(feat);
        sorted_low[feat] = get_sorted_indices(col, false);
        sorted_high[feat] = get_sorted_indices(col, true);
    }

    // Loop until we have k unique indices
    while ((int)selected_indices.size() < k) {
        for (int feat = 0; feat < p; ++feat) {
            // Add next batch of lowest
            int added = 0;
            while (low_ptr[feat] < len && added < num_each && (int)selected_indices.size() < k) {
                int idx = sorted_low[feat][low_ptr[feat]++];
                if (selected_indices.insert(idx).second)
                    ++added;
            }
            // Add next batch of highest
            added = 0;
            while (high_ptr[feat] < len && added < num_each && (int)selected_indices.size() < k) {
                int idx = sorted_high[feat][high_ptr[feat]++];
                if (selected_indices.insert(idx).second)
                    ++added;
            }
            if ((int)selected_indices.size() >= k) break;
        }
    }

    // Extract selected subset for regression
    MatrixXf X_sel(selected_indices.size(), p);
    VectorXf y_sel(selected_indices.size());
    int row = 0;
    for (int idx : selected_indices) {
        X_sel.row(row) = mat.row(idx).head(p);
        y_sel(row) = mat(idx, 5);
        ++row;
    }

    // Linear regression
    MatrixXf X_with_intercept(X_sel.rows(), p + 1);
    X_with_intercept.col(0) = VectorXf::Ones(X_sel.rows());
    X_with_intercept.block(0, 1, X_sel.rows(), p) = X_sel;

    VectorXf beta = X_with_intercept.colPivHouseholderQr().solve(y_sel);

    cout << "Linear regression coefficients (on selected subset):" << endl;
    cout << "Intercept: " << beta(0) << endl;
    for (int i = 1; i < beta.size(); ++i) {
        cout << "Feature " << i << ": " << beta(i) << endl;
    }

    // Optional: R^2
    VectorXf y_pred = X_with_intercept * beta;
    float ss_tot = (y_sel.array() - y_sel.mean()).square().sum();
    float ss_res = (y_sel - y_pred).squaredNorm();
    float r2 = 1.0f - ss_res / ss_tot;
    cout << "R^2: " << r2 << endl;

    return 0;
}
