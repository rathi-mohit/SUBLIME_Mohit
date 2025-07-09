#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class RidgeRegression {
private:
    VectorXd beta;
    double intercept;
    double lambda;
    VectorXd feature_means;
    VectorXd feature_stds;

public:
    RidgeRegression(double lambda_val = 0.1)
        : lambda(lambda_val) {}

    // Standardize features (mean=0, std=1)
    MatrixXd standardize_features(const MatrixXd& X, bool fit_transform = true) {
        MatrixXd X_std = X;
        if (fit_transform) {
            feature_means = X.colwise().mean();
            feature_stds = VectorXd(X.cols());
            X_std.rowwise() -= feature_means.transpose();
            for (int j = 0; j < X.cols(); ++j) {
                double variance = X_std.col(j).array().square().mean();
                feature_stds(j) = sqrt(variance);
                if (feature_stds(j) > 1e-8) {
                    X_std.col(j) /= feature_stds(j);
                } else {
                    feature_stds(j) = 1.0;
                }
            }
        } else {
            X_std.rowwise() -= feature_means.transpose();
            for (int j = 0; j < X.cols(); ++j) {
                if (feature_stds(j) > 1e-8) {
                    X_std.col(j) /= feature_stds(j);
                }
            }
        }
        return X_std;
    }

    // Fit the Ridge regression model using closed-form solution
    void fit(const MatrixXd& X, const VectorXd& y) {
        int n = X.rows();
        int p = X.cols();

        // Standardize features and store scaling parameters
        MatrixXd X_std = standardize_features(X, true);

        // Center target variable
        double y_mean = y.mean();
        VectorXd y_centered = y.array() - y_mean;

        // Closed-form solution: beta = (X^T X + lambda * I)^(-1) X^T y
        MatrixXd I = MatrixXd::Identity(p, p);
        beta = (X_std.transpose() * X_std + lambda * I).ldlt().solve(X_std.transpose() * y_centered);

        // Intercept is the mean of y
        intercept = y_mean;
    }

    // Make predictions
    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_std = X;
        X_std.rowwise() -= feature_means.transpose();
        for (int j = 0; j < X.cols(); ++j) {
            if (feature_stds(j) > 1e-8) {
                X_std.col(j) /= feature_stds(j);
            }
        }
        return X_std * beta + VectorXd::Constant(X.rows(), intercept);
    }

    VectorXd get_coefficients() const { return beta; }
    double get_intercept() const { return intercept; }

    double r_squared(const MatrixXd& X, const VectorXd& y) const {
        VectorXd predictions = predict(X);
        double y_mean = y.mean();
        double ss_res = (y - predictions).array().square().sum();
        double ss_tot = (y.array() - y_mean).array().square().sum();
        if (ss_tot < 1e-8) return 1.0;
        return 1.0 - (ss_res / ss_tot);
    }

    double mse(const MatrixXd& X, const VectorXd& y) const {
        VectorXd predictions = predict(X);
        return (y - predictions).array().square().mean();
    }
};

// CSV parsing and data loading functions remain unchanged
vector<string> parse_csv_line(const string& line) {
    vector<string> fields;
    stringstream ss(line);
    string field;
    while (getline(ss, field, ',')) {
        field.erase(0, field.find_first_not_of(" \t"));
        field.erase(field.find_last_not_of(" \t") + 1);
        fields.push_back(field);
    }
    return fields;
}

pair<MatrixXd, VectorXd> load_data(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    vector<vector<double>> data;
    string line;
    int line_number = 0;
    while (getline(file, line)) {
        line_number++;
        if (line.empty()) continue;
        vector<string> fields = parse_csv_line(line);
        if (fields.size() != 11) {
            cout << "Warning: Line " << line_number << " has " << fields.size()
                 << " fields, expected 11. Skipping." << endl;
            continue;
        }
        vector<double> row;
        try {
            for (const string& field : fields) {
                row.push_back(stod(field));
            }
            data.push_back(row);
        }
        catch (const exception& e) {
            cout << "Error parsing line " << line_number << ": " << e.what() << endl;
            continue;
        }
    }
    file.close();
    if (data.empty()) {
        throw runtime_error("No valid data found in file");
    }
    int n_samples = data.size();
    int n_features = 10;
    MatrixXd X(n_samples, n_features);
    VectorXd y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }
        y(i) = data[i][n_features];
    }
    cout << "Loaded " << n_samples << " samples with " << n_features << " features" << endl;
    return make_pair(X, y);
}

int main() {
    try {
        string filename;
        cout << "Enter the CSV filename (with 10 features + 1 target): ";
        cin >> filename;

        cout << "Loading data from " << filename << "..." << endl;
        auto data = load_data(filename);
        MatrixXd X = data.first;
        VectorXd y = data.second;

        cout << "Data Statistics:" << endl;
        cout << "Feature means: " << X.colwise().mean().transpose() << endl;
        cout << "Target mean: " << y.mean() << endl;
        cout << "Target std: " << sqrt((y.array() - y.mean()).square().mean()) << endl;

        double lambda;
        cout << "\nEnter lambda value for Ridge regularization (e.g., 0.1): ";
        cin >> lambda;

        cout << "Training Ridge regression model..." << endl;
        RidgeRegression ridge(lambda);
        ridge.fit(X, y);

        cout << "\n=== Ridge Regression Results ===" << endl;
        cout << "Lambda (regularization parameter): " << lambda << endl;
        cout << "Intercept: " << ridge.get_intercept() << endl;

        VectorXd coefficients = ridge.get_coefficients();
        cout << "\nFeature Coefficients:" << endl;
        for (int i = 0; i < coefficients.size(); ++i) {
            cout << "Feature " << (i + 1) << ": " << coefficients(i) << endl;
        }

        double r2 = ridge.r_squared(X, y);
        double mse_val = ridge.mse(X, y);

        cout << "\nModel Performance:" << endl;
        cout << "R-squared: " << r2 << endl;
        cout << "Mean Squared Error: " << mse_val << endl;
        cout << "Root Mean Squared Error: " << sqrt(mse_val) << endl;

        char make_prediction;
        cout << "\nDo you want to make a prediction on new data? (y/n): ";
        cin >> make_prediction;

        if (make_prediction == 'y' || make_prediction == 'Y') {
            cout << "Enter 10 feature values separated by spaces: ";
            vector<double> new_features(10);
            for (int i = 0; i < 10; ++i) {
                cin >> new_features[i];
            }
            MatrixXd new_X(1, 10);
            for (int i = 0; i < 10; ++i) {
                new_X(0, i) = new_features[i];
            }
            VectorXd prediction = ridge.predict(new_X);
            cout << "Predicted value: " << prediction(0) << endl;
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
