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

class LassoRegression {
private:
    VectorXd beta;
    double intercept;
    double lambda;
    int max_iterations;
    double tolerance;
    VectorXd feature_means;  // Store means for prediction
    VectorXd feature_stds;   // Store standard deviations for prediction

public:
    LassoRegression(double lambda_val = 0.1, int max_iter = 1000, double tol = 1e-6)
        : lambda(lambda_val), max_iterations(max_iter), tolerance(tol) {}

    // Soft thresholding function for Lasso
    double soft_threshold(double x, double threshold) {
        if (x > threshold) return x - threshold;
        else if (x < -threshold) return x + threshold;
        else return 0.0;
    }

    // Standardize features (mean=0, std=1)
    MatrixXd standardize_features(const MatrixXd& X, bool fit_transform = true) {
        MatrixXd X_std = X;
        
        if (fit_transform) {
            // Calculate and store means and standard deviations
            feature_means = X.colwise().mean();
            feature_stds = VectorXd(X.cols());
            
            // Center the data
            X_std.rowwise() -= feature_means.transpose();
            
            // Calculate standard deviations
            for (int j = 0; j < X.cols(); ++j) {
                double variance = X_std.col(j).array().square().mean();
                feature_stds(j) = sqrt(variance);
                if (feature_stds(j) > 1e-8) {
                    X_std.col(j) /= feature_stds(j);
                } else {
                    feature_stds(j) = 1.0;  // Avoid division by zero
                }
            }
        } else {
            // Use stored means and standard deviations for prediction
            X_std.rowwise() -= feature_means.transpose();
            for (int j = 0; j < X.cols(); ++j) {
                if (feature_stds(j) > 1e-8) {
                    X_std.col(j) /= feature_stds(j);
                }
            }
        }
        
        return X_std;
    }

    // Fit the Lasso regression model using coordinate descent
    void fit(const MatrixXd& X, const VectorXd& y) {
        int n = X.rows();
        int p = X.cols();
        
        // Standardize features and store scaling parameters
        MatrixXd X_std = standardize_features(X, true);
        
        // Center target variable
        double y_mean = y.mean();
        VectorXd y_centered = y.array() - y_mean;
        
        // Initialize coefficients
        beta = VectorXd::Zero(p);
        intercept = y_mean;
        
        // Coordinate descent algorithm
        for (int iter = 0; iter < max_iterations; ++iter) {
            VectorXd beta_old = beta;
            
            for (int j = 0; j < p; ++j) {
                // Calculate residual without j-th feature
                VectorXd r = y_centered - X_std * beta + beta(j) * X_std.col(j);
                
                // Calculate correlation with j-th feature
                double rho = X_std.col(j).dot(r) / n;
                
                // Update j-th coefficient using soft thresholding
                beta(j) = soft_threshold(rho, lambda / n);
            }
            
            // Check for convergence
            if ((beta - beta_old).norm() < tolerance) {
                cout << "Converged after " << iter + 1 << " iterations" << endl;
                break;
            }
        }
    }

    // Make predictions
    VectorXd predict(const MatrixXd& X) const {
        MatrixXd X_std = X;
        
        // Apply same standardization as training data
        X_std.rowwise() -= feature_means.transpose();
        for (int j = 0; j < X.cols(); ++j) {
            if (feature_stds(j) > 1e-8) {
                X_std.col(j) /= feature_stds(j);
            }
        }
        
        return X_std * beta + VectorXd::Constant(X.rows(), intercept);
    }

    // Get coefficients
    VectorXd get_coefficients() const { return beta; }
    double get_intercept() const { return intercept; }

    // Calculate R-squared
    double r_squared(const MatrixXd& X, const VectorXd& y) const {
        VectorXd predictions = predict(X);
        double y_mean = y.mean();
        
        double ss_res = (y - predictions).array().square().sum();
        double ss_tot = (y.array() - y_mean).array().square().sum();
        
        if (ss_tot < 1e-8) return 1.0;  // Perfect prediction case
        return 1.0 - (ss_res / ss_tot);
    }

    // Calculate Mean Squared Error
    double mse(const MatrixXd& X, const VectorXd& y) const {
        VectorXd predictions = predict(X);
        return (y - predictions).array().square().mean();
    }
};

// Function to parse CSV line
vector<string> parse_csv_line(const string& line) {
    vector<string> fields;
    stringstream ss(line);
    string field;
    
    while (getline(ss, field, ',')) {
        // Remove leading and trailing whitespace
        field.erase(0, field.find_first_not_of(" \t"));
        field.erase(field.find_last_not_of(" \t") + 1);
        fields.push_back(field);
    }
    
    return fields;
}

// Function to load data from CSV file
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
        
        // Expecting 11 columns (10 features + 1 target)
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
    int n_features = 10;  // First 10 columns are features
    
    // Create feature matrix X and target vector y
    MatrixXd X(n_samples, n_features);
    VectorXd y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }
        y(i) = data[i][n_features];  // Last column is target
    }
    
    cout << "Loaded " << n_samples << " samples with " << n_features << " features" << endl;
    return make_pair(X, y);
}

int main() {
    try {
        string filename;
        cout << "Enter the CSV filename (with 10 features + 1 target): ";
        cin >> filename;
        
        // Load data
        cout << "Loading data from " << filename << "..." << endl;
        auto data = load_data(filename);
        MatrixXd X = data.first;
        VectorXd y = data.second;
        
        // Display basic statistics
        cout << "Data Statistics:" << endl;
        cout << "Feature means: " << X.colwise().mean().transpose() << endl;
        cout << "Target mean: " << y.mean() << endl;
        cout << "Target std: " << sqrt((y.array() - y.mean()).square().mean()) << endl;
        
        // Get lambda value from user
        double lambda;
        cout << "\nEnter lambda value for Lasso regularization (e.g., 0.1): ";
        cin >> lambda;
        
        // Create and train Lasso regression model
        cout << "Training Lasso regression model..." << endl;
        LassoRegression lasso(lambda);
        lasso.fit(X, y);
        
        // Display results
        cout << "\n=== Lasso Regression Results ===" << endl;
        cout << "Lambda (regularization parameter): " << lambda << endl;
        cout << "Intercept: " << lasso.get_intercept() << endl;
        
        VectorXd coefficients = lasso.get_coefficients();
        cout << "\nFeature Coefficients:" << endl;
        for (int i = 0; i < coefficients.size(); ++i) {
            cout << "Feature " << (i + 1) << ": " << coefficients(i) << endl;
        }
        
        // Calculate and display performance metrics
        double r2 = lasso.r_squared(X, y);
        double mse_val = lasso.mse(X, y);
        
        cout << "\nModel Performance:" << endl;
        cout << "R-squared: " << r2 << endl;
        cout << "Mean Squared Error: " << mse_val << endl;
        cout << "Root Mean Squared Error: " << sqrt(mse_val) << endl;
        
        // Count non-zero coefficients
        int non_zero_count = 0;
        for (int i = 0; i < coefficients.size(); ++i) {
            if (abs(coefficients(i)) > 1e-8) {
                non_zero_count++;
            }
        }
        cout << "Number of non-zero coefficients: " << non_zero_count << " out of " 
             << coefficients.size() << endl;
        
        // Show which features are selected
        cout << "\nSelected features (non-zero coefficients):" << endl;
        for (int i = 0; i < coefficients.size(); ++i) {
            if (abs(coefficients(i)) > 1e-8) {
                cout << "Feature " << (i + 1) << ": " << coefficients(i) << endl;
            }
        }
        
        // Optional: Make predictions on sample data
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
            
            VectorXd prediction = lasso.predict(new_X);
            cout << "Predicted value: " << prediction(0) << endl;
        }
        
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
