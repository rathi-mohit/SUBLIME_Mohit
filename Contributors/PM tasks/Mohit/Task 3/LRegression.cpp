#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <omp.h>
#include <Eigen/Dense>
#include <limits>

#define r 6370

using namespace Eigen;
using namespace std;

struct datapoint {
    float fare_amount;
    string pickup_datetime;
    double pickup_longitude, pickup_latitude;
    double dropoff_longitude, dropoff_latitude;
    double passenger_count;

    double trip_hour;
    double hav_dist;
};
struct ModelMetrics {
    double R2;
    double adjR2;
    double AIC;
    double BIC;
    double Cp;
};

vector<datapoint> load_csv(const string &filepath, const bool header = true);
void print_data_point(const vector<datapoint> &data, const size_t n);
double haversine_distance(datapoint point);

double RSS(const MatrixXd &X, const VectorXd &y);
double RSS(const VectorXd &y, const VectorXd &y_pred);
double R2_score(MatrixXd &X, VectorXd &y, VectorXd &beta);
double RSE(MatrixXd &X, VectorXd &y, VectorXd &beta);
VectorXd t_stat(MatrixXd &X, VectorXd &y, VectorXd &beta);

pair<vector<int>, vector<double>> forward_selection(const MatrixXd &X_full, const VectorXd &y, int max_features = -1);
ModelMetrics compute_metrics(const MatrixXd &X, const VectorXd &y, const VectorXd &beta, double sigma2_full_model);
double AIC(const MatrixXd &X, const VectorXd &y);

int main() {
    auto start = chrono::high_resolution_clock::now();
    const string filepath = "uber.csv";

    vector<datapoint> data = load_csv(filepath);
    shuffle(data.begin(), data.end(), default_random_engine(0));

    size_t N = data.size(); int p = 10;
    MatrixXd X(N, p+1);
    VectorXd y(N, 1);

    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        X(i, 0) = 1;
        X(i, 1) = data[i].hav_dist;
        X(i, 2) = data[i].passenger_count;
        X(i, 3) = data[i].trip_hour;

        y(i, 0) = data[i].fare_amount;
    }

    X.col(4) = X.col(1).array() * X.col(1).array(); // dist^2
    X.col(5) = X.col(2).array() * X.col(2).array(); // passenger_count^2
    X.col(6) = X.col(3).array() * X.col(3).array(); // trip_hour^2

    X.col(7) = X.col(1).array() * X.col(2).array(); // dist*passenger_count
    X.col(8) = X.col(3).array() * X.col(2).array(); // trip_hour*passenger count
    X.col(9) = X.col(1).array() * X.col(3).array(); // dist*trip_hour
    X.col(10) = X.col(4).array() * X.col(1).array(); // dist^3

    // Performing forward selection
    pair<vector<int>, vector<double>> selection = forward_selection(X, y);
    vector<int> selection_order = selection.first;
    vector<double> ordered_AIC = selection.second;

    // for (int i = 0; i < selection_order.size(); ++i) {
    //     cout << "Model " << (i+1) << " -- " << "Selected feature: " << selection_order[i] << " | " << "AIC: " << ordered_AIC[i] << '\n'; 
    // }

    // // Output of the above for loop: 1 10 4 9 6 3 2 and then marginal decrease in AIC for 8 7 5

    MatrixXd X_selected(N, 8);
    X_selected.col(0) = X.col(0);

    #pragma omp parallel for
    for (int i = 1; i < 8; ++i) {
        X_selected.col(i) = X.col(selection_order[i-1]);
    }

    // train_test_split
    int N8 = 0.8*N;
    int N2 = N - N8;

    MatrixXd X_train = X_selected.topRows(N8);
    MatrixXd X_test = X_selected.bottomRows(N2);
    VectorXd y_train = y.topRows(X_train.rows());
    VectorXd y_test = y.bottomRows(X_test.rows());

    VectorXd betaOLS = ((X_train.transpose() * X_train).inverse()) * (X_train.transpose()) * y_train;
    // MatrixXd I = MatrixXd::Identity(X_train.cols(), X_train.cols()); I(0, 0) = 0; 
    // double lambda = 10.0;
    // VectorXd betaRidge = (X_train.transpose() * X_train + lambda * I).inverse() * (X_train.transpose() * y_train);

    cout << "R2 score: " << R2_score(X_test, y_test, betaOLS) << '\n';
    cout << t_stat(X_train, y_train, betaOLS) << '\n';

    // MatrixXd X_unselected(N, 4);
    // X_unselected.col(0) = X.col(0); X_unselected.col(1) = X.col(5);
    // X_unselected.col(2) = X.col(7); X_unselected.col(3) = X.col(8);

    // double RSSM_unselected = RSS(X_unselected, y);
    // double RSSF = RSS(X, y);

    // cout << "F-stat: " << ((RSSM_unselected - RSSF)/7)/(RSSF/(N - 11)) << '\n';

    auto end = chrono::high_resolution_clock::now();
    chrono:: duration <double> duration = end - start;
    cout << "Time Taken: " << duration.count();
    return 0;
}

vector<datapoint> load_csv(const string &filepath, const bool header) {
    ifstream file(filepath);
    string line;

    if (header) getline(file, line); // skip the header

    vector<datapoint> data;
    while (getline(file, line)) {
        stringstream ss(line);
        string field;
        datapoint point;

        try {
            // skip col1, key
            getline(ss, field, ',');
            getline(ss, field, ',');

            // fare_amount
            getline(ss, field, ',');
            point.fare_amount = stof(field);
            // // if (point.fare_amount <= 0) continue;

            // pick_datetime and trip_hour
            getline(ss, field, ',');
            point.pickup_datetime = field;
            point.trip_hour = stod(field.substr(11, 2));

            // pickup_longitude
            getline(ss, field, ',');
            point.pickup_longitude = stod(field);

            // pickup_latitude
            getline(ss, field, ',');
            point.pickup_latitude = stod(field);

            // dropoff_longitude
            getline(ss, field, ',');
            point.dropoff_longitude = stod(field);

            // dropoff_latitude
            getline(ss, field, ',');
            point.dropoff_latitude = stod(field);

            // passenger_count
            getline(ss, field, ',');
            point.passenger_count = stoi(field);

            // great-circle distance
            point.hav_dist = haversine_distance(point);

            if (point.hav_dist < 0.1 || point.hav_dist > 100) continue; // nearly 7000
            if (point.passenger_count <= 0 || point.passenger_count > 6) continue; // nearly 600

            data.push_back(point);
        } catch (...) {
            continue; // skip rows with invalid or missing data (only one row in this dataset)
        }
    }
    return data;   
}
void print_data_point(const vector<datapoint> &data, const size_t n) {
    if (!data.empty() && n < data.size()) {
        const datapoint& point = data[n-1];
        cout << "The datapoint:\n";
        cout << "Fare: " << point.fare_amount << '\n';
        cout << "Datetime: " << point.pickup_datetime << '\n';
        cout << "Pickup: (" << point.pickup_latitude << ", " << point.pickup_longitude << ")\n";
        cout << "Dropoff: (" << point.dropoff_latitude << ", " << point.dropoff_longitude << ")\n";
        cout << "Passengers: " << point.passenger_count << '\n';
        cout << "Distance (km): " << point.hav_dist << '\n';
    }
    else cout << "Data is empty or too short" << '\n';
    return;
}
double haversine_distance(datapoint point) {
    double phi1 = 2*M_PI*point.dropoff_latitude/360.0;
    double phi2 = 2*M_PI*point.pickup_latitude/360.0;

    double delPhi = phi1 - phi2;
    double delLambda = 2*M_PI*(point.dropoff_longitude - point.pickup_longitude)/360.0;

    double sinDelPhi = sin(delPhi/2);
    double sinDelLambda = sin(delLambda/2);
    double hav_dist = 2.0*r*asin(sqrt(sinDelPhi*sinDelPhi + cos(phi1)*cos(phi2)*sinDelLambda*sinDelLambda));

    return hav_dist;
}
double RSS(const MatrixXd &X, const VectorXd &y) {
    VectorXd beta = (X.transpose() * X).inverse() * (X.transpose() * y);
    VectorXd residuals = y - X * beta;
    return residuals.squaredNorm();
}
double RSS(const VectorXd &y, const VectorXd &y_pred) {
    VectorXd residuals = y - y_pred;
    return residuals.squaredNorm();
}
double R2_score(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    int N = X.rows();

    VectorXd y_pred = X * beta;
    VectorXd y_mean_vec = VectorXd::Constant(y.rows(), y.mean());

    VectorXd e = y - y_pred;

    double RSS = e.squaredNorm();  
    double TSS = (y - y_mean_vec).squaredNorm();
    double R2_Score = 1 - (RSS/TSS);
    
    return R2_Score;
}
double RSE(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    int N = X.rows();

    VectorXd y_pred = X * beta;
    VectorXd y_mean_vec = VectorXd::Constant(y.rows(), y.mean());

    VectorXd e = y - y_pred;

    double RSS = e.squaredNorm();
    double RSE = sqrt(RSS/(N-2));

    return RSE;
}
VectorXd t_stat(MatrixXd &X, VectorXd &y, VectorXd &beta) {
    VectorXd y_pred = X * beta;
    VectorXd residuals = y - y_pred;
    double RSS = residuals.squaredNorm();

    int N = X.rows(), p = X.cols() - 1;
    double sigma2_hat = RSS / (N - p - 1);

    MatrixXd XtX_inv = (X.transpose() * X).inverse();
    MatrixXd cov_beta = sigma2_hat * XtX_inv;

    VectorXd t_stats(p + 1);
    for (int j = 0; j <= p; ++j) {
        double se = sqrt(cov_beta(j, j));
        t_stats(j) = beta(j) / se;
    }

    return t_stats;
}
pair<vector<int>, vector<double>> threaded_forward_selection(const MatrixXd &X_full, const VectorXd &y, int max_features) {
    int N = X_full.rows();
    int p = X_full.cols();
    set<int> selected;
    set<int> remaining;
    for (int i = 1; i < p; ++i) remaining.insert(i); 

    vector<int> selected_order;
    vector<double> AIC_in_same_order;

    while (!remaining.empty() && (max_features == -1 || selected.size() < max_features)) {
        int best_feature = -1;
        double best_candidate_AIC = numeric_limits<double>::max();

        vector<int> remaining_vec(remaining.begin(), remaining.end());
        int num_candidates = remaining_vec.size();

        vector<int> thread_best_feature(omp_get_max_threads(), -1);
        vector<double> thread_best_AIC(omp_get_max_threads(), numeric_limits<double>::max());

        #pragma omp parallel for
        for (int i = 0; i < num_candidates; ++i) {
            int feature = remaining_vec[i];
            vector<int> candidate(selected.begin(), selected.end());
            candidate.push_back(feature);

            MatrixXd X_candidate(N, candidate.size() + 1);
            X_candidate.col(0) = X_full.col(0); 
            for (int j = 0; j < candidate.size(); ++j) {
                X_candidate.col(j + 1) = X_full.col(candidate[j]);
            }
            double aic = AIC(X_candidate, y);

            int thread_id = omp_get_thread_num();
            if (aic < thread_best_AIC[thread_id]) {
                thread_best_AIC[thread_id] = aic;
                thread_best_feature[thread_id] = feature;
            }
        }

        for (int i = 0; i < thread_best_AIC.size(); ++i) {
            if (thread_best_AIC[i] < best_candidate_AIC) {
                best_candidate_AIC = thread_best_AIC[i];
                best_feature = thread_best_feature[i];
            }
        }

        selected_order.push_back(best_feature);
        AIC_in_same_order.push_back(best_candidate_AIC);
        selected.insert(best_feature);
        remaining.erase(best_feature);
    }

    return make_pair(selected_order, AIC_in_same_order);
}
pair<vector<int>, vector<double>> forward_selection(const MatrixXd &X_full, const VectorXd &y, int max_features) {
    int N = X_full.rows();
    int p = X_full.cols();
    set<int> selected;
    set<int> remaining;
    for (int i = 1; i < p; ++i) remaining.insert(i); 

    vector<int> selected_order;
    vector<double> AIC_in_same_order;
    double best_rss = numeric_limits<double>::max();

    while (!remaining.empty() && (max_features == -1 || selected.size() < max_features)) {
        int best_feature = -1;
        double best_candidate_rss = numeric_limits<double>::max();
        double best_candidate_AIC = numeric_limits<double>::max();

        for (int feature : remaining) {
            vector<int> candidate(selected.begin(), selected.end());
            candidate.push_back(feature);

            MatrixXd X_candidate(N, candidate.size() + 1);
            X_candidate.col(0) = X_full.col(0); // intercept
            for (int i = 0; i < candidate.size(); ++i)
                X_candidate.col(i + 1) = X_full.col(candidate[i]);

            double rss = RSS(X_candidate, y);
            double aic = AIC(X_candidate, y);
            if (rss < best_candidate_rss) {
                best_candidate_rss = rss;
                best_candidate_AIC = aic;
                best_feature = feature;
            }
        }
        selected_order.push_back(best_feature);
        AIC_in_same_order.push_back(best_candidate_AIC);
        selected.insert(best_feature);
        remaining.erase(best_feature);
    }
    
    return make_pair(selected_order, AIC_in_same_order);
}
ModelMetrics compute_metrics(const MatrixXd &X, const VectorXd &y, const VectorXd &beta, double sigma2_full_model) {
    int N = X.rows();
    int p = X.cols(); 

    VectorXd residuals = y - X * beta;
    double RSS = residuals.squaredNorm();
    double TSS = (y.array() - y.mean()).matrix().squaredNorm();

    double R2 = 1.0 - RSS / TSS;
    double adjR2 = 1.0 - ((RSS / (N - p)) / (TSS / (N - 1)));

    double AIC = N * log(RSS / N) + 2 * p;
    double BIC = N * log(RSS / N) + p * log(N);

    double Cp = RSS / sigma2_full_model - (N - 2 * p);

    return { R2, adjR2, AIC, BIC, Cp };
}
double AIC(const MatrixXd &X, const VectorXd &y) {
    int N = X.rows();
    int p = X.cols(); 

    VectorXd beta = (X.transpose() * X).inverse() * X.transpose() * y;
    VectorXd residuals = y - X * beta;
    double RSS = residuals.squaredNorm();

    double AIC = N * log(RSS / N) + 2 * p;
    return AIC;
}

