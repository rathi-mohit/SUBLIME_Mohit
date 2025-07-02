#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include <unordered_map>
#include <functional>

#include <fstream>
#include <sstream>

#include <ctime>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
using namespace Eigen;

using namespace std;

// Definitions
double pi = M_1_PI;

vector<string> features = {"index", "key", "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "ride_distance", "pickup_hour", "pickup_date"};

struct df // A structure to hold vectors columns
{
    vector<int> index;
    vector<string> key;
    vector<double> fare_amount;
    vector<double> pickup_datetime;
    vector<double> pickup_longitude;
    vector<double> pickup_latitude;
    vector<double> dropoff_longitude;
    vector<double> dropoff_latitude;
    vector<double> passenger_count;
    vector<double> ride_distance;
    vector<double> pickup_date;

    df() = default;
};

// Referencing columns by strings
unordered_map<string, function<vector<double>(const df & ) >> create_feature_map()
{
    return
    {
        {"passenger_count", [](const df & d) { return d.passenger_count; }},
        {"pickup_longitude", [](const df & d) { return d.pickup_longitude; }},
        {"pickup_latitude", [](const df & d) { return d.pickup_latitude; }},
        {"dropoff_longitude", [](const df & d) { return d.dropoff_longitude; }},
        {"dropoff_latitude", [](const df & d) { return d.dropoff_latitude; }},
        {"pickup_datetime", [](const df & d) { return d.pickup_datetime; }},
        {"ride_distance", [](const df & d) { return d.ride_distance; }},
        {"pickup_date", [](const df & d) { return d.ride_distance; }}
    };
}

// Convert datetime to 24-hour format
vector<double> dt_to_features(const string & dt)
{
    istringstream iss(dt);
    tm tm_struct = {};
    iss >> get_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
    if (iss.fail()) {
        cerr << "Failed to parse time string." << endl;
        return {};
    }
    double time = tm_struct.tm_hour + tm_struct.tm_min/60.0 + tm_struct.tm_sec/3600.0;
    double date = tm_struct.tm_mday;
    return {time, date};
}

// Based on haversine formula
// Not multiplying earth radius r
double coords_to_dist(const double & lat1, const double & lon1, const double & lat2, const double & lon2)
{
    double dlat = (lat2 - lat1)*pi/180;
    double dlon = (lon2 - lon1)*pi/180;
    double a = sin(dlat/2)*sin(dlat/2) + cos(lat1*pi/180)*cos(lat2*pi/180)*sin(dlon/2)*sin(dlon/2);
    double d = 2*atan2(sqrt(a), sqrt(1-a));
    return d;
}

// CSV reading function
df readCSV(const string & filename)
{
    df data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    getline(file, line);  // Skip header

    while (getline(file, line)) {
        try {
            stringstream ss(line);
            string token;

            // Parse each field
            getline(ss, token, ',');
            int index = stoi(token);

            string key;
            getline(ss, key, ',');

            getline(ss, token, ',');
            double fare_amount = stod(token);

            string pickup_datetime;
            getline(ss, pickup_datetime, ',');
            auto feats = dt_to_features(pickup_datetime);
            double time_24h = feats[0], date = feats[1];

            getline(ss, token, ',');
            double pickup_longitude = stod(token);

            getline(ss, token, ',');
            double pickup_latitude = stod(token);

            getline(ss, token, ',');
            double dropoff_longitude = stod(token);

            getline(ss, token, ',');
            double dropoff_latitude = stod(token);

            getline(ss, token, ',');
            int passenger_count = stoi(token);

            // Outlier removal
            if (pickup_longitude == 0 or dropoff_longitude == 0 or pickup_latitude == 0 or dropoff_latitude == 0)
            {
                continue;
            }

            double distance = coords_to_dist(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude);
            if (distance*6357 > 100 or distance*6357 < 0.1)
            {
                continue;
            }
            // Store data
            data.index.push_back(index);
            data.key.push_back(key);
            data.fare_amount.push_back(fare_amount);
            data.pickup_datetime.push_back(time_24h);
            data.pickup_longitude.push_back(pickup_longitude);
            data.pickup_latitude.push_back(pickup_latitude);
            data.dropoff_longitude.push_back(dropoff_longitude);
            data.dropoff_latitude.push_back(dropoff_latitude);
            data.passenger_count.push_back(passenger_count);
            data.ride_distance.push_back(distance);
            data.pickup_date.push_back(date);
        }
        catch (...) {
            cout << "L. " << line << endl;
            continue;
        }
    }
    return data;
}

// Linear regression using Eigen with SVD
VectorXd linear_regression_svd(const MatrixXd& X, const VectorXd& y) {
    auto XT = X.transpose();
    auto beta = (XT*X).inverse()*XT*y;
    return beta;
}

double calc_r2(const VectorXd& y, const VectorXd & yhat)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double tss = (y.array() - y.mean()).square().sum();
    double r2 = 1 - rss/tss;
    return r2;
}

double calc_cp(const VectorXd & y, const VectorXd & yhat, const int & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double varhat = (y.array() - y.mean()).square().sum()/(n - p - 1);
    double cp = (rss + 2*p*varhat)/n;
    return cp;
}

double calc_aic(const VectorXd & y, const VectorXd & yhat, const int & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double varhat = (y.array() - y.mean()).square().sum()/(n - p - 1);
    double aic = 2.0*p/n + rss/(n*varhat);
    return aic;
}

double calc_bic(const VectorXd & y, const VectorXd & yhat, const int & p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double varhat = (y.array() - y.mean()).square().sum()/(n - p - 1);
    double bic = rss/n + log(n)*p*varhat;
    return bic;
}

// Forward stepwise selection with Eigen
vector<string> fwd_selection(const df & data, const vector<string>& candidate_features, int max_features)
{
    auto feature_map = create_feature_map();
    vector<double> Y = data.fare_amount;
    int n = Y.size();
    VectorXd y = Map<VectorXd>(Y.data(), n);
    
    unordered_map<string, VectorXd> norm_features;
    for (const string & feat : candidate_features)
    {
        if (feat == "passenger_count")
        {
            vector<double> v = feature_map[feat](data);
            VectorXd V = Map<VectorXd>(v.data(), v.size());
            norm_features[feat] = V; // Don't normalize passenger count
        }
        else
        {
            vector<double> v = feature_map[feat](data);
            VectorXd V = Map<VectorXd>(v.data(), v.size());
            norm_features[feat] = V.normalized();
        }
    }

    vector<string> selected;
    double best_r2 = -1e9;
    vector<string> best_features;

    for (int s = 0; s < max_features; s++)
    {
        string best_feature;
        double temp_r2 = best_r2;
        VectorXd best_beta;

        for (const string& candidate : candidate_features)
        {
            if (find(selected.begin(), selected.end(), candidate) != selected.end())
                continue;

            int c = selected.size() + 1; // +1 for adding feature
            MatrixXd X(n, c + 1); // +1 for intercept term
            X.col(0) = VectorXd::Ones(n);

            // Add selected features
            for (int i = 1; i < c; ++i)
            {
                X.col(i) = norm_features[selected[i-1]];
            }

            // Add candidate feature
            X.col(c) = norm_features[candidate];

            // Solve with SVD
            VectorXd beta = linear_regression_svd(X, y);
            VectorXd y_pred = X*beta;

            double r2 = calc_r2(y, y_pred);

            // Update best candidate
            if (r2 > temp_r2) {
                temp_r2 = r2;
                best_feature = candidate;
                best_beta = beta;
            }
        }

        if (temp_r2 > best_r2)
        {
            best_r2 = temp_r2;
            selected.push_back(best_feature);
            cout << "Added '" << best_feature << "', R2 = " << temp_r2 << endl;
        }
        else
        {
            cout << "Closing." << endl;
            break;
        }
    }
    return selected;
}

int main()
{
    string filename = "uber.csv";
    auto data = readCSV(filename);
    
    if (data.index.empty())
    {
        cerr << "Cooked." << endl;
        return -1;
    }

    vector<string> candidate_features = {"passenger_count", "ride_distance", "pickup_datetime", "pickup_date"};
    auto n = candidate_features.size();
    auto selected = fwd_selection(data, candidate_features, n);
    
    cin.get();
    return 0;
}