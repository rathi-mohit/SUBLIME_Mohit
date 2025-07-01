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

vector<string> features = {"index", "key", "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"};

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
        {"pickup_datetime", [](const df & d) { return d.pickup_datetime; }}
    };
}

// Functions

// Normalise a column (x_i - mu)/sigma
VectorXd normalize(const vector<double> & v)
{
    int n = v.size();
    vector<double> vec = v;
    VectorXd V = Map<VectorXd>(vec.data(), n);
    double mu = V.mean();
    double sigma = V.squaredNorm()/double(n) - mu*mu;
    // cout << "Trying" << endl;
    try
    {
        VectorXd norm_v(n);
        // cout << "Finished" << endl;
        for (int i = 0; i < n; i++)
        {
            norm_v[i] = (v[i] - mu)/sigma;
        }
        return norm_v;
    }
    catch(...)
    {
        return V;
    }
}

// Convert date-time to time in hours
double dt_to_24h(const string & dt)
{
    istringstream iss(dt);
    tm tm_struct = {};
    iss >> get_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
    if (iss.fail())
    {
        cerr << "Failed to parse time string." << endl;
        return 1;
    }
    return tm_struct.tm_hour + tm_struct.tm_min/60.0 + tm_struct.tm_sec/3600.0;
}

df readCSV(const string & filename) // Function to read CSV
{
    df data;
    ifstream file(filename);
    
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    
    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        try
        {
            stringstream ss(line);
            string token;

            getline(ss, token, ',');
            int index = stoi(token);

            string key;
            getline(ss, key, ',');

            getline(ss, token, ',');
            double fare_amount = stod(token);

            string pickup_datetime;
            getline(ss, pickup_datetime, ',');
            double time_24h = dt_to_24h(pickup_datetime);

            getline(ss, token, ',');
            double pickup_longitude = stod(token);

            getline(ss, token, ',');
            double pickup_latitude = stod(token);

            getline(ss, token, ',');
            double dropoff_longitude = stod(token);

            getline(ss, token, ',');
            double dropoff_latitude = stod(token);

            getline(ss, token, ',');
            double passenger_count = stod(token);

            data.index.push_back(index);
            data.key.push_back(key);
            data.fare_amount.push_back(fare_amount);
            data.pickup_datetime.push_back(time_24h);
            data.pickup_longitude.push_back(pickup_longitude);
            data.pickup_latitude.push_back(pickup_latitude);
            data.dropoff_longitude.push_back(dropoff_longitude);
            data.passenger_count.push_back(passenger_count);
        }
        catch (...)
        {
            cout << "This is a bad line. " << line << endl;
            cout << "-------------------" << endl;
            continue;
        }
    }
    return data;
}

// Tf is this linear regression!?
VectorXd linear_regression(const MatrixXd & X, const VectorXd & y)
{
    JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
    return svd.solve(y);
}

// R2 calc
// p = number of features
double calc_r2(const VectorXd & y, const VectorXd & yhat, int p)
{
    int n = y.size();
    double rss = (y - yhat).squaredNorm();
    double tss = (y.array() - y.mean()).square().sum();
    return 1 - rss/tss;
}

// double calc_fstat(const VectorXd & y, const VectorXd & y_pred, int p)
// {
//     int n = y.size();
//     double rss = (y - y_pred).squaredNorm();
//     double tss = (y.array() - y.mean()).square().sum();
//     return 1 - rss/tss;
// }

vector<string> fwd_step_sel(const df & data, const vector<string> & features, int max_features)
{
    auto feature_map = create_feature_map();
    vector<double> Y = data.fare_amount;
    int n = Y.size();
    VectorXd y = Map<VectorXd>(Y.data(), n);

    // Normalise and map everything to VectorXd
    unordered_map<string, VectorXd> normalised;
    for (const string & feature: features)
    {
        vector<double> v = feature_map[feature](data);
        // VectorXd V = Map<VectorXd>(v.data(), v.size());
        normalised[feature] = normalize(v);
    }

    vector<string> selects;
    double best_rss = 1e10;

    for (int s = 0; s < max_features; s++)
    {
        string best_feature;
        VectorXd best_beta;
        double temp_rss = best_rss;

        for (const string & feature: features)
        {
            if (find(selects.begin(), selects.end(), feature) != selects.end())
            {
                break;
            }

            int c = selects.size() + 1; // +1 for adding feature
            MatrixXd X(n, c + 1); // +1 for intercept
            X.col(0) = VectorXd::Ones(n); // Setting intercept column to be 1

            for (int i = 1; i < c; i++)
            {
                X.col(i) = normalised[selects[i-1]];
            }
            X.col(c) = normalised[feature];

            // Overpowered LinReg
            VectorXd beta = linear_regression(X, y);
            cout << beta.size();
            VectorXd yhat = X*beta;
            cout << yhat.size();

            double rss = (y - yhat).squaredNorm();
            if (rss < best_rss)
            {
                temp_rss = best_rss;
                best_feature = feature;
                best_beta = beta;
            }
        }

        if (temp_rss > best_rss)
        {
            best_rss = temp_rss;
            selects.push_back(best_feature);
            cout << best_feature << " Added." << endl;
            cout << "RSS: " << best_rss << endl;           
        }
    }
    return selects;
}

int main()
{
    string filename = "uber.csv";
    auto data = readCSV(filename);
    if (data.index.empty()) 
    {
        cerr << "Cooked" << endl;
        return -1;
    }

    vector<string> reg_features = {"passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "pickup_datetime"};
    cout << "Max features (<" << reg_features.size() << "): " << endl;
    int f;
    cin >> f;
    auto selects = fwd_step_sel(data, reg_features, f);
    cout << "Exit..." << endl;
    cin.get();
    return 0;
}