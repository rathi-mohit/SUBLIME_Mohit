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
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct df
{
    vector<string> headers;
    vector<string> date;
    vector<double> month;
    vector<double> day;
    vector<double> time;
    vector<double> global_active_power;
    vector<double> global_reactive_power;
    vector<double> voltage;
    vector<double> global_intensity;
    vector<double> sub_metering_1;
    vector<double> sub_metering_2;
    vector<double> sub_metering_3;
    vector<vector<double>*> ref_cols;

    df()
    { ref_cols = { &month, &day, &time, &global_active_power, &global_reactive_power, &voltage, &global_intensity, &sub_metering_1, &sub_metering_2, &sub_metering_3}; }
};

double dt_to_24h(const string & dt)
{
    istringstream iss(dt);
    tm tm_struct = {};
    iss >> get_time(&tm_struct, "%H:%M:%S");
    if (iss.fail()) {return 0.0;}
    return tm_struct.tm_hour + tm_struct.tm_min/60.0 + tm_struct.tm_sec/3600.0;
}

pair<double, double> date_to_feature(const string & date)
{
    istringstream iss(date);
    tm tm_struct = {};
    iss >> get_time(& tm_struct, "%d/%m/%Y");
    if (iss.fail()) {return {0.0, 0.0};}
    return {tm_struct.tm_mon + 1.0, tm_struct.tm_mday + 0.0};
}

df readCSV(const string & filename)
{
    df data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }
    string line;
    getline(file, line); // Header
    while (getline(file, line))
    {
        try
        {
            stringstream ss(line);
            string item;

            string date, time;
            double global_active_power = 0, global_reactive_power = 0, voltage = 0, global_intensity = 0;
            double sub_metering_1 = 0, sub_metering_2 = 0, sub_metering_3 = 0;

            getline(ss, date, ';');
            getline(ss, time, ';');

            string s_global_active_power, s_global_reactive_power, s_voltage, s_global_intensity;
            string s_sub_metering_1, s_sub_metering_2, s_sub_metering_3;

            getline(ss, s_global_active_power, ';');
            getline(ss, s_global_reactive_power, ';');
            getline(ss, s_voltage, ';');
            getline(ss, s_global_intensity, ';');
            getline(ss, s_sub_metering_1, ';');
            getline(ss, s_sub_metering_2, ';');
            getline(ss, s_sub_metering_3, ';');

            data.date.push_back(date);

            double month = 0.0, day = 0.0;
            tie(month, day) = date_to_feature(date);
            data.month.push_back(month);
            data.day.push_back(day);

            data.time.push_back(dt_to_24h(time));
            data.global_active_power.push_back(stod(s_global_active_power));
            data.global_reactive_power.push_back(stod(s_global_reactive_power));
            data.voltage.push_back(stod(s_voltage));
            data.global_intensity.push_back(stod(s_global_intensity));
            data.sub_metering_1.push_back(stod(s_sub_metering_1));
            data.sub_metering_2.push_back(stod(s_sub_metering_2));
            data.sub_metering_3.push_back(stod(s_sub_metering_3));
        }
        catch (...)
        { cout << "Skipping line " << line << endl;}
    }
    return data;
}

pair<MatrixXd, VectorXd> convert(df & dframe)
{
    int n = dframe.date.size();
    VectorXd y = Map<VectorXd>(dframe.global_reactive_power.data(), n);
    MatrixXd X(n, 9);
    for (int i = 0; i < 9; i++)
    {
        if (i == 4) continue;
        auto col = *dframe.ref_cols[i];
        X.col(i - (i > 4)) = Map<VectorXd>(col.data(), n);
    }
    X.col(8) = VectorXd(n, 1);
    return {X, y};
}

struct DOptSelection
{
    private:
    MatrixXd X;
    VectorXd y;
    int k;
    int p, r, n;

    public:
    DOptSelection(const MatrixXd & X, const VectorXd & y, const int & k) : X (X), y (y), k (k)
    {
        int p = X.cols();
        int r = k/(2*p);
        int n = X.rows();
    }

    // Function to get the rows with r highest, lowest values of column p-1 and r lowest of column p-1
    pair<MatrixXd, VectorXd> get_dopt_selection()
    {
        vector<VectorXd> opt_X;
        vector<double> opt_y;

        for (int feature = 0; feature < p; feature++)
        {
            vector<double> top_r(r, -1e10);         
            vector<VectorXd> top_rows(r);
            vector<double> top_y(r);
            
            vector<double> bottom_r(r, 1e20);         
            vector<VectorXd> bottom_rows(r);
            vector<double> bottom_y(r);

            # pragma omp for
            for (int i = 0; i < n; ++i)
            {
                double x = X(i, feature-1);
                int min_idx = 0;
                int max_idx = 0;

                # pragma omp for
                for (int j = 0; j < r; j++)
                {
                    if (top_r[j] < top_r[min_idx]) { min_idx = j; }
                    if (bottom_r[j] > bottom_r[max_idx]) { max_idx = j; }
                }

                if (x > top_r[min_idx])
                {
                    top_r[min_idx] = x;
                    top_rows[min_idx] = X.row(i);
                    top_y[min_idx] = y[i];
                }

                if (x < bottom_r[max_idx])
                {
                    bottom_r[max_idx] = x;
                    bottom_rows[max_idx] = X.row(i);
                    bottom_y[max_idx] = y[i];
                }
                
            }
                // Append top rows and bottom rows to our vector
                int inc = top_rows.size() + bottom_rows.size();
                opt_X.resize(opt_X.size() + inc);
                opt_X.insert(opt_X.end(), top_rows.begin(), top_rows.end());
                opt_X.insert(opt_X.end(), bottom_rows.begin(), bottom_rows.end());

                opt_y.resize(opt_y.size() + inc);
                opt_y.insert(opt_y.end(), top_y.begin(), top_y.end());
                opt_y.insert(opt_y.end(), bottom_y.begin(), bottom_y.end());
        }

        int s = opt_X.size();
        MatrixXd sel_X(s, p);
        for (int i = 0; i < s; i++) { sel_X.row(i) = opt_X[i]; }
        Map<VectorXd> sel_y(opt_y.data(), opt_y.size());

        return {sel_X, sel_y};
    }
};

VectorXd ols_reg(const MatrixXd & X, const VectorXd & y)
{
    MatrixXd XT = X.transpose();
    VectorXd beta = (XT*X).inverse()*(XT*y);
    return beta;
}

int main()
{
    string filename = "household_power_consumption.txt";
    df dframe = readCSV(filename);

    auto xy = convert(dframe);
    MatrixXd X = xy.first;
    VectorXd y = xy.second;
    
    DOptSelection dopt(X, y, 100000);
    auto sel = dopt.get_dopt_selection();
    MatrixXd sel_X = sel.first;
    VectorXd sel_y = sel.second;
    
    VectorXd beta = ols_reg(sel_X, sel_y);
    cin.get();
    return 0;
}
