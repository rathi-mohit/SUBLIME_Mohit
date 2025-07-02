#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

struct datapoint {
    string timestamp;
    double associated_value;
};

int count(const vector<datapoint>& data);
double mean(const vector<datapoint>& data);
double variance(const vector<datapoint>& data);
double maximum_obs(const vector<datapoint>& data);
double minimum_obs(const vector<datapoint>& data);
double range(const vector<datapoint>& data);
double median(vector<datapoint> data);

int main() {
    const string filepath = "./Datasets/ambient_temperature_system_failure.csv";

    ifstream file(filepath);
    string line;

    // skipping the header
    getline(file, line);

    vector<datapoint> data;
    data.reserve(7300);

    while (getline(file, line)) {
        datapoint point;
        stringstream ss(line);

        string field;

        getline(ss, field, ',');
        point.timestamp = field;
        getline(ss, field, ',');
        point.associated_value = stod(field);

        data.push_back(point);
    }

    cout << count(data) << '\n';
    cout << mean(data) << '\n';
    cout << variance(data) << '\n';
    cout << minimum_obs(data) << '\n';
    cout << maximum_obs(data) << '\n';
    cout << range(data) << '\n';
    cout << median(data) << '\n';

    return 0;
}

int count(const vector<datapoint>& data) {
    return data.size();
}

double mean(const vector<datapoint>& data) {
    int size = data.size();
    long double sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += data[i].associated_value;
    }
    return sum / size;
}

double variance(const vector<datapoint>& data) {
    int size = data.size();
    long double sum_of_squares = 0;
    double mew = mean(data);
    for (int i = 0; i < size; ++i) {
        sum_of_squares += (data[i].associated_value - mew) * (data[i].associated_value - mew);
    }
    return sum_of_squares / size;
}

double maximum_obs(const vector<datapoint>& data) {
    int size = data.size();
    double maximum = INT_MIN;
    for (int i = 0; i < size; ++i) {
        if (data[i].associated_value > maximum) maximum = data[i].associated_value;
    }
    return maximum;
}

double minimum_obs(const vector<datapoint>& data) {
    int size = data.size();
    double minimum = INT_MAX;
    for (int i = 0; i < size; ++i) {
        if (data[i].associated_value < minimum) minimum = data[i].associated_value;
    }
    return minimum;
}

double range(const vector<datapoint>& data) {
    return maximum_obs(data) - minimum_obs(data);
}

double median(vector<datapoint> data) {
    int size = data.size();
    vector<double> associated_values(size);
    for (int i = 0; i < size; ++i) {
        associated_values[i] = data[i].associated_value;
    }
    sort(associated_values.begin(), associated_values.end());

    if (size % 2 == 0) {
        return (associated_values[size / 2 - 1] + associated_values[size / 2]) / 2.0;
    } else {
        return associated_values[size / 2];
    }
}
