#include <iostream>

#include <vector>
#include <string>
#include <iomanip>
#include <typeinfo>

#include <string_view>
#include <fstream>
#include <sstream>

#include <cmath>

#include <algorithm>
#include <stdexcept>
#include <set>

using namespace std;

// Using string_view to read_csv
// Better performance for largers CSV files by avoiding unnecessary copies

vector<string_view> parseRow(string_view row)
{
    
}