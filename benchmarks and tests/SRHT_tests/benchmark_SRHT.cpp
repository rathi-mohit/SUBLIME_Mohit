#include <benchmark/benchmark.h>
#include <iostream>
#include "../../src/SRHT.hpp"
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Eigen; 


static void BM_SomeFunction(benchmark::State& state) {
 ifstream file("../chem_data.csv");
    string a;
    MatrixXd X(13910,128);
    VectorXd y(13910);
    for(int i=0; i<13910; i++)
    {
        X(i,0)=1;
        getline(file, a, ',');
        y(i)=stod(a);
        for(int j=1; j<127; j++)
        {
            getline(file, a, ',');
            X(i,j)=stod(a);
        }
        getline(file, a, '\n');
        X(i,127)=stod(a);
    }
  for (auto _ : state) {
    VectorXd b=LinearRegression(X,y); //timing this
    benchmark::DoNotOptimize(b);
  }
}

static void BM_SomeFunction2(benchmark::State& state) {
 ifstream file("../chem_data.csv");
    string a;
    MatrixXd X(13910,128);
    VectorXd y(13910);
    for(int i=0; i<13910; i++)
    {
        X(i,0)=1;
        getline(file, a, ',');
        y(i)=stod(a);
        for(int j=1; j<127; j++)
        {
            getline(file, a, ',');
            X(i,j)=stod(a);
        }
        getline(file, a, '\n');
        X(i,127)=stod(a);
    }
  int r=(int)(128*log(13910));
  for (auto _ : state) {
    VectorXd b=SRHT(X,y,r); //timing this
    benchmark::DoNotOptimize(b);
  }
}

// Register the function as a benchmark
BENCHMARK(BM_SomeFunction)->Iterations(1000);
BENCHMARK(BM_SomeFunction2)->Iterations(1000);
// Run the benchmark
BENCHMARK_MAIN();
