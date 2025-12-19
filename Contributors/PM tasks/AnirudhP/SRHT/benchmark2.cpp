#include <benchmark/benchmark.h>
#include <iostream>
#include "SRHT.cpp"
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Eigen; 


static void BM_SomeFunction(benchmark::State& state) {
 random_device rd;
    mt19937 mt(rd());
    double mean_data=0;
    double stddev_data=2;

    normal_distribution<double> gen_data(mean_data, stddev_data);

    double mean_par=0;
    double stddev_par=5;

    normal_distribution<double> gen_par(mean_par, stddev_par);

    double mean_err=0;
    double stddev_err=1;

    normal_distribution<double> gen_err(mean_err, stddev_err);

    int rows=500000;
    int cols=1000;
    MatrixXd X(rows, cols);
    VectorXd b(cols);
    VectorXd err(rows);

    for(int i=0; i<rows; i++)
    {
        X(i,0)=1;
        for(int j=1; j<cols; j++)
        {
            X(i,j)=gen_data(mt);
        }

        err(i)=gen_err(mt);
    }

    for(int j=0; j<cols; j++)
    {
        b(j)=gen_par(mt);
    }

    VectorXd y=X*b+err;

  for (auto _ : state) {
    VectorXd b=LinearRegression(X,y); //timing this
    benchmark::DoNotOptimize(b);
  }
}

static void BM_SomeFunction2(benchmark::State& state) {
random_device rd;
    mt19937 mt(rd());
    double mean_data=0;
    double stddev_data=2;

    normal_distribution<double> gen_data(mean_data, stddev_data);

    double mean_par=0;
    double stddev_par=5;

    normal_distribution<double> gen_par(mean_par, stddev_par);

    double mean_err=0;
    double stddev_err=1;

    normal_distribution<double> gen_err(mean_err, stddev_err);

    int rows=500000;
    int cols=1000;
    MatrixXd X(rows, cols);
    VectorXd b(cols);
    VectorXd err(rows);

    for(int i=0; i<rows; i++)
    {
        X(i,0)=1;
        for(int j=1; j<cols; j++)
        {
            X(i,j)=gen_data(mt);
        }

        err(i)=gen_err(mt);
    }

    for(int j=0; j<cols; j++)
    {
        b(j)=gen_par(mt);
    }

    VectorXd y=X*b+err;
  for (auto _ : state) {
    VectorXd b=SRHT(X,y); //timing this
    benchmark::DoNotOptimize(b);
  }
}

// Register the function as a benchmark
BENCHMARK(BM_SomeFunction)->Iterations(10);
BENCHMARK(BM_SomeFunction2)->Iterations(10);
// Run the benchmark
BENCHMARK_MAIN();
