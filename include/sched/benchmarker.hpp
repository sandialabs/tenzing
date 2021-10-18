#pragma once

#include <map>
#include <vector>
#include <string>

#include "schedule.hpp"


struct Benchmark {
    struct Result {
        double pct01;
        double pct10;
        double pct50;
        double pct90;
        double pct99;
        double stddev;
    };

    struct Opts {

    };
};

/* actually run the code to do the benchmark
*/
struct EmpiricalBenchmarker : public Benchmark {
    Result benchmark(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm, const BenchOpts &opts = BenchOpts());
    std::vector<Result> benchmark(std::vector<Schedule> &schedules, MPI_Comm comm, const BenchOpts &opts = BenchOpts()); 
};


/* find the results in a loaded database and return them
*/
struct CsvBenchmarker : public Benchmark {
    // what CSV file to read in
    CsvBenchmarker(const std::string &path);
    Result benchmark(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm, const BenchOpts &opts = BenchOpts());

    std::map<std::vector<std::string>, Result> data_;
};