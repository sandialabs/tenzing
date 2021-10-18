#include "sched/benchmarker.hpp"

#include "sched/randomness.hpp"
#include "sched/numeric.hpp"

#include <vincentlaucsb/csv-parser/csv.hpp>


#include <cmath>
#include <algorithm>
#include <numeric>

using Result = Benchmark::Result;

std::vector<Result> EmpiricalBenchmarker::benchmark(std::vector<Schedule> &schedules, MPI_Comm comm, const BenchOpts &opts) {


    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // order to run schedules in each iteration
    std::vector<int> perm(schedules.size());
    std::iota(perm.begin(), perm.end(), 0);

    // each iteration's time for each schedule
    std::vector<std::vector<double>> times(schedules.size());

    // each iteration, do schedules in a random order
    for (size_t i = 0; i < opts.nIters; ++i) {
        if (0 == rank) {
            std::cerr << " " << i;
        }
        if (0 == rank) {
            std::random_shuffle(perm.begin(), perm.end());
        }
        MPI_Bcast(perm.data(), perm.size(), MPI_INT, 0, MPI_COMM_WORLD);
        for (int si : perm)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            double rstart = MPI_Wtime();
            schedules[si].run();
            double elapsed = MPI_Wtime() - rstart;
            times[si].push_back(elapsed);
        }
    }
    if (0 == rank) {
        std::cerr << std::endl;
    }

    // for each schedule
    for (size_t si = 0; si < times.size(); ++si)
    {
        // each iteration's time is the maximum observed across all ranks
        MPI_Allreduce(MPI_IN_PLACE, times[si].data(), times[si].size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    std::vector<Result> ret;
    for (auto &st : times)
    {
        std::sort(st.begin(), st.end());
        Result result;
        result.pct01  = st[st.size() * 01 / 100];
        result.pct10  = st[st.size() * 10 / 100];
        result.pct50  = st[st.size() * 50 / 100];
        result.pct90  = st[st.size() * 90 / 100];
        result.pct99  = st[st.size() * 99 / 100];
        result.stddev = stddev(st);
        ret.push_back(result);
    }
    return ret;
}



struct Measurement {
    size_t nSamples; // how many samples make up the measurement
    double time; // estimated operation time
};

Measurement measure(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm,
 double nSamplesHint, double targetSecs = 0.01 // target measurement time in seconds
) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Measurement result;
    result.nSamples = nSamplesHint;

    while (true) {
        MPI_Barrier(comm);
        double start = MPI_Wtime();
        for (size_t i = 0; i < result.nSamples; ++i) {
            for (auto &op : order) {
                op->run();
            }
        }
        double elapsed = MPI_Wtime() - start;

        // "true" time is max observed by any rank
        MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);

        // measurement time did not reach the target
        if (elapsed < targetSecs) {
            // estimate how many samples we need based off the past measurement
            double perSample = elapsed / result.nSamples;
            double estSamples = targetSecs / perSample;
            estSamples *= 1.1; // scale up to try to overshoot

            // take a step in that direction
            result.nSamples += std::ceil((estSamples - result.nSamples) * 0.5);
        } else {
            result.time = elapsed / result.nSamples;
            break;
        }
    }

    return result;
}

Result EmpiricalBenchmarker::benchmark(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm comm, const BenchOpts &opts) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
retry:
    // determine the number of samples needed for a measurement
    Measurement mmt = measure(order, comm, 1);
    // if (0 == rank) STDERR("initial estimate: " << mmt.nSamples << " samples");

// statistical test may help with this
#if 0
    // run the benchmark until the time stops decreasing
    {
        double current = mmt.time;
        double last;
        do {
            last = current;
            current = measure(order, comm, mmt.nSamples).time;
            // if (0 == rank) STDERR("current: " << current << "");

        } while (current < last);
    }

    // re-estimate the number of samples needed for a measurement
    mmt = measure(order, comm, mmt.nSamples);
    // if (0 == rank) STDERR("revised estimate: " << mmt.nSamples << " samples");
#endif
    size_t nSamplesHint = mmt.nSamples;

    // get the requested number of measurements

    std::vector<double> times;
    for (size_t i = 0; i < opts.nIters; ++i) {
        mmt = measure(order, comm, nSamplesHint);
        nSamplesHint = std::max(mmt.nSamples, nSamplesHint); // update the hint with the max number of samples ever needed
        times.push_back(mmt.time);
    }

    // each iteration's time is the maximum observed across all ranks
    MPI_Allreduce(MPI_IN_PLACE, times.data(), times.size(), MPI_DOUBLE, MPI_MAX, comm);

    if (randomness::compound_test(times)) {
        goto retry;
    }

    std::sort(times.begin(), times.end());
    Result ret;
    ret.pct01  = times[times.size() * 01 / 100];
    ret.pct10  = times[times.size() * 10 / 100];
    ret.pct50  = times[times.size() * 50 / 100];
    ret.pct90  = times[times.size() * 90 / 100];
    ret.pct99  = times[times.size() * 99 / 100];
    ret.stddev = stddev(times);

    return ret;
}

CsvBenchmarker::CsvBenchmarker(const std::string &path) {

    STDERR("open " << path);

    using namespace csv;

    CSVFormat format;
    format.header_row(0);
    CSVReader reader(path);

    for (CSVRow &row : reader) {
        Benchmark::Result result;
        std::vector<std::string> order;

        for (size_t i = 0; i < row.size(); ++i) {
            if (0 == i) result.pct01 = row[i].get<double>();
            else if (1 == i) result.pct10 = row[i].get<double>();
            else if (2 == i) result.pct50 = row[i].get<double>();
            else if (3 == i) result.pct90 = row[i].get<double>();
            else if (4 == i) result.pct99 = row[i].get<double>();
            else if (5 == i) result.stddev = row[i].get<double>();
            else {
                auto s = row[i].get<std::string>();
                order.push_back(s);
            }
        }

        auto p = data_.insert(std::make_pair(order, result));
        if (!p.second) {
            STDERR("duplicate ordering in input:");

            std::stringstream ss;
            for (CSVField& field: row) {
                    ss << field.get<>() << " ";
            }

            STDERR(ss.str());
            throw std::runtime_error(AT);
        }

    }

    STDERR("got " << data_.size() << " records");

}

Result CsvBenchmarker::benchmark(std::vector<std::shared_ptr<CpuNode>> &order, MPI_Comm, const BenchOpts &) {
    std::vector<std::string> names;
    for (const auto &op : order) {
        names.push_back(op->name());
    }

    Result r = data_.at(names);

    return r;
}