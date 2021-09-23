#include "sched/numeric.hpp"
#include "sched/operation.hpp"
#include "sched/schedule.hpp"
#include "sched/graph.hpp"
#include "sched/mcts.hpp"

#include <mpi.h>

#include <vector>
#include <memory>

#include <thread> //sleep
#include <chrono>

bool epoch;

class SlowFirst : public CpuNode {
protected:
    std::string name_;
public:
    SlowFirst(const std::string &name) : name_(name) {}
    // Node functions
    std::string name() const override { return name_; }
    EQ_DEF(SlowFirst);
    LT_DEF(SlowFirst);
    CLONE_DEF(SlowFirst);
    virtual int tag() const override { return 13; }
    bool operator<(const SlowFirst &rhs) const {return name() < rhs.name(); }
    bool operator==(const SlowFirst &rhs) const {return name() == rhs.name(); }
    virtual void run() {
        if (0 == epoch) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            epoch = 1;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            epoch = 0;
        }
    }
};

class FastFirst : public CpuNode {
protected:
    std::string name_;
public:
    FastFirst(const std::string &name) : name_(name) {}
    // Node functions
    std::string name() const override { return name_; }
    EQ_DEF(FastFirst);
    LT_DEF(FastFirst);
    CLONE_DEF(FastFirst);
    virtual int tag() const override { return 14; }
    bool operator<(const FastFirst &rhs) const {return name() < rhs.name(); }
    bool operator==(const FastFirst &rhs) const {return name() == rhs.name(); }
    virtual void run() {
        if (0 == epoch) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            epoch = 1;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            epoch = 0;
        }
    }
};

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    STDERR("create nodes...");
    std::shared_ptr<Start> start = std::make_shared<Start>();
    auto ff = std::make_shared<FastFirst>("ff");
    auto sf = std::make_shared<SlowFirst>("sf");
    std::shared_ptr<End> end = std::make_shared<End>();

    STDERR("create graph...");
    Graph<CpuNode> orig(start);
    orig.then(start, ff);
    orig.then(start, sf);
    orig.then(ff, end);
    orig.then(sf, end);

    epoch = 0;

    STDERR("mcts...");
    mcts::mcts(orig, MPI_COMM_WORLD);

    MPI_Finalize();
}