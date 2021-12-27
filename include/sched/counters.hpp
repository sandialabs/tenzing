#pragma once

#define SCHED_ENABLE_COUNTERS

#include <cstdint>

namespace counters {

template<typename T>
class ZeroInit
{
    T val_;
public:
    ZeroInit() : val_(static_cast<T>(0)) { }
    ZeroInit(T val) : val_(val) { }
    operator T&() { return val_; }
    operator T() const { return val_; }
};


struct Mcts {
    ZeroInit<double> SELECT_TIME;
    ZeroInit<double> EXPAND_TIME;
    ZeroInit<double> ROLLOUT_TIME;
    ZeroInit<double> REDUNDANT_SYNC_TIME;
    ZeroInit<double> RMAP_TIME;
    ZeroInit<double> BENCHMARK_TIME;
    ZeroInit<double> BACKPROP_TIME;
};


extern Mcts mcts;


} // namespace counters

#ifdef SCHED_ENABLE_COUNTERS
#define SCHED_COUNTER_EXPR(expr) expr
#define SCHED_COUNTER(group, key) counters::group.key
#define SCHED_COUNTER_OP(group, key, _op) (counters::group.key) _op
#else
#define SCHED_COUNTER_EXPR(expr)
#define SCHED_COUNTER(group, key)
#define SCHED_COUNTER_OP(group, key, op)
#endif