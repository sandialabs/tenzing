#pragma once

/*! \file Model an execution platform
*/

#include <map>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include "cuda_runtime.h"
#include "macro_at.hpp"

/* handle representing a CUDA stream
*/
struct Stream {

    typedef unsigned id_t;

    id_t id_;
    Stream(id_t id) : id_(id) {}
    Stream() : Stream(0) {} // default stream

    bool operator<(const Stream &rhs) const {
        return id_ < rhs.id_;
    }
    bool operator>(const Stream &rhs) const {
        return id_ > rhs.id_;
    }
    bool operator==(const Stream &rhs) const {
        return id_ == rhs.id_;
    }
    bool operator!=(const Stream &rhs) const {
        return !(rhs == *this);
    }
};


void to_json(nlohmann::json& j, const Stream &s);
void from_json(const nlohmann::json& j, Stream &s);

inline std::ostream &operator<<(std::ostream &os, const Stream &s) {
    os << s.id_;
    return os;
}

/* handle representing a CUDA event
*/
struct Event {

    typedef unsigned id_t;

    id_t id_;
    Event(id_t id) : id_(id) {}
    Event() : Event(0) {} // default Event
    explicit operator id_t() const {return id_;}

    bool operator<(const Event &rhs) const {
        return id_ < rhs.id_;
    }
    bool operator>(const Event &rhs) const {
        return id_ > rhs.id_;
    }
    bool operator==(const Event &rhs) const {
        return id_ == rhs.id_;
    }
    bool operator!=(const Event &rhs) const {
        return !(rhs == *this);
    }
};


void to_json(nlohmann::json& j, const Event &s);
void from_json(const nlohmann::json& j, Event &s);

inline std::ostream &operator<<(std::ostream &os, const Event &s) {
    os << s.id_;
    return os;
}

struct CPU {
    int id_;
    CPU(int id) : id_(id) {}
    operator int() const {return id_;}
};

/* FIXME: how is lifetime of stream handled
*/
struct Platform {

    std::vector<Stream> streams_;

    Platform(MPI_Comm comm) : comm_(comm) {}
    Platform(const Platform &other) = delete;
    Platform(Platform &&other) = default;

private:
    std::vector<cudaStream_t> cStreams_;
    std::vector<cudaEvent_t> events_;
    MPI_Comm comm_;
public:

    cudaStream_t cuda_stream(const Stream &stream) const {
        if (UNLIKELY(0 == stream.id_)) {
            return 0;
        } else {
            return cStreams_[stream.id_ - 1];
        }
    }

    cudaEvent_t cuda_event(const Event &event) const {
        return events_[event.id_];
    }

    Event new_event() {
        Event evt(events_.size());
        cudaEvent_t e;
        CUDA_RUNTIME(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        events_.push_back(e);
        return evt;
    }

    const MPI_Comm &comm() const {
        return comm_;
    }
    MPI_Comm &comm() {
        return comm_;
    }


    //create a platform with `n` streams
    static Platform make_n_streams(int n, MPI_Comm comm) {
        Platform ret(comm);
        for (int i = 0; i < n; ++i) {
            ret.streams_.push_back(Stream(i+1));
            cudaStream_t s;
            CUDA_RUNTIME(cudaStreamCreate(&s));
            ret.cStreams_.push_back(s);
        }
        return ret;
    }
};

