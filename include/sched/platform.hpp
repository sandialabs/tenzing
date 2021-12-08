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


struct Platform {

    std::vector<Stream> streams_;

    Platform(MPI_Comm comm) : comm_(comm) {
        // default stream
        streams_.push_back(Stream(0));
        cStreams_.push_back(0);
    }
    ~Platform() {
        for (auto &event : events_) {
            CUDA_RUNTIME(cudaEventDestroy(event));
        }
        // don't try to delete default stream
        for (size_t i = 1; i < cStreams_.size(); ++i) {
            CUDA_RUNTIME(cudaStreamDestroy(cStreams_[i]));
        }
    }
    Platform(const Platform &other) = delete; // stream lifetime?
    Platform(Platform &&other) = default;

private:
    std::vector<cudaStream_t> cStreams_;
    std::vector<cudaEvent_t> events_;
    MPI_Comm comm_;
public:

    // return the number of streams, not counting the default stream
    int num_streams() const {
        if (streams_.size() != cStreams_.size()) {
            THROW_RUNTIME("internal platform stream bookkeeping error");
        }
        if (streams_.empty()) {
            THROW_RUNTIME("platform missing default stream")
        }
        return streams_.size() - 1;
    }

    int num_events() const {
        return events_.size();
    }

    cudaStream_t cuda_stream(const Stream &stream) const {
        if (UNLIKELY(stream.id_ >= streams_.size())) {
            THROW_RUNTIME("requested non-existent stream " << stream.id_);
        }
        return cStreams_[stream.id_];
    }

    cudaEvent_t cuda_event(const Event &event) const {
        if (UNLIKELY(event.id_ >= events_.size())) {
            THROW_RUNTIME("requested non-existent event " << event.id_);
        }
        return events_[event.id_];
    }

    Event new_event() {
        Event evt(events_.size());
        cudaEvent_t e;
        CUDA_RUNTIME(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        events_.push_back(e);
        return evt;
    }

    Stream new_stream() {
        Stream stream(streams_.size());
        streams_.push_back(stream);
        cudaStream_t s;
        CUDA_RUNTIME(cudaStreamCreate(&s));
        cStreams_.push_back(s);
        return stream;
    }

    const MPI_Comm &comm() const {
        return comm_;
    }
    MPI_Comm &comm() {
        return comm_;
    }


    void ensure_streams(int n) {
        while(num_streams() < n) {
            new_stream();
        }
    }

    void ensure_events(int n) {
        while(num_events() < n) {
            new_event();
        }
    }

    //create a platform with `n` streams
    static Platform make_n_streams(int n, MPI_Comm comm) {
        Platform ret(comm);
        ret.ensure_streams(n);
        return ret;
    }
};

