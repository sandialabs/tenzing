#pragma once

/*! \file Model an execution platform
*/

#include <map>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include "cuda_runtime.h"

struct Stream {

    typedef int id_t;

    id_t id_;
    Stream(id_t id) : id_(id) {}
    Stream() : Stream(0) {} // default stream
    explicit operator id_t() const {return id_;}

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

struct CPU {
    int id_;
    CPU(int id) : id_(id) {}
    operator int() const {return id_;}
};

/* FIXME: how is lifetime of stream handled
*/
struct Platform {
    std::map<Stream, cudaStream_t> streams_;
    std::map<CPU, int> cpus_;
    MPI_Comm comm_;

    Platform(MPI_Comm comm) : comm_(comm) {}

    cudaStream_t cuda_stream(const Stream stream) const {
        return streams_.at(stream);
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
            cudaStream_t stream;
            CUDA_RUNTIME(cudaStreamCreate(&stream));
            ret.streams_.insert(std::make_pair(Stream(n), stream));
        }
        return ret;
    }
};

