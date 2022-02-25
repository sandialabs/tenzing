/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

/* mpi-specific operations
 */

#pragma once

#include "tenzing/operation.hpp"

#include <mpi.h>

#include <vector>

class Irecv : public CpuOp {
public:
  struct Args {
    void *buf;
    int count;
    MPI_Datatype datatype;
    int source;
    int tag;
    MPI_Comm comm;
    MPI_Request *request;
    bool operator==(const Args &rhs) const {
      return buf == rhs.buf && count == rhs.count && datatype == rhs.datatype &&
             source == rhs.source && tag == rhs.tag && comm == rhs.comm && request == rhs.request;
    }
  };

protected:
  Args args_;
  std::string name_;

public:
  Irecv(Args args, const std::string &name) : args_(args), name_(name) {}
  std::string name() const override { return name_; }
  virtual void run(Platform &plat) override;

  CLONE_DEF(Irecv);
  EQ_DEF(Irecv);
  LT_DEF(Irecv);
  bool operator==(const Irecv &rhs) const { return args_ == rhs.args_; }
  bool operator<(const Irecv &rhs) const { return name() < rhs.name(); }
};

class Isend : public CpuOp {
public:
  struct Args {
    const void *buf;
    int count;
    MPI_Datatype datatype;
    int dest;
    int tag;
    MPI_Comm comm;
    MPI_Request *request;
    bool operator==(const Args &rhs) const {
      return buf == rhs.buf && count == rhs.count && datatype == rhs.datatype && dest == rhs.dest &&
             tag == rhs.tag && comm == rhs.comm && request == rhs.request;
    }
  };

protected:
  Args args_;
  std::string name_;

public:
  Isend(Args args, const std::string &name) : args_(args), name_(name) {}
  std::string name() const override { return name_; }

  virtual void run(Platform &plat) override;

  CLONE_DEF(Isend);
  EQ_DEF(Isend);
  LT_DEF(Isend);
  bool operator==(const Isend &rhs) const { return args_ == rhs.args_; }
  bool operator<(const Isend &rhs) const { return name() < rhs.name(); }
};

class Ialltoallv : public CpuOp {
public:
  struct Args {
    const void *sendbuf;
    const int *sendcounts;
    const int *sdispls;
    MPI_Datatype sendtype;
    void *recvbuf;
    const int *recvcounts;
    const int *rdispls;
    MPI_Datatype recvtype;
    MPI_Comm comm;
    MPI_Request *request;
    bool operator==(const Args &rhs) const {
      return sendbuf == rhs.sendbuf && sendcounts == rhs.sendcounts && sdispls == rhs.sdispls &&
             sendtype == rhs.sendtype && recvbuf == rhs.recvbuf && recvcounts == rhs.recvcounts &&
             rdispls == rhs.rdispls && recvtype == rhs.recvtype && comm == rhs.comm &&
             request == rhs.request;
    }
  };

protected:
  Args args_;
  std::string name_;

public:
  Ialltoallv(Args args, const std::string &name) : args_(args), name_(name) {}
  std::string name() const override { return name_; }

  virtual void run(Platform &plat) override;

  CLONE_DEF(Ialltoallv);
  EQ_DEF(Ialltoallv);
  LT_DEF(Ialltoallv);

  bool operator==(const Ialltoallv &rhs) const { return args_ == rhs.args_; }
  bool operator<(const Ialltoallv &rhs) const { return name() < rhs.name(); }
};

class Wait : public CpuOp {
public:
  struct Args {
    MPI_Request *request;
    MPI_Status *status;
    bool operator==(const Args &rhs) const {
      return request == rhs.request && status == rhs.status;
    }
  };

protected:
  Args args_;
  std::string name_;

public:
  Wait(Args args, const std::string &name) : args_(args), name_(name) {}
  std::string name() const override { return name_; }

  virtual void run(Platform &plat) override;

  CLONE_DEF(Wait);
  EQ_DEF(Wait);
  LT_DEF(Wait);
  bool operator==(const Wait &rhs) const { return args_ == rhs.args_; }
  bool operator<(const Wait &rhs) const { return name() < rhs.name(); }
};

/* an MPI Waitall operation which owns its own handles
 */
class OwningWaitall : public CpuOp {
protected:
  std::vector<MPI_Request> reqs_;
  std::string name_;

public:
  OwningWaitall(const std::string &name) : name_(name) {}
  OwningWaitall(const size_t n, const std::string &name) : reqs_(n), name_(name) {}
  std::string name() const override { return name_; }

  virtual void run(Platform &plat) override;

  CLONE_DEF(OwningWaitall);
  EQ_DEF(OwningWaitall);
  LT_DEF(OwningWaitall);
  bool operator==(const OwningWaitall &rhs) const { return reqs_ == rhs.reqs_; }
  bool operator<(const OwningWaitall &rhs) const { return name() < rhs.name(); }

  void add_request(MPI_Request req) { reqs_.push_back(req); }
  std::vector<MPI_Request> &requests() { return reqs_; }
};

/* call MPI_Wait on all operations
 */
class MultiWait : public CpuOp {
protected:
  std::vector<MPI_Request *> reqs_;
  std::string name_;

public:
  MultiWait(const std::string &name) : name_(name) {}
  std::string name() const override { return name_; }

  virtual void run(Platform &plat) override;

  CLONE_DEF(MultiWait);
  EQ_DEF(MultiWait);
  LT_DEF(MultiWait);
  bool operator==(const MultiWait &rhs) const { return reqs_ == rhs.reqs_; }
  bool operator<(const MultiWait &rhs) const { return name() < rhs.name(); }

  void add_request(MPI_Request *req) { reqs_.push_back(req); }
};