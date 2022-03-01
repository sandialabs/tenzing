/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 */

#pragma once

#include <mpi.h>

#include "tenzing/cuda/cuda_runtime.hpp"

#include "csr_mat.hpp"
#include "partition.hpp"
#include "split_mat.hpp"

#include <cassert>

enum Tag : int { row_ptr, col_ind, val, x, num_cols };

template <typename Ordinal, typename Scalar>
int send_matrix(int dst, CsrMat<Where::host, Ordinal, Scalar> &&m,
                MPI_Comm comm);

template <>
inline int send_matrix<int, float>(int dst, CsrMat<Where::host, int, float> &&m,
                            MPI_Comm comm) {

  if (!m.row_ptr()) {
    THROW_RUNTIME("bad row_ptr");
  }
  if (!m.col_ind()) {
    THROW_RUNTIME("bad col_ind");
  }
  if (!m.val()) {
    THROW_RUNTIME("bad val");
  }

  MPI_Request reqs[4];
  int numCols = m.num_cols();
  STDERR("send numCols MPI_INT");
  MPI_Isend(&numCols, 1, MPI_INT, dst, Tag::num_cols, comm, &reqs[0]);
  STDERR("send row_ptr");
  MPI_Isend(m.row_ptr(), m.num_rows() + 1, MPI_INT, dst, Tag::row_ptr, comm,
            &reqs[1]);
            STDERR("send col_ind");
  MPI_Isend(m.col_ind(), m.nnz(), MPI_INT, dst, Tag::col_ind, comm, &reqs[2]);
  STDERR("send val");
  MPI_Isend(m.val(), m.nnz(), MPI_FLOAT, dst, Tag::val, comm, &reqs[3]);
  STDERR("waitall");
  MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

  STDERR("return from send_matrix");
  return 0;
}

template <typename Ordinal, typename Scalar>
CsrMat<Where::host, Ordinal, Scalar> receive_matrix(MPI_Comm comm);

template <>
inline CsrMat<Where::host, int, float> receive_matrix<int, float>(MPI_Comm comm) {

  int numCols;
  MPI_Recv(&numCols, 1, MPI_INT, 0, Tag::num_cols, comm, MPI_STATUS_IGNORE);

  // probe for number of rows
  MPI_Status stat;
  MPI_Probe(0, Tag::row_ptr, comm, &stat);
  int numRows;
  MPI_Get_count(&stat, MPI_INT, &numRows);
  if (numRows > 0) {
    --numRows;
  }

  // probe for nnz
  MPI_Probe(0, Tag::col_ind, comm, &stat);
  int nnz;
  MPI_Get_count(&stat, MPI_INT, &nnz);

  std::cerr << "recv " << numRows << "x" << numCols << " w/ " << nnz << "\n";
  CsrMat<Where::host, int, float> csr(numRows, numCols, nnz);

  // receive actual data into matrix
  MPI_Recv(csr.row_ptr(), numRows + 1, MPI_INT, 0, Tag::row_ptr, comm,
           MPI_STATUS_IGNORE);
  MPI_Recv(csr.col_ind(), nnz, MPI_INT, 0, Tag::col_ind, comm,
           MPI_STATUS_IGNORE);
  MPI_Recv(csr.val(), nnz, MPI_FLOAT, 0, Tag::val, comm, MPI_STATUS_IGNORE);

  return csr;
}


/* Ax=y , partitioned evenly by rows of A

   always have to pack on the send side, since not all local x values will be
   needed so may as well pack in a way that recver doesn't have to unpack

   serializing local and remote means you don't have to worry about
   concurrent adds to the product vector
   if kernels are sufficiently large, no real opportunity for these to overlap
   anyway if they're small, communication time will be longer anway


*/
template <typename Ordinal, typename Scalar> class RowPartSpmv {

  typedef CsrMat<Where::device, Ordinal, Scalar> csr_device_type;
  typedef CsrMat<Where::host, Ordinal, Scalar> csr_host_type;

private:
  MPI_Comm comm_;

  int loff_;                        // first row in global index
  csr_device_type la_;              // local A
  csr_device_type ra_;              // remote A
  Array<Where::device, Scalar> lx_; // local x
  Array<Where::device, Scalar> rx_; // remote x
  Array<Where::device, Scalar> ly_;

  // info for sending x
  struct SendParam {
    int dst; // destination rank
    int displ;
    int count;
    MPI_Request req;
  };
  std::vector<SendParam> sendParams_; // parameters for each rank
  Array<Where::device, Ordinal>
      xSendIdx_; // which entry of local X will be in each xSendBuf_;
  Array<Where::device, Scalar> xSendBuf_; // send local x entries to other ranks

  std::vector<Ordinal> gCols_; // global index from local

  std::map<Ordinal, std::vector<Ordinal>>
      sendEntr; // which entries of x need to be sent to each rank

  struct RecvParam {
    int src;   // source rank
    int displ; // displacement in
    int count; // number of entries
    MPI_Request req;
  };
  std::vector<RecvParam> recvParams_;

public:
  const csr_device_type &lA() const { return la_; }
  const csr_device_type &rA() const { return ra_; }
  const Array<Where::device, Scalar> &lx() const { return lx_; }
  Array<Where::device, Scalar> &rx() { return rx_; }
  const Array<Where::device, Scalar> &rx() const { return rx_; }
  const Array<Where::device, Scalar> &ly() const { return ly_; }
  Array<Where::device, Scalar> &x_send_buf() { return xSendBuf_; }
  const Array<Where::device, Ordinal> &x_send_idx() const { return xSendIdx_; }
  std::vector<SendParam> &send_params() { return sendParams_; }
  std::vector<RecvParam> &recv_params() { return recvParams_; }

  /* create from a matrix at root
   */
  RowPartSpmv(const csr_host_type &wholeA, const int root, MPI_Comm comm)
      : comm_(comm) {


    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    csr_host_type a;
    if (root == rank) {
      std::cerr << "partition matrix\n";
      std::vector<csr_host_type> as = part_by_rows(wholeA, size);
      for (int dst = 0; dst < size; ++dst) {
        if (root != dst) {
          std::cerr << "send A to " << dst << "\n";
          send_matrix(dst, std::move(as[dst]), comm_);
        }
      }
      a = as[rank];
    } else {
      std::cerr << "recv A at " << rank << "\n";
      a = receive_matrix<Ordinal, Scalar>(comm_);
    }

    // split row part of a into local and global
    SplitMat<csr_host_type> scm = split_local_remote(a, comm);

    // create local part of x array
    // undefined entries
    Range xrange = get_partition(a.num_cols(), rank, size);
    lx_ = Array<Where::device, float>(xrange.extent());
    ly_ = Array<Where::device, float>(scm.local.num_rows());

    // create remote part of x array
    // one entry per remote column
    rx_ = Array<Where::device, float>(scm.globals.size());
    if (0 == rx_.size()) {
      std::cerr << "WARN: not receiving anything\n";
    }

#ifdef SANITY_CHECKS
    // check local columns to ensure they're inside lx
    for (Ordinal r = 0; r < scm.local.num_rows(); ++r) {
      for (Ordinal ci = scm.local.row_ptr(r); ci < scm.local.row_ptr(r + 1);
           ++ci) {
        Ordinal c = scm.local.col_ind(ci);
        if (c >= lx_.size()) {
          throw std::runtime_error(AT);
        } else if (c < 0) {
          throw std::runtime_error(AT);
        }
      }
    }
#endif

#ifdef SANITY_CHECKS
    // check remote columns to ensure they're inside rx
    for (Ordinal r = 0; r < scm.remote.num_rows(); ++r) {
      for (Ordinal ci = scm.remote.row_ptr(r); ci < scm.remote.row_ptr(r + 1);
           ++ci) {
        Ordinal c = scm.remote.col_ind(ci);
        if (c >= rx_.size()) {
          throw std::runtime_error(AT);
        } else if (c < 0) {
          throw std::runtime_error(AT);
        }
      }
    }
#endif

#ifdef SANITY_CHECKS
    // check that globals are inside the matrix
    for (int c : scm.globals) {
      if (c < 0) {
        std::cerr << "column index " << c << " less than 0\n";
        throw std::runtime_error(AT);
      } else if (c >= a.num_cols()) {
        std::cerr << "global index " << c << ">= numcols " << a.num_cols()
                  << "\n";
        throw std::runtime_error(AT);
      }
    }
#endif

    la_ = std::move(scm.local);
    ra_ = std::move(scm.remote);
    if (la_.nnz() + ra_.nnz() != a.nnz()) {
      std::cerr << "lost a non-zero during split\n";
      throw std::runtime_error(AT);
    }
    if (la_.num_rows() != ra_.num_rows()) {
      std::cerr << "local and remote num_rows mismatch\n";
      throw std::runtime_error(AT);
    }

    loff_ = scm.loff;

    // ry_ = Array<Where::device, float>(la_.num_rows());

    // determine which columns needed from others
    std::map<int, std::vector<int>> recvCols;
    for (int c : scm.globals) {

      auto src = get_owner(a.num_cols(), c, size);
#ifdef SANITY_CHECKS
      if (rank == src) {
        std::cerr << "should not need my own columns in remote part";
        throw std::runtime_error(AT);
      }
      if (src < 0) {
        std::cerr << "owning rank " << src << " is < 0\n";
        throw std::runtime_error(AT);
      } else if (src > size) {
        std::cerr << "owning rank " << src << " is >= size(comm)\n";
      }
#endif
      recvCols[src].push_back(c);
    }

#if 1
    for (int r = 0; r < size; ++r) {
      MPI_Barrier(comm_);
      if (r == rank) {
        std::cerr << "rank " << rank << " recvCols:\n";
        for (auto it = recvCols.begin(); it != recvCols.end(); ++it) {
          std::cerr << "from " << it->first << ": ";
          std::cerr << it->second.size() << "\n";
          // for (auto &c : it->second) {
          //     std::cerr << c << " ";
          // }
          // std::cerr << "\n";
        }
      }
      MPI_Barrier(comm_);
    }
#endif

    // create receive parameters
    int offset = 0;
    for (auto it = recvCols.begin(); it != recvCols.end(); ++it) {
      RecvParam param;
      param.displ = offset;
      param.src = it->first;
      offset += it->second.size();
      param.count = offset - param.displ;
      recvParams_.push_back(param);
    }

#if 1
    for (int r = 0; r < size; ++r) {
      MPI_Barrier(comm_);
      if (r == rank) {
        std::cerr << "rank " << rank << " recvParams:\n";
        for (RecvParam &p : recvParams_) {
          std::cerr << "src=" << p.src << " displ=" << p.displ
                    << " count=" << p.count << "\n";
        }
      }
      MPI_Barrier(comm_);
    }
#endif

    // tell others which cols I need (send 0 if nothing)
    std::vector<MPI_Request> reqs(size);
    for (int dest = 0; dest < size; ++dest) {
      auto it = recvCols.find(dest);
      if (it != recvCols.end()) {
        if (!it->second.data()) {
          throw std::runtime_error(AT);
        }
#ifdef SANITY_CHECKS
        // check that globals are inside the matrix
        for (int c : it->second) {
          if (c < 0) {
            std::cerr << "requesting column " << c << " less than 0\n";
            throw std::runtime_error(AT);
          } else if (c >= a.num_cols()) {
            std::cerr << "requesting column " << c << ">= numcols "
                      << a.num_cols() << "\n";
            throw std::runtime_error(AT);
          } else {
            int r = get_owner(a.num_cols(), c, size);
            if (r != dest) {
              std::cerr << "requesting column " << c << " from " << dest
                        << " but should be " << r << "\n";
              throw std::runtime_error(AT);
            }
          }
        }
#endif

        MPI_Isend(it->second.data(), it->second.size(), MPI_INT, dest, 0, comm_,
                  &reqs[dest]);
      } else {
        int _;
        MPI_Isend(&_ /*junk*/, 0, MPI_INT, dest, 0, comm_, &reqs[dest]);
      }
    }

    // which global x rows other ranks need from me
    std::map<int, std::vector<int>> sendCols;
    for (int src = 0; src < size; ++src) {
      MPI_Status status;
      MPI_Probe(src, 0, comm, &status);
      int count;
      MPI_Get_count(&status, MPI_INT, &count);
      if (count != 0) {
        sendCols[src].resize(count);
        MPI_Recv(sendCols[src].data(), count, MPI_INT, src, 0, comm_,
                 MPI_STATUS_IGNORE);

#ifdef SANITY_CHECKS
        // src should only request columns in the matrix
        for (auto &e : sendCols[src]) {
          if (e < 0) {
            std::cerr << "(" << rank << ") rank " << src << " requested col "
                      << e << "/" << a.num_cols() << "\n";
            throw std::runtime_error(AT);
          } else if (e >= a.num_cols()) {
            std::cerr << "(" << rank << ") rank " << src << " requested col "
                      << e << "/" << a.num_cols() << "\n";
            throw std::runtime_error(AT);
          }
        }
        // src should only request columns that I own from me
        for (auto &e : sendCols[src]) {
          if (rank != get_owner(a.num_cols(), e, size)) {
            std::cerr << "(" << rank << ") rank " << src << " requested col "
                      << e << "/" << a.num_cols() << "\n";
            MPI_Finalize();
            throw std::runtime_error(AT);
          }
        }
#endif

      } else {
        int _;
        MPI_Recv(&_, 0, MPI_INT, src, 0, comm_, MPI_STATUS_IGNORE);
      }
    }

    // wait for sends
    for (size_t i = 0; i < reqs.size(); ++i) {
      MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);
    }

    // create the offsets from lx that we will send out
    // TODO: should be device array
    std::vector<int> offsets;
    for (auto it = sendCols.begin(); it != sendCols.end(); ++it) {
      // TODO - adjust for changed local array columns
      SendParam param;
      param.displ = offsets.size();
      param.dst = it->first;
      for (int gc : it->second) {
        int lc = gc - scm.loff;
        if (lc < 0)
          throw std::runtime_error(AT);
        if (lc >= lx_.size())
          throw std::runtime_error(AT);
        offsets.push_back(lc);
      }
      param.count = offsets.size() - param.displ;
      sendParams_.push_back(param);
    }

#ifdef SANITY_CHECKS
    for (int off : offsets) {
      if (off < 0) {
        THROW_RUNTIME("scatter index too small (<0)");
      }
      if (off >= lx_.size()) {
        THROW_RUNTIME("scatter index too large (> lx size)")
      }
    }
#endif

    // device version of offsets for packing
    xSendIdx_ = offsets;
    // buffer that x values will be placed into for sending
    STDERR("resize xSendBuf_ to " << xSendIdx_.size());
    xSendBuf_.resize(xSendIdx_.size());

    assert(lx_.size() > 0);
    assert(ly_.size() > 0);
  }
};