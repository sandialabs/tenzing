#pragma once

#include "coo_mat.hpp"
#include "partition.hpp"

#include "mpi.h"
#include "where.hpp"

#include <map>


/* local matrix has cols renumbered to be 0..N
   into the local dense vector

   remote matrix has cols renumbered for the remote dense vector
*/
template<typename CsrMat>
struct SplitMat {
    int loff; // global row for local matrix 0
    CsrMat local; // local matrix
    CsrMat remote; // remote matrix (with local column indices)
    std::map<int, int> locals;  // get local column from global column (in remote matrix)
    std::vector<int> globals; // get global column for local column
};

/* Row partition of a matrix into a "local" and "remote" part
   If locally there are rows i...j, then the local part also has columns i...j
   The remote part will have all other columns

   Each rank will also renumber the column indices in the remote part
   This rank will recv the corresponding remote x vector entries, but
   doesn't want to materialize the whole distributed x vector in memory
   so the column indices must be packed into a contiguous range 0...N

   Furthermore, we want all entries from rank 0 to come first, then rank 1, etc.
   This is so we can just get entries from rank 0 and recv them directly into the
   remote x vector at the correct offset

   To do this, relabel the matrix in the following way:
   Get a list of unique required global ids, and then sort them.
   The first will be local 0, then local 1, etc

*/

template<typename CsrMat>
SplitMat<CsrMat> split_local_remote(const CsrMat &m, MPI_Comm comm) {

    using coo_type = CooMat<typename CsrMat::ordinal_type, typename CsrMat::scalar_type>;

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // which rows of x are local
    Range localRange = get_partition(m.num_cols(), rank, size);
    std::cerr << "[" << localRange.lb <<","<< localRange.ub << ")\n";
    int loff = localRange.lb;

    // build two matrices, local gets local non-zeros, remote gets remote non-zeros
    coo_type local(m.num_rows(), localRange.ub - localRange.lb);
    coo_type remote(m.num_rows(), m.num_cols());


    // build local matrix, row partition gives needed offset to convert
    // the global colum directly into a local column
    std::vector<int> globals; // track all remote global columns
    for (int r = 0; r < m.num_rows(); ++r) {
        for (int ci = m.row_ptr(r); ci < m.row_ptr(r+1); ++ci) {
            int c = m.col_ind(ci);
            float v = m.val(ci);
            if (c >= localRange.lb && c < localRange.ub) {
                // adjust local columns to begin at 0 since the local x vector will begin at 0
                int lc = c - loff;
                if (lc < 0) {
                    throw std::runtime_error(AT);
                }
                if (lc >= local.num_cols()) {
                    throw std::runtime_error(AT);
                }
                local.push_back(r,lc,v);
            } else {
                globals.push_back(c);
                // this global column index will be renumbered later
                remote.push_back(r, c, v);
            }
        }
    }

    // sort the global columns present in the local rows
    // remove duplicates
    // then, they can be easily relabeled contiguously and be sorted by rank
    // now, globals[c] is the global column of remote column c 
    std::sort(globals.begin(), globals.end());
    auto it = std::unique(globals.begin(), globals.end());
    globals.resize(it - globals.begin());

    // locals[c] is the local column for remote global column c
    std::map<int, int> locals;
    for (size_t lc = 0; lc < globals.size(); ++lc) {
        int gc = globals[lc];
        locals[gc] = lc;
    }

#if 0
    for (int l = 0; l < globals.size(); ++l) {
        int g = globals[l];
        std::cerr << g << " -> " << l << "\n";
    }
#endif

    // adjust any column in the remote part of the matrix to be its local id instead
    for (typename coo_type::Entry &e : remote) {
        e.j = locals[e.j];
    }

    // now we know how many columns the remote matrix actually is
    // and can construct it with relabeled columns
    coo_type relabelRemote(remote.num_rows(), globals.size());
    for (typename coo_type::Entry &e : remote) {
        relabelRemote.push_back(e.i, locals[e.j], e.e);
    }

    return SplitMat<CsrMat> {
        .loff=loff,
        .local=local,
        .remote=relabelRemote,
        .locals=locals,
        .globals=globals
    };

}