/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include "tenzing/macro_at.hpp"

#include "array.hpp"
#include "coo_mat.hpp"
#include "algorithm.hpp"

#include <cassert>


template <Where where, typename Ordinal, typename Scalar>
class CsrMat
{
public:



    CsrMat();
    Ordinal nnz() const;
    Ordinal num_rows() const;
};

template <typename Ordinal, typename Scalar>
class CsrMat<Where::host, Ordinal, Scalar>;
template <typename Ordinal, typename Scalar>
class CsrMat<Where::device, Ordinal, Scalar>;

/* host sparse matrix */
template <typename Ordinal, typename Scalar>
class CsrMat<Where::host, Ordinal, Scalar>
{
    friend class CsrMat<Where::device, Ordinal, Scalar>; // device can see inside
    std::vector<Ordinal> rowPtr_;
    std::vector<Ordinal> colInd_;
    std::vector<Scalar> val_;
    Ordinal numCols_;

public:
    typedef Ordinal ordinal_type;
    typedef Scalar scalar_type;

    CsrMat() = default;
    CsrMat(Ordinal numRows, Ordinal numCols, Ordinal nnz) : rowPtr_(numRows + 1), colInd_(nnz), val_(nnz), numCols_(numCols) {}

    CsrMat(const CooMat<Ordinal, Scalar> &coo) : numCols_(coo.num_cols())
    {
        for (auto &e : coo.entries())
        {
            while (Ordinal(rowPtr_.size()) <= e.i)
            {
                rowPtr_.push_back(colInd_.size());
            }
            colInd_.push_back(e.j);
            val_.push_back(e.e);
        }
        while (Ordinal(rowPtr_.size()) < coo.num_rows() + 1)
        {
            rowPtr_.push_back(colInd_.size());
        }
    }

    Ordinal num_rows() const
    {
        if (rowPtr_.size() <= 1)
        {
            return 0;
        }
        else
        {
            return rowPtr_.size() - 1;
        }
    }

    Ordinal num_cols() const
    {
        return numCols_;
    }

    Ordinal nnz() const
    {
        if (colInd_.size() != val_.size())
        {
            throw std::logic_error("bad invariant");
        }
        return colInd_.size();
    }

    const Ordinal &row_ptr(Ordinal i) const
    {
        return rowPtr_[i];
    }
    const Ordinal &col_ind(Ordinal i) const
    {
        return colInd_[i];
    }
    const Scalar &val(Ordinal i) const
    {
        return val_[i];
    }

    const Ordinal *row_ptr() const { return rowPtr_.data(); }
    Ordinal *row_ptr() { return rowPtr_.data(); }
    const Ordinal *col_ind() const { return colInd_.data(); }
    Ordinal *col_ind() { return colInd_.data(); }
    const Scalar *val() const { return val_.data(); }
    Scalar *val() { return val_.data(); }

    /* keep rows [rowStart, rowEnd)
    */
    void retain_rows(Ordinal rowStart, Ordinal rowEnd)
    {

        if (0 == rowEnd)
        {
            throw std::logic_error("unimplemented");
        }
        // erase rows after
        // dont want to keep rowEnd, so rowEnd points to end of rowEnd-1
        std::cerr << "rowPtr_ = rowPtr[:" << rowEnd + 1 << "]\n";
        rowPtr_.resize(rowEnd + 1);
        std::cerr << "resize entries to " << rowPtr_.back() << "\n";
        colInd_.resize(rowPtr_.back());
        val_.resize(rowPtr_.back());

        // erase early row pointers
        std::cerr << "rowPtr <<= " << rowStart << "\n";
        shift_left(rowPtr_.begin() + rowStart, rowPtr_.end(), rowStart);
        std::cerr << "resize rowPtr to " << rowEnd - rowStart + 1 << "\n";
        rowPtr_.resize(rowEnd - rowStart + 1);

        const int off = rowPtr_[0];
        // erase entries for first rows
        std::cerr << "entries <<= " << off << "\n";
        shift_left(colInd_.begin() + off, colInd_.end(), off);
        shift_left(val_.begin() + off, val_.end(), off);

        // adjust row pointer offset
        std::cerr << "subtract rowPtrs by " << off << "\n";
        for (auto &e : rowPtr_)
        {
            e -= off;
        }

        // resize entries
        std::cerr << "resize entries to " << rowPtr_.back() << "\n";
        colInd_.resize(rowPtr_.back());
        val_.resize(rowPtr_.back());
    }
};

/* device sparse matrix
*/
template <typename Ordinal, typename Scalar>
class CsrMat<Where::device, Ordinal, Scalar>
{
    Array<Where::device, Ordinal> rowPtr_;
    Array<Where::device, Ordinal> colInd_;
    Array<Where::device, Scalar> val_;
    Ordinal numCols_;

public:
    struct View
    {
        ArrayView<Ordinal> rowPtr_;
        ArrayView<Ordinal> colInd_;
        ArrayView<Scalar> val_;
        Ordinal numCols_;

        bool operator==(const View &rhs) const {
            return rowPtr_ == rhs.rowPtr_ && colInd_ == rhs.colInd_ && val_ == rhs.val_ && numCols_ == rhs.numCols_;
        }

        __host__ __device__ Ordinal num_rows() const
        {
            if (rowPtr_.size() > 0)
            {
                return rowPtr_.size() - 1;
            }
            else
            {
                return 0;
            }
        }
        __host__ __device__ Ordinal num_cols() const {
            return numCols_;
        }

        __host__ __device__ Ordinal nnz() const {
            return colInd_.size();
        }

        __host__ __device__ const Ordinal *row_ptr() const
        {
            return rowPtr_.data();
        }
        __host__ __device__ Ordinal *row_ptr()
        {
            return rowPtr_.data();
        }

        __host__ __device__ const Ordinal *col_ind() const
        {
            return colInd_.data();
        }
        __host__ __device__ Ordinal *col_ind()
        {
            return colInd_.data();
        }

        __host__ __device__ const Scalar *val() const
        {
            return val_.data();
        }
        __host__ __device__ Scalar *val()
        {
            return val_.data();
        }

        __device__ const Ordinal &row_ptr(Ordinal i) const
        {
            return rowPtr_(i);
        }

        __device__ const Ordinal &col_ind(Ordinal i) const
        {
            return colInd_(i);
        }

        __device__ const Scalar &val(Ordinal i) const
        {
            return val_(i);
        }
    }; // View

    CsrMat() = default;
    CsrMat(CsrMat &&other) = delete;
    CsrMat(const CsrMat &other) = delete;

    CsrMat &operator=(CsrMat &&rhs)
    {
        if (this != &rhs)
        {
            rowPtr_ = std::move(rhs.rowPtr_);
            colInd_ = std::move(rhs.colInd_);
            val_ = std::move(rhs.val_);
            numCols_ = std::move(rhs.numCols_);
        }
        return *this;
    }

    // create device matrix from host
    CsrMat(const CsrMat<Where::host, Ordinal, Scalar> &m) : rowPtr_(m.rowPtr_), colInd_(m.colInd_), val_(m.val_), numCols_(m.numCols_)
    {
        if (colInd_.size() != val_.size())
        {
            throw std::logic_error("bad invariant");
        }
    }
    ~CsrMat()
    {
    }
    Ordinal num_rows() const
    {
        if (rowPtr_.size() <= 1)
        {
            return 0;
        }
        else
        {
            return rowPtr_.size() - 1;
        }
    }
    Ordinal num_cols() const
    {
        return numCols_;
    }

    Ordinal nnz() const
    {
        return colInd_.size();
    }

    View view() const
    {
        View v;
        v.rowPtr_ = rowPtr_.view();
        v.colInd_ = colInd_.view();
        v.val_ = val_.view();
        v.numCols_ = numCols_;
        return v;
    }
};

// mxn random matrix with nnz
template <typename Ordinal, typename Scalar>
CsrMat<Where::host, Ordinal, Scalar> random_matrix(const int64_t m, const int64_t n, const int64_t nnz)
{

    if (m * n < nnz)
    {
        throw std::logic_error(AT);
    }

    CooMat<Ordinal, Scalar> coo(m, n);
    while (coo.nnz() < nnz)
    {

        int64_t toPush = nnz - coo.nnz();
        std::cerr << "adding " << toPush << " non-zeros\n";
        for (int64_t _ = 0; _ < toPush; ++_)
        {
            Ordinal r = rand() % m;
            Ordinal c = rand() % n;
            Scalar e = 1.0;
            coo.push_back(r, c, e);
        }
        std::cerr << "removing duplicate non-zeros\n";
        coo.remove_duplicates();
    }
    coo.sort();
    std::cerr << "coo: " << coo.num_rows() << "x" << coo.num_cols() << "\n";
    CsrMat<Where::host, Ordinal, Scalar> csr(coo);
    std::cerr << "csr: " << csr.num_rows() << "x" << csr.num_cols() << " w/ " << csr.nnz() << "\n";
    return csr;
};

// nxn diagonal matrix with bandwidth b
template <typename Ordinal, typename Scalar>
CsrMat<Where::host, Ordinal, Scalar> random_band_matrix(const int64_t n, const int64_t bw, const int64_t nnz)
{
    CooMat<Ordinal, Scalar> coo(n, n);
    while (coo.nnz() < nnz)
    {

        int64_t toPush = nnz - coo.nnz();
        std::cerr << "adding " << toPush << " non-zeros\n";
        for (int64_t _ = 0; _ < toPush; ++_)
        {
            int r = rand() % n; // random row

            // column in the band
            int lb = r - bw;
            int ub = r + bw + 1;
            int64_t c = rand() % (ub - lb) + lb;
            if (c < 0 || c >= n)
            {
                // retry, don't over-weight first or last column
                continue;
            }
            float e = 1.0;

            assert(c < n);
            assert(r < n);
            coo.push_back(r, c, e);
        }
        std::cerr << "removing duplicate non-zeros\n";
        coo.remove_duplicates();
    }
    coo.sort();
    std::cerr << "coo: " << coo.num_rows() << "x" << coo.num_cols() << "\n";
    CsrMat<Where::host, Ordinal, Scalar> csr(coo);
    std::cerr << "csr: " << csr.num_rows() << "x" << csr.num_cols() << " w/ " << csr.nnz() << "\n";
    return csr;
};