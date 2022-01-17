/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

#include <vector>
#include <algorithm>

template<typename Ordinal, typename Scalar>
class CooMat {
public:

    struct Entry {
        Ordinal i;
        Ordinal j;
        Scalar e;

        Entry() = default;
        Entry(Ordinal _i, Ordinal _j, Scalar _e) : i(_i), j(_j), e(_e) {}
        Entry(const Entry &other) = default;
        Entry(Entry &&other) = default;
        Entry & operator=(Entry &&rhs) {
            i = std::move(rhs.i);
            j = std::move(rhs.j);
            e = std::move(rhs.e);
            return *this;
        }
        

        static bool by_ij(const Entry &a, const Entry &b) {
            if (a.i < b.i) {
                return true;
            } else if (a.i > b.i) {
                return false;
            } else {
                return a.j < b.j;
            }
        }

        static bool same_ij(const Entry &a, const Entry &b) {
            return a.i == b.i && a.j == b.j;
        }
    };

private:

    // sorted during construction
    std::vector<Entry> data_;
    Ordinal numRows_;
    Ordinal numCols_;

public:
    CooMat(Ordinal m, Ordinal n) : numRows_(m), numCols_(n) {}
    const std::vector<Entry> &entries() const {return data_;}
    void push_back(Ordinal i, Ordinal j, Scalar e) {
        data_.push_back(Entry(i,j,e));  
    }

    void sort() {
        std::sort(data_.begin(), data_.end(), Entry::by_ij);
    }

    void remove_duplicates() {
        std::sort(data_.begin(), data_.end(), Entry::by_ij);
        auto it = std::unique(data_.begin(), data_.end(), Entry::same_ij);
        data_.resize(it - data_.begin());
    }

    int64_t num_rows() const {return numRows_;}
    int64_t num_cols() const {return numCols_;}
    int64_t nnz() const {return data_.size();}

    typename std::vector<Entry>::iterator begin() {return data_.begin();}
    typename std::vector<Entry>::iterator end() {return data_.end();}
};