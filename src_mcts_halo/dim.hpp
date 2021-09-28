#pragma once

#include <ostream>

template <unsigned N, typename T>
struct Dim;

template <typename T>
struct Dim<3, T> {
    T x;
    T y;
    T z;
    Dim(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    Dim() : Dim(0, 0, 0) {}
    bool operator==(const Dim &rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }

    Dim &operator+=(const Dim &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Dim operator+(const Dim &rhs) const {
        Dim ret = *this;
        ret += rhs;
        return ret;
    }

    Dim operator-() const {
        Dim ret = *this;
        x = -x;
        y = -y;
        z = -z;
        return ret;
    }
};

template <typename T>
using Dim3 = Dim<3, T>;

template<typename T>
std::ostream &operator<<(std::ostream &os, const Dim3<T> &d) {
    os << "<" << d.x << "," << d.y << "," << d.z << ">";
    return os;
}