#pragma once

#include <ostream>

template <typename T>
struct Dim2 {
    T x;
    T y;
    Dim2(T _x, T _y) : x(_x), y(_y) {}
    Dim2() : Dim2(0,0) {}
    bool operator==(const Dim2 &rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    Dim2 &operator+=(const Dim2 &rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    Dim2 operator+(const Dim2 &rhs) const {
        Dim2 ret = *this;
        ret += rhs;
        return ret;
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Dim2<T> &d) {
    os << "<" << d.x << ", " << d.y << ">";
    return os;
}