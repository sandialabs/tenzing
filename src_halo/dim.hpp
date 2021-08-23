#pragma once

template <typename T>
struct Dim2 {
    T x;
    T y;
    Dim2(T _x, T _y) : x(_x), y(_y) {}
    Dim2() : Dim2(0,0) {}
    bool operator==(const Dim2 &rhs) const {
        return x == rhs.x && y == rhs.y;
    }
};