/* Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the
 * terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
 * software.
 */

#pragma once

template<typename ForwardIt>
void shift_left(ForwardIt first, ForwardIt last, size_t n) {
    while(first != last) {
        *(first-n) = *first;
        ++first;
    }
}