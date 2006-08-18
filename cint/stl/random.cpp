// -*- C++ -*-

// Copyright (C) 2001 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 2, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

// As a special exception, you may use this file as part of a free software
// library without restriction.  Specifically, if other files instantiate
// templates or use macros or inline functions from this file, or you compile
// this file and link it with other files to produce an executable, this
// file does not by itself cause the resulting executable to be covered by
// the GNU General Public License.  This exception does not however
// invalidate any other reasons why the executable file might be covered by
// the GNU General Public License.

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 */

#include <stddef.h>

#define __SEED 161803398

class __random_generator {
protected:
    unsigned long table[55];
    size_t index1;
    size_t index2;
public:
    unsigned long operator()(unsigned long limit) {
	index1 = (index1 + 1) % 55;
	index2 = (index2 + 1) % 55;
	table[index1] = table[index1] - table[index2];
	return table[index1] % limit;
    }
    void seed(unsigned long j);
    __random_generator(unsigned long j) { seed(j); }
};

void __random_generator::seed(unsigned long j) {
    unsigned long k = 1;
    table[54] = j;
    for (size_t i = 0; i < 54; i++) {
        size_t ii = 21 * i % 55;
        table[ii] = k;
        k = j - k;
        j = table[ii];
    }
    for (int loop = 0; loop < 4; loop++) {
        for (i = 0; i < 55; i++)
            table[i] = table[i] - table[(1 + i + 30) % 55];
    }
    index1 = 0;
    index2 = 31;
}

__random_generator rd(__SEED);

unsigned long __long_random(unsigned long limit) {
    return rd(limit);
}



