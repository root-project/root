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

#ifndef TEMPBUF_H
#define TEMPBUF_H

#include <limits.h>
#include <pair.h>

#ifndef __stl_buffer_size
#define __stl_buffer_size 16384 // 16k
#endif

extern char __stl_temp_buffer[__stl_buffer_size];

//not reentrant code

template <class T>
pair<T*, int> get_temporary_buffer(int len, T*) {
    while (len > __stl_buffer_size / sizeof(T)) {
	set_new_handler(0);
        T* tmp = (T*)(::operator new((unsigned int)len * sizeof(T)));
        if (tmp) return pair<T*, int>(tmp, len);
        len = len / 2;
    }
    return pair<T*, int>((T*)__stl_temp_buffer, 
                         (int)(__stl_buffer_size / sizeof(T)));
}

template <class T>
void return_temporary_buffer(T* p) {
    if ((char*)(p) != __stl_temp_buffer) deallocate(p);
}

template <class T>
pair<T*, long> get_temporary_buffer(long len, T* p) {
    if (len > INT_MAX/sizeof(T)) 
	len = INT_MAX/sizeof(T);
    pair<T*, int> tmp = get_temporary_buffer((int)len, p);
    return pair<T*, long>(tmp.first, (long)(tmp.second));
}

#endif
