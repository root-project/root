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

#ifndef FARALLOC_H
#define FARALLOC_H

#include <new.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream.h>
#include <algobase.h>

template <class T>
inline random_access_iterator_tag iterator_category(const T __far *) {
    return random_access_iterator_tag();
}

template <class T>
inline T* value_type(const T __far *) { return (T*)(0); }

template <class T>
inline long* distance_type(const T __far*) { return (long*)(0); }

inline void destroy(char __far *) {}
inline void destroy(unsigned char __far *) {}
inline void destroy(short __far *) {}
inline void destroy(unsigned short __far *) {}
inline void destroy(int __far *) {}
inline void destroy(unsigned int __far *) {}
inline void destroy(long __far *) {}
inline void destroy(unsigned long __far *) {}
inline void destroy(float __far *) {}
inline void destroy(double __far *) {}

inline void destroy(char __far *, char __far *) {}
inline void destroy(unsigned char __far *, unsigned char __far *) {}
inline void destroy(short __far *, short __far *) {}
inline void destroy(unsigned short __far *, unsigned short __far *) {}
inline void destroy(int __far *, int __far *) {}
inline void destroy(unsigned int __far *, unsigned int __far *) {}
inline void destroy(long __far *, long __far *) {}
inline void destroy(unsigned long __far *, unsigned long __far *) {}
inline void destroy(float __far *, float __far *) {}
inline void destroy(double __far *, double __far *) {}

inline void __far * operator new(size_t, void __far *p) { return p; }

template <class T>
inline T __far * allocate(long size, T __far * p) {
    set_new_handler(0);
    T __far * tmp = 
        (T __far *)(::operator new((unsigned long)(size * sizeof(T))));
    if (tmp == 0) {
	cout << "out of memory" << endl; 
	exit(1);
    }
    return tmp;
}

template <class T>
inline void deallocate(T __far * buffer) {
    ::operator delete(buffer);
}

template <class T1, class T2>
inline void construct( T1 __far *p, const T2& value )
{
    new(p)T1(value);
}

template <class T>
inline void destroy( T __far * pointer ) {
    pointer->~T();
}

template <class T>
class far_allocator {
public:
    typedef T value_type;
    typedef T __far * pointer;
    typedef const T __far * const_pointer;
    typedef T __far & reference;
    typedef const T __far & const_reference;
    typedef unsigned long size_type;
    typedef long difference_type;
    pointer allocate(size_type n) {
        return ::allocate((difference_type)n, (pointer)0);
    }
    void deallocate(pointer p) { ::deallocate(p); }
    pointer address(reference x) { return (pointer)&x; }
    const_pointer const_address(const_reference x) { 
	return (const_pointer)&x; 
    }
    size_type init_page_size() { 
	return max(size_type(1), size_type(4096/sizeof(T))); 
    }
    size_type max_size() const { 
	return max(size_type(1), size_type(ULONG_MAX/sizeof(T))); 
    }
};

class far_allocator<void> {
public:
    typedef void __far * pointer;
};

#endif
