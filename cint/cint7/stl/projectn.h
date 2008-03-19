/* -*- C++ -*- */

/************************************************************************
 *
 * Copyright(c) 1995~2006  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

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

#ifndef PROJECTN_H
#define PROJECTN_H

#include <function.h>

template <class T>
struct singleton {
    T first;
    singleton(const T& a) : first(a) {}
    operator T() { return first; }
    singleton() : first(T()) {}
};

template <class T, class U>
struct select1st : unary_function<T, U> {
    const U& operator()(const T& x)  { return x.first; }
};

#endif


