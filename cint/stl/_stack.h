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

#ifndef STACK_H
#define STACK_H

#include <bool.h>
#include <heap.h>

template <class Container>
class stack {
friend bool operator==(const stack<Container>& x, const stack<Container>& y);
friend bool operator<(const stack<Container>& x, const stack<Container>& y);
public:
    typedef Container::value_type value_type;
    typedef Container::size_type size_type;
protected:
    Container c;
public:
    bool empty() const { return c.empty(); }
    size_type size() const { return c.size(); }
    value_type& top() { return c.back(); }
    const value_type& top() const { return c.back(); }
    void push(const value_type& x) { c.push_back(x); }
    void pop() { c.pop_back(); }
};

template <class Container>
bool operator==(const stack<Container>& x, const stack<Container>& y) {
    return x.c == y.c;
}

template <class Container>
bool operator<(const stack<Container>& x, const stack<Container>& y) {
    return x.c < y.c;
}

template <class Container>
class queue {
friend bool operator==(const queue<Container>& x, const queue<Container>& y);
friend bool operator<(const queue<Container>& x, const queue<Container>& y);
public:
    typedef Container::value_type value_type;
    typedef Container::size_type size_type;
protected:
    Container c;
public:
    bool empty() const { return c.empty(); }
    size_type size() const { return c.size(); }
    value_type& front() { return c.front(); }
    const value_type& front() const { return c.front(); }
    value_type& back() { return c.back(); }
    const value_type& back() const { return c.back(); }
    void push(const value_type& x) { c.push_back(x); }
    void pop() { c.pop_front(); }
};

template <class Container>
bool operator==(const queue<Container>& x, const queue<Container>& y) {
    return x.c == y.c;
}

template <class Container>
bool operator<(const queue<Container>& x, const queue<Container>& y) {
    return x.c < y.c;
}

template <class Container, class Compare> 
// Compare = less<Container::value_type> >
class  priority_queue {
public:
    typedef Container::value_type value_type;
    typedef Container::size_type size_type;
protected:
    Container c;
    Compare comp;
public:
    priority_queue(const Compare& x = Compare()) :  c(), comp(x) {}
    priority_queue(const value_type* first, const value_type* last, 
		   const Compare& x = Compare()) : c(first, last), comp(x) {
	make_heap(c.begin(), c.end(), comp);
    }
/*
    template <class InputIterator>
    priority_queue(InputIterator first, InputIterator last, 
		   const Compare& x = Compare()) : c(first, last), comp(x) {
	make_heap(c.begin(), c.end(), comp);
    }
*/
    bool empty() const { return c.empty(); }
    size_type size() const { return c.size(); }
    value_type& top() { return c.front(); }
    const value_type& top() const { return c.front(); }
    void push(const value_type& x) { 
	c.push_back(x); 
	push_heap(c.begin(), c.end(), comp);
    }
    void pop() { 
	pop_heap(c.begin(), c.end(), comp);
	c.pop_back(); 
    }
};

// no equality is provided

#endif
