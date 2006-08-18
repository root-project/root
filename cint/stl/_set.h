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

#ifndef SET_H
#define SET_H

#ifndef Allocator
#define Allocator allocator
#include <defalloc.h>
#endif

#include <tree.h>

template <class Key, class Compare>
class set {
public:
// typedefs:

    typedef Key key_type;
    typedef Key value_type;
    typedef Compare key_compare;
    typedef Compare value_compare;
private:
    typedef singleton<const Key> svalue_type;
    typedef rb_tree<key_type, svalue_type, 
                    select1st<svalue_type, key_type>, key_compare> rep_type;
    rep_type t;  // red-black tree representing map
public:
    typedef rep_type::reference reference;
    typedef rep_type::const_reference const_reference;
    typedef rep_type::iterator iterator;
    typedef iterator const_iterator;
    typedef rep_type::reverse_iterator reverse_iterator;
    typedef reverse_iterator const_reverse_iterator;
    typedef rep_type::size_type size_type;
    typedef rep_type::difference_type difference_type;

// allocation/deallocation

    set(const Compare& comp = Compare()) : t(comp, false) {}
    set(const value_type* first, const value_type* last, 
        const Compare& comp = Compare()) : t(comp, false) {
        for (const value_type* i = first; i != last; ++i)
           t.insert(svalue_type(*i));
    }
    set(const set<Key, Compare>& x) : t(x.t, false) {}
    set<Key, Compare>& operator=(const set<Key, Compare>& x) { 
        t = x.t; 
        return *this;
    }

// accessors:

    key_compare key_comp() const { return t.key_comp(); }
    value_compare value_comp() const { return t.key_comp(); }
    iterator begin() const { return ((rep_type&)t).begin(); }
    iterator end() const { return ((rep_type&)t).end(); }
    reverse_iterator rbegin() const { return ((rep_type&)t).rbegin(); } 
    reverse_iterator rend() const { return ((rep_type&)t).rend(); }
    bool empty() const { return t.empty(); }
    size_type size() const { return t.size(); }
    size_type max_size() const { return t.max_size(); }
    void swap(set<Key, Compare>& x) { t.swap(x.t); }

// insert/erase
    typedef  pair<iterator, bool> pair_iterator_bool; 
    // typedef done to get around compiler bug
    pair_iterator_bool insert(const value_type& x) { return t.insert(x); }
    iterator insert(iterator position, const value_type& x) {
        return t.insert(position, x);
    }
    void insert(const value_type* first, const value_type* last) {
        for (const value_type* i = first; i != last; ++i)
           t.insert(svalue_type(*i));
    }
    void erase(iterator position) { t.erase(position); }
    size_type erase(const key_type& x) { return t.erase(x); }
    void erase(iterator first, iterator last) { t.erase(first, last); }

// set operations:

    iterator find(const key_type& x) const { return ((rep_type&)t).find(x); }
    size_type count(const key_type& x) const { return ((rep_type&)t).count(x); }
    iterator lower_bound(const key_type& x) const {
        return ((rep_type&)t).lower_bound(x);
    }
    iterator upper_bound(const key_type& x) const {
        return ((rep_type&)t).upper_bound(x); 
    }
    typedef  pair<iterator, iterator> pair_iterator_iterator; 
    // typedef done to get around compiler bug
    pair_iterator_iterator equal_range(const key_type& x) const {
        return ((rep_type&)t).equal_range(x);
    }
};

template <class Key, class Compare>
inline bool operator==(const set<Key, Compare>& x, 
                       const set<Key, Compare>& y) {
    return x.size() == y.size() && equal(x.begin(), x.end(), y.begin());
}

template <class Key, class Compare>
inline bool operator<(const set<Key, Compare>& x, 
                      const set<Key, Compare>& y) {
    return lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

#undef Allocator

#endif
