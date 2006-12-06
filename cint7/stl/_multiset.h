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

#ifndef MULTISET_H
#define MULTISET_H

#ifndef Allocator
#define Allocator allocator
#include <defalloc.h>
#endif

#include <tree.h>

template <class Key, class Compare>
class multiset {
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

    multiset(const Compare& comp = Compare()) : t(comp, true) {}
    multiset(const value_type* first, const value_type* last, 
             const Compare& comp = Compare()) : t(comp, true) {
        for (const value_type* i = first; i != last; ++i)
           t.insert(svalue_type(*i));
    }
    multiset(const multiset<Key, Compare>& x) : t(x.t, true) {}
    multiset<Key, Compare>& operator=(const multiset<Key, Compare>& x) {
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
    void swap(multiset<Key, Compare>& x) { t.swap(x.t); }

// insert/erase
    iterator insert(const value_type& x) { return t.insert(x).first; }
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
   
// multiset operations:

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
inline bool operator==(const multiset<Key, Compare>& x, 
                       const multiset<Key, Compare>& y) {
    return x.size() == y.size() && equal(x.begin(), x.end(), y.begin());
}

template <class Key, class Compare>
inline bool operator<(const multiset<Key, Compare>& x, 
                      const multiset<Key, Compare>& y) {
    return lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

#undef Allocator

#endif
