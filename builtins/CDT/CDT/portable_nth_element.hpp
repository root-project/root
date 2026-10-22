// This code is copied from LLVM's libc++
// https://github.com/llvm-mirror/libcxx/blob/4dde9ccef57d50e50620408a0b7a902f0aba803e/include/algorithm
// It is needed to provide portable nth_element on different platforms

// -*- C++ -*-
//===-------------------------- algorithm ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CDT_PORTABLE_NTH_ELEMENT
#define CDT_PORTABLE_NTH_ELEMENT

#include <algorithm>
#include <iterator>
#ifdef CDT_CXX11_IS_SUPPORTED
#include <type_traits>
#else
#include <boost/type_traits/add_lvalue_reference.hpp>
#endif

namespace CDT
{
namespace detail
{

// sort

// stable, 2-3 compares, 0-2 swaps

template <class Compare, class ForwardIterator>
unsigned
sort3(ForwardIterator x, ForwardIterator y, ForwardIterator z, Compare c)
{
    unsigned r = 0;
    if(!c(*y, *x)) // if x <= y
    {
        if(!c(*z, *y))     // if y <= z
            return r;      // x <= y && y <= z
                           // x <= y && y > z
        std::swap(*y, *z); // x <= z && y < z
        r = 1;
        if(c(*y, *x)) // if x > y
        {
            std::swap(*x, *y); // x < y && y <= z
            r = 2;
        }
        return r; // x <= y && y < z
    }
    if(c(*z, *y)) // x > y, if y > z
    {
        std::swap(*x, *z); // x < y && y < z
        r = 1;
        return r;
    }
    std::swap(*x, *y); // x > y && y <= z
    r = 1;             // x < y && x <= z
    if(c(*z, *y))      // if y > z
    {
        std::swap(*y, *z); // x <= y && y < z
        r = 2;
    }
    return r;
} // x <= y && y <= z

// Assumes size > 0
template <class Compare, class BirdirectionalIterator>
void selection_sort(
    BirdirectionalIterator first,
    BirdirectionalIterator last,
    Compare comp)
{
    BirdirectionalIterator lm1 = last;
    for(--lm1; first != lm1; ++first)
    {
#ifdef CDT_CXX11_IS_SUPPORTED
        BirdirectionalIterator i = std::min_element<
            BirdirectionalIterator,
            typename std::add_lvalue_reference<Compare>::type>(
            first, last, comp);
#else
        BirdirectionalIterator i = std::min_element<
            BirdirectionalIterator,
            typename boost::add_lvalue_reference<Compare>::type>(
            first, last, comp);
#endif
        if(i != first)
            std::swap(*first, *i);
    }
}

// nth_element

template <class Compare, class RandomAccessIterator>
void nth_element(
    RandomAccessIterator first,
    RandomAccessIterator nth,
    RandomAccessIterator last,
    Compare comp)
{
    // Compare is known to be a reference type
    typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
        difference_type;
    const difference_type limit = 7;
    while(true)
    {
    restart:
        if(nth == last)
            return;
        difference_type len = last - first;
        switch(len)
        {
        case 0:
        case 1:
            return;
        case 2:
            if(comp(*--last, *first))
                std::swap(*first, *last);
            return;
        case 3: {
            RandomAccessIterator m = first;
            detail::sort3<Compare>(first, ++m, --last, comp);
            return;
        }
        }
        if(len <= limit)
        {
            detail::selection_sort<Compare>(first, last, comp);
            return;
        }
        // len > limit >= 3
        RandomAccessIterator m = first + len / 2;
        RandomAccessIterator lm1 = last;
        unsigned n_swaps = detail::sort3<Compare>(first, m, --lm1, comp);
        // *m is median
        // partition [first, m) < *m and *m <= [m, last)
        // (this inhibits tossing elements equivalent to m around
        // unnecessarily)
        RandomAccessIterator i = first;
        RandomAccessIterator j = lm1;
        // j points beyond range to be tested, *lm1 is known to be <= *m
        // The search going up is known to be guarded but the search coming
        // down isn't. Prime the downward search with a guard.
        if(!comp(*i, *m)) // if *first == *m
        {
            // *first == *m, *first doesn't go in first part
            // manually guard downward moving j against i
            while(true)
            {
                if(i == --j)
                {
                    // *first == *m, *m <= all other elements
                    // Parition instead into [first, i) == *first and
                    // *first < [i, last)
                    ++i; // first + 1
                    j = last;
                    if(!comp(*first, *--j)) // we need a guard if
                                            // *first == *(last-1)
                    {
                        while(true)
                        {
                            if(i == j)
                                return; // [first, last) all equivalent
                                        // elements
                            if(comp(*first, *i))
                            {
                                std::swap(*i, *j);
                                ++n_swaps;
                                ++i;
                                break;
                            }
                            ++i;
                        }
                    }
                    // [first, i) == *first and *first < [j,
                    // last) and j == last - 1
                    if(i == j)
                        return;
                    while(true)
                    {
                        while(!comp(*first, *i))
                            ++i;
                        while(comp(*first, *--j))
                            ;
                        if(i >= j)
                            break;
                        std::swap(*i, *j);
                        ++n_swaps;
                        ++i;
                    }
                    // [first, i) == *first and *first < [i,
                    // last) The first part is sorted,
                    if(nth < i)
                        return;
                    // nth_element the secod part
                    // nth_element<Compare>(i, nth, last, comp);
                    first = i;
                    goto restart;
                }
                if(comp(*j, *m))
                {
                    std::swap(*i, *j);
                    ++n_swaps;
                    break; // found guard for downward moving j, now use
                           // unguarded partition
                }
            }
        }
        ++i;
        // j points beyond range to be tested, *lm1 is known to be <= *m
        // if not yet partitioned...
        if(i < j)
        {
            // known that *(i - 1) < *m
            while(true)
            {
                // m still guards upward moving i
                while(comp(*i, *m))
                    ++i;
                // It is now known that a guard exists for downward moving
                // j
                while(!comp(*--j, *m))
                    ;
                if(i >= j)
                    break;
                std::swap(*i, *j);
                ++n_swaps;
                // It is known that m != j
                // If m just moved, follow it
                if(m == i)
                    m = j;
                ++i;
            }
        }
        // [first, i) < *m and *m <= [i, last)
        if(i != m && comp(*m, *i))
        {
            std::swap(*i, *m);
            ++n_swaps;
        }
        // [first, i) < *i and *i <= [i+1, last)
        if(nth == i)
            return;
        if(n_swaps == 0)
        {
            // We were given a perfectly partitioned sequence.  Coincidence?
            if(nth < i)
            {
                // Check for [first, i) already sorted
                j = m = first;
                while(++j != i)
                {
                    if(comp(*j, *m))
                        // not yet sorted, so sort
                        goto not_sorted;
                    m = j;
                }
                // [first, i) sorted
                return;
            }
            else
            {
                // Check for [i, last) already sorted
                j = m = i;
                while(++j != last)
                {
                    if(comp(*j, *m))
                        // not yet sorted, so sort
                        goto not_sorted;
                    m = j;
                }
                // [i, last) sorted
                return;
            }
        }
    not_sorted:
        // nth_element on range containing nth
        if(nth < i)
        {
            // nth_element<Compare>(first, nth, i, comp);
            last = i;
        }
        else
        {
            // nth_element<Compare>(i+1, nth, last, comp);
            first = ++i;
        }
    }
}

template <class _RandomAccessIterator, class _Compare>
inline void portable_nth_element(
    _RandomAccessIterator first,
    _RandomAccessIterator nth,
    _RandomAccessIterator last,
    _Compare comp)
{
#ifdef CDT_CXX11_IS_SUPPORTED
    detail::nth_element<typename std::add_lvalue_reference<_Compare>::type>(
        first, nth, last, comp);
#else
    detail::nth_element<typename boost::add_lvalue_reference<_Compare>::type>(
        first, nth, last, comp);
#endif
}

} // namespace detail
} // namespace CDT

#endif // CDT_PORTABLE_NTH_ELEMENT
