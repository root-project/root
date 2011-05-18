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

#ifndef VECTOR_H
#define VECTOR_H

#include <algobase.h>
#include <bool.h>

#ifndef Allocator
#define Allocator allocator
#include <defalloc.h>
#endif

#ifndef __CINT__
#ifndef vector
#define vector vector
#endif
#endif

//template <class T,class Allocator=allocator>
template <class T>
class vector {
public:
    
    typedef Allocator<T> vector_allocator;
    typedef T value_type;
    typedef vector_allocator::pointer pointer;
    typedef vector_allocator::pointer iterator;
    typedef vector_allocator::const_pointer const_iterator;
    typedef vector_allocator::reference reference;
    typedef vector_allocator::const_reference const_reference;
    typedef vector_allocator::size_type size_type;
    typedef vector_allocator::difference_type difference_type;
    typedef reverse_iterator<const_iterator, value_type, const_reference, 
                             difference_type>  const_reverse_iterator;
    typedef reverse_iterator<iterator, value_type, reference, difference_type>
        reverse_iterator;
protected:
    static Allocator<T> static_allocator;
    iterator start;
    iterator finish;
    iterator end_of_storage;
    void insert_aux(iterator position, const T& x);
public:
    iterator begin() { return start; }
    const_iterator begin() const { return start; }
    iterator end() { return finish; }
    const_iterator end() const { return finish; }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { 
        return const_reverse_iterator(end()); 
    }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { 
        return const_reverse_iterator(begin()); 
    }
    size_type size() const { return size_type(end() - begin()); }
    size_type max_size() const { return static_allocator.max_size(); }
    size_type capacity() const { return size_type(end_of_storage - begin()); }
    bool empty() const { return begin() == end(); }
    reference at(size_type n) { if (n>=size()) throw "out-of-bounds"; return *(begin() + n); }
    const_reference at(size_type n) const { if (n>=size()) throw "out-of-bounds"; return *(begin() + n); }
    reference operator[](size_type n) { return *(begin() + n); }
    const_reference operator[](size_type n) const { return *(begin() + n); }
    vector() : start(0), finish(0), end_of_storage(0) {}
    vector(size_type n, const T& value = T()) {
      start = static_allocator.allocate(n);
      uninitialized_fill_n(start, n, value);
      finish = start + n;
      end_of_storage = finish;
    }
    vector(const vector<T>& x) {
      start = static_allocator.allocate(x.end() - x.begin());
      finish = uninitialized_copy(x.begin(), x.end(), start);
      end_of_storage = finish;
    }
    vector(const_iterator first, const_iterator last) {
      size_type n = 0;
      distance(first, last, n);
      start = static_allocator.allocate(n);
      finish = uninitialized_copy(first, last, start);
      end_of_storage = finish;
    }
    ~vector() { 
      destroy(start, finish);
      static_allocator.deallocate(start);
    }
    vector<T>& operator=(const vector<T>& x);
    void reserve(size_type n) {
      if (capacity() < n) {
          iterator tmp = static_allocator.allocate(n);
          uninitialized_copy(begin(), end(), tmp);
          destroy(start, finish);
          static_allocator.deallocate(start);
          finish = tmp + size();
          start = tmp;
          end_of_storage = begin() + n;
      }
    }
    reference front() { return *begin(); }
    const_reference front() const { return *begin(); }
    reference back() { return *(end() - 1); }
    const_reference back() const { return *(end() - 1); }
    void push_back(const T& x) {
      if (finish != end_of_storage) {
          /* Borland bug */
          construct(finish, x);
          finish++;
      } else
          insert_aux(end(), x);
    }
    void swap(vector<T>& x) {
      ::swap(start, x.start);
      ::swap(finish, x.finish);
      ::swap(end_of_storage, x.end_of_storage);
    }
    iterator insert(iterator position, const T& x) {
      size_type n = position - begin();
      if (finish != end_of_storage && position == end()) {
          /* Borland bug */
          construct(finish, x);
          finish++;
      } else
          insert_aux(position, x);
      return begin() + n;
    }
    void insert (iterator position, const_iterator first, 
             const_iterator last);
    void insert (iterator position, size_type n, const T& x);
    void pop_back() {
      /* Borland bug */
        --finish;
        destroy(finish);
    }
    void erase(iterator position) {
      if (position + 1 != end())
          copy(position + 1, end(), position);
      /* Borland bug */
      --finish;
      destroy(finish);
    }
    void erase(iterator first, iterator last) {
      vector<T>::iterator i = copy(last, end(), first);
      destroy(i, finish);
      // work around for destroy(copy(last, end(), first), finish);
      finish = finish - (last - first); 
    }
    void resize(size_type __new_size, const T& __x) {
      if (__new_size < size()) 
        erase(begin() + __new_size, end());
      else
        insert(end(), __new_size - size(), __x);
    }
    void resize(size_type __new_size) { resize(__new_size, T()); }
    void clear() { erase(begin(), end()); }
};

template <class T>
inline bool operator==(const vector<T>& x, const vector<T>& y) {
    return x.size() == y.size() && equal(x.begin(), x.end(), y.begin());
}

template <class T>
inline bool operator<(const vector<T>& x, const vector<T>& y) {
    return lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}


template <class T>
vector<T>::vector_allocator vector<T>::static_allocator;


template <class T>
vector<T>& vector<T>::operator=(const vector<T>& x) {
    if (&x == this) return *this;
    if (x.size() > capacity()) {
      destroy(start, finish);
      static_allocator.deallocate(start);
      start = static_allocator.allocate(x.end() - x.begin());
      end_of_storage = uninitialized_copy(x.begin(), x.end(), start);
    } else if (size() >= x.size()) {
      vector<T>::iterator i = copy(x.begin(), x.end(), begin());
      destroy(i, finish);
      // work around for destroy(copy(x.begin(), x.end(), begin()), finish);
    } else {
      copy(x.begin(), x.begin() + size(), begin());
      uninitialized_copy(x.begin() + size(), x.end(), begin() + size());
    }
    finish = begin() + x.size();
    return *this;
}

template <class T>
void vector<T>::insert_aux(iterator position, const T& x) {
    if (finish != end_of_storage) {
      construct(finish, *(finish - 1));
      copy_backward(position, finish - 1, finish);
      *position = x;
      ++finish;
    } else {
      size_type len = size() ? 2 * size() 
          : static_allocator.init_page_size();
#ifdef __CINT__
        iterator tmp;
      tmp = static_allocator.allocate(len);
#else
      iterator tmp = static_allocator.allocate(len);
#endif
      uninitialized_copy(begin(), position, tmp);
      construct(tmp + (position - begin()), x);
      uninitialized_copy(position, end(), tmp + (position - begin()) + 1); 
      destroy(begin(), end());
      static_allocator.deallocate(begin());
      end_of_storage = tmp + len;
      finish = tmp + size() + 1;
      start = tmp;
    }
}

template <class T>
void vector<T>::insert(iterator position, size_type n, const T& x) {
    if (n == 0) return;
    if (end_of_storage - finish >= n) {
      if (end() - position > n) {
          uninitialized_copy(end() - n, end(), end());
          copy_backward(position, end() - n, end());
          fill(position, position + n, x);
      } else {
          uninitialized_copy(position, end(), position + n);
          fill(position, end(), x);
          uninitialized_fill_n(end(), n - (end() - position), x);
      }
      finish += n;
    } else {
      size_type len = size() + max(size(), n);
      iterator tmp = static_allocator.allocate(len);
      uninitialized_copy(begin(), position, tmp);
      uninitialized_fill_n(tmp + (position - begin()), n, x);
      uninitialized_copy(position, end(), tmp + (position - begin() + n));
      destroy(begin(), end());
      static_allocator.deallocate(begin());
      end_of_storage = tmp + len;
      finish = tmp + size() + n;
      start = tmp;
    }
}

template <class T>
void vector<T>::insert(iterator position, 
                   const_iterator first, 
                   const_iterator last) {
    if (first == last) return;
    size_type n = 0;
    distance(first, last, n);
    if (end_of_storage - finish >= n) {
      if (end() - position > n) {
          uninitialized_copy(end() - n, end(), end());
          copy_backward(position, end() - n, end());
          copy(first, last, position);
      } else {
          uninitialized_copy(position, end(), position + n);
          copy(first, first + (end() - position), position);
          uninitialized_copy(first + (end() - position), last, end());
      }
      finish += n;
    } else {
      size_type len = size() + max(size(), n);
      iterator tmp = static_allocator.allocate(len);
      uninitialized_copy(begin(), position, tmp);
      uninitialized_copy(first, last, tmp + (position - begin()));
      uninitialized_copy(position, end(), tmp + (position - begin() + n));
      destroy(begin(), end());
      static_allocator.deallocate(begin());
      end_of_storage = tmp + len;
      finish = tmp + size() + n;
      start = tmp;
    }
}

#undef Allocator
#undef vector

#endif

 



