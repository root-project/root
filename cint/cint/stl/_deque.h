/* -*- C++ -*- */

/************************************************************************
 *
 * Copyright(c) 1995~2006  Masaharu Goto (root-cint@cern.ch)
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

#ifndef DEQUE_H
#define DEQUE_H

#include <function.h>
#include <algobase.h>
#include <_iterator.h>
#include <bool.h>

#ifndef Allocator
#define Allocator allocator
#include <defalloc.h>
#endif

#ifndef __CINT__
#ifndef deque 
#define deque deque
#endif
#endif

template <class T> 
class deque {
public:
    class iterator;
    class const_iterator;
    friend class iterator;
    friend class const_iterator;
public:
    typedef T value_type;
    typedef Allocator<T> data_allocator_type;
    typedef Allocator<T>::pointer pointer;
    typedef Allocator<T>::reference reference;
    typedef Allocator<T>::const_reference const_reference;
    typedef Allocator<T>::size_type size_type;
    typedef Allocator<T>::difference_type difference_type;
    typedef Allocator<pointer> map_allocator_type;   
protected:
    static data_allocator_type data_allocator;
    static size_type buffer_size;
    static map_allocator_type map_allocator;
    typedef Allocator<pointer>::pointer map_pointer;
public:
    class iterator : public random_access_iterator<T, difference_type> {
    friend class deque<T>;
    friend class const_iterator;
    protected:
	pointer current;
	pointer first;
	pointer last;
	map_pointer node;
	iterator(pointer x, map_pointer y) 
	    : current(x), first(*y), last(*y + buffer_size), node(y) {}
    public:
	iterator() : current(0), first(0), last(0), node(0) {}
	reference operator*() const { return *current; }
	difference_type operator-(const iterator& x) const {
	    return node == x.node 
		? current - x.current 
		: difference_type(buffer_size * (node - x.node - 1) +
				  (current - first) + (x.last - x.current));
	}
	iterator& operator++() {
	    if (++current == last) {
		first = *(++node);
		current = first;
		last = first + buffer_size;
	    }
	    return *this; 
	}
	iterator operator++(int)  {
	    iterator tmp = *this;
	    ++*this;
	    return tmp;
	}
	iterator& operator--() {
	    if (current == first) {
		first = *(--node);
		last = first + buffer_size;
		current = last;
	    }
	    --current;
	    return *this;
	}
	iterator operator--(int) {
	    iterator tmp = *this;
	    --*this;
	    return tmp;
	}
	iterator& operator+=(difference_type n) {
	    difference_type offset = n + (current - first);
	    difference_type num_node_to_jump = offset >= 0
		? offset / buffer_size
		: -((-offset + buffer_size - 1) / buffer_size);
	    if (num_node_to_jump == 0)
		current += n;
	    else {
		node = node + num_node_to_jump;
		first = *node;
		last = first + buffer_size;
		current = first + (offset - num_node_to_jump * buffer_size);
	    }
	    return *this;
	}
	iterator& operator-=(difference_type n) { return *this += -n; }
	iterator operator+(difference_type n) const {
	    iterator tmp = *this;
	    return tmp += n;
	}
	iterator operator-(difference_type n) const {
	    iterator tmp = *this;
	    return tmp -= n;
	}
	reference operator[](difference_type n) { return *(*this + n); }
	bool operator==(const iterator& x) const {      
	    return current == x.current || 
		((current == first || x.current == x.first) && 
		 *this - x == 0);
	}
	bool operator<(const iterator& x) const {
	    return (node == x.node) ? (current < x.current) : (node < x.node);
	}
    };
    class const_iterator : public random_access_iterator<T, difference_type> {
    friend class deque<T>;
    protected:
	pointer current;
	pointer first;
	pointer last;
	map_pointer node;
	const_iterator(pointer x, map_pointer y) 
	    : current(x), first(*y), last(*y + buffer_size), node(y) {}
    public:
	const_iterator() : current(0), first(0), last(0), node(0) {}
	const_iterator(const iterator& x) 
	    : current(x.current), first(x.first), last(x.last), node(x.node) {}     
	const_reference operator*() const { return *current; }
	difference_type operator-(const const_iterator& x) const {
	    return node == x.node 
		? current - x.current 
		: difference_type(buffer_size * (node - x.node - 1) +
				  (current - first) + (x.last - x.current));
	}
	const_iterator& operator++() {
	    if (++current == last) {
		first = *(++node);
		current = first;
		last = first + buffer_size;
	    }
	    return *this; 
	}
	const_iterator operator++(int)  {
	    const_iterator tmp = *this;
	    ++*this;
	    return tmp;
	}
	const_iterator& operator--() {
	    if (current == first) {
		first = *(--node);
		last = first + buffer_size;
		current = last;
	    }
	    --current;
	    return *this;
	}
	const_iterator operator--(int) {
	    const_iterator tmp = *this;
	    --*this;
	    return tmp;
	}
	const_iterator& operator+=(difference_type n) {
	    difference_type offset = n + (current - first);
	    difference_type num_node_to_jump = offset >= 0
		? offset / buffer_size
		: -((-offset + buffer_size - 1) / buffer_size);
	    if (num_node_to_jump == 0)
		current += n;
	    else {
		node = node + num_node_to_jump;
		first = *node;
		last = first + buffer_size;
		current = first + (offset - num_node_to_jump * buffer_size);
	    }
	    return *this;
	}
	const_iterator& operator-=(difference_type n) { return *this += -n; }
	const_iterator operator+(difference_type n) const {
	    const_iterator tmp = *this;
	    return tmp += n;
	}
	const_iterator operator-(difference_type n) const {
	    const_iterator tmp = *this;
	    return tmp -= n;
	}
	const_reference operator[](difference_type n) { 
	    return *(*this + n); 
	}
	bool operator==(const const_iterator& x) const {      
	    return current == x.current || 
		((current == first || x.current == x.first) && 
		 *this - x == 0);
	}
	bool operator<(const const_iterator& x) const {
	    return (node == x.node) ? (current < x.current) : (node < x.node);
	}
    };
    typedef reverse_iterator<const_iterator, value_type, const_reference, 
                             difference_type>  const_reverse_iterator;
    typedef reverse_iterator<iterator, value_type, reference, difference_type>
        reverse_iterator; 
protected:    
    iterator start;
    iterator finish;
    size_type length;
    map_pointer map;
    size_type map_size;

    void allocate_at_begin();
    void allocate_at_end();
    void deallocate_at_begin();
    void deallocate_at_end();

public:
    deque() : start(), finish(), length(0), map(0), map_size(0) {
	buffer_size = data_allocator.init_page_size();
    }
#ifdef __CINT__
    iterator begin() { return (iterator(start)); }
    iterator end() { return (iterator(finish)); }
#else
    iterator begin() { return start; }
    const_iterator begin() const { return start; }
    iterator end() { return finish; }
    const_iterator end() const { return finish; }
#endif
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { 
        return const_reverse_iterator(end()); 
    }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { 
        return const_reverse_iterator(begin()); 
    } 
    bool empty() const { return length == 0; }
    size_type size() const { return length; }
    size_type max_size() const { return data_allocator.max_size(); }
    reference operator[](size_type n) { return *(begin() + n); }
    const_reference operator[](size_type n) const { return *(begin() + n); }
    reference front() { return *begin(); }
    const_reference front() const { return *begin(); }
    reference back() { return *(end() - 1); }
    const_reference back() const { return *(end() - 1); }
    void push_front(const T& x) {
	if (empty() || begin().current == begin().first)
	    allocate_at_begin();
	--start.current;
	construct(start.current, x);
	++length;
    }
    void push_back(const T& x) {
	if (empty() || end().current == end().last)
	    allocate_at_end();
	construct(finish.current, x);
	++finish.current;
	++length;
    }
    void pop_front() {
	destroy(start.current);
	++start.current;
	--length; 
	if (empty() || begin().current == begin().last)
	    deallocate_at_begin();
    }
    void pop_back() {
	--finish.current;
	destroy(finish.current);
	--length; 
	if (empty() || end().current == end().first)
	    deallocate_at_end();
    }
    void swap(deque<T>& x) {
	::swap(start, x.start);
	::swap(finish, x.finish);
	::swap(length, x.length);
	::swap(map, x.map);
	::swap(map_size, x.map_size);
    }
    iterator insert(iterator position, const T& x);
    void insert(iterator position, size_type n, const T& x);
//  template <class Iterator> void insert(iterator position,
//                                        Iterator first, Iterator last);
    void insert(iterator position, const T* first, const T* last);
    void erase(iterator position);
    void erase(iterator first, iterator last);    
    deque(size_type n, const T& value = T())
	: start(), finish(), length(0), map(0), map_size(0) {
	buffer_size = data_allocator.init_page_size();  
	insert(begin(), n, value);
    }
//  template <class Iterator> deque(Iterator first, Iterator last);
    deque(const T* first, const T* last)
	: start(), finish(), length(0), map(0), map_size(0) {
	buffer_size = data_allocator.init_page_size();  
	copy(first, last, back_inserter(*this));
    }
    deque(const deque<T>& x)
	: start(), finish(), length(0), map(0), map_size(0) {
	buffer_size = data_allocator.init_page_size();  
	copy(x.begin(), x.end(), back_inserter(*this));
    }
    deque<T>& operator=(const deque<T>& x) {
	if (this != &x)
	    if (size() >= x.size()) 
		erase(copy(x.begin(), x.end(), begin()), end());
	    else 
		copy(x.begin() + size(), x.end(),
		     inserter(*this, copy(x.begin(), x.begin() + size(),
					  begin())));
	return *this;
    }
    ~deque();
};

template <class T>
deque<T>::data_allocator_type deque<T>::data_allocator;

template <class T>
deque<T>::map_allocator_type deque<T>::map_allocator;

template <class T>
deque<T>::size_type deque<T>::buffer_size = 0; 
// should be data_allocator.init_page_size(); // Borland bug

template <class T>
bool operator==(const deque<T>& x, const deque<T>& y) {
    return x.size() == y.size() && equal(x.begin(), x.end(), y.begin());
}

template <class T>
bool operator<(const deque<T>& x, const deque<T>& y) {
    return lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

template <class T>
deque<T>::~deque() { while (!empty()) pop_front(); }     

template <class T>
void deque<T>::allocate_at_begin() {
    pointer p = data_allocator.allocate(buffer_size);
    if (!empty()) {
	if (start.node == map) {
	    difference_type i = finish.node - start.node;
	    map_size = (i + 1) * 2;
	    map_pointer tmp = map_allocator.allocate(map_size);
	    copy(start.node, finish.node + 1, tmp + map_size / 4 + 1);
	    map_allocator.deallocate(map);
	    map = tmp;
	    map[map_size / 4] = p;
	    start = iterator(p + buffer_size, map + map_size / 4);
	    finish = iterator(finish.current, map + map_size / 4 + i + 1);
	} else {
	    *--start.node = p;
	    start = iterator(p + buffer_size, start.node);
	}
    } else {
	map_size = map_allocator.init_page_size();
	map = map_allocator.allocate(map_size);
	map[map_size / 2] = p;
	start = iterator(p + buffer_size / 2 + 1, map + map_size / 2);
	finish = start;
    }
}

template <class T>
void deque<T>::allocate_at_end() {
    pointer p = data_allocator.allocate(buffer_size);
    if (!empty()) {
	if (finish.node == map + map_size - 1) {
	    difference_type i = finish.node - start.node;
		 map_size = (i + 1) * 2;
	    map_pointer tmp = map_allocator.allocate(map_size);
	    copy(start.node, finish.node + 1, tmp + map_size / 4);
	    map_allocator.deallocate(map);
	    map = tmp;
		 map[map_size / 4 + i + 1] = p;
	    start = iterator(start.current, map + map_size / 4);
	    finish = iterator(p, map + map_size / 4 + i + 1);
	} else {
	    *++finish.node = p;
	    finish = iterator(p, finish.node);
	}
    } else {
	map_size = map_allocator.init_page_size();
	map = map_allocator.allocate(map_size);
	map[map_size / 2] = p;
	start = iterator(p + buffer_size / 2, map + map_size / 2);
	finish = start;
    }
}

template <class T>
void deque<T>::deallocate_at_begin() {
    data_allocator.deallocate(*start.node++);
    if (empty()) {
	start = iterator();
	finish = start;
	map_allocator.deallocate(map);
    } else
	start = iterator(*start.node, start.node);
}

template <class T>
void deque<T>::deallocate_at_end() {
    data_allocator.deallocate(*finish.node--);
    if (empty()) {
	start = iterator();
	finish = start;
	map_allocator.deallocate(map);
    } else
	finish = iterator(*finish.node + buffer_size, finish.node);
}

template <class T>
deque<T>::iterator deque<T>::insert(iterator position, const T& x) {
    if (position == begin()) {
	push_front(x);
	return begin();
    } else if (position == end()) {
	push_back(x);
	return end() - 1;
    } else if (end() - position > position - begin()) {
	push_front(*begin());
	copy(begin() + 2, position, begin() + 1); 
	*(position - 1) = x;
	return position - 1;
    } else {
	push_back(*(end() - 1));
	copy_backward(position, end() - 2, end() - 1); 
	*position = x;
	return position;
    }
}

template <class T>
void deque<T>::insert(iterator position, size_type n, const T& x) {
    if (end() - position > position - begin()) {
	iterator old_begin = begin();
	if (n > position - old_begin) {
		 size_type m = n - (position - old_begin);
	    while (m-- > 0) push_front(x);
	    iterator i = position;
	    while (i != old_begin) push_front(*--i);
	    fill(old_begin, position, x);
	} else {
	    iterator i = old_begin + n;
	    while (i != old_begin) push_front(*--i);
	    copy(old_begin + n, position, old_begin);
	    fill(position - n, position, x);
	}
    } else {
	iterator old_end = end();
	if (n > old_end - position) {
		 size_type m = n - (old_end - position);
	    while (m-- > 0) push_back(x);
	    iterator i = position;
	    while (i != old_end) push_back(*i++);
	    fill(position, old_end, x);
	} else {
	    iterator i = old_end - n;
	    while (i != old_end) push_back(*i++);
	    copy_backward(position, old_end - n, old_end);
	    fill(position, position + n, x);
	}
    }
}

template <class T>
void deque<T>::insert(iterator position, const T* first, const T* last) {
    size_type n = 0;
    distance(first, last, n);
    if (end() - position > position - begin()) {
	iterator old_begin = begin();
	if (n > position - old_begin) {
		 const T* m = last - (position - old_begin);
	    while (m != first) push_front(*--m);
	    iterator i = position;
	    while (i != old_begin) push_front(*--i);
	    copy(last - (position - old_begin), last, old_begin);
	} else {
	    iterator i = old_begin + n;
	    while (i != old_begin) push_front(*--i);
	    copy(old_begin + n, position, old_begin);
	    copy(first, last, position - n);
	}
    } else {
	iterator old_end = end();
	if (n > old_end - position) {
		 const T* m = first + (old_end - position);
	    while (m != last) push_back(*m++);
	    iterator i = position;
	    while (i != old_end) push_back(*i++);
		 copy(first, first + (old_end - position), position);
	} else {
	    iterator i = old_end - n;
	    while (i != old_end) push_back(*i++);
	    copy_backward(position, old_end - n, old_end);
	    copy(first, last, position);
	}
    }
}

template <class T>
void deque<T>::erase(iterator position) {
    if (end() - position > position - begin()) {
	copy_backward(begin(), position, position + 1);
	pop_front();
    } else {
	copy(position + 1, end(), position);
	pop_back();
    }
}
    
template <class T>
void deque<T>::erase(iterator first, iterator last) {
	 difference_type n = last - first;
    if (end() - last > first - begin()) {
	copy_backward(begin(), first, last);
	while(n-- > 0) pop_front();
    } else   {
	copy(last, end(), first);
	while(n-- > 0) pop_back();
    }
}

#undef Allocator
#undef deque

#endif
