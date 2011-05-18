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

#ifndef ITERATOR_H
#define ITERATOR_H

#include <stddef.h>
#include <iostream.h>
#include <bool.h>

struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};

template <class T, class Distance> struct input_iterator {};
struct output_iterator {};
template <class T, class Distance> struct forward_iterator {};
template <class T, class Distance> struct bidirectional_iterator {};
template <class T, class Distance> struct random_access_iterator {};

template <class T, class Distance> 
inline input_iterator_tag 
iterator_category(const input_iterator<T, Distance>&) {
    return input_iterator_tag();
}

inline output_iterator_tag iterator_category(const output_iterator&) {
    return output_iterator_tag();
}

template <class T, class Distance> 
inline forward_iterator_tag
iterator_category(const forward_iterator<T, Distance>&) {
    return forward_iterator_tag();
}

template <class T, class Distance> 
inline bidirectional_iterator_tag
iterator_category(const bidirectional_iterator<T, Distance>&) {
    return bidirectional_iterator_tag();
}

template <class T, class Distance> 
inline random_access_iterator_tag
iterator_category(const random_access_iterator<T, Distance>&) {
    return random_access_iterator_tag();
}

template <class T>
inline random_access_iterator_tag iterator_category(const T*) {
    return random_access_iterator_tag();
}


template <class T, class Distance> 
inline T* value_type(const input_iterator<T, Distance>&) {
    return (T*)(0); 
}

template <class T, class Distance> 
inline T* value_type(const forward_iterator<T, Distance>&) {
    return (T*)(0);
}

template <class T, class Distance> 
inline T* value_type(const bidirectional_iterator<T, Distance>&) {
    return (T*)(0);
}

template <class T, class Distance> 
inline T* value_type(const random_access_iterator<T, Distance>&) {
    return (T*)(0);
}

template <class T>
inline T* value_type(const T*) { return (T*)(0); }

template <class T, class Distance> 
inline Distance* distance_type(const input_iterator<T, Distance>&) {
    return (Distance*)(0);
}

template <class T, class Distance> 
inline Distance* distance_type(const forward_iterator<T, Distance>&) {
    return (Distance*)(0);
}

template <class T, class Distance> 
inline Distance* 
distance_type(const bidirectional_iterator<T, Distance>&) {
    return (Distance*)(0);
}

template <class T, class Distance> 
inline Distance* 
distance_type(const random_access_iterator<T, Distance>&) {
    return (Distance*)(0);
}

template <class T>
inline ptrdiff_t* distance_type(const T*) { return (ptrdiff_t*)(0); }


template <class Container>
class back_insert_iterator : public output_iterator {
protected:
#ifdef __CINT__
    Container* container;
#else
    Container& container;
#endif
public:
#ifdef __CINT__
    back_insert_iterator(Container& x) : container(&x) {}
#else
    back_insert_iterator(Container& x) : container(x) {}
#endif
    back_insert_iterator<Container>&
    operator=(const Container::value_type& value) { 
	container.push_back(value);
	return *this;
    }
    back_insert_iterator<Container>& operator*() { return *this; }
    back_insert_iterator<Container>& operator++() { return *this; }
    back_insert_iterator<Container> operator++(int) { return *this; }
};

template <class Container>
back_insert_iterator<Container> back_inserter(Container& x) {
    return back_insert_iterator<Container>(x);
}

template <class Container>
class front_insert_iterator : public output_iterator {
protected:
#ifdef __CINT__
    Container* container;
#else
    Container& container;
#endif
public:
#ifdef __CINT__
    front_insert_iterator(Container& x) : container(&x) {}
#else
    front_insert_iterator(Container& x) : container(x) {}
#endif
    front_insert_iterator<Container>&
    operator=(const Container::value_type& value) { 
	container.push_front(value);
	return *this;
    }
    front_insert_iterator<Container>& operator*() { return *this; }
    front_insert_iterator<Container>& operator++() { return *this; }
    front_insert_iterator<Container> operator++(int) { return *this; }
};

template <class Container>
front_insert_iterator<Container> front_inserter(Container& x) {
    return front_insert_iterator<Container>(x);
}

template <class Container>
class insert_iterator : public output_iterator {
protected:
#ifdef __CINT__
    Container* container;
#else
    Container& container;
#endif
    Container::iterator iter;
public:
#ifdef __CINT__
    insert_iterator(Container& x, Container::iterator i) 
	: container(&x), iter(i) {}
#else
    insert_iterator(Container& x, Container::iterator i) 
	: container(x), iter(i) {}
#endif
    insert_iterator<Container>&
    operator=(const Container::value_type& value) { 
	iter = container.insert(iter, value);
	++iter;
	return *this;
    }
    insert_iterator<Container>& operator*() { return *this; }
    insert_iterator<Container>& operator++() { return *this; }
    insert_iterator<Container>& operator++(int) { return *this; }
};

template <class Container, class Iterator>
insert_iterator<Container> inserter(Container& x, Iterator i) {
    return insert_iterator<Container>(x, Container::iterator(i));
}

template <class BidirectionalIterator, class T, class Reference, 
          class Distance> 
// Reference = T& 
// Distance = ptrdiff_t
class reverse_bidirectional_iterator 
    : public bidirectional_iterator<T, Distance> {
    typedef reverse_bidirectional_iterator<BidirectionalIterator, T, Reference,
                                           Distance> self;
    friend bool operator==(const reverse_bidirectional_iterator::self& x, const reverse_bidirectional_iterator::self& y);
protected:
    BidirectionalIterator current;
public:
    reverse_bidirectional_iterator() {}
    reverse_bidirectional_iterator(BidirectionalIterator x) : current(x) {}
    BidirectionalIterator base() { return current; }
    Reference operator*() const {
	BidirectionalIterator tmp = current;
	return *--tmp;
    }
    self& operator++() {
	--current;
	return *this;
    }
    self operator++(int) {
	self tmp = *this;
	--current;
	return tmp;
    }
    self& operator--() {
	++current;
	return *this;
    }
    self operator--(int) {
	self tmp = *this;
	++current;
	return tmp;
    }
};

template <class BidirectionalIterator, class T, class Reference,
          class Distance>
inline bool operator==(
    const reverse_bidirectional_iterator<BidirectionalIterator, T, Reference,
		                         Distance>& x, 
    const reverse_bidirectional_iterator<BidirectionalIterator, T, Reference,
		                         Distance>& y) {
    return x.current == y.current;
}

template <class RandomAccessIterator, class T, class Reference,
          class Distance> 
// Reference = T&
// Distance = ptrdiff_t
class reverse_iterator : public random_access_iterator<T, Distance> {
    typedef reverse_iterator<RandomAccessIterator, T, Reference, Distance>
	self;
    friend bool operator==(const reverse_iterator::self& x, const reverse_iterator::self& y);
    friend bool operator<(const reverse_iterator::self& x, const reverse_iterator::self& y);
    friend Distance operator-(const reverse_iterator::self& x, const reverse_iterator::self& y);
    friend reverse_iterator::self operator+(Distance n, const reverse_iterator::self& x);
protected:
    RandomAccessIterator current;
public:
    reverse_iterator() {}
    reverse_iterator(RandomAccessIterator x) : current(x) {}
    RandomAccessIterator base() { return current; }
    Reference operator*() const { return *(current - 1); }
    self& operator++() {
	--current;
	return *this;
    }
    self operator++(int) {
	self tmp = *this;
	--current;
	return tmp;
    }
    self& operator--() {
	++current;
	return *this;
    }
    self operator--(int) {
	self tmp = *this;
	++current;
	return tmp;
    }
    self operator+(Distance n) const {
	return self(current - n);
    }
    self& operator+=(Distance n) {
	current -= n;
	return *this;
    }
    self operator-(Distance n) const {
	return self(current + n);
    }
    self& operator-=(Distance n) {
	current += n;
	return *this;
    }
    Reference operator[](Distance n) { return *(*this + n); }
};

template <class RandomAccessIterator, class T, class Reference, class Distance>
inline bool operator==(const reverse_iterator<RandomAccessIterator, T,
		                              Reference, Distance>& x, 
		       const reverse_iterator<RandomAccessIterator, T,
		                              Reference, Distance>& y) {
    return x.current == y.current;
}

template <class RandomAccessIterator, class T, class Reference, class Distance>
inline bool operator<(const reverse_iterator<RandomAccessIterator, T,
		                             Reference, Distance>& x, 
		      const reverse_iterator<RandomAccessIterator, T,
		                             Reference, Distance>& y) {
    return y.current < x.current;
}

template <class RandomAccessIterator, class T, class Reference, class Distance>
inline Distance operator-(const reverse_iterator<RandomAccessIterator, T,
			                         Reference, Distance>& x, 
			  const reverse_iterator<RandomAccessIterator, T,
			                         Reference, Distance>& y) {
    return y.current - x.current;
}

template <class RandomAccessIterator, class T, class Reference, class Distance>
inline reverse_iterator<RandomAccessIterator, T, Reference, Distance> 
operator+(Distance n,
	  const reverse_iterator<RandomAccessIterator, T, Reference,
	                         Distance>& x) {
    return reverse_iterator<RandomAccessIterator, T, Reference, Distance>
	(x.current - n);
}


template <class OutputIterator, class T>
class raw_storage_iterator : public output_iterator {
protected:
    OutputIterator iter;
public:
    raw_storage_iterator(OutputIterator x) : iter(x) {}
    raw_storage_iterator<OutputIterator, T>& operator*() { return *this; }
    raw_storage_iterator<OutputIterator, T>& operator=(const T& element) {
	construct(iter, element);
	return *this;
    }        
    raw_storage_iterator<OutputIterator, T>& operator++() {
	++iter;
	return *this;
    }
    raw_storage_iterator<OutputIterator, T> operator++(int) {
	raw_storage_iterator<OutputIterator, T> tmp = *this;
	++iter;
	return tmp;
    }
};


template <class T, class Distance> // Distance == ptrdiff_t
class istream_iterator : public input_iterator<T, Distance> {
friend bool operator==(const istream_iterator<T, Distance>& x,
		       const istream_iterator<T, Distance>& y);
protected:
    istream* stream;
    T value;
    bool end_marker;
    void read() {
	end_marker = (*stream) ? true : false;
	if (end_marker) *stream >> value;
	end_marker = (*stream) ? true : false;
    }
public:
    istream_iterator() : stream(&cin), end_marker(false) {}
    istream_iterator(istream& s) : stream(&s) { read(); }
    const T& operator*() const { return value; }
    istream_iterator<T, Distance>& operator++() { 
	read(); 
	return *this;
    }
    istream_iterator<T, Distance> operator++(int)  {
	istream_iterator<T, Distance> tmp = *this;
	read();
	return tmp;
    }
};

template <class T, class Distance>
bool operator==(const istream_iterator<T, Distance>& x,
		const istream_iterator<T, Distance>& y) {
    return x.stream == y.stream && x.end_marker == y.end_marker ||
	x.end_marker == false && y.end_marker == false;
}

template <class T>
class ostream_iterator : public output_iterator {
protected:
    ostream* stream;
    char* string;
public:
    ostream_iterator(ostream& s) : stream(&s), string(0) {}
    ostream_iterator(ostream& s, char* c) : stream(&s), string(c)  {}
    ostream_iterator<T>& operator=(const T& value) { 
	*stream << value;
	if (string) *stream << string;
	return *this;
    }
    ostream_iterator<T>& operator*() { return *this; }
    ostream_iterator<T>& operator++() { return *this; } 
    ostream_iterator<T> operator++(int) { return *this; } 
};


#ifdef G__VISUAL
// added for CINT VC++5.0 configuration

template<class T,class Distance>
inline bidirectional_iterator_tag
iterator_category(const _Bidit<T,Distance>&) {
  return bidirectional_iterator_tag();
}

template<class T,class Distance>
inline random_access_iterator_tag
iterator_category(const _Ranit<T,Distance>&) {
  return random_access_iterator_tag();
}

template <class T, class Distance> 
inline T* value_type(const _Bidit<T, Distance>&) {
    return (T*)(0);
}

template <class T, class Distance> 
inline T* value_type(const _Randit<T, Distance>&) {
    return (T*)(0);
}
#endif

#if (G__GNUC>=3)

#if (G__GNUC_VER>=3001) 

// vector ///////////////////////////////////////////////////////////////
template <class T> 
inline random_access_iterator_tag iterator_category(const vector<T>::iterator&)
  {return random_access_iterator_tag();}

template <class T> 
inline T* value_type(const vector<T>::iterator&) {return (T*)(0);}

template <class T, class Distance> 
inline Distance* 
distance_type(const vector<T>::iterator&) {return (Distance*)(0);}

// list ///////////////////////////////////////////////////////////////
template <class T> 
inline bidirectional_iterator_tag iterator_category(const list<T>::iterator&)
  {return bidirectional_iterator_tag();}

template <class T> 
inline T* value_type(const list<T>::iterator&) {return (T*)(0);}

template <class T, class Distance> 
inline Distance* 
distance_type(const list<T>::iterator&) {return (Distance*)(0);}

// deque ///////////////////////////////////////////////////////////////
template <class T> 
inline random_access_iterator_tag iterator_category(const deque<T>::iterator&)
  {return random_access_iterator_tag();}

template <class T> 
inline T* value_type(const deque<T>::iterator&) {return (T*)(0);}

template <class T,class Distance>
inline Distance* 
distance_type(const deque<T>::iterator&) {return (Distance*)(0);}

// map ///////////////////////////////////////////////////////////////
template <class Key,class T> 
inline bidirectional_iterator_tag iterator_category(const map<Key,T>::iterator&)
  {return bidirectional_iterator_tag();}

template <class Key,class T> 
inline T* value_type(const map<Key,T>::iterator&) {return (T*)(0);}

template <class Key,class T,class Distance> 
inline Distance* 
distance_type(const map<Key,T>::iterator&) {return (Distance*)(0);}

// set ///////////////////////////////////////////////////////////////
template <class T> 
inline bidirectional_iterator_tag iterator_category(const set<T>::iterator&)
  {return bidirectional_iterator_tag();}

template <class T> 
inline T* value_type(const set<T>::iterator&) {return (T*)(0);}

template <class T,class Distance> 
inline Distance* 
distance_type(const set<T>::iterator&) {return (Distance*)(0);}

#endif // (GNUC_VER>=3001)

// This iterator adapter is 'normal' in the sense that it does not
// change the semantics of any of the operators of its itererator
// parameter.  Its primary purpose is to convert an iterator that is
// not a class, e.g. a pointer, into an iterator that is a class.
// The _Container parameter exists solely so that different containers
// using this template can instantiate different types, even if the
// _Iterator parameter is the same.
template<typename _Iterator, typename _Container>
class __normal_iterator
  : public iterator<iterator_traits<_Iterator>::iterator_category,
                    iterator_traits<_Iterator>::value_type,
                    iterator_traits<_Iterator>::difference_type,
                    iterator_traits<_Iterator>::pointer,
                    iterator_traits<_Iterator>::reference>
{

protected:
  _Iterator _M_current;

public:
  typedef __normal_iterator<_Iterator, _Container> normal_iterator_type;
  typedef iterator_traits<_Iterator>  			__traits_type;
  typedef typename __traits_type::iterator_category 	iterator_category;
  typedef typename __traits_type::value_type 		value_type;
  typedef typename __traits_type::difference_type 	difference_type;
  typedef typename __traits_type::pointer          	pointer;
  typedef typename __traits_type::reference 		reference;

  __normal_iterator() : _M_current(_Iterator()) { }

  explicit __normal_iterator(const _Iterator& __i) : _M_current(__i) { }

  // Allow iterator to const_iterator conversion
  template<typename _Iter>
  inline __normal_iterator(const __normal_iterator<_Iter, _Container>& __i)
    : _M_current(__i.base()) { }

  // Forward iterator requirements
  reference
  operator*() const { return *_M_current; }

  pointer
  operator->() const { return _M_current; }

  normal_iterator_type&
  operator++() { ++_M_current; return *this; }

  normal_iterator_type
  operator++(int) { return __normal_iterator(_M_current++); }

  // Bidirectional iterator requirements
  normal_iterator_type&
  operator--() { --_M_current; return *this; }

  normal_iterator_type
  operator--(int) { return __normal_iterator(_M_current--); }

  // Random access iterator requirements
  reference
  operator[](const difference_type& __n) const
  { return _M_current[__n]; }

  normal_iterator_type&
  operator+=(const difference_type& __n)
  { _M_current += __n; return *this; }

  normal_iterator_type
  operator+(const difference_type& __n) const
  { return __normal_iterator(_M_current + __n); }

  normal_iterator_type&
  operator-=(const difference_type& __n)
  { _M_current -= __n; return *this; }

  normal_iterator_type
  operator-(const difference_type& __n) const
  { return __normal_iterator(_M_current - __n); }

  difference_type
  operator-(const normal_iterator_type& __i) const
  { return _M_current - __i._M_current; }

  const _Iterator& 
  base() const { return _M_current; }
};

// forward iterator requirements

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator==(const __normal_iterator<_IteratorL, _Container>& __lhs,
	   const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator!=(const __normal_iterator<_IteratorL, _Container>& __lhs,
	   const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return !(__lhs == __rhs); }

// random access iterator requirements

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool 
operator<(const __normal_iterator<_IteratorL, _Container>& __lhs,
	  const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>(const __normal_iterator<_IteratorL, _Container>& __lhs,
	  const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __rhs < __lhs; }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<=(const __normal_iterator<_IteratorL, _Container>& __lhs,
	   const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return !(__rhs < __lhs); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>=(const __normal_iterator<_IteratorL, _Container>& __lhs,
	   const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return !(__lhs < __rhs); }

template<typename _Iterator, typename _Container>
inline __normal_iterator<_Iterator, _Container>
operator+(__normal_iterator<_Iterator, _Container>::difference_type __n,
          const __normal_iterator<_Iterator, _Container>& __i)
{ return __normal_iterator<_Iterator, _Container>(__i.base() + __n); }

template <class T, class Container> 
inline random_access_iterator_tag iterator_category(const __normal_iterator<T, Container>&) {
    return random_access_iterator_tag();
}
template <class T, class Container> 
inline T* value_type(const __normal_iterator<T, Container>&) {
    return (T*)(0); 
}
#endif

#endif
