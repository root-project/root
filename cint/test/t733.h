/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


template <class Iterator>
struct iterator_traits {
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
  pointer a;
  Iterator b;
  Iterator* c;
};

#if 1
// template partial specialization, need to implement in CINT
template <class T>
struct iterator_traits<T*> {
  typedef T*                         pointer;
  typedef T&                         reference;
  pointer a;
  T b;
  T* c;
};
#endif

#if  1
template <class T>
struct iterator_traits<const T*> {
  typedef const T*                   pointer;
  typedef const T&                   reference;
  pointer a;
  const T b;
  const T* c;
};
#endif

class iter {
 public:
  typedef void* pointer ;
  typedef void* reference ;
};

iterator_traits<iter> a1;
iterator_traits<int*> b2;
iterator_traits<const int*> c3;

