/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <list>

template<class T> class normal_iterator {};

template<class T> class Allocator {};

template<class T,class alloc=Allocator<T> > class List {
 public:
  typedef normal_iterator<int> iterator;
};

class BaseStar {
 public:
};

//void f( list<BaseStar*>& x) { }

#ifdef __MAKECINT__
#pragma link C++ class list<BaseStar*>::iterator;
//#pragma link C++ typedef list<BaseStar*>::iterator;
#pragma link C++ typedef List<BaseStar*>::iterator;
#endif

