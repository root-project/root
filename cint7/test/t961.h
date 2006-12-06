/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// problem remains with Borland C++ 5.5

#if 1

#include <vector>

#else

typedef unsigned int size_type;
template<class T> class allocator { };

template<class T,class Allocator=allocator<T> >
class vector {
 public:
  vector(size_type n,const T& value=T()) ;
};

#endif


template <class T>  class c {

  std::vector<c<T>*> fDummy;
  //std::vector<c<T> > fDummy;

 public:
  c() ;
  c(const c &cc);
   
   //ClassDefT(c<T>,1)
};

template<class T> c<T>::c(const c &cc) {}
template<class T> c<T>::c() {}


#ifdef __MAKECINT__
#pragma link C++ class c<float>;
#else
c<float> x;
#endif


