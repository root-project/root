/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if 1

#include <stdio.h>

double y=123.45;
typedef double* DListIterator;
void f(DListIterator& x) { printf("void f(DListIterator&) %g\n",*x); }

int main() {
  DListIterator iter = &y;
  f(*(DListIterator*)(&iter));
  return 0;
}

#else

template<class T> class vector {
  T* x;
public:
  typedef T* iterator ;
  iterator begin() { return(x); }
};

typedef vector<double> DList;
typedef vector<double>::iterator DListIterator;

class Compiled1 {
 public:
  Compiled1()  {}
  virtual void publicFunc1( void ) { }
  virtual void publicFunc2( DListIterator &iter ) { }
};

main() {
  Compiled1 a;
  DList x;
  DListIterator i=x.begin();
  char buf[200];
  sprintf(buf,"a.publicFunc2(*(DListIterator*)(%ld))",(long)(&i));
  G__calc(buf);
}

#endif
