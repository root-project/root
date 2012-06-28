/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef T928_H
#define T928_H

#include <cstdio>
#include <typeinfo>
using namespace std;

template<class T> 
class A {
 public:
  T ddd;
  template<class E> void f(void) { 
    T t;
    E e;
    printf("void A<%s>::f<%s>(void)\n",typeid(t).name(),typeid(e).name());
  }
  template<class E> void g(E& e) { 
    T t;
    printf("void A<%s>::e(%s&)\n",typeid(t).name(),typeid(e).name());
  }
  void x() {
    f<short>();
  }
};


class B {
 public:
};

//void A<int>::f<B>(void);
//void A<int>::f<int>(void);

template<class E> void F(void) { 
  E e;
  printf("void ::F<%s>(void)\n",typeid(e).name());
}

void f() {
  A<int> x;
  x.f<int>();
  F<int>();
}


#ifndef T928A
#ifdef __MAKECINT__
#pragma link C++ function A<int>::f<B>;
#pragma link C++ function A<int>::f<int>;
#pragma link C++ function A<short>::f<B>();
#pragma link C++ function A<short>::f<long>();
#pragma link C++ function A<int>::g(B);
#pragma link C++ function A<int>::g(int);
#endif
#endif


#endif
