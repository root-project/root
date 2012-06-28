/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

template<class T> class A;

A<int> *pa;

template<class T> class A {
  T* p;
 public:
  A() ;
  void set(T in) ;
  ~A();
  void disp() ;
};

//A<int> *pa;
A<double> *pb;

template<class T> A<T>::A() {
  p=0;
}

template<class T> A<T>::~A() {
  if(p) delete p;
}

template<class T> void A<T>::set(T in) {
  if(p) delete p;
  p=new T(in);
}

template<class T> void A<T>::disp() {
  if(p) {
    cout << *p << endl;
  }
}

A<char> c;

int main() {
  A<short> d;
  pa=new A<int>;
  pa->set(0xfffff);
  pb=new A<double>;
  pb->set(3.1416);
  c.set(123);
  d.set(0xfff);
  pa->disp();
  pb->disp();
  c.disp();
  d.disp();
  delete pb;
  delete pa;
  return 0;
}
