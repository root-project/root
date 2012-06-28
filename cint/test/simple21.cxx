/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


template<class T> class A {
 public:
  void set(T in) ;
};


template<class T> void A<T>::set(T in) {
}

int main() {
  A<int> *pa;
  pa=new A<int>;
  pa->set(0xfffff);
  return 0;
}
