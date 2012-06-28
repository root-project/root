/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef CINT_HIDE_FAILURE
#define TEST2  // TEST2 still has problem
#endif
#define NS

#include <stdio.h>

template<class T,class U>  T convert(const U& x) { return(T(x)); }

class A { 
public:
  double d;
  int i;
  A(const A& x) : d(x.d), i(x.i) { }
  explicit A(double x) : d(x), i(0) { }
  explicit A(int x) : d(0), i(x) { }
  operator double() const { return d; }
  operator int() const { return i; }
  void disp() const {printf("A() %g %d\n",d,i);}
};

namespace ns {

  class B { 
  public:
    double d;
    int i;
    B(const B& x) : d(x.d), i(x.i) { }
    explicit B(double x) : d(x), i(0) { }
  explicit B(int x) : d(0), i(x) { }
    operator double() const { return d; }
    operator int() const { return i; }
    void disp() const {printf("B() %g %d\n",d,i);}
  };
  template<class T,class U>  T conv(const U& x) { printf("template "); return(T(x)); }
  template<> B conv<B>(const int& x) { printf("specialization "); return(B(x)); }
}

void test1() {
  double d=6.28;
  A ad(3.14);
  A ai(123);

  ad.disp();
  ai.disp();

  for(int i=0;i<3;i++) {
    ad = convert<A>(d+i);
    ai = convert<A>(i+567);
    ad.disp();
    ai.disp();
  }
}

#ifdef NS
using namespace ns;
#endif
void test2() {
  double d=6.28;
  ns::B ad(3.14);
  ns::B ai(123);

  ad.disp();
  ai.disp();

  for(int i=0;i<3;i++) {
#ifdef NS
    ad = conv<B>(d+i);
    ai = conv<B>(i+567);
#endif
    ad = ns::conv<ns::B>(d+i);
    ai = ns::conv<ns::B>(i+567);
    ad.disp();
    ai.disp();
  }
}

int main() {
  //printf("t695 causes problem due to member function template specialization\n");
  test1();
#ifdef TEST2
  test2();
#endif
  return 0;
}
