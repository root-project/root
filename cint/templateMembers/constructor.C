#include <iostream>
using namespace std;

struct C { };

struct A {
  A() { };

  template <class U>
  A(const U& u) { cout << __PRETTY_FUNCTION__ << endl; }

  template <class U>
  void doit(const U& u) { cout << __PRETTY_FUNCTION__ << endl; } };

#ifdef __MAKECINT__
#pragma link C++ class A+;
#pragma link C++ class C+;
#pragma link C++ function A::A(const C&);
#pragma link C++ function A::doit(const C&);
#endif

