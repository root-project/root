#include <iostream>
using namespace std;

struct C { };

const char* nameof(C*) { return "C"; }

struct A {
  A() { };

  template <class U>
  A(const U& u) { 
#if defined(__CINT__) || defined(_MSC_VER)
	  cout << "A::A(const U &) [with U = " << nameof((U*)0) << "]" << endl;
#else
	  cout << __PRETTY_FUNCTION__ << endl; 
#endif
  }

  template <class U>
  void doit(const U& u) { 
#if defined(__CINT__) || defined(_MSC_VER)
	  cout << "void A::doit(const U &) [with U = " << nameof((U*)0) << "]" << endl;
#else
	  cout << __PRETTY_FUNCTION__ << endl; 
#endif
  } 

};

#ifdef __MAKECINT__
#pragma link C++ class A+;
#pragma link C++ class C+;
#pragma link C++ function A::A(const C&);
#pragma link C++ function A::doit(const C&);
#endif

