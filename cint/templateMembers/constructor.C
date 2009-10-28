#include <iostream>
using namespace std;

struct C { };

class TRootIOCtor;
const char* nameof(C*) { return "C"; }
const char* nameof(TRootIOCtor*) { return "TRootIOCtor"; }
const char* nameof(TRootIOCtor**) { return "TRootIOCtor*"; }

struct A {
  A() { };

  template <class U>
#if defined(__CINT__) || defined(_MSC_VER)
  A(const U& u) { 
	  cout << "A::A(const U &) [with U = " << nameof((U*)0) << "]" << endl;
#else
  A(const U&) { 
	  cout << __PRETTY_FUNCTION__ << endl; 
#endif
  }

  template <class U>
#if defined(__CINT__) || defined(_MSC_VER)
  void doit(const U& u) { 
	  cout << "void A::doit(const U &) [with U = " << nameof((U*)0) << "]" << endl;
#else
  void doit(const U&) { 
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

