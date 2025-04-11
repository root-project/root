#include <cstring>
#include <iostream>
using namespace std;

struct TemplateArgument {};
class TRootIOCtor;
const char* nameof(TemplateArgument*) { return "TemplateArgument"; }
const char* nameof(TRootIOCtor*) { return "TRootIOCtor"; }
const char* nameof(TRootIOCtor**) { return "TRootIOCtor*"; }

const char* PF2cl(const char* PF) {
   if (strstr(PF,"TemplateArgument")) return "TemplateArgument";
   return "NOT TemplateArgument!";
}

struct A {
  A() { };

  template <class U>
   A(const U&) { 
      cout << "c'tor taking " << 
#if defined(__CINT__) || defined(_MSC_VER)
      nameof((U*)0)
#else
      PF2cl(__PRETTY_FUNCTION__)
#endif
      << endl;
  }

  template <class U>
   void doit(const U&) { 
      cout << "doit taking " << 
#if defined(__CINT__) || defined(_MSC_VER)
      nameof((U*)0)
#else
      PF2cl(__PRETTY_FUNCTION__)
#endif
      << endl;
  } 

};

#ifdef __MAKECINT__
#pragma link C++ class A+;
#pragma link C++ class TemplateArgument+;
#pragma link C++ function A::A(const TemplateArgument&);
#pragma link C++ function A::doit(const TemplateArgument&);
#endif

