#include "TFile.h"
#include "vector"

class MyClass0 {
public:
  int a;
  class MyNestedClass {
  public:
    int b;
  };
};

//#define NOSPACE
#ifndef NOSPACE
namespace MySpace {
#endif

  class A : public TObject {
  public:
    int a;
    ClassDef(MySpace::A,1)
  };

  class MyClass : public TObject {
  public:
#ifdef NESTING_PROBLEM_SOLVED
    std::vector<A> member;
#else
    std::vector<MySpace::A> member;
#endif
    ClassDef(MyClass,1)
  };

#ifndef NOSPACE
} // end namespace

void testNamespaceWrite();

#endif
