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

namespace MySpace {

  class A : public TObject {
  public:
    int a;
    //    ClassDef(MySpace::A,1)
    ClassDefOverride(A,1)
  };

  class MyClass : public A {
  public:
    int a;
    A b;
    MyClass() {}
    MyClass(int a_) : a(a_) {}
    std::vector<A> member;
    ClassDefOverride(MyClass,1)
  };

} // end namespace

void namespace_driver();
void testNamespaceWrite();
