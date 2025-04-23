// test for ROOT-7265
namespace Test {
  class OuterClass {
  public:
    class InnerClass { };
    typedef InnerClass InnerClassAlias;
  };
}

#if defined(__MAKECINT__)
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ namespace Test;
#pragma link C++ class Test::OuterClass::InnerClassAlias+;
#endif

using namespace Test;

void execnamespaceconflict(){};
