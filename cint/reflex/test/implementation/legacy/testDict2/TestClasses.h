#include <list>
#include <cstdlib>

namespace {
class ForwardedUnnamedType;
}

namespace xmlTypedefSelection {
class RealXmlSelClass {};

typedef RealXmlSelClass TypedefXmlSelClass;
typedef TypedefXmlSelClass TypedefXmlSelClass2;

} // ns xmlTypedefSelection

namespace testclasses {
class WithArray {
public:
   WithArray() {
      m_a[0] = 1;
      m_a[1] = 2;
      m_a[2] = 3;
      m_a[3] = 4;
      m_a[4] = 5;
   }


private:
   int m_a[5];
};


template <int i> class testTemplInt {
   int
   foo() { return i; }

};

static const int staticint = 1;

class TestFunctionReturnTypes {
public:
   int
   returnFundamental() { return 0; }

   A
   returnCompound() { A* a = new A(); return *a; }

   A*
   returnPointer1() { return new A(); }

   int*
   returnPointer2() { return 0; }

   void
   returnVoid() {}

   const int&
   returnRef1() { return staticint; }

   A&
   returnRef2() { A* a = new A(); return *a; }

};


class Outer {
public:
   class Inner {
   public:
      class IInner {};
   };
};


class TemplFun {
public:
   template <class T> void
   foooo(T /*t*/) {}

};


class DataMembers {
   typedef int (*MYFUNPTR)(const std::vector<double>&, const std::vector<double>&);

   int i;
   int* pi;
   int** ppi;
   int pa[5];
   int paa[5][5];
   int paa2[5][4][3][2][1];
   int** paa3[5][3][1];
   MYFUNPTR fPtr2;
   int (* fPtr)(int,
                char,
                bool);

};

class Typedefs {
   typedef int MYINT;
   typedef MYINT* PMYINT;
   typedef const MYINT** PPMYINT;
   typedef const PMYINT& RPMYINT;
   typedef const PPMYINT* PPPMYINT;

   struct A {};
   typedef A A;

   struct B {};
   typedef B C;
   typedef C B;

};

typedef int MyInt;
typedef std::vector<MyInt> MyVector;

class WithTypedefMember {
   int m_i;
   MyInt m_mi;
   std::vector<MyInt> m_v;
   MyVector m_mv;
};

template <class T> class WithTypedefMemberT {
   T m_t;
};


class WithTypedef {
   typedef int MyInt;
};

class UnnamedTypes {
   struct {
      int i;
      double d;
   } m_struct;

   union {
      struct { int i; } m_str1;
      struct { int j; } m_str2;
   } m_union;

private:
   struct Private_Inner_Struct {
      int i;
      double j;
   };

};

struct PrivateConstructors {
   int i;
   void
   foo() {}

};

class WithTransientMember {
private:
   int m_transient;
   WithTransientMember* m_nottransient;
};


struct ConstNonConstMembers {
   ConstNonConstMembers(): m_i(0),
      m_ci(1),
      m_vi(2) {}

   int
   foo(int i) { return m_i * i; }

   int
   foo(int i) const { return m_i * i; }

   int
   constfoo() const { return 0; }

   int
   nonconstfoo() { return 1; }

   int m_i;
   const int m_ci;
   volatile int m_vi;

};

namespace OverloadedOperators {
struct NoOp {
public:
   std::vector<int> makeItComplex; };

struct OpNew {
public:
   void*
   operator new(size_t n) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }

};

struct PlOpNew {
public:
   void*
   operator new(size_t n,
                void* v) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }

};

struct PlOpOpNew {
public:
   void*
   operator new(size_t n) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }

   void*
   operator new(size_t n,
                void* v) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }

};

struct OpANew {
public:
   void* operator new
   [](size_t n) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }

};

struct PlOpANew {
public:
   void* operator new
   [](size_t n,
      void* v) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }

};

struct PlOpAOpANew {
public:
   void* operator new
   [](size_t n) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }

   void* operator new
   [](size_t n,
      void* v) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }

};

}    // ns OverloadedOperators

namespace NonPublicDestructor {
class BaseWithProtectedDestructor {
protected:
   BaseWithProtectedDestructor() {}

   BaseWithProtectedDestructor(const BaseWithProtectedDestructor& /* arg */) {}

   ~BaseWithProtectedDestructor() {}

};

class Derived20: public BaseWithProtectedDestructor {
};

class Derived21: public BaseWithProtectedDestructor {
public:
   ~Derived21() {}

};

}    // ns NonPublicDestructor


namespace ConvOp {
struct ConversionOperator {
   typedef const int* (ConversionOperator::* ptr_to_mem_fun)() const;
   operator ptr_to_mem_fun() const { return & ConversionOperator::i; }

   typedef int* ConversionOperator::* ptr_to_mem_data;
   operator ptr_to_mem_data() const { return &ConversionOperator::m_ip; }
   const int*
   i() const { return &m_i; }

   int m_i;
   int* m_ip;
};

template <class T> struct ConversionOperatorT {
   typedef const T* (ConversionOperatorT<T>::* ptr_to_mem_fun)() const;
   operator ptr_to_mem_fun() const { return & ConversionOperatorT<T>::i; }

   typedef T* ConversionOperatorT<T>::* ptr_to_mem_data;
   operator ptr_to_mem_data() const { return &ConversionOperatorT<T>::m_ip; }
   const T*
   i() const { return &m_i; }

   T m_i;
   T* m_ip;
};

}    // ns ConvOp


namespace { class ForwardedUnnamedNestedType; }

namespace FwUnnamedNSType {
struct ForwardUnnamedNamespaceType {
   void
   foo(const ForwardedUnnamedType* /* fp */) {}

   void
   foo2(const ForwardedUnnamedNestedType* /* fp */) {}

};

}    // ns FwUnnamedNSType

struct Base { virtual ~Base() {} };
struct DerivedA: public Base {};
struct DerivedB: public Base {};

class MyClass {};
struct MyStruct {};


} // namespace testclasses


class BadDictionary {
public:
   const A*
   rioOnTrack(unsigned int indx) const {
      //return new A();
      return m_vect->operator [](indx);
   }


   std::vector<const A*>* m_vect;


}; // namespace testclasses

namespace testclasses2 {
template <class T> class WithTypedefMemberT {
   T m_t;
};

}

typedef int MYINT;
typedef float MYFLOAT;

// template instances
namespace {
struct _testclasses_instances {
   std::vector<MyClass> m_v2;

   std::vector<MYINT> m_v0;
   std::vector<MYFLOAT> m_v1;

   std::list<MYINT> m_l0;
   std::list<MYFLOAT> m_l1;

   struct A {};
   _testclasses_instances() {
      A a;
      testclasses::TemplFun tf;
      tf.foooo(a);
   }


   std::vector<testclasses::Base*> mv1;
   testclasses::testTemplInt<1> mi1;
   testclasses::testTemplInt<-1> mim1;
   testclasses::ConvOp::ConversionOperatorT<int> m1;
   testclasses::WithTypedefMemberT<testclasses::MyVector> m2;
   testclasses::WithTypedefMemberT<testclasses::MyInt> m3;
   testclasses2::WithTypedefMemberT<testclasses::MyVector> m4;
   testclasses2::WithTypedefMemberT<testclasses::MyInt> m5;
};
}
