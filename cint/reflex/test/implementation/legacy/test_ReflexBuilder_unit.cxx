// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// CppUnit include file
#include "cppunit/extensions/HelperMacros.h"

// Seal include files
#include "Reflex/Builder/ClassBuilder.h"
#include "Reflex/Builder/TypedefBuilder.h"
#include "Reflex/Builder/EnumBuilder.h"
#include "Reflex/Builder/FunctionBuilder.h"
#include "Reflex/Builder/NamespaceBuilder.h"
#include "Reflex/Builder/VariableBuilder.h"

#include "Reflex/Reflex.h"

// Standard C++ include files
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <map>

using namespace Reflex;
using namespace std;

#if defined(_WIN32) && !defined(__CINT__)
typedef __int64 longlong;
typedef unsigned __int64 ulonglong;
#else
typedef long long int longlong;    /* */
typedef unsigned long long int /**/ ulonglong;
#endif

/** test_SealKenel_Properties.cpp
 *
 * Simple test cases to exercise Framework properties
 */

class ReflexBuilderUnitTest: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(ReflexBuilderUnitTest);
   CPPUNIT_TEST(classbuilder);
   CPPUNIT_TEST(propertybuilder);
   CPPUNIT_TEST(inheritancebuilder);
   CPPUNIT_TEST(datamemberaccess);
   CPPUNIT_TEST(methodinvocation);
   CPPUNIT_TEST(objectinstantiation);
   CPPUNIT_TEST(typedefbuilder);
   CPPUNIT_TEST(classAddTypedef);
   CPPUNIT_TEST(enumbuilder);
   CPPUNIT_TEST(classAddEnum);
   CPPUNIT_TEST(unionbuilder);
   CPPUNIT_TEST(freefunctionbuilder);
   CPPUNIT_TEST(freevariablebuilder);
   CPPUNIT_TEST(demangler);
   CPPUNIT_TEST(normalizer);
   CPPUNIT_TEST(forwarddeclarations);
   CPPUNIT_TEST(subscopebuilder);
   CPPUNIT_TEST(type_equivalence);
   CPPUNIT_TEST(type_template);
   CPPUNIT_TEST(member_template);
   CPPUNIT_TEST(typebuilder);
   CPPUNIT_TEST(bases);
   CPPUNIT_TEST(hiddentypes);
   CPPUNIT_TEST(shutdown);
   CPPUNIT_TEST_SUITE_END();

public:
   void
   setUp() {}

   void classbuilder();
   void propertybuilder();
   void inheritancebuilder();
   void datamemberaccess();
   void methodinvocation();
   void objectinstantiation();
   void typedefbuilder();
   void classAddTypedef();
   void enumbuilder();
   void classAddEnum();
   void unionbuilder();
   void freefunctionbuilder();
   void freevariablebuilder();
   void demangler();
   void normalizer();
   void forwarddeclarations();
   void subscopebuilder();
   void offset();
   void type_equivalence();
   void type_template();
   void member_template();
   void typebuilder();
   void bases();
   void hiddentypes();

   void shutdown();
   void
   tearDown() {}

};

//==============================================================================================
// offsetof test
//==============================================================================================

// this would be a work around for offset calculation with templated classes with more than one
// template parameter, but it's not needed for the time being as we are using shadow
// classes which are not templated (thx Axel Naumann)
#define OffsetOf_m(c1, mem) ((size_t) (&(c1 64)->mem) - 64)

struct OffsetOf1 { int fI; };
template <class T> struct OffsetOf2 { int fI; };
template <class T1, class T2> struct OffsetOf3 { int fI; };
template <class T1, class T2, class T3, class T4, class T5> struct OffsetOf6 { int fI; };

void
ReflexBuilderUnitTest::offset() {
   CPPUNIT_ASSERT(OffsetOf(OffsetOf1, fI));
   CPPUNIT_ASSERT(OffsetOf_m((OffsetOf1*), fI));
   CPPUNIT_ASSERT(OffsetOf_m((OffsetOf2<int>*), fI));
   CPPUNIT_ASSERT(OffsetOf_m((OffsetOf3<int, int>*), fI));
   CPPUNIT_ASSERT(OffsetOf_m((OffsetOf6<int, int, int, int, int>*), fI));
}


//==============================================================================================
// Demangle test
//==============================================================================================
namespace a {
struct b {};
}
struct bar {};
void
ReflexBuilderUnitTest::demangler() {
   CPPUNIT_ASSERT_EQUAL(string("int"), Tools::Demangle(typeid(int)));
   CPPUNIT_ASSERT_EQUAL(string("int*"), Tools::Demangle(typeid(int*)));
#if defined(__GNUC__) && __GNUC__ >= 4 || (__GNUC__ >= 3 && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 4)
   CPPUNIT_ASSERT_EQUAL(string("float ()(int)"), Tools::Demangle(typeid(float (int))));
   CPPUNIT_ASSERT_EQUAL(string("float* ()(int)"), Tools::Demangle(typeid(float* (int))));
#else
   CPPUNIT_ASSERT_EQUAL(string("float (int)"), Tools::Demangle(typeid(float (int))));
   CPPUNIT_ASSERT_EQUAL(string("float* (int)"), Tools::Demangle(typeid(float* (int))));
#endif
   CPPUNIT_ASSERT_EQUAL(string("float (*)(int)"), Tools::Demangle(typeid(float(*) (int))));
   CPPUNIT_ASSERT_EQUAL(string("a::b"), Tools::Demangle(typeid(a::b)));
   CPPUNIT_ASSERT_EQUAL(string("std::vector<bar,std::allocator<bar> >"),
                        Tools::Demangle(typeid(vector<bar> )));
}


//==============================================================================================
// Normalize test
//==============================================================================================
void
ReflexBuilderUnitTest::normalizer() {
   CPPUNIT_ASSERT_EQUAL(string(""), Tools::NormalizeName(""));
   CPPUNIT_ASSERT_EQUAL(string(""), Tools::NormalizeName(" "));
   CPPUNIT_ASSERT_EQUAL(string(""), Tools::NormalizeName("  "));
   CPPUNIT_ASSERT_EQUAL(string("x"), Tools::NormalizeName("x"));
   CPPUNIT_ASSERT_EQUAL(string("x"), Tools::NormalizeName(" x"));
   CPPUNIT_ASSERT_EQUAL(string("int"), Tools::NormalizeName("int"));
   CPPUNIT_ASSERT_EQUAL(string("int"), Tools::NormalizeName("   int"));
   CPPUNIT_ASSERT_EQUAL(string("int"), Tools::NormalizeName("int  "));
   CPPUNIT_ASSERT_EQUAL(string("int*"), Tools::NormalizeName("int *"));
   CPPUNIT_ASSERT_EQUAL(string("int**"), Tools::NormalizeName("int **"));
   CPPUNIT_ASSERT_EQUAL(string("int**"), Tools::NormalizeName("int **  "));
   CPPUNIT_ASSERT_EQUAL(string("int*&"), Tools::NormalizeName("int *&"));
   CPPUNIT_ASSERT_EQUAL(string("float (int)"), Tools::NormalizeName("float (int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* (int)"), Tools::NormalizeName("float *(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* (int)"), Tools::NormalizeName("float * (int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* (int)"), Tools::NormalizeName("float*(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float** (int)"), Tools::NormalizeName("float **(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float*& (int)"), Tools::NormalizeName("float *&(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float*& (int)"), Tools::NormalizeName("float * & (int)"));
   CPPUNIT_ASSERT_EQUAL(string("float*& (int)"), Tools::NormalizeName("float  *  &  (int)"));
   CPPUNIT_ASSERT_EQUAL(string("float**& (int)"), Tools::NormalizeName("float **&(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float (*)(int)"), Tools::NormalizeName("float (*)(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float (*)(int)"), Tools::NormalizeName("float(*)(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float (*)(int)"), Tools::NormalizeName("  float  (  *  )  (  int  )  "));
   CPPUNIT_ASSERT_EQUAL(string("float& (*)(int)"), Tools::NormalizeName("float& (*)(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float**& (*)(int)"), Tools::NormalizeName("float**& (*)(int)"));
   CPPUNIT_ASSERT_EQUAL(string("a::b"), Tools::NormalizeName("a::b"));
   CPPUNIT_ASSERT_EQUAL(string("std::vector<bar,std::allocator<bar> >"), Tools::NormalizeName("std::vector<bar,std::allocator<bar> >"));
   CPPUNIT_ASSERT_EQUAL(string("std::vector<bar,std::allocator<bar> >"), Tools::NormalizeName("std::vector<bar, std::allocator<bar>>"));
   CPPUNIT_ASSERT_EQUAL(string("std::vector<bar,std::vector<bar,std::allocator<bar> > >"), Tools::NormalizeName("std::vector<bar, std::vector<bar, std::allocator<bar>>>"));
   CPPUNIT_ASSERT_EQUAL(string("float ()(int)"), Tools::NormalizeName("float ()(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float ()(int)"), Tools::NormalizeName("float ( )( int)"));
   CPPUNIT_ASSERT_EQUAL(string("float ()(int)"), Tools::NormalizeName("float (  )( int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* ()(int)"), Tools::NormalizeName("float* ()(int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* ()(int)"), Tools::NormalizeName("float * () (int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* ()(int)"), Tools::NormalizeName("float* ( ) (int)"));
   CPPUNIT_ASSERT_EQUAL(string("float* ()(int)"), Tools::NormalizeName("float * (   )   (   int   )"));
   CPPUNIT_ASSERT_EQUAL(string("unsigned int"), Tools::NormalizeName("unsigned int"));
   CPPUNIT_ASSERT_EQUAL(string("unsigned int"), Tools::NormalizeName("unsigned    int"));
   CPPUNIT_ASSERT_EQUAL(string("unsigned int"), Tools::NormalizeName("   unsigned    int  "));
   CPPUNIT_ASSERT_EQUAL(string("const char*"), Tools::NormalizeName("   const  char   *"));
   CPPUNIT_ASSERT_EQUAL(string("volatile const char*"), Tools::NormalizeName(" volatile   const  char   *"));
} // normalizer


//==============================================================================================
// Class Building test
//==============================================================================================

namespace a {
struct bar {
   bar() {}

   bar(float i): b(i) {}

   float b;
};

class foo {
public:
   //Constructors-Destructors
   foo (): cv(0) {}

   foo (int /* i */): cv(0) {}

   foo (const foo&): cv(0) {}

   ~foo() {}

   //Methods
   void
   f0(void) {}

   bool
   f1(bool) { return true; }

   char
   f2(char,
      char*) { return 0; }

   short
   f3(short,
      short*,
      short&) { return 0; }

   int
   f4(int,
      const int*,
      int* const,
      int**) { return 0; }

   long
   f5(long,
      long[6],
      long**,
      long*&,
      long**&) { return 0; }

   longlong
   f6(bar[10],
      const bar&,
      const bar*,
      bar const*,
      bar* const,
      bar**) { return 0; }

   float
   f7(float,
      float,
      float,
      float,
      float,
      float,
      float) { return 0.0; }

   double
   f8(unsigned int,
      unsigned long,
      bool,
      char,
      unsigned char,
      int,
      int,
      int) { return 0.0; }

   // Data members

public:
   bool b;
   char c;
   short s;
   int i;
   float f;
   double d;
   float* fp;
   bar ba;
   bar* bap;
   //private:
   int pi;
   const int* ci;
   volatile bool v;
   const volatile char cv;
   // Static methods
   static void
   s0(void) {}

   static bool
   b1(bool) { return true; }

   template <class T> void
   t1(T) {}

   template <class T1, class T2> int
   t2(T1,
      T2) { return 0; }

   // Special cases
   void(*ff1 (void))(void) { return s0; }            // returning a function pointer
        bool(*ff2 (double))(bool) { return b1; }     // returning a function pointer
             float(*ff3(float (* ff)(int))) (int) { return ff; } // returning and receiving a function pointer
};
}

void
ReflexBuilderUnitTest::classbuilder() {
   ClassBuilderT<a::bar>(STRUCT)
   .AddFunctionMember<void (void)>("bar", 0, 0, 0, PUBLIC | CONSTRUCTOR)
   .AddFunctionMember<void (float)>("bar", 0, 0, 0, CONSTRUCTOR)
   .AddDataMember<float>("b", OffsetOf(a::bar, b));

   ClassBuilderT<a::foo>()
   .AddFunctionMember<void (void)>("foo", 0, 0, 0, CONSTRUCTOR)
   .AddFunctionMember<void (int)>("foo", 0, 0, 0, CONSTRUCTOR)
   .AddFunctionMember<void (const a::foo&)>("foo", 0, 0, 0, CONSTRUCTOR)
   .AddFunctionMember<void (void)>("~foo", 0, 0, 0, DESTRUCTOR)
   .AddFunctionMember<void (void)>("f0", 0)
   .AddFunctionMember<bool (bool)>("f1", 0)
   .AddFunctionMember<char (char, char*)>("f2", 0)
   .AddFunctionMember<short (short, short*, short&)>("f3", 0)
   .AddFunctionMember<int (int, const int*, int* const, int**)>("f4", 0)
   .AddFunctionMember<long (long, long[6], long**, long*&, long**&)>("f5", 0)
   .AddFunctionMember<longlong(bar[10], const bar&, const bar*, bar const*, bar* const, bar**)>("f6", 0)
   .AddFunctionMember<float (float, float, float, float, float, float, float)>("f7", 0)
   .AddFunctionMember<double (unsigned int, unsigned long, bool, char, unsigned char, int, int, int)>("f8", 0)
   .AddDataMember<bool>("b", OffsetOf(a::foo, b), PUBLIC)
   .AddDataMember<char>("c", OffsetOf(a::foo, c), PUBLIC)
   .AddDataMember<short>("s", OffsetOf(a::foo, s), PUBLIC)
   .AddDataMember<int>("i", OffsetOf(a::foo, i), PUBLIC)
   .AddDataMember<float>("f", OffsetOf(a::foo, f), PUBLIC)
   .AddDataMember<double>("d", OffsetOf(a::foo, d), PUBLIC)
   .AddDataMember<float*>("fp", OffsetOf(a::foo, fp), PUBLIC)
   .AddDataMember<a::bar>("ba", OffsetOf(a::foo, ba), PUBLIC)
   .AddDataMember<a::bar*>("bap", OffsetOf(a::foo, bap), PUBLIC)
   .AddDataMember<int>("pi", OffsetOf(a::foo, pi), PRIVATE)
   .AddDataMember<int*>("ci", OffsetOf(a::foo, ci), PRIVATE)
   .AddDataMember<volatile bool>("v", OffsetOf(a::foo, v), PRIVATE)
   .AddDataMember<const volatile char>("cv", OffsetOf(a::foo, cv), PRIVATE)
   .AddFunctionMember<void (void)>("s0", 0, 0, 0, STATIC | PRIVATE)
   .AddFunctionMember<void (bool)>("b1", 0, 0, 0, STATIC | PRIVATE)
   .AddFunctionMember<void (int)>("t1<int>", 0, 0, 0, PRIVATE)
   .AddFunctionMember<void (float)>("t1<float>", 0, 0, 0, PRIVATE)
   .AddFunctionMember<int (int, float)>("t2<int,float>", 0, 0, 0, PRIVATE)
   .AddFunctionMember<int (double, float)>("t2<double,float>", 0, 0, 0, PRIVATE)
   .AddFunctionMember<void(*(void))(void)>("g1", 0, 0, 0, PRIVATE)
   .AddFunctionMember<bool(*(double))(bool)>("g2", 0, 0, 0, PRIVATE);
   //fixme//.AddFunctionMember<float (* (float(*)(int)) )(int)>("g3", 0, 0, 0, PRIVATE );

   Type t = Type::ByName("a::foo");
   Scope s = Scope::ByName("a");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT_EQUAL(t.Id(), Type::ByName("::a::foo").Id());
   CPPUNIT_ASSERT_EQUAL(t.Id(), TypeBuilder("::a::foo").Id());
   CPPUNIT_ASSERT_EQUAL(s.Id(), Scope::ByName("::a").Id());
   CPPUNIT_ASSERT_EQUAL(t.Id(), Type::ByTypeInfo(typeid(a::foo)).Id());
   CPPUNIT_ASSERT_EQUAL(size_t(21), t.FunctionMemberSize());
   CPPUNIT_ASSERT_EQUAL(size_t(13), t.DataMemberSize());

   for (size_t i = 0; i < t.FunctionMemberSize(); i++) {
      Member f = t.FunctionMemberAt(i);

      if (f.Name() == "foo") {
         CPPUNIT_ASSERT(f.IsConstructor());
      } else if (f.Name() == "~foo") {
         CPPUNIT_ASSERT(f.IsDestructor());
      } else if (f.Name()[0] == 'f') {
         CPPUNIT_ASSERT_EQUAL(size_t(f.Name()[1] - '0'), f.TypeOf().FunctionParameterSize());
      } else if (f.Name()[0] == 's') {
         CPPUNIT_ASSERT(f.IsStatic());
         CPPUNIT_ASSERT(f.IsPrivate());
      } else if (f.Name()[0] == 't') {
         CPPUNIT_ASSERT_EQUAL(size_t(f.Name()[1] - '0'), f.TypeOf().FunctionParameterSize());
      }
   }

   // Check data members and their offsets....
   a::foo obj;

   for (size_t i = 0; i < t.DataMemberSize(); i++) {
      Member d = t.DataMemberAt(i);

      if (d.Name() == "b") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.b - (size_t) &obj, d.Offset());
      } else if (d.Name() == "c") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.c - (size_t) &obj, d.Offset());
      } else if (d.Name() == "s") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.s - (size_t) &obj, d.Offset());
      } else if (d.Name() == "i") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.i - (size_t) &obj, d.Offset());
      } else if (d.Name() == "f") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.f - (size_t) &obj, d.Offset());
      } else if (d.Name() == "d") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.d - (size_t) &obj, d.Offset());
      } else if (d.Name() == "fp") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.fp - (size_t) &obj, d.Offset());
      } else if (d.Name() == "ba") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.ba - (size_t) &obj, d.Offset());
      } else if (d.Name() == "bap") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.bap - (size_t) &obj, d.Offset());
      } else if (d.Name() == "pi") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.pi - (size_t) &obj, d.Offset());
      } else if (d.Name() == "ci") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.ci - (size_t) &obj, d.Offset());
      } else if (d.Name() == "v") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.v - (size_t) &obj, d.Offset());
      } else if (d.Name() == "cv") {
         CPPUNIT_ASSERT_EQUAL((size_t) &obj.cv - (size_t) &obj, d.Offset());
      } else { CPPUNIT_ASSERT(false); }
   }

   // Check that after adding new types the old fundamental types are still intact.
   // This may indicate a problem with the hash table using "const char *"
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(void)));
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(int)));
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(bool)));
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(long)));
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(float)));
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(double)));

} // classbuilder


//==============================================================================================
// Properties to classes, members, ...
//==============================================================================================
namespace b {
   struct foo {
      void
      f(void) {}

      int v;
      int d;
   };
}

void
ReflexBuilderUnitTest::propertybuilder() {
   std::string str = "string";
   ClassBuilderT<b::foo>()
   .AddProperty("foo_p0", "value")
   .AddProperty("foo_p1", 100.0)
   .AddFunctionMember<void (void)>("f", 0)
   .AddProperty("p0", 10L)
   .AddProperty("p1", make_pair(10, 10.))
   .AddDataMember<int>("v", OffsetOf(b::foo, v))
   .AddDataMember<int>("d", OffsetOf(b::foo, d))
   .AddProperty("p0", str)
   .AddProperty("p1", true);
   Type t = Type::ByName("b::foo");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(t.Properties().HasProperty("foo_p0"));
   CPPUNIT_ASSERT_EQUAL(string("value"),
                        string(any_cast<const char*>(t.Properties().PropertyValue("foo_p0"))));
   CPPUNIT_ASSERT_EQUAL(100.0, any_cast<double>(t.Properties().PropertyValue("foo_p1")));
   CPPUNIT_ASSERT(t.MemberByName("f").Properties().HasProperty("p0"));
   CPPUNIT_ASSERT(t.MemberByName("f").Properties().HasProperty("p1"));
   CPPUNIT_ASSERT(t.MemberByName("v").Properties().PropertyCount() == 0);
   CPPUNIT_ASSERT(t.MemberByName("d").Properties().HasProperty("p0"));
   CPPUNIT_ASSERT(t.MemberByName("d").Properties().HasProperty("p1"));
   CPPUNIT_ASSERT_EQUAL(true, any_cast<bool>(t.MemberByName("d").Properties().PropertyValue("p1")));
} // propertybuilder


//==============================================================================================
// Inheritance member access
//==============================================================================================
namespace c {
   struct base { int b; virtual ~base() {} };
   struct dev: public base { int d; };
   struct dev1: virtual public base { int d1; };
   struct dev2: virtual public base { int d2; };
   struct dia: public dev1,
      public dev2 { int dd; };
}

void
ReflexBuilderUnitTest::inheritancebuilder() {
   ClassBuilderT<c::base>(VIRTUAL).AddDataMember<int>("b", OffsetOf(c::base, b));
   ClassBuilderT<c::dev>().AddBase<c::base>().AddDataMember<int>("d", OffsetOf(c::dev, d));
   ClassBuilderT<c::dev1>().AddBase<c::base>(VIRTUAL).AddDataMember<int>("d1", OffsetOf(c::dev1, d1));
   ClassBuilderT<c::dev2>().AddBase<c::base>(VIRTUAL).AddDataMember<int>("d2", OffsetOf(c::dev2, d2));
   ClassBuilderT<c::dia>().AddBase<c::dev1>().AddBase<c::dev2>().AddDataMember<int>("dd", OffsetOf(c::dia, dd));
   Type base = Type::ByName("c::base");
   Type dev = Type::ByName("c::dev");
   Type dev1 = Type::ByName("c::dev1");
   Type dev2 = Type::ByName("c::dev2");
   Type dia = Type::ByName("c::dia");
   CPPUNIT_ASSERT(base);
   CPPUNIT_ASSERT(dev);
   CPPUNIT_ASSERT(dev1);
   CPPUNIT_ASSERT(dev2);
   CPPUNIT_ASSERT(dia);
   CPPUNIT_ASSERT((bool) dev.HasBase(base));
   CPPUNIT_ASSERT(dev1.HasBase(base));
   CPPUNIT_ASSERT(dev2.HasBase(base));
   CPPUNIT_ASSERT(dia.HasBase(base));

   c::dia obj;
   c::dev1* o1 = &obj;
   c::dev2* o2 = &obj;
   c::base* b = &obj;

   // Up-cast
   CPPUNIT_ASSERT_EQUAL((void*) o1, dia.CastObject(dev1, Object(Type(), &obj)).Address());
   CPPUNIT_ASSERT_EQUAL((void*) o2, dia.CastObject(dev2, Object(Type(), &obj)).Address());
   // Down-cast
   CPPUNIT_ASSERT_EQUAL((void*) &obj, base.CastObject(dia, Object(Type(), b)).Address());
   // Down cast for non virutal does not work
   //CPPUNIT_ASSERT_EQUAL((void*)&obj, dev1.CastObject(dia, o1));
   //CPPUNIT_ASSERT_EQUAL((void*)&obj, dev2.CastObject(dia, o2));
} // inheritancebuilder


//==============================================================================================
// Data member access
//==============================================================================================
namespace d {
   struct foo {
      foo(int j): i(j) {}

      virtual ~foo() {}

      int i;
   };
}
void
ReflexBuilderUnitTest::datamemberaccess() {
   ClassBuilderT<d::foo>().AddDataMember<int>("i", OffsetOf(d::foo, i));
   Type c = Type::ByTypeInfo(typeid(d::foo));
   CPPUNIT_ASSERT(c);
   d::foo instance(99);
   Object obj(c, &instance);
   Object f = obj.Get("i");
   CPPUNIT_ASSERT(f.Address());
   CPPUNIT_ASSERT_EQUAL(instance.i, *(int*) f.Address());
}


//==============================================================================================
// Invocation test
//==============================================================================================
namespace e {
   struct foo {
      foo(int i): c(i) {}

      int
      add(int a,
          int b) { return a + b + c; }

      int c;
   };
}
void
foo_stub(void* retaddr,
         void* o,
         const vector<void*>& args,
         void* /*ctx*/) {
   *(int*) retaddr = ((e::foo*) o)->add(*(int*) args[0], *(int*) args[1]);
}


void
ReflexBuilderUnitTest::methodinvocation() {
   ClassBuilderT<e::foo>().AddFunctionMember<int (int, int)>("add", &foo_stub);
   Type c = Type::ByTypeInfo(typeid(e::foo));
   CPPUNIT_ASSERT(c);
   e::foo instance(10);
   Object obj(c, &instance);
   int a1 = 3;
   int a2 = 2;
   int ret = -42;
   obj.Invoke("add", ret, Tools::MakeVector((void*) &a1, (void*) &a2));
   CPPUNIT_ASSERT_EQUAL(15, ret);
}


//==============================================================================================
// Construction and destruction test
//==============================================================================================
namespace f {
   struct foo {
      foo(int i, float f): fI(i),
         fF(f) {}

      foo(const foo& o): fI(o.fI),
         fF(o.fF) {}

      ~foo() {}

      int fI;
      float fF;
   };
   void
   ctor1(void* retaddr,
         void* o,
         const vector<void*>& args,
         void*) {
      if (retaddr) {
         *(void**) retaddr = new (o) foo(*(int*) args[0], *(float*) args[1]);
      } else { new (o) foo(*(int*) args[0], *(float*) args[1]); }
   }


   void
   ctor2(void* retaddr,
         void* o,
         const vector<void*>& args,
         void*) {
      if (retaddr) {
         *(void**) retaddr = new (o) foo(*(foo*) args[0]);
      } else { new (o) foo(*(foo*) args[0]); }
   }


   void
   dtor(void*,
        void* o,
        const vector<void*>&,
        void*) {
      ((foo*) o)->~foo();
   }


}
void
ReflexBuilderUnitTest::objectinstantiation() {
   ClassBuilderT<f::foo>()
   .AddFunctionMember<void (int, float)>("foo", &f::ctor1, 0, 0, CONSTRUCTOR)
   .AddFunctionMember<void (const f::foo&)>("foo", &f::ctor2, 0, 0, CONSTRUCTOR)
   .AddFunctionMember<void (void)>("~foo", &f::dtor, 0, 0, DESTRUCTOR | PUBLIC)
   .AddDataMember<int>("fI", OffsetOf(f::foo, fI))
   .AddDataMember<float>("fF", OffsetOf(f::foo, fF));
   Type c = Type::ByTypeInfo(typeid(f::foo));
   CPPUNIT_ASSERT(c);
   int i = 100;
   float f = 400.0;
   Object o1 = c.Construct(Type::ByTypeInfo(typeid(void (int, float))),
                           Tools::MakeVector((void*) &i, (void*) &f));
   CPPUNIT_ASSERT(o1.TypeOf());
   CPPUNIT_ASSERT(o1.Address());
   Object fr = o1.Get("fI");
   CPPUNIT_ASSERT(fr.Address());
   CPPUNIT_ASSERT_EQUAL(100, *(int*) fr.Address());
   CPPUNIT_ASSERT_EQUAL(400.0F, *(float*) o1.Get("fF").Address());
   CPPUNIT_ASSERT_EQUAL(100, Object_Cast<int>(o1.Get("fI")));

   Object o2 = c.Construct(Type::ByTypeInfo(typeid(void (const f::foo &))), Tools::MakeVector(o1.Address()));
   CPPUNIT_ASSERT(o2.TypeOf());
   CPPUNIT_ASSERT(o2.Address());
   CPPUNIT_ASSERT_EQUAL(100, *(int*) o2.Get("fI").Address());
   CPPUNIT_ASSERT_EQUAL(400.0F, *(float*) o2.Get("fF").Address());

   o2.Destruct();
   o1.Destruct();
} // objectinstantiation


//==============================================================================================
// Typedefs
//==============================================================================================
void
ReflexBuilderUnitTest::typedefbuilder() {
   TypedefBuilder<int>("MyInt1").AddProperty("owner", "me").AddProperty("price", 10.0);
   Type t = Type::ByName("MyInt1");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsTypedef());
   Type t2 = Type::ByName("int");
   Type t3 = Type::ByTypeInfo(typeid(int));
   Type t4 = t.ToType();
   CPPUNIT_ASSERT(Type::ByTypeInfo(typeid(int)) == t.ToType());
   // Within namespace or class
   NamespaceBuilder("AA");
   TypedefBuilder<float>("AA::MyInt1");
   t = Type::ByName("AA::MyInt1");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsTypedef());
   Scope s = Scope::ByName("AA");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT(s == t.DeclaringScope());
   CPPUNIT_ASSERT_EQUAL(size_t(1), s.SubTypeSize());
   CPPUNIT_ASSERT(t == s.SubTypeAt(0));
} // typedefbuilder


class classWithTypedefs {
public:
   typedef int MyInt2;
   typedef MyInt2 MyOtherInt;
   typedef MyOtherInt ThisBecomesStupid;
};

void
ReflexBuilderUnitTest::classAddTypedef() {
   ClassBuilderT<classWithTypedefs>()
   .AddTypedef<classWithTypedefs::MyInt2>("classWithTypedefs::MyInt2")
   .AddTypedef<classWithTypedefs::MyOtherInt>("classWithTypedefs::MyOtherInt")
   .AddTypedef("classWithTypedefs::MyOtherInt",
               "classWithTypedefs::ThisBecomesStupid");

   Type t = Type::ByName("classWithTypedefs");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(!t.DataMemberSize());
   CPPUNIT_ASSERT(!t.MemberSize());

   Scope s = Scope::ByName("classWithTypedefs");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT_EQUAL(3, (int) s.SubTypeSize());

   Type td0 = Type::ByName("classWithTypedefs::MyInt2");
   CPPUNIT_ASSERT(td0);
   CPPUNIT_ASSERT(td0.ToType());
   CPPUNIT_ASSERT(Type::ByName("int") == td0.ToType());
   CPPUNIT_ASSERT("MyInt2" == td0.Name());
   CPPUNIT_ASSERT("classWithTypedefs::MyInt2" == td0.Name(SCOPED));

   Type td2 = s.SubTypeAt(2);
   CPPUNIT_ASSERT(td2);
   CPPUNIT_ASSERT(td2.ToType());
   CPPUNIT_ASSERT(Type::ByName("classWithTypedefs::MyOtherInt") == td2.ToType());
   CPPUNIT_ASSERT("ThisBecomesStupid" == td2.Name());
} // classAddTypedef


//==============================================================================================
// Enums
//==============================================================================================
enum enum1 { val_a, val_b, val_c };
namespace g {
   enum enum2 { i = 10, j = 20, k = 30 };
}
void
ReflexBuilderUnitTest::enumbuilder() {
   EnumBuilderT<enum1>().AddItem("a", val_a).AddItem("b", val_b).AddItem("c", val_c);
   EnumBuilderT<g::enum2>().AddItem("i", g::i).AddItem("j", g::j).AddItem("k", g::k);
   Type e1 = Type::ByName("enum1");
   CPPUNIT_ASSERT(e1);
   CPPUNIT_ASSERT(e1.IsEnum());
   CPPUNIT_ASSERT_EQUAL(size_t(3), e1.MemberSize());
   CPPUNIT_ASSERT_EQUAL(size_t(val_b), *(size_t*) e1.MemberAt(1).Get().Address());

   Type e2 = Type::ByName("g::enum2");
   CPPUNIT_ASSERT(e2);
   CPPUNIT_ASSERT(e2.IsEnum());
   CPPUNIT_ASSERT_EQUAL(size_t(3), e2.MemberSize());
   CPPUNIT_ASSERT_EQUAL(size_t(g::i), *(size_t*) e2.MemberAt(0).Get().Address());
}


class classWithEnums {
public:
   typedef enum { A, B, C } enum1;
   enum ENUM2 { D, E, F };
   enum ENUM3 { G, H, I };
};

void
ReflexBuilderUnitTest::classAddEnum() {
   ClassBuilderT<classWithEnums>()
   .AddEnum<classWithEnums::enum1>("A=5;B=6;C=7")
   .AddEnum<classWithEnums::ENUM2>("D=0;E=1;F=2")
   .AddEnum("classWithEnums::ENUM3", "G=0;H=1;I=2", &typeid(classWithEnums::ENUM3));

   Type t = Type::ByName("classWithEnums");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT("classWithEnums" == t.Name(SCOPED));

   Scope s = Scope::ByName("classWithEnums");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT_EQUAL(3, (int) s.SubTypeSize());

   Type e0 = s.SubTypeAt(0);
   CPPUNIT_ASSERT(e0);
   CPPUNIT_ASSERT("enum1" == e0.Name());
   CPPUNIT_ASSERT_EQUAL(3, (int) e0.DataMemberSize());
   CPPUNIT_ASSERT_EQUAL(3, (int) e0.MemberSize());
   CPPUNIT_ASSERT("classWithEnums::A" == e0.MemberAt(0).Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(5, Object_Cast<int>(e0.MemberAt(0).Get(Object())));

   Type e2 = s.SubTypeAt(2);
   CPPUNIT_ASSERT(e2);
   CPPUNIT_ASSERT("classWithEnums::ENUM3" == e2.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(3, (int) e2.DataMemberSize());
   CPPUNIT_ASSERT_EQUAL(2, Object_Cast<int>(e2.DataMemberAt(2).Get(Object())));

} // classAddEnum


//==============================================================================================
// Union Builder
//==============================================================================================
void
ReflexBuilderUnitTest::unionbuilder() {
}


//==============================================================================================
// Free functions test
//==============================================================================================
namespace h {
   int
   func(int a) { return 2 * a; }

   int s_i;
}
int
gfunc(int a) { return 3 * a; }

void
func_stub(void* retaddr,
          void*,
          const vector<void*>& args,
          void*) {
   *(int*) retaddr = h::func(*(int*) args[0]);
}


void
ReflexBuilderUnitTest::freefunctionbuilder() {
   NamespaceBuilder h("h");
   FunctionBuilderT<int (int)>("h::func", func_stub, 0, 0, 0);
   FunctionBuilderT<int (int)>("gfunc", 0, 0, 0, 0);

   Scope sh = Scope::ByName("h");
   Member m = sh.MemberByName("func");
   CPPUNIT_ASSERT(m);
   Member f = m;
   CPPUNIT_ASSERT(f);
   CPPUNIT_ASSERT(!f.IsStatic());
   int a = 3;
   int ret = -42;
   f.Invoke(ret, Tools::MakeVector((void*) &a));
   CPPUNIT_ASSERT_EQUAL(6, ret);
   //
   Type t = Type::ByTypeInfo(typeid(int (int)));
   CPPUNIT_ASSERT(t);
   FunctionBuilder(t, "g::func2", func_stub, 0, 0, 0);
   Scope g = Scope::ByName("g");
   CPPUNIT_ASSERT(g);
   CPPUNIT_ASSERT(g.IsNamespace());
   Member m2 = g.MemberByName("func2");
   Member f2 = m2;
   CPPUNIT_ASSERT(f2);
   CPPUNIT_ASSERT(!f2.IsStatic());
} // freefunctionbuilder


void
ReflexBuilderUnitTest::freevariablebuilder() {
   NamespaceBuilder sh("h");
   VariableBuilderT<int>("h::s_i", (size_t) &h::s_i, STATIC);

   // In the follwing ASSERT_EQUALs an additional function
   // from the previous example has to be added
   Scope h = Scope::ByName("h");
   CPPUNIT_ASSERT(h);
   CPPUNIT_ASSERT_EQUAL(2, (int) h.MemberSize());
   CPPUNIT_ASSERT_EQUAL(1, (int) h.DataMemberSize());
   CPPUNIT_ASSERT_EQUAL(1, (int) h.FunctionMemberSize());
   Member m = h.MemberByName("s_i");
   CPPUNIT_ASSERT(m);

   VariableBuilderT<int>("s_j", 99);
   Scope h2 = Scope::ByName("");
   CPPUNIT_ASSERT(h2);
   CPPUNIT_ASSERT_EQUAL(2, (int) h2.MemberSize());
   CPPUNIT_ASSERT_EQUAL(1, (int) h2.DataMemberSize());
   CPPUNIT_ASSERT_EQUAL(1, (int) h2.FunctionMemberSize());
   Member m2 = h2.MemberByName("s_j");
   CPPUNIT_ASSERT(m2);

} // freevariablebuilder


//==============================================================================================
// Forward declarations test
//==============================================================================================
struct ftype;
namespace i {
   struct foo {
      ftype* m;
      ftype& f(ftype*);
   };
}

void
ReflexBuilderUnitTest::forwarddeclarations() {
   Type t_f = TypeBuilder("ftype");
   Type t_fp = PointerBuilder(t_f);
   Type t_fr = ReferenceBuilder(t_f);
   Type t_ftb = FunctionTypeBuilder(t_fr, t_fp);
   ClassBuilderT<i::foo>()
   .AddDataMember(t_fp, "m", OffsetOf(i::foo, m))
   .AddFunctionMember(t_ftb, "f", 0);
   Type t = Type::ByName("i::foo");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   Member m = t.MemberByName("m");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT(m.TypeOf().IsPointer());
   CPPUNIT_ASSERT(m.TypeOf().ToType().Id());
   CPPUNIT_ASSERT_EQUAL(string("ftype"), m.TypeOf().ToType().Name());
   Member f = t.MemberByName("f");
   CPPUNIT_ASSERT(f);
   CPPUNIT_ASSERT(f.TypeOf().IsFunction());
   CPPUNIT_ASSERT_EQUAL(string("ftype& (ftype*)"), f.TypeOf().Name(Q));


} // forwarddeclarations


//==============================================================================================
// Sunbscope builder test
//==============================================================================================

namespace j {
   struct base1 { virtual ~base1() {} };
   struct derived1: public base1 { int i; };
   struct base2 { virtual ~base2() {} };
   struct derived2: public base2 { int i; };
}

void
ReflexBuilderUnitTest::subscopebuilder() {
   NamespaceBuilder("j");
   ClassBuilderT<j::base1>();
   ClassBuilderT<j::derived1>().AddBase<j::base1>().AddDataMember<int>("i", OffsetOf(j::derived1, i));
   ClassBuilderT<j::base2>();
   ClassBuilderT<j::derived2>().AddBase<j::base2>().AddDataMember<int>("i", OffsetOf(j::derived2, i));

   Scope s = Scope::ByName("j");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT_EQUAL(size_t(4), s.SubScopeSize());
}


//==============================================================================================
// type equivalence test
//==============================================================================================

void
ReflexBuilderUnitTest::type_equivalence() {
   TypedefBuilder<int>("MyInt");
   Type f1 = Type::ByName("int");
   Type f2 = Type::ByName("MyInt");
   Type f3 = Type(f1, CONST | VOLATILE);
   Type f4 = Type(f2, VOLATILE);
   Type f5 = Type(f2, CONST | VOLATILE);
   CPPUNIT_ASSERT(f1.IsEquivalentTo(f2));
   CPPUNIT_ASSERT(!f1.IsEquivalentTo(f3));
   CPPUNIT_ASSERT(!f1.IsEquivalentTo(f4));
   CPPUNIT_ASSERT(!f1.IsEquivalentTo(f5));
   CPPUNIT_ASSERT(!f2.IsEquivalentTo(f3));
   CPPUNIT_ASSERT(!f2.IsEquivalentTo(f4));
   CPPUNIT_ASSERT(!f2.IsEquivalentTo(f5));
   CPPUNIT_ASSERT(!f3.IsEquivalentTo(f4));
   CPPUNIT_ASSERT(f3.IsEquivalentTo(f5));
   CPPUNIT_ASSERT(!f4.IsEquivalentTo(f5));

   Type f1p = PointerBuilder(f1);
   Type f2p = PointerBuilder(f2);
   Type f3p = PointerBuilder(f3);
   Type f4p = PointerBuilder(f4);
   Type f5p = PointerBuilder(f5);
   CPPUNIT_ASSERT(f1p.IsEquivalentTo(f2p));
   CPPUNIT_ASSERT(!f1p.IsEquivalentTo(f3p));
   CPPUNIT_ASSERT(!f1p.IsEquivalentTo(f4p));
   CPPUNIT_ASSERT(!f1p.IsEquivalentTo(f5p));
   CPPUNIT_ASSERT(!f2p.IsEquivalentTo(f3p));
   CPPUNIT_ASSERT(!f2p.IsEquivalentTo(f4p));
   CPPUNIT_ASSERT(!f2p.IsEquivalentTo(f5p));
   CPPUNIT_ASSERT(!f3p.IsEquivalentTo(f4p));
   CPPUNIT_ASSERT(f3p.IsEquivalentTo(f5p));
   CPPUNIT_ASSERT(!f4p.IsEquivalentTo(f5p));
} // type_equivalence


//==============================================================================================
// type/member template test
//==============================================================================================


void
ReflexBuilderUnitTest::type_template() {
   NamespaceBuilder("std");
   ClassBuilderT<std::vector<int> >();
   ClassBuilderT<std::vector<float> >();
   ClassBuilderT<std::map<int, int> >();
   ClassBuilderT<std::map<int, float> >();
   ClassBuilderT<std::map<float, std::string> >("std::map<float, std::string>");
   ClassBuilderT<std::list<std::string> >();

   Scope s = Scope::ByName("std");

   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT_EQUAL(4, (int) s.SubTypeTemplateSize());
   CPPUNIT_ASSERT_EQUAL(std::string("std::vector"), s.SubTypeTemplateAt(0).Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("std::map"), s.SubTypeTemplateAt(1).Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("std::map"), s.SubTypeTemplateAt(2).Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("std::list"), s.SubTypeTemplateAt(3).Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("vector"), s.SubTypeTemplateAt(0).Name());
   CPPUNIT_ASSERT_EQUAL(std::string("map"), s.SubTypeTemplateAt(1).Name());
   CPPUNIT_ASSERT_EQUAL(std::string("map"), s.SubTypeTemplateAt(2).Name());
   CPPUNIT_ASSERT_EQUAL(std::string("list"), s.SubTypeTemplateAt(3).Name());

   TypeTemplate tt0 = s.SubTypeTemplateAt(0);
   CPPUNIT_ASSERT(tt0);
   CPPUNIT_ASSERT_EQUAL(2, (int) tt0.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(2, (int) tt0.TemplateInstanceSize());

   Type tt0i0 = tt0.TemplateInstanceAt(0);
   CPPUNIT_ASSERT(tt0i0);
   CPPUNIT_ASSERT("std::vector<int,std::allocator<int> >" == tt0i0.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(2, (int) tt0i0.TemplateArgumentSize());
   CPPUNIT_ASSERT("int" == tt0i0.TemplateArgumentAt(0).Name());

   Type tt0i1 = tt0.TemplateInstanceAt(1);
   CPPUNIT_ASSERT(tt0i1);
   CPPUNIT_ASSERT("std::vector<float,std::allocator<float> >" == tt0i1.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(2, (int) tt0i1.TemplateArgumentSize());
   CPPUNIT_ASSERT("float" == tt0i1.TemplateArgumentAt(0).Name());

   TypeTemplate tt1 = s.SubTypeTemplateAt(1);
   CPPUNIT_ASSERT(tt1);
   CPPUNIT_ASSERT_EQUAL(4, (int) tt1.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(2, (int) tt1.TemplateInstanceSize());

   Type tt1i0 = tt1.TemplateInstanceAt(0);
   CPPUNIT_ASSERT(tt1i0);
   CPPUNIT_ASSERT_EQUAL(4, (int) tt1i0.TemplateArgumentSize());
   CPPUNIT_ASSERT("int" == tt1i0.TemplateArgumentAt(0).Name());
   CPPUNIT_ASSERT("int" == tt1i0.TemplateArgumentAt(1).Name());

   Type tt1i1 = tt1.TemplateInstanceAt(1);
   CPPUNIT_ASSERT(tt1i1);
   CPPUNIT_ASSERT_EQUAL(4, (int) tt1i1.TemplateArgumentSize());
   CPPUNIT_ASSERT("int" == tt1i1.TemplateArgumentAt(0).Name());
   CPPUNIT_ASSERT("float" == tt1i1.TemplateArgumentAt(1).Name());

   TypeTemplate tt2 = s.SubTypeTemplateAt(2);
   CPPUNIT_ASSERT(tt2);
   CPPUNIT_ASSERT_EQUAL(2, (int) tt2.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(1, (int) tt2.TemplateInstanceSize());

   Type tt2i0 = tt2.TemplateInstanceAt(0);
   CPPUNIT_ASSERT(tt2i0);
   CPPUNIT_ASSERT_EQUAL(2, (int) tt2i0.TemplateArgumentSize());
   CPPUNIT_ASSERT_EQUAL(string("float"), tt2i0.TemplateArgumentAt(0).Name());
   CPPUNIT_ASSERT_EQUAL(string("std::string"), tt2i0.TemplateArgumentAt(1).Name(SCOPED));

} // type_template


struct MyMemberTemplateClass {
   template <typename T> void
   foo(T /* t */) {}

   template <typename T0, typename T1> void
   foo(T0 /* t0 */,
       T1 /* t1 */) {}

   template <class T0> int
   bar(T0 t) { return t; }

};

namespace TTT {
   template <typename T> void
   foo(T /* t */) {}

   template <typename T0, typename T1> void
   foo(T0 /* t0 */,
       T1 /* t1 */) {}

   template <class T0> int
   bar(T0 t) { return t; }

}

template <typename T> void
foo(T /* t */) {}

template <typename T0, typename T1> void
foo(T0 /* t0 */,
    T1 /* t1 */) {}

//template<class T0> int bar2(T0 t) { return t; }

void
ReflexBuilderUnitTest::member_template() {
   ClassBuilderT<MyMemberTemplateClass>()
   .AddFunctionMember<void (int)>("foo<int>", 0)
   .AddFunctionMember<void (float)>("foo<float>", 0)
   .AddFunctionMember<void (int, int)>("foo<int,int>", 0)
   .AddFunctionMember<void (int, float)>("foo<int,float>", 0)
   .AddFunctionMember<int (int)>("bar<int>", 0)
   .AddFunctionMember<int (float)>("bar<float>", 0)
   .AddFunctionMember<int (bool)>("bar<bool>", 0);

   Type t = Type::ByName("MyMemberTemplateClass");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT_EQUAL(3, (int) t.MemberTemplateSize());

   MemberTemplate mt0 = t.MemberTemplateAt(0);
   CPPUNIT_ASSERT_EQUAL(std::string("MyMemberTemplateClass::foo"), mt0.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(1, (int) mt0.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(2, (int) mt0.TemplateInstanceSize());

   Member mt0i0 = mt0.TemplateInstanceAt(0);
   CPPUNIT_ASSERT(mt0i0);
   CPPUNIT_ASSERT(mt0i0.IsTemplateInstance());
   CPPUNIT_ASSERT_EQUAL(std::string("MyMemberTemplateClass::foo<int>"), mt0i0.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(1, (int) mt0i0.TemplateArgumentSize());
   CPPUNIT_ASSERT("int" == mt0i0.TemplateArgumentAt(0).Name());


   FunctionBuilderT<void (int)>("TTT::foo<int>", 0, 0, "t", 0);
   FunctionBuilderT<void (float)>("TTT::foo<float>", 0, 0, "t", 0);
   FunctionBuilderT<void (bool)>("TTT::foo<bool>", 0, 0, "t", 0);
   FunctionBuilderT<void (int, int)>("TTT::foo<int,int>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, float)>("TTT::foo<bool,float>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, bool)>("TTT::foo<bool,bool>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (int, int)>("TTT::bar<int,int>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, float)>("TTT::bar<bool,float>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, bool)>("TTT::bar<bool,bool>", 0, 0, "t0;t1", 0);

   Scope s = Scope::ByName("TTT");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT_EQUAL(3, (int) s.MemberTemplateSize());

   MemberTemplate mt1 = s.MemberTemplateAt(0);
   CPPUNIT_ASSERT(mt1);
   CPPUNIT_ASSERT("TTT::foo" == mt1.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(1, (int) mt1.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(3, (int) mt1.TemplateInstanceSize());

   Member mt1i1 = mt1.TemplateInstanceAt(1);
   CPPUNIT_ASSERT(mt1i1);
   CPPUNIT_ASSERT(mt1i1.IsTemplateInstance());
   CPPUNIT_ASSERT("TTT::foo<float>" == mt1i1.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(1, (int) mt1i1.TemplateArgumentSize());
   CPPUNIT_ASSERT(mt1i1.TemplateArgumentAt(0));
   CPPUNIT_ASSERT("float" == mt1i1.TemplateArgumentAt(0).Name(SCOPED | QUALIFIED | FINAL));

   MemberTemplate mt2 = s.MemberTemplateAt(2);
   CPPUNIT_ASSERT(mt2);
   CPPUNIT_ASSERT("TTT::bar" == mt2.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(2, (int) mt2.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(3, (int) mt2.TemplateInstanceSize());

   ClassBuilderT<std::string>();
   ClassBuilderT<std::string>("std::string");
   FunctionBuilderT<void (int)>("foo<int>", 0, 0, "t", 0);
   FunctionBuilderT<void (float)>("foo<float>", 0, 0, "t", 0);
   FunctionBuilderT<void (std::string)>("foo<std::string>", 0, 0, "t", 0);
   FunctionBuilderT<void (int, int)>("foo<int,int>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, float)>("foo<bool,float>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, bool)>("foo<bool,bool>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (int, int)>("bar<int,int>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, float)>("bar<bool,float>", 0, 0, "t0;t1", 0);
   FunctionBuilderT<void (bool, bool)>("bar<bool,bool>", 0, 0, "t0;t1", 0);

   Scope s1 = Scope::ByName("");
   CPPUNIT_ASSERT(s1);
   CPPUNIT_ASSERT_EQUAL(3, (int) s.MemberTemplateSize());

   MemberTemplate mt3 = s1.MemberTemplateAt(0);
   CPPUNIT_ASSERT(mt3);
   CPPUNIT_ASSERT("foo" == mt3.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(1, (int) mt3.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(3, (int) mt3.TemplateInstanceSize());

   Member mt3i2 = mt3.TemplateInstanceAt(2);
   CPPUNIT_ASSERT(mt3i2);
   CPPUNIT_ASSERT(mt3i2.IsTemplateInstance());
   CPPUNIT_ASSERT("foo<std::string>" == mt3i2.Name(SCOPED));
   CPPUNIT_ASSERT("foo<std::string>" == mt3i2.Name(SCOPED | FINAL | QUALIFIED));
   CPPUNIT_ASSERT_EQUAL(1, (int) mt3i2.TemplateArgumentSize());
   CPPUNIT_ASSERT(mt3i2.TemplateArgumentAt(0));
   CPPUNIT_ASSERT("std::string" == mt3i2.TemplateArgumentAt(0).Name(SCOPED | QUALIFIED | FINAL));
   CPPUNIT_ASSERT("string" == mt3i2.TemplateArgumentAt(0).Name());

   MemberTemplate mt4 = s1.MemberTemplateAt(2);
   CPPUNIT_ASSERT(mt4);
   CPPUNIT_ASSERT("bar" == mt4.Name(SCOPED | FINAL));
   CPPUNIT_ASSERT_EQUAL(2, (int) mt4.TemplateParameterSize());
   CPPUNIT_ASSERT_EQUAL(3, (int) mt4.TemplateInstanceSize());

} // member_template


void
ReflexBuilderUnitTest::typebuilder() {
   CPPUNIT_ASSERT_EQUAL(7, int (Tools::MakeVector(1, 2, 3, 4, 5, 6, 7).size()));
}


namespace cc {
   struct base {};
   struct dev: public base {};
}

void
ReflexBuilderUnitTest::bases() {
   ClassBuilder("cc::dev", typeid(cc::dev), sizeof(cc::dev), PUBLIC | CLASS | ABSTRACT | VIRTUAL)
   .AddBase(TypeBuilder("cc::base"), BaseOffset<cc::dev, cc::base>::Get(), VIRTUAL | PUBLIC);

   Type base = Type::ByName("cc::base");
   Type dev = Type::ByName("cc::dev");
   Type foo = Type::ByName("foo");
   CPPUNIT_ASSERT(base.Id());
   CPPUNIT_ASSERT(!base);
   CPPUNIT_ASSERT(dev);
   CPPUNIT_ASSERT(!foo);

   CPPUNIT_ASSERT(!dev.HasBase(Type::ByName("foo")));
   CPPUNIT_ASSERT_EQUAL(false, dev.HasBase(foo));
   CPPUNIT_ASSERT(dev.HasBase(base));
   CPPUNIT_ASSERT_EQUAL(true, dev.HasBase(base));

   ClassBuilder("cc::base", typeid(cc::base), sizeof(cc::base), PUBLIC | CLASS);

   CPPUNIT_ASSERT(base);
   CPPUNIT_ASSERT(dev);
   CPPUNIT_ASSERT(!foo);

   CPPUNIT_ASSERT(!dev.HasBase(foo));
   CPPUNIT_ASSERT_EQUAL(false, dev.HasBase(foo));
   CPPUNIT_ASSERT(dev.HasBase(base));
   CPPUNIT_ASSERT_EQUAL(true, dev.HasBase(base));
} // bases


void
ReflexBuilderUnitTest::hiddentypes() {
   TypeBuilder("MyType", 0);
   Type t0 = Type::ByName("MyType");
   CPPUNIT_ASSERT(t0.Id());
   CPPUNIT_ASSERT_EQUAL(std::string("MyType"), t0.Name());
   CPPUNIT_ASSERT(!t0);
   Scope s0 = Scope::ByName("MyType");
   CPPUNIT_ASSERT(!s0.Id());
   CPPUNIT_ASSERT(!s0);

   TypedefTypeBuilder("MyType", t0);
   Type t1 = Type::ByName("MyType");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT(t1.IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("MyType"), t1.Name());
   CPPUNIT_ASSERT(t0 != t1);
   CPPUNIT_ASSERT(!t1.FinalType());
   CPPUNIT_ASSERT(t1.ToType() == t0);
   CPPUNIT_ASSERT_EQUAL(std::string("MyType @HIDDEN@"), t0.Name());
   Scope s1 = Scope::ByName("MyType");
   CPPUNIT_ASSERT(!s1);   // should be different once a typedef is a scope

   Type t2 = Type::ByName("MyType @HIDDEN@");
   CPPUNIT_ASSERT(t2.Id());
   CPPUNIT_ASSERT(t0 == t2);
   Scope s2a = Scope::ByName("MyType");
   CPPUNIT_ASSERT(!s2a);
   Scope s2 = Scope::ByName("MyType @HIDDEN@");
   CPPUNIT_ASSERT(!s2.Id());
   CPPUNIT_ASSERT(!s2);

   size_t nt = Type::TypeSize();
   TypedefTypeBuilder("MyType", t0);
   CPPUNIT_ASSERT_EQUAL(nt, Type::TypeSize());

   ClassBuilder("MyType", typeid(void), 0, 0);
   Type t3 = Type::ByName("MyType");
   CPPUNIT_ASSERT(t3 == t1);
   CPPUNIT_ASSERT(t0);
   CPPUNIT_ASSERT(t3.FinalType() == t0);
   CPPUNIT_ASSERT(t3 == t1);
   CPPUNIT_ASSERT(t3.FinalType() == t2);
   Scope s3 = Scope::ByName("MyType");
   CPPUNIT_ASSERT(s3);
   CPPUNIT_ASSERT(s3.IsClass());
   CPPUNIT_ASSERT_EQUAL(std::string("MyType @HIDDEN@"), s3.Name());
   Scope s4 = Scope::ByName("MyType @HIDDEN@");
   CPPUNIT_ASSERT(s4);
   CPPUNIT_ASSERT(s4.IsClass());

} // hiddentypes


void
ReflexBuilderUnitTest::shutdown() {
}


// Class registration on cppunit framework
CPPUNIT_TEST_SUITE_REGISTRATION(ReflexBuilderUnitTest);

// CppUnit test-driver common for all the cppunit test classes
#include <CppUnit_testdriver.cpp>
