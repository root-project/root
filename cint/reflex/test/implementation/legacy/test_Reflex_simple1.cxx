// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// CppUnit include file
#include "cppunit/extensions/HelperMacros.h"

// Seal include files
#include "Reflex/Reflex.h"

// Standard C++ include files
#include <iostream>
#include <iomanip>
#include <limits>
#ifdef _WIN32
  # include <windows.h>
#else
  # include <dlfcn.h>
#endif

using namespace Reflex;

/**
 * test_Reflex_simple1.cpp
 * testing Reflex with the dictionary of Reflex itself
 */

class ReflexSimple1Test: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(ReflexSimple1Test);

   CPPUNIT_TEST(loadLibrary);
   CPPUNIT_TEST(testSizeT);
   CPPUNIT_TEST(testBase);
   CPPUNIT_TEST(testTypeCount);
   CPPUNIT_TEST(testMembers);
   CPPUNIT_TEST(testVirtual);
   CPPUNIT_TEST(unloadLibrary);
   CPPUNIT_TEST(shutdown);

   CPPUNIT_TEST_SUITE_END();

public:
   void
   setUp() {}

   void loadLibrary();
   void testSizeT();
   void testBase();
   void testTypeCount();
   void testMembers();
   void testVirtual();
   void unloadLibrary();
   void
   shutdown() {}

   void
   tearDown() {}

}; // class ReflesSimple1Test

#if defined(_WIN32)
static HMODULE s_libInstance = 0;
#else
static void* s_libInstance = 0;
#endif

void
ReflexSimple1Test::loadLibrary() {
#if defined(_WIN32)
   s_libInstance = LoadLibrary("libtest_ReflexRflx.dll");
#else
   s_libInstance = dlopen("libtest_ReflexRflx.so", RTLD_NOW);

   if (!s_libInstance) {
      std::cout << dlerror() << std::endl;
   }
#endif
   CPPUNIT_ASSERT(s_libInstance);
}


void
ReflexSimple1Test::testSizeT() {
   Type t = Type::ByName("size_t");
   CPPUNIT_ASSERT(t);
#if defined(__GNUC__)
# if (__GNUC__ <= 3) && (!__x86_64__)
   std::string size_t_T = "j";
# else
   std::string size_t_T = "m";
# endif
#elif defined(_WIN32)
   std::string size_t_T = "unsigned int";
#endif
   CPPUNIT_ASSERT_EQUAL(size_t_T, std::string(t.TypeInfo().name()));
   CPPUNIT_ASSERT_EQUAL(size_t_T, std::string(t.ToType().TypeInfo().name()));
}


void
ReflexSimple1Test::testBase() {
   Type t1 = Type::ByName("Reflex::ScopeName");
   Type t2 = Type::ByName("Reflex::ScopeBase");

   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT(t2);

   CPPUNIT_ASSERT_EQUAL(std::string(""), t1.ToType().Name());
   CPPUNIT_ASSERT_EQUAL(t1.Name(), t1.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(t1.Name(), t1.RawType().Name());

   CPPUNIT_ASSERT(!t1.HasBase(t2));
   CPPUNIT_ASSERT(!t2.HasBase(t1));
}


void
ReflexSimple1Test::testTypeCount() {
   CPPUNIT_ASSERT((int (Type::TypeSize()) > 500) && (int (Type::TypeSize()) < 1000));
}


void
ReflexSimple1Test::testMembers() {
   Member m;
   Type t = Type::ByName("Reflex::PropertyList");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.Id());
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), t.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList"), t.Name(SCOPED));

   Object o = t.Construct();

   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT(o.Address());

   CPPUNIT_ASSERT_EQUAL(1, int (t.DataMemberSize()));
   CPPUNIT_ASSERT_EQUAL(31, int (t.FunctionMemberSize()));
   CPPUNIT_ASSERT_EQUAL(32, int (t.MemberSize()));

   t.UpdateMembers();

   CPPUNIT_ASSERT_EQUAL(1, int (t.DataMemberSize()));
   CPPUNIT_ASSERT_EQUAL(31, int (t.FunctionMemberSize()));
   CPPUNIT_ASSERT_EQUAL(32, int (t.MemberSize()));
   Member_Iterator iM = t.DataMember_Begin();

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("fPropertyListImpl"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::fPropertyListImpl"), m.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL((void*) 0, Object_Cast<void*>(m.Get(o)));

   iM = t.FunctionMember_Begin();
   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::PropertyList"), m.Name(SCOPED));
   CPPUNIT_ASSERT(m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::PropertyList"), m.Name(SCOPED));
   CPPUNIT_ASSERT(m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("~PropertyList"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::~PropertyList"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::PropertyList"), m.Name(SCOPED));
   CPPUNIT_ASSERT(m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("__getNewDelFunctions"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::__getNewDelFunctions"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("operator="), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::operator="), m.Name(SCOPED));
   CPPUNIT_ASSERT(m.IsOperator());
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("operator bool"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::operator bool"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("AddProperty"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::AddProperty"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("AddProperty"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::AddProperty"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("AddProperty"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::AddProperty"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("AddProperty"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::AddProperty"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("ClearProperties"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::ClearProperties"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   /*
      removed from dictionary
      m = *iM++;
      CPPUNIT_ASSERT(m);
      CPPUNIT_ASSERT_EQUAL(std::string("HasKey"), m.Name());
      CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::HasKey"), m.Name(SCOPED));
      CPPUNIT_ASSERT(!m.IsConstructor());
      CPPUNIT_ASSERT(!m.IsDestructor());
    */

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("HasProperty"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::HasProperty"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("HasProperty"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::HasProperty"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("Key_Begin"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::Key_Begin"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   m = *iM++;
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("Key_End"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::PropertyList::Key_End"), m.Name(SCOPED));
   CPPUNIT_ASSERT(!m.IsConstructor());
   CPPUNIT_ASSERT(!m.IsDestructor());

   o.Destruct();
   CPPUNIT_ASSERT(!o);
} // testMembers


void
ReflexSimple1Test::testVirtual() {
   Type t1 = Type::ByName("Reflex::Type");
   Type t2 = Type::ByName("Reflex::TypeBase");

   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT_EQUAL(std::string("Type"), t1.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::Type"), t1.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::Type"), t1.Name(SCOPED | QUALIFIED | FINAL));
   CPPUNIT_ASSERT(!t1.IsVirtual());
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT_EQUAL(std::string("TypeBase"), t2.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::TypeBase"), t2.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("Reflex::TypeBase"), t2.Name(SCOPED | QUALIFIED | FINAL));
   CPPUNIT_ASSERT(t2.IsVirtual());

}


void
ReflexSimple1Test::unloadLibrary() {
#if defined(_WIN32)
   int ret = FreeLibrary(s_libInstance);

   if (ret == 0) {
      std::cout << "Unload of dictionary library failed. Reason: " << GetLastError() << std::endl;
   }
   CPPUNIT_ASSERT(ret);
#else
   int ret = dlclose(s_libInstance);

   if (ret == -1) {
      std::cout << "Unload of dictionary library failed. Reason: " << dlerror() << std::endl;
   }
   CPPUNIT_ASSERT(!ret);
#endif
   //std::cout << "Endless" << std::endl;
   //while (true) {}

} // unloadLibrary


// Class registration on cppunit framework
CPPUNIT_TEST_SUITE_REGISTRATION(ReflexSimple1Test);

// CppUnit test-driver common for all the cppunit test classes
#include <CppUnit_testdriver.cpp>
