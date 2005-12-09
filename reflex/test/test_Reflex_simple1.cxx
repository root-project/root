// @(#)root/reflex:$Name:  $:$Id: test_Reflex_simple1.cxx,v 1.1 2005/11/14 15:08:01 roiser Exp $
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
  #include<windows.h>
#else
  #include<dlfcn.h>
#endif

using namespace ROOT::Reflex;

/**
 * test_Reflex_simple1.cpp
 * testing Reflex with the dictionary of Reflex itself 
 */

class ReflexSimple1Test : public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE( ReflexSimple1Test );
  CPPUNIT_TEST( loadLibrary );
  CPPUNIT_TEST( testSizeT );
  CPPUNIT_TEST( testBase );
  CPPUNIT_TEST( testTypeCount );
  CPPUNIT_TEST( testMembers );
  CPPUNIT_TEST( testVirtual );
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp() {}

  void loadLibrary();
  void testSizeT();
  void testBase();
  void testTypeCount();
  void testMembers();
  void testVirtual();

  void tearDown() {}

}; // class ReflesSimple1Test



void ReflexSimple1Test::loadLibrary() {
 void * libInstance = 0;
 #if defined (_WIN32)
   libInstance = LoadLibrary("test_ReflexRflx.dll");
 #else
   libInstance = dlopen("libtest_ReflexRflx.so", RTLD_LAZY);
 #endif
   CPPUNIT_ASSERT(libInstance);
}

void ReflexSimple1Test::testSizeT() {

  Type t = Type::ByName("size_t");
  CPPUNIT_ASSERT(t);
#if defined(__GNUC__)
#if __GNUC__ <= 3
  std::string size_t_T = "j";
#else 
  std::string size_t_T = "m";
#endif
#elif defined(_WIN32)
  std::string size_t_T = "size_t";
#endif
  CPPUNIT_ASSERT_EQUAL(size_t_T,std::string(t.TypeInfo().name()));
  CPPUNIT_ASSERT_EQUAL(size_t_T,std::string(t.ToType().TypeInfo().name()));
}

void ReflexSimple1Test::testBase() {

  Type t1 = Type::ByName("ROOT::Reflex::ScopeName");
  Type t2 = Type::ByName("ROOT::Reflex::ScopeBase");

  CPPUNIT_ASSERT(t1);
  CPPUNIT_ASSERT(t2);

  CPPUNIT_ASSERT(!t1.HasBase(t2));
  CPPUNIT_ASSERT(!t2.HasBase(t1));
}
  
void ReflexSimple1Test::testTypeCount() {
  CPPUNIT_ASSERT( (int(Type::TypeSize()) > 200) && (int(Type::TypeSize()) < 400) );
}


void ReflexSimple1Test::testMembers() {
  
  Member m;
  Type t = Type::ByName("ROOT::Reflex::PropertyList");

  CPPUNIT_ASSERT(t);
  CPPUNIT_ASSERT(t.Id());
  CPPUNIT_ASSERT(t.IsClass());
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"),t.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList"), t.Name(SCOPED));
  
  Object o = t.Construct();

  CPPUNIT_ASSERT(o);
  CPPUNIT_ASSERT(o.Address());

  CPPUNIT_ASSERT_EQUAL(1, int(t.DataMemberSize()));
  CPPUNIT_ASSERT_EQUAL(14, int(t.FunctionMemberSize()));
  CPPUNIT_ASSERT_EQUAL(15, int(t.MemberSize()));

  t.UpdateMembers();

  CPPUNIT_ASSERT_EQUAL(1, int(t.DataMemberSize()));
  CPPUNIT_ASSERT_EQUAL(14, int(t.FunctionMemberSize()));
  CPPUNIT_ASSERT_EQUAL(15, int(t.MemberSize()));

  m = t.DataMemberAt(0);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("fPropertyListImpl"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::fPropertyListImpl"), m.Name(SCOPED));
  CPPUNIT_ASSERT_EQUAL(0, int(Object_Cast<void*>(m.Get(o))));

  m = t.FunctionMemberAt(0);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertyList"), m.Name(SCOPED));
  CPPUNIT_ASSERT(m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(1);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertyList"), m.Name(SCOPED));
  CPPUNIT_ASSERT(m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(2);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("~PropertyList"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::~PropertyList"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(m.IsDestructor());

  m = t.FunctionMemberAt(3);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("operator bool"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::operator bool"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(4);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("AddProperty"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::AddProperty"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(5);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("AddProperty"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::AddProperty"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(6);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("ClearProperties"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::ClearProperties"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(7);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("RemoveProperty"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::RemoveProperty"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(8);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("HasKey"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::HasKey"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(9);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertySize"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertySize"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(10);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyKeys"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertyKeys"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(11);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyAsString"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertyAsString"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(12);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyValue"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertyValue"), m.Name(SCOPED));
  CPPUNIT_ASSERT(!m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

  m = t.FunctionMemberAt(13);
  CPPUNIT_ASSERT(m);
  CPPUNIT_ASSERT_EQUAL(std::string("PropertyList"), m.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::PropertyList::PropertyList"), m.Name(SCOPED));
  CPPUNIT_ASSERT(m.IsConstructor());
  CPPUNIT_ASSERT(!m.IsDestructor());

}

void ReflexSimple1Test::testVirtual() {

  Type t1 = Type::ByName("ROOT::Reflex::Type");
  Type t2 = Type::ByName("ROOT::Reflex::TypeBase");

  CPPUNIT_ASSERT(t1);
  CPPUNIT_ASSERT_EQUAL(std::string("Type"),t1.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::Type"), t1.Name(SCOPED));
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::Type"), t1.Name(SCOPED|QUALIFIED|FINAL));
  CPPUNIT_ASSERT(!t1.IsVirtual());
  CPPUNIT_ASSERT(t2);
  CPPUNIT_ASSERT_EQUAL(std::string("TypeBase"),t2.Name());
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::TypeBase"), t2.Name(SCOPED));
  CPPUNIT_ASSERT_EQUAL(std::string("ROOT::Reflex::TypeBase"), t2.Name(SCOPED|QUALIFIED|FINAL));
  CPPUNIT_ASSERT(t2.IsVirtual());

}

// Class registration on cppunit framework
CPPUNIT_TEST_SUITE_REGISTRATION(ReflexSimple1Test);

// CppUnit test-driver common for all the cppunit test classes 
#include<CppUnit_testdriver.cpp>
