// @(#)root/reflex:$Name:  $:$Id:  $
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
 * test_Reflex_lookup.cxx
 * testing name lookup, trying to comply to the ISO standard 
 */

class ReflexLookupTest : public CppUnit::TestFixture {

   CPPUNIT_TEST_SUITE( ReflexLookupTest );

   CPPUNIT_TEST( loadLibrary );
   CPPUNIT_TEST( lookupMember );
   CPPUNIT_TEST( lookupType );
   CPPUNIT_TEST( lookupScope );

   CPPUNIT_TEST_SUITE_END();

public:

   void setUp() {}

   void loadLibrary();
   void lookupMember();
   void lookupType();
   void lookupScope();

   void tearDown() {}

}; // class ReflexLookupTest

static void * s_libInstance = 0;

void ReflexLookupTest::loadLibrary() {
#if defined (_WIN32)
   s_libInstance = LoadLibrary("libtest_Class2DictRflx.dll");
#else
   s_libInstance = dlopen("libtest_Class2DictRflx.so", RTLD_NOW);
#endif
   CPPUNIT_ASSERT( s_libInstance );
}

void ReflexLookupTest::lookupMember() {

   Scope s0 = Scope::ByName("ClassA");
   CPPUNIT_ASSERT( s0 );

   Member m0 = s0.LookupMember( "fA" );
   CPPUNIT_ASSERT( m0 );

   Member m1 = s0.LookupMember( "fAA" );
   CPPUNIT_ASSERT( ! m1 );

   Member m2 = s0.LookupMember( "fM" );
   CPPUNIT_ASSERT( m2 );

   Scope s1 = Scope::ByName("");
   CPPUNIT_ASSERT( s1 );
   CPPUNIT_ASSERT( s1.IsTopScope());

   Member m3 = s1.LookupMember( "ClassA::fA" );
   CPPUNIT_ASSERT( m3 );

}

void ReflexLookupTest::lookupType() {

   Scope s0 = Scope::ByName("ClassF");
   CPPUNIT_ASSERT( s0 );

   Type t0 = s0.LookupType("ClassA");
   CPPUNIT_ASSERT( t0 );

   Type t1 = s0.LookupType("ClassM");
   CPPUNIT_ASSERT( t1 );

   Scope s1 = Scope::ByName("testclasses");
   CPPUNIT_ASSERT( s1 );

   Type t2 = s1.LookupType("MyInt");
   CPPUNIT_ASSERT( t2 );
   CPPUNIT_ASSERT( t2.IsTypedef());

   Scope s2 = Scope::ByName("");
   CPPUNIT_ASSERT( s2 );
   CPPUNIT_ASSERT( s2.IsTopScope());

   Type t3 = s2.LookupType("testclasses::MyInt");
   CPPUNIT_ASSERT( t3 );

   Type t4 = s2.LookupType("ClassA");
   CPPUNIT_ASSERT(t4);

}

void ReflexLookupTest::lookupScope() {

}

// Class registration on cppunit framework
CPPUNIT_TEST_SUITE_REGISTRATION(ReflexLookupTest);

// CppUnit test-driver common for all the cppunit test classes 
#include<CppUnit_testdriver.cpp>
