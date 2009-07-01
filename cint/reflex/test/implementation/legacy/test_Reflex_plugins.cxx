// @(#)root/reflex:$Id$
// Author: Pere Mato 2006

// CppUnit include file
#include "cppunit/extensions/HelperMacros.h"

#include "Reflex/PluginService.h"
#include "testPlugins/ICommon.h"

using namespace ROOT::Reflex;
using namespace std;

class PluginServiceUnitTest: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(PluginServiceUnitTest);
   CPPUNIT_TEST(very_basic);
   CPPUNIT_TEST(with_namespace);
   CPPUNIT_TEST(check_arguments);
   CPPUNIT_TEST(templated);
   CPPUNIT_TEST(const_arguments);
   CPPUNIT_TEST(with_id);
   CPPUNIT_TEST(with_id_other_module);
   CPPUNIT_TEST_SUITE_END();

public:
   void setUp();
   void very_basic();
   void with_namespace();
   void check_arguments();
   void templated();
   void const_arguments();
   void with_id();
   void with_id_other_module();
   void tearDown();
};

void
PluginServiceUnitTest::setUp() {
   PluginService::SetDebug(2);
}


void
PluginServiceUnitTest::tearDown() {
   PluginService::SetDebug(0);
}


void
PluginServiceUnitTest::very_basic() {
   ICommon* o = 0;
   o = PluginService::Create<ICommon*>("MyClass");
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(10, o->do_nothing(10));
   CPPUNIT_ASSERT_EQUAL(string("MyClass doing something"), o->do_something());
   CPPUNIT_ASSERT_EQUAL(99.99, o->get_f());
   //Fail cases
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyOtherClass") == 0); // should fail
   delete o;
}


void
PluginServiceUnitTest::with_namespace() {
   ICommon* o = 0;
   o = PluginService::Create<ICommon*>("MyNS::Another", 99.9F);
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(10, o->do_nothing(10));
   CPPUNIT_ASSERT_EQUAL(99.9F, (float) o->get_f());
   delete o;
   ::Base* b;
   b = PluginService::Create<::Base*>("MyNS::Another", 99.9, o);
   CPPUNIT_ASSERT(b);
   CPPUNIT_ASSERT_EQUAL(10., b->do_base(10.));
   delete b;
}


void
PluginServiceUnitTest::templated() {
   ICommon* o = 0;
   o = PluginService::Create<ICommon*>(" MyTClass<double,int>  ");
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(99., o->get_f());
   delete o;
   o = PluginService::Create<ICommon*>("MyTClass<double,int>", 88.8);
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(88.8, o->get_f());
   delete o;
   string cname = "MyTClass<int,std::basic_string<char,std::char_traits<char>,std::allocator<char> > >";
   o = PluginService::Create<ICommon*>(cname, 88, string("this is a string"));
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(88., o->get_f());
   delete o;

} // templated


void
PluginServiceUnitTest::check_arguments() {
   ICommon* o = 0;
   // No automatic conversions
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", 9.9) == 0); // should fail
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", 99) == 0); // should fail
   o = PluginService::Create<ICommon*>("MyClass", 9.9F);
   CPPUNIT_ASSERT(o);
   delete o;
   // 2 arguments
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", 9.9F, o) == 0); // should fail
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", 99, &o) == 0); // should fail
   o = PluginService::Create<ICommon*>("MyClass", 99.0, o);
   CPPUNIT_ASSERT(o);
   delete o;
   // 2 arguments
   double d = 99.99;
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", (char*) "this is a string", &d) == 0); // should fail
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", string("this is a string"), d) == 0); // should fail
   o = PluginService::Create<ICommon*>("MyClass", string("this is a string"), &d);
   CPPUNIT_ASSERT(o);
   delete o;
   // 3 arguments or more
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", 99, &o, 17) == 0); // should fail
   CPPUNIT_ASSERT(PluginService::Create<ICommon*>("MyClass", 99, &o, 1, 2, 3, 4) == 0); // should fail
} // check_arguments


void
PluginServiceUnitTest::const_arguments() {
   ICommon* o = 0;
   o = PluginService::Create<ICommon*>("MyClassConst", (const ICommon*) 0);
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(0.0, o->get_f());
   delete o;
}


void
PluginServiceUnitTest::with_id() {
   ICommon* o = 0;
   o = PluginService::CreateWithId<ICommon*>(ID(2, 5));
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(99.99, o->get_f());
   delete o;
}


void
PluginServiceUnitTest::with_id_other_module() {
   ICommon* o = 0;
   o = PluginService::CreateWithId<ICommon*>(ID(7, 7));
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT_EQUAL(99.99, o->get_f());
   delete o;
}


// Class registration on cppunit framework
CPPUNIT_TEST_SUITE_REGISTRATION(PluginServiceUnitTest);

// CppUnit test-driver common for all the cppunit test classes
#include <CppUnit_testdriver.cpp>
