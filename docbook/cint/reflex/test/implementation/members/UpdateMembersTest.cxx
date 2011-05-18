// Make sure that types can be unloaded even after UpdateMembers has been called.
// See https://savannah.cern.ch/bugs/?34051

#include "util/HelperMacros.hpp"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   Type tA = Type::ByName("A");
   CPPUNIT_ASSERT(tA);
   CPPUNIT_ASSERT(tA.MemberByName("a"));
   CPPUNIT_ASSERT(tA.DataMemberByName("a"));

   CPPUNIT_ASSERT_EQUAL((size_t) 1, tA.DataMemberSize(INHERITEDMEMBERS_NO));
   CPPUNIT_ASSERT_EQUAL((size_t) 1, tA.DataMemberSize());

   // triggers tA.UpdateMembers():
   CPPUNIT_ASSERT_EQUAL((size_t) 1, tA.DataMemberSize(INHERITEDMEMBERS_ALSO));

   CPPUNIT_ASSERT(tA);
   CPPUNIT_ASSERT(tA.MemberByName("a"));
   CPPUNIT_ASSERT(tA.DataMemberByName("a"));
   CPPUNIT_ASSERT_EQUAL((size_t) 1, tA.DataMemberSize(INHERITEDMEMBERS_DEFAULT));
}

REFLEX_TEST(test002)
{
   Type tA = Type::ByName("A");

   Type tB = Type::ByName("B");
   CPPUNIT_ASSERT(tB);
   CPPUNIT_ASSERT(tB.MemberByName("b"));
   CPPUNIT_ASSERT(tB.DataMemberByName("b"));

   tB.UpdateMembers();

   CPPUNIT_ASSERT(tB.MemberByName("a"));
   CPPUNIT_ASSERT(tB.DataMemberByName("a"));

   tA.Unload();

   // WILL FAIL! (no MemberName)
   // CPPUNIT_ASSERT(!tB.MemberByName("a"));

   CPPUNIT_ASSERT(tB.MemberByName("b", Type(), INHERITEDMEMBERS_NO));
   CPPUNIT_ASSERT(tB.MemberByName("b"));
}
