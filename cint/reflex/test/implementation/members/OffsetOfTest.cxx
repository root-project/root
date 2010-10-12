#include "util/HelperMacros.hpp"
#include "OffsetOf.hpp"

#include "Reflex/Type.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   // Test offset for a member of unnamed type.
   // This is an issue e.g. for BOOST's OffsetOf
   // See https://savannah.cern.ch/bugs/?33071

   Scope t = Type::ByName("UnNamedMember");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(((Type) t).IsComplete());
   CPPUNIT_ASSERT_EQUAL((size_t) 2, t.DataMemberSize());

   Member_Iterator im = t.DataMember_Begin();
   CPPUNIT_ASSERT_EQUAL((size_t) 0, im->Offset());

   ++im;
   UnNamedMember um;
   size_t unnamed_member_offset = (size_t) (((const char*) &um.fUnNamed)
                                            - ((const char*) &um));
   CPPUNIT_ASSERT_EQUAL(unnamed_member_offset, im->Offset());
}

REFLEX_TEST(test002)
{
   // Test offset for a member of a type overloading operator&().
   // This is an issue for a naive OffsetOf implementation using &member.
   // See https://savannah.cern.ch/bugs/?33071

   Scope t = Type::ByName("Test_OffsetOf_With_AddrOfOp");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(((Type) t).IsComplete());
   CPPUNIT_ASSERT_EQUAL((size_t) 2, t.DataMemberSize());

   Member_Iterator im = t.DataMember_Begin();
   CPPUNIT_ASSERT_EQUAL((size_t) 0, im->Offset());

   ++im;
   CPPUNIT_ASSERT(im->Offset() > 0 && im->Offset() < 16);
}
