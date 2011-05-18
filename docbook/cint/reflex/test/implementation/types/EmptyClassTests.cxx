#include "util/HelperMacros.hpp"

#include "Reflex/Type.h"
#include "Reflex/Object.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   Reflex::Type t = Type::ByName("EmptyClass");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(t.IsComplete());

   CPPUNIT_ASSERT_EQUAL((size_t) 0, t.BaseSize());
   CPPUNIT_ASSERT_EQUAL((size_t) 0, t.DataMemberSize());
   CPPUNIT_ASSERT_EQUAL((size_t) 5, t.FunctionMemberSize()); // def c'tor, copt c'tor, d'tor, assign op, __getNewDelFunctions
   CPPUNIT_ASSERT_EQUAL((size_t) 5, t.MemberSize());
   CPPUNIT_ASSERT_EQUAL((size_t) 0, t.MemberTemplateSize());
   CPPUNIT_ASSERT_EQUAL((size_t) 0, t.SubScopeSize());
   CPPUNIT_ASSERT_EQUAL((size_t) 0, t.SubTypeSize());
   CPPUNIT_ASSERT_EQUAL((size_t) 0, t.SubTypeTemplateSize());
}
