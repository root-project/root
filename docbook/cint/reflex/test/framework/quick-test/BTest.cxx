#include "util/HelperMacros.hpp"

#include "Reflex/Type.h"

using namespace Reflex;

// test that only class A was included due to the selection file
REFLEX_TEST(test001)
{
   Type t = Type::ByName("A");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(t.IsComplete());

   t = Type::ByName("B");

   CPPUNIT_ASSERT(!t);
}
