#include "util/HelperMacros.hpp"

#include "Reflex/Type.h"
#include "Reflex/Object.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   // just make sure we have a dictionary
   Type t = Type::ByName("A");
   CPPUNIT_ASSERT(t);
}
