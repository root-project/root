#include "util/HelperMacros.hpp"

#include "Reflex/Type.h"
#include "Reflex/Object.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   Type t = Type::ByName("shape");
   CPPUNIT_ASSERT_THROW(t.Construct(), RuntimeError);
}
