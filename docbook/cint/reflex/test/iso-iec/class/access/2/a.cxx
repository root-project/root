#include "util/HelperMacros.hpp"

#include "Reflex/Type.h"
#include "Reflex/Object.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   Type classX = Type::ByName("X");
   Type structS = Type::ByName("S");

   Object classXObject = classX.Construct();
   Object structSObject = structS.Construct();

   Object xMember = classXObject.Get("a");
   Object sMember = structSObject.Get("a");

   //CPPUNIT_ASSERT(xMember.Address() == 0);
   CPPUNIT_ASSERT(sMember.Address() != 0);
}
