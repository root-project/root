// Check getting of members

#include "util/HelperMacros.hpp"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

#include "Get.hpp"

using namespace Reflex;

int S::A::s = 0;

REFLEX_TEST(test001)
{
// See e.g. https://savannah.cern.ch/bugs/?65759
   Type tT = Type::ByName("S::T");
   CPPUNIT_ASSERT(tT);

   S::A::s = 43;
   S::T o;
   o.a = 42;

   Member mA = tT.DataMemberByName("a");
   CPPUNIT_ASSERT(mA);
   Object objA = mA.Get(Object::Create(o));
   CPPUNIT_ASSERT(objA);
   CPPUNIT_ASSERT_EQUAL(42, *(int*)objA.Address());
   
   Member mS = tT.DataMemberByName("s");
   CPPUNIT_ASSERT(mS);
   Object objS = mS.Get();
   CPPUNIT_ASSERT(objS);
   CPPUNIT_ASSERT_EQUAL(43, *(int*)objS.Address());

}
