#include "util/HelperMacros.hpp"

#include "Reflex/Type.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"

using namespace Reflex;

REFLEX_TEST(test001)
{
   float test = static_cast<float>(3.14159);
   Type fppConv = Type::ByName("FloatingPointPromotion");

   CPPUNIT_ASSERT(fppConv);

   Reflex::Member setter = fppConv.FunctionMemberByName("setDouble");
   CPPUNIT_ASSERT(setter);

   Reflex::Member getter = fppConv.FunctionMemberByName("getDouble");
   CPPUNIT_ASSERT(getter);

   std::vector<void*> args;
   args.push_back(&test);
   //setter.Invoke(fppConv, args);

   // have already performed the conversion but test equivalence anyway
   // Reflex::Object setDouble = getter.Invoke(fppConv);
   // double doubleValue = Reflex::Object_Cast<double>(setDouble);
   // CPPUNIT_ASSERT_DOUBLES_EQUAL(3.14159, doubleValue, 0.01);
}
