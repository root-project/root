#include "util/HelperMacros.hpp"

#include "Reflex/Scope.h"
#include "Reflex/Type.h"

using namespace Reflex;

REFLEX_TEST(TemplateDefaultArg)
{
   // Check that template default arguments are handled correctly.
   // Name caching can creak it, see https://savannah.cern.ch/bugs/?43356
   Scope t = Type::ByName("DataVector<int>");

   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(((Type) t).IsComplete());

   Type_Iterator it = t.SubType_Begin();
   CPPUNIT_ASSERT(*it);

   CPPUNIT_ASSERT_EQUAL(std::string(it->Name(0)), std::string("self"));
}
