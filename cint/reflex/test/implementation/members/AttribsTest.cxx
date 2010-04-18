// Check for members' attributes

#include "util/HelperMacros.hpp"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

#include "Attribs.hpp"

using namespace Reflex;

REFLEX_TEST(test001)
{
// See e.g. https://savannah.cern.ch/bugs/?65758
   Type t = Type::ByName("X<float>");
   CPPUNIT_ASSERT(t);               
   CPPUNIT_ASSERT(t.MemberSize() > 12);

   for (Member_Iterator iM = t.Member_Begin(),
           eM = t.Member_End(); iM != eM; ++iM) {
      Member m = *iM;
      std::string name = m.Name();
      for (const char* iName = name.c_str(); *iName; ++iName) {
         switch (*iName) {
         case 'v': {
            CPPUNIT_ASSERT(t.IsVirtual());               
            CPPUNIT_ASSERT(m.IsVirtual());
         }
            break;
         case 'a': {
            CPPUNIT_ASSERT(t.IsAbstract());
            CPPUNIT_ASSERT(m.IsAbstract());
            CPPUNIT_ASSERT(t.IsVirtual());               
            CPPUNIT_ASSERT(m.IsVirtual());
            break;
         }
         case 'c': CPPUNIT_ASSERT(m.IsConst()); break;
         case 's': CPPUNIT_ASSERT(m.IsStatic()); break;
         case 'm': CPPUNIT_ASSERT(m.IsMutable()); break;
         case 'X': CPPUNIT_ASSERT(m.IsExplicit()); break;
         case 't': CPPUNIT_ASSERT(m.IsTransient()); break;
         case '0': CPPUNIT_ASSERT(m.IsPrivate()); break;
         case '1': CPPUNIT_ASSERT(m.IsProtected()); break;
         case '2': CPPUNIT_ASSERT(m.IsPublic()); break;
         case '~': CPPUNIT_ASSERT(m.IsDestructor()); ++iName; break; // d'tor, skip
         case '_': break; // delimiter
         case 'i': break; // variable
         case 'f': break; // function
         default:  CPPUNIT_ASSERT_EQUAL((const char*)"",(std::string("AttribsTest/test001: unhandled character ") + *iName + " in member named " + m.Name()).c_str());
         }
      }
   }
}
