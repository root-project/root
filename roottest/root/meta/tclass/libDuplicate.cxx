#include <vector>

#ifdef __ROOTCLING__
#pragma link C++ class vector<double>+;
#endif

#include "TClass.h"
#include "TError.h"

bool testDuplicate() {
   auto cn = "vector<double>";
   auto members = TClass::GetClass(cn)->GetListOfDataMembers();
   if (members == nullptr) {
      Error("testDuplicate",
            "For class %s the list of data members is a nullptr.",
            cn);
      return false;
   }
   return true;
}

