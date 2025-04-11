#include "Cintex/Cintex.h"
#include "Reflex/Type.h"
#include "TClass.h"
#include "TList.h"

// Ensure that UpdateMembers() does not mess with ROOT's list of real data members.
// This corrupts I/O by duplicating members, because ROOT I/O alreadyiterates over base classes itself.

using namespace ROOT::Cintex;
using namespace ROOT::Reflex;

void processClass(const char* name, bool update) {

   Type t = Type::ByName(name);
   if (!t) {
      std::cout << name << " is not known to Reflex!" << std::endl;
      return;
   }
   if (update) t.UpdateMembers();
   TClass* cl = TClass::GetClass(name);
   if (!cl) {
      std::cout << name << " is not known to ROOT!" << std::endl;
      return;
   }
   cl->BuildRealData();

   std::cout << name << " has " << cl->GetListOfRealData()->GetSize()
             << " data members according to TClass, and "
             << t.DataMemberSize() << " according to Reflex." << std::endl;
}

void updateMembers_test() {
   Cintex::Enable();
   processClass("N::NoBase", true);
   processClass("Derived0", true);
   processClass("Derived1", true);
   processClass("Derived2", false);
}
