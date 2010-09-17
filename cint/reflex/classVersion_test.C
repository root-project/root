#include "Cintex/Cintex.h"
#include "Reflex/Type.h"
#include "TClass.h"

using namespace ROOT::Cintex;
using namespace ROOT::Reflex;

void classVersion_test() {
   Type tWithClassVersion = Type::ByName("WithClassVersion");
   RflxAssert(tWithClassVersion);
   Cintex::Enable();
   TClass* cl = TClass::GetClass("WithClassVersion");
   RflxEqual(cl->GetClassVersion(), 42);

   cl = TClass::GetClass("MyTemp<std::vector<std::string> >");
   // fprintf(stdout,"cl=%p\n",cl);

}
