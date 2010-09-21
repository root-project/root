#include "Cintex/Cintex.h"
#include "Reflex/Type.h"
#include "TClass.h"

using namespace ROOT::Cintex;
using namespace ROOT::Reflex;

void checkVersion(const char* name, short version, bool inhFromTObj = false) {
   printf("\nCLASS %s, version %d, isATObject=%d\n", name,
          (int)version, (int)inhFromTObj);
   TClass* cl = TClass::GetClass(name);
   RflxAssert(cl);
   RflxEqual(cl->GetClassVersion(), version);
   RflxEqual(inhFromTObj, cl->InheritsFrom(TObject::Class()));
}

void classVersion_test() {
   Type tWithClassVersion = Type::ByName("WithClassVersion");
   RflxAssert(tWithClassVersion);
   Cintex::Enable();
   checkVersion("WithClassVersion", 42);
   checkVersion("WithClassDef", 12);
   checkVersion("NoIO", 0);
   checkVersion("FromTObject", 13, true);
   RflxAssert(!TClass::GetClass("NoDictionary"));
   RflxAssert(!TClass::GetClass("NoDictionaryTObj"));

   RflxAssert(TClass::GetClass("MyTemp<std::vector<std::string> >"));
}
