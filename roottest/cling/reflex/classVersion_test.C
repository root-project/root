#include "TClass.h"
#include "TestHelper.h"

void checkVersion(const char* name, short version, bool inhFromTObj = false) {
   printf("\nCLASS %s, version %d, isATObject=%d\n", name,
          (int)version, (int)inhFromTObj);
   TClass* cl = TClass::GetClass(name);
   RflxAssert(cl);
   if (cl) {
      RflxEqual(cl->GetClassVersion(), version);
      RflxEqual(inhFromTObj, cl->InheritsFrom(TObject::Class()));
   }
   cl->GetStreamerInfo();
}

void classVersion_test() {
   TClass *tWithClassVersion = TClass::GetClass("WithClassVersion");
   RflxAssert(tWithClassVersion);

   checkVersion("WithClassVersion", 42);
   checkVersion("WithClassDef", 12);
   checkVersion("NoIO", 0);
   checkVersion("FromTObject", 13, true);
   RflxAssert(!TClass::GetClass("NoDictionary"));
   RflxAssert(!TClass::GetClass("NoDictionaryTObj"));

   RflxAssert(TClass::GetClass("MyTemp<std::vector<std::string> >"));

   checkVersion("TemplateWithVersion<int>",10);
   checkVersion("TemplateWithVersion<double>",10);
}
