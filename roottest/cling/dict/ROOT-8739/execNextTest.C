#include "TError.h"

int check(TString classname) {

   TClass *cl = nullptr;

   cl = TClass::GetClass(classname);

   if (!cl) {
      Error("nextTest","Could not find TClass for %s",classname.Data());
      return 1;
   }

   if (classname != cl->GetName()) {

      Error("nextTest","Find %s instead of %s",cl->GetName(), classname.Data());
      return 2;
   }

   return 0;
}

int execNextTest()
{

   if (0 != gSystem->Load("nextDict")) {
      return 100;
   }

   int result = 0;
   result += check("next");
   // result += check("next::Inside_next");
   result += check("Next");
   result += check("Next::Inside_Next");
   result += check("OtherNext");
   result += check("OtherNext::Inside_OtherNext");
   result += check("YetAnotherNext");
   result += check("YetAnotherNext::Inside_YetAnotherNext");
   return result;
}
