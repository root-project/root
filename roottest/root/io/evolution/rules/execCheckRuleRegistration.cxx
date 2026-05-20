#include "TClass.h"
#include "TObject.h"
#include "TROOT.h"
#include "TSchemaRuleSet.h"
#include "TTree.h"

#ifdef __ROOTCLING__
#pragma read sourceClass="Canary" targetClass="TObject";
#pragma read sourceClass="Canary" targetClass="TTree";
#endif


int Check(TClass *cl)
{
   auto rules = cl->GetSchemaRules();
   if (!rules) {
      fprintf(stdout, "Error: no schema rules for %s\n", cl->GetName());
      return 1;
   }
   if (!rules->HasRuleWithSourceClass("Canary")) {
      fprintf(stdout, "Error: no schema rule for source class Canary in %s\n", cl->GetName());
      return 2;
   }
   return 0;
}

int execCheckRuleRegistration()
{
   auto cl = (TClass*)gROOT->GetListOfClasses()->FindObject("TObject");
   if (!cl) {
      fprintf(stderr, "Failed to find TObject TClass ... its loading is delayed\n");
      return 1;
   }
   auto res = Check(cl);
   // Testing that the rules is attached to a TClass that was not yet created
   // when the library was loaded.
   cl = (TClass*)gROOT->GetListOfClasses()->FindObject("TTree");
   if (cl) {
      fprintf(stderr, "TTree TClass is already loaded, testing is inacccurate.\n");
      return 1;
   }
   // Now load the TTree TClass
   cl = TClass::GetClass("TTree");
   if (!cl) {
      fprintf(stderr, "Failed to load TTree TClass ... its loading is delayed\n");
      return 1;
   }
   res += Check(cl);

   return res;
}

