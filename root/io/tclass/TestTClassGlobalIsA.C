#include "TNamed.h"
#include "TClass.h"
#include <iostream>
#include "TApplication.h"

TClass* globalIsA(const TClass *cl, const void* obj) {
   if (gDebug>0) {
      std::cerr << "running globalIsA for " << (void*)cl << " and " << obj << std::endl;
   } else {
      std::cout << "running globalIsA for " << cl->GetName() << std::endl;
   }
   if (! cl->InheritsFrom(TObject::Class())) {
      std::cerr << "this global function does not work for " << cl->GetName() << " because it does not inherit from TObject\n";
      return 0;
   }
   TObject *tobj = (TObject*)obj;
   return tobj->IsA();
}

void TestTClassGlobalIsA() {
   TNamed * m = new TNamed("example","test");

   TClass * cltobj = TObject::Class();
   TClass * cltnam = TNamed::Class();

   TObject* o = m;
   
   bool hasError = kFALSE;
   
   // First the normal test
   if ( cltobj == cltobj->GetActualClass(o) ) {
      std::cerr << "cltobj->IsA(o) returns the pointer class not the object class\n";
      hasError = true;
   }
   if ( cltnam != cltobj->GetActualClass(o) ) {
      std::cerr << "cltobj->IsA(o) does not return the object class\n";
      hasError = true;
   }
   if ( cltnam != cltnam->GetActualClass(m) ) {
      std::cerr << "cltnam->IsA(o) does not return the object class\n";
      hasError = true;
   }
   if (hasError) gApplication->Terminate(1);

   
   // we are in a dictionary so we can do this (well not on windows though :( ).
   cltobj->fIsA = 0;
   cltnam->fIsA = 0;
   cltobj->SetGlobalIsA(globalIsA);
   cltnam->SetGlobalIsA(globalIsA);
   if ( cltnam != cltobj->GetActualClass(o) ) {
      std::cerr << "cltobj->IsA(o) does not return the object class\n";
      hasError = true;
   }
   if ( cltnam != cltnam->GetActualClass(m) ) {
      std::cerr << "cltnam->IsA(o) does not return the object class\n";
      hasError = true;
   }
   
   if (hasError) gApplication->Terminate(1);
}
