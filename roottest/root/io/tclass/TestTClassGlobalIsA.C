#include "TClass.h"
#include "TNamed.h"
#include "TClass.h"
#include <iostream>
#include "TApplication.h"
#include "TStopwatch.h"
#include "TList.h"
#include "TDataMember.h"

TClass* globalIsA(const TClass *cl, const void* obj) {
   if (gDebug>0) {
      std::cerr << "running globalIsA for " << (void*)cl << " and " << obj << std::endl;
   } else if (gDebug==0) {
      std::cout << "running globalIsA for " << cl->GetName() << std::endl;
   }
   if (! cl->InheritsFrom(TObject::Class())) {
      std::cerr << "this global function does not work for " << cl->GetName() << " because it does not inherit from TObject\n";
      return 0;
   }
   TObject *tobj = (TObject*)obj;
   return tobj->IsA();
}

void SetIsA(TClass *cl, TClass *newvalue)
{
   static long offset_fIsA = -1;
   if (offset_fIsA == -1) {
      TDataMember *d_fIsA = (TDataMember *)TClass::Class()->GetListOfDataMembers()->FindObject("fIsA");
      if (d_fIsA == 0) {
         std::cerr << "Could not find the offset of TClass::fIsA.\n";
      }
      offset_fIsA = d_fIsA->GetOffset();
   }
   TClass **ptr_fIsA = (TClass**)(((char*)cl)+offset_fIsA);
   *ptr_fIsA = newvalue;
}

void TestTClassGlobalIsA() {
   TNamed * m = new TNamed("example","test");

   TClass * cltobj = TObject::Class();
   TClass * cltnam = TNamed::Class();

   
   TObject* o = m;
   
   bool hasError = kFALSE;
   
   std::cout << "Running with fIsA\n";

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
   
   SetIsA(cltobj,0);
   SetIsA(cltnam,0);

   std::cout << "Running without fIsA\n";
   if ( cltnam != cltobj->GetActualClass(o) ) {
#ifndef ClingWorkAroundCallfuncAndVirtual
      std::cerr << "cltobj->IsA(o) does not return the object class\n";
      hasError = true;
#endif
   }
   if ( cltnam != cltnam->GetActualClass(m) ) {
      std::cerr << "cltnam->IsA(o) does not return the object class\n";
      hasError = true;
   }

   cltobj->SetGlobalIsA(globalIsA);
   cltnam->SetGlobalIsA(globalIsA);

   std::cout << "Running with  fGlobalIsA\n";
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

void TestTClassGlobalIsAPerf(int n=10) {
   TNamed * m = new TNamed("example","test");

   TClass * cltobj = TObject::Class();
   TClass * cltnam = TNamed::Class();

   TObject* o = m;
   
   bool hasError = kFALSE;

   TStopwatch w;


   
   // First the normal test
   if ( cltobj == cltobj->GetActualClass(o) ) {
      std::cerr << "cltobj->IsA(o) returns the pointer class not the object class\n";
      hasError = true;
   }
   
   int i;

   std::cout << "Running with fIsA\n";
   w.Start();
   for(i=0;i<n;++i) {
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
   w.Print();

   SetIsA(cltobj,0);
   SetIsA(cltnam,0);

   std::cout << "Running without fIsA\n";
   w.Start();
   for(i=0;i<n;++i) {
      if ( cltnam != cltobj->GetActualClass(o) ) {
         std::cerr << "cltobj->IsA(o) does not return the object class\n";
         hasError = true;
      }
      if ( cltnam != cltnam->GetActualClass(m) ) {
         std::cerr << "cltnam->IsA(o) does not return the object class\n";
         hasError = true;
      }
   }
   w.Print();

   cltobj->SetGlobalIsA(globalIsA);
   cltnam->SetGlobalIsA(globalIsA);
   gDebug = -1;
   std::cout << "Running with  fGlobalIsA\n";
   w.Start();
   for(i=0;i<n;++i) {
   
      if ( cltnam != cltobj->GetActualClass(o) ) {
         std::cerr << "cltobj->IsA(o) does not return the object class\n";
         hasError = true;
      }
      if ( cltnam != cltnam->GetActualClass(m) ) {
         std::cerr << "cltnam->IsA(o) does not return the object class\n";
         hasError = true;
      }
   }
   w.Print();
   if (hasError) gApplication->Terminate(1);
}
