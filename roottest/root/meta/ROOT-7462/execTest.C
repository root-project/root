{
   gROOT->ProcessLine(".L instHeader.h");
   //#include "../instHeader.h"
   auto c = TClass::GetClass("Outer");
   printf("Name of real 'variable' seen as: '%s'\n",c->GetListOfRealData()->At(1)->GetName());

   instHeaderTestValid(false);
   instHeaderTestDecl(true);

   TFile *file = TFile::Open("inst.root");
   if (file == nullptr || file->IsZombie()) gSystem->Exit(1);
   void *obj = file->Get("object");

   // Delay parsing as long as possible by using ProcessLine
   if (obj) gROOT->ProcessLine(TString::Format("{Outer *o = (Outer*)%p; o->Print();}",obj));

   // Loading the library (or even auto parsing the header since it is the same for both library
   // solves the I/O problem.
   gSystem->Load("inst2lib");

   instHeaderTestValid(true);
   instHeaderTestDecl(false);
      
   printf("Name of real 'variable' seen as: '%s'\n",c->GetListOfRealData()->At(1)->GetName());
   obj = file->Get("object");
   if (obj) gROOT->ProcessLine(TString::Format("{Outer *o = (Outer*)%p; o->Print();}",obj));

   return 0;
}
