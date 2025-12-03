#include "TError.h"
#include "TFile.h"
#include "TClass.h"
#include "TSystem.h"
#include "TROOT.h"

int execReadTmplt() {
   TFile *f1 = TFile::Open("tmpltd32.root");
   TFile *f2 = TFile::Open("tmpltd.root");
  
   if (f1 == 0 || f2 == 0) {
      Error("execReadTmplt","Missing files");
      return 11;
   }
 
   TClass *cl = TClass::GetClass("Wrapper<double>");
   TClass *cl32 = TClass::GetClass("Wrapper<Double32_t>");
   TClass *cl64 = TClass::GetClass("Wrapper<Long64_t>");

   if (!cl) {
      Error("execReadTmplt","Could not find the TClass for Wrapper<double>");
      return 1;
   }
   if (!cl32) {
      Error("execReadTmplt","Could not find the TClass for Wrapper<Double32_t>");
      return 2;
   }
   if (!cl64) {
      Error("execReadTmplt","Could not find the TClass for Wrapper<Long64_t>");
      return 3;
   }
   if (cl->IsLoaded()) {
      Error("execReadTmplt","TClass for Wrapper<double> is already loaded!");
      return 4;
   }
   if (cl32->IsLoaded()) {
      Error("execReadTmplt","TClass for Wrapper<Double32_t> is already loaded!");
      return 5;
   }
   if (cl64->IsLoaded()) {
      Error("execReadTmplt","TClass for Wrapper<Long64_t> is already loaded!");
      return 6;
   }

   if (cl->GetClassInfo()) {
      Error("execReadTmplt","TClass for Wrapper<double> has already a classinfo!");
      return 7;
   }
   if (cl32->GetClassInfo()) {
      Error("execReadTmplt","TClass for Wrapper<Double32_t> has already a classinfo!");
      return 8;
   }
   if (cl64->GetClassInfo()) {
      Error("execReadTmplt","TClass for Wrapper<Long64_t> has already a classinfo!");
      return 9;
   }

   // Now try loading the Double32_t
   // gSystem->Load("execTmpltD32_cxx");
   gROOT->ProcessLine("#include \"tmplt.h\"");
   gROOT->ProcessLine("template class Wrapper<Double32_t>;");
   gROOT->ProcessLine("template class Wrapper<Long64_t>;");
 
   TClass *alt = TClass::GetClass("Wrapper<Double32_t");
   if (alt != cl32 ) {
      Error("execReadTmplt","Wrapper<Double32_t> was replaced.");
      return 7;
   }
   alt = TClass::GetClass("Wrapper<double");
   if (alt != cl ) {
      Error("execReadTmplt","Wrapper<double> was replaced.");
      return 8;
   }
   alt = TClass::GetClass("Wrapper<Long64_t>");
   if (alt != cl64 ) {
      Error("execReadTmplt","Wrapper<Long64_t> was replaced.");
      return 9;
   }

   if (cl->IsLoaded()) {
      Error("execReadTmplt","TClass for Wrapper<double> is already loaded!");
      return 10;
   }
   if (cl32->IsLoaded()) {
      Error("execReadTmplt","TClass for Wrapper<Double32_t> is already loaded!");
      return 11;
   }
   if (cl64->IsLoaded()) {
      Error("execReadTmplt","TClass for Wrapper<Long64_t> is already loaded!");
      return 12;
   }


   if (!cl->GetClassInfo()) {
      Error("execReadTmplt","TClass for Wrapper<double> does not have a classinfo!");
      return 13;
   }
   if (!cl64->GetClassInfo()) {
      Error("execReadTmplt","TClass for Wrapper<Long64_t> does not have a classinfo!");
      return 14;
   }
#ifndef ClingWorkAroundTClassUpdateDouble32
   if (!cl32->GetClassInfo()) {
      Error("execReadTmplt","TClass for Wrapper<Double32_t> does not have a classinfo!");
      return 15;
   }
#endif
   return 0;
}
