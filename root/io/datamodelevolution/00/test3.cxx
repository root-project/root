#include "DataModelV3.h"
#include "TFile.h"
#include "TError.h"

#include "TBufferFile.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TTree.h"

void test3() {

   TFile *f = TFile::Open("test1.root","READ");

   TClassRef cl = TClass::GetClass("ACache");
   TClassRef newcl = TClass::GetClass("Axis");

   cout << "Stream out object\n";
   Axis *a;  f->GetObject("obj",a);

   if (!a) {
      ::Error("test3","Could not find the object");
      return;
   }

   
   a->Print();

   cout << "Stream out embedded object\n";
   Container *c; f->GetObject("cont",c);
   if (!c) {
      ::Error("test2","Could not find the container object");
      return;
   }

   c->a.Print();

   TTree *tree; f->GetObject("tree",tree);
   tree->Scan("*");

   Axis *ptr = 0;
   tree->SetBranchAddress("obj",&ptr);
   tree->GetEntry(0);
   ptr->Print();

   // tree->Print("debugInfo");

   delete f;
}
