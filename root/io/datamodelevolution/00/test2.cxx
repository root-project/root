#include "DataModelV2.h"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"

#include "TBufferFile.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"

#include "TVirtualObject.h"


void test2() {

   TFile *f = TFile::Open("test1.root","READ");

   TClassRef cl = TClass::GetClass("ACache");

   // cl->GetStreamerInfos()->ls();

   cout << "Stream out object\n";
   ACache *a;  f->GetObject("obj",a);

   // cl->GetStreamerInfos()->ls();

   if (!a) {
      ::Error("test2","Could not find the object");
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

   ACache *ptr = 0;
   tree->SetBranchAddress("obj",&ptr);
   tree->GetEntry(0);
   if (ptr==0) {
      ::Error("test2","No object read from the TTree");
   } else {
      ptr->Print();
   }

   //tree->Print("debugInfo");
   //TClass::GetClass("ACache")->GetStreamerInfos()->ls();

   delete f;
}
