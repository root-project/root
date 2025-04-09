#include "DataModelV2.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TError.h"
#include "TBranch.h"

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
   cout << "TTree::Scan object branch\n";
   tree->Scan("*");

   ACache *ptr = new ACache;
   tree->SetBranchAddress("obj",&ptr);
   cout << "TTree::GetEntry object branch\n";
   tree->GetEntry(0);
   if (ptr==0) {
      ::Error("test2","No object read from the TTree");
   } else {
      ptr->Print();
   }

   delete ptr;
   ACache *ptr2 = new ACache;
   // tree->SetBranchAddress("obj",&ptr2);
   auto b = tree->GetBranch("obj");
   b->SetAddress(&ptr2);
   cout << "TTree::GetEntry object branch with 'new' object.\n";
   tree->GetEntry(0);
   if (ptr2 == nullptr) {
      ::Error("test2","No object read from the TTree");
   } else {
      ptr2->Print();
   }

   //tree->Print("debugInfo");
   //TClass::GetClass("ACache")->GetStreamerInfos()->ls();

   delete f;
}
