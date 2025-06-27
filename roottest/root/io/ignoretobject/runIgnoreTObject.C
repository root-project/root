// runIgnoreTObject.C
//
// Make sure that the TObject base branch
// is not created for a class which has had
// TClass::IgnoreTObjectStreamer() called.
//

#include "TTree.h"
#include "TBranchElement.h"
#include "TFile.h"
#include "TROOT.h"

#include <iostream>

#include "A.C"

using namespace std;


class example : public TObject {
public:
   example() : number(9.99) {
      example::Class()->IgnoreTObjectStreamer();  // don't store TObject's fBits and fUniqueID
   }
   double number;
   ClassDefOverride(example, 1)
};

UInt_t test3() {

   UInt_t failcount = 0;

   TFile* file = TFile::Open("test.root", "READ");
   TTree* tree; file->GetObject("tree",tree);

   if (!tree) return 1;

   if ( tree->GetListOfLeaves()->GetEntries() != 2) {
      cout << "Error: There is too many leaves in the tree. It should be 2 (it is " << tree->GetListOfLeaves()->GetEntries() << ")\n";
      tree->Print();
      ++failcount;
      return failcount;
   };

   example* content = new example;
   tree->SetBranchAddress("content", &content);

   for (Long64_t i=0; i<tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      if ( abs(content->number - i) > 0.01 ) {
         cout << "Error: the value is incorrect, expect " << i  << " and we have: " << content->number << '\n';
         ++failcount;
      }
   }

   delete tree;
   delete file;

   return failcount;
}

UInt_t test2() {

   UInt_t failcount = 0;

   TFile file("test.root", "RECREATE");
   TTree* tree = new TTree("tree", "tree");

   example* content = new example;
   auto b = tree->Branch("content", &content);
   if ( b->GetListOfBranches()->GetEntries() != 1) {
      cout << "Error: There is too many branches in \"content\". It should be 1 (it is " << b->GetListOfBranches()->GetEntries() << ")\n";
      tree->Print();
      ++failcount;
      return failcount;
   };
   if ( b->GetListOfLeaves()->GetEntries() != 1) {
      cout << "Error: There is too many branches in \"content\". It should be 1 (it is " << b->GetListOfLeaves()->GetEntries() << ")\n";
      tree->Print();
      ++failcount;
      return failcount;
   };
   if ( tree->GetListOfLeaves()->GetEntries() != 2) {
      cout << "Error: There is too many leaves in the tree. It should be 2 (it is " << tree->GetListOfLeaves()->GetEntries() << ")\n";
      tree->Print();
      ++failcount;
      return failcount;
   };

   for (int i=0; i<10000; ++i) {
      content->number = i;
      tree->Fill();
   }

   tree->Write();

   file.Close();
   return 0;
}

UInt_t test1()
{
   UInt_t failcount = 0;

   A* a = new A();
   TTree* t1 = new TTree("t1", "Test Tree");
   TBranchElement* br1 = reinterpret_cast<TBranchElement*>(t1->Branch("br1.", "A", &a));
   TBranchElement* brx = reinterpret_cast<TBranchElement*>(br1->GetListOfBranches()->At(0));
   cout << "name: " << brx->GetName() << " id: " << brx->GetID() << endl;
   if ( brx->GetID() != 1 ) {
      cout << "Error: the first branch id should be 1 (it is " << brx->GetID() << ")\n";
      ++failcount;
   }
   if ( br1->GetListOfBranches()->GetEntries() != 3 ) {
      cout << "Error: There is too many branches in \"br1\". It should be 3 (it is " << br1->GetListOfBranches()->GetEntries() << ")\n";
      ++failcount;
      t1->Print();
   }
   delete t1;
   delete a;

   return failcount;
}

int runIgnoreTObject()
{
   UInt_t failcount = 0;
   failcount += test1();
   failcount += test2();
   failcount += test3();

   return failcount;

}
