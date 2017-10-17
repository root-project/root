#include "TError.h"
#include "TH1F.h"
#include "TMemFile.h"
#include "TTree.h"

#include <vector>

TFile *Create() {
   TMemFile *file = new TMemFile("checkMakeClass","RECREATE");
   TTree *tree = new TTree("T","T");
   std::vector<int> input;
   input.push_back(1);
   tree->Branch("data",&input);
   tree->Fill();
   // Put TTree in the odd state
   tree->SetMakeClass(true);
   file->Write();
   delete tree;
   return file;
}

int Read(TFile *file) {
   TTree *t;
   file->GetObject("T",t);
   if (!t) {
      Error("Read","Could not reat the TTree");
      return 2;
   }

   if (t->GetMakeClass()) {
      Error("Read","Tree is in MakeClass mode");
      return 3;
   }

   TBranch *b = t->GetBranch("data");

   if (!b) {
      Error("Read","Could not reat the TBranch named data");
      return 4;
   }

   if (b->GetMakeClass()) {
      Error("Read","The branch named data is in MakeClass mode");
      // return 5;
   }

#if 0
   if (0 == t->GetEntry(0)) {
      Error("Read","Reading the entry failed");
      // return 6;
   }
#endif

   t->Draw("data","","goff");
   TH1F *htemp = dynamic_cast<TH1F*>(gDirectory->Get("htemp"));
   if (nullptr == htemp) {
      Error("Read","Could not find TTree::Draw's htemp histogram");
      return 7;
   }
   
   if (htemp->GetMean() != 1) {
      Error("Read","Error the histo mean is %f rather than 1",htemp->GetMean());
      return 8;
   }

   std::vector<int> *output = nullptr;
   t->SetBranchAddress("data",&output);
   if (0 == t->GetEntry(0)) {
      Error("Read","Reading the entry (after calling SetBranchAddress) failed");
      // return 9;
   }

   if (nullptr == output) {
      Error("Read","After the output pointer is still null");
      return 10;
   }

   if (0 == output->size()) {
      Error("Read","After reading the vector is empty");
      // return 11;
   }

   if (1 != output->size()) {
      Error("Read","After reading the vector is size is %d rather than the expected 1",(int)output->size());
      // return 12;
   }

   if (1 != output->at(0)) {
      Error("Read","After reading the vector's value is %d rather than the expected 1",output->at(0));
      // return 13;
   }

   return 0;
}

int assertMakeClass() {
   TFile *file = Create();
   if (!file) {
      Error("assertMakeClass","Could not create TMemFile");
      return 1;
   }
   int result = Read(file);
   delete file;
   return result;
}

