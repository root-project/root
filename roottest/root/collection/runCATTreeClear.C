// Test for r42332

#include "TClonesArray.h"
#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"

void writeCA() {
   TFile* file = TFile::Open("ca.root", "RECREATE");
   TTree* tree = new TTree("T", "CA tree");
   TClonesArray* ca = new TClonesArray("TNamed");
   tree->Branch("CA", &ca);

   (*ca)[0] = new TNamed("One","One title");
   (*ca)[1] = new TNamed("Two","Short lived");
   tree->Fill();

   ca->Clear();
   (*ca)[0] = new TNamed("One","New one");
   printf("During writing, tree event 1, ca entry 1 is %s NULL\n",
          ca->At(1) ? "!=" : "==");
   tree->Fill();

   tree->Write();
   delete file;
}

void readCA() {
   TFile* file = TFile::Open("ca.root");
   TTree* tree = 0;
   file->GetObject("T", tree);
   tree->Scan(); // BUG: notice how "One title" is shown twice!

   TClonesArray* ca = 0;
   tree->SetBranchAddress("CA", &ca);

   for (int i = 0; i < tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      printf("TCA[1] %s NULL\n", ca->At(1) ? "!=" : "==");
      TIter iCA(ca);
      TNamed* n = 0;
      printf("CA contains %d elements\n", ca->GetEntries());
      while ((n = (TNamed*)iCA()))
         // BUG fixed by r42332: "two" remained in the CA for tree entry 1
         printf(" Name: %s Title: %s\n", n->GetName(), n->GetTitle());
   }
}

void runCATTreeClear() {
   writeCA();
   readCA();
}
