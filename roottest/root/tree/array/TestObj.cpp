/*
        TestObj.cpp
*/

#include "TestObj.h"

#include "TFile.h"
#include "TTree.h"

ClassImp(Reconstruction)
//const bool Reconstruction::gInitted = Reconstruction::StaticInit();




void save()
{
   AllReconstructions* pRec = new AllReconstructions;
   TFile* file = new TFile("test.root", "RECREATE");
   TTree* Tree = new TTree("TestTree", "Test");
   Tree->Branch("Main", "AllReconstructions", &pRec, 32000, 99);
   pRec->Init();
   for (int i = 0; i < 10; ++i) {
      pRec->Init(i);
      Tree->Fill();
   };
   file->Write();
   pRec->print();

   delete file;
   file = 0;
   Tree = 0;

   delete pRec;
   pRec = 0;

   fprintf(stderr, "reloading\n");
   file = new TFile("test.root");
   TTree* tree = (TTree*) file->Get("TestTree");
   tree->SetBranchAddress("Main", &pRec);
   tree->GetEntry(3);
   pRec->print();
   // Disconnect the branch from the local variable.
   tree->SetBranchAddress("Main", 0);
}
