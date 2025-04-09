#include "TTree.h"
#include <memory>

TTree *create() {
   TTree *tree = new TTree("tree", "Tree with Char_t");
   Int_t ntrk;
   tree->Branch("ntrk", &ntrk, "ntrk/I");
   Char_t charge[3];
   ntrk = 3;
   tree->Branch("charge", &charge, "charge[ntrk]/B");
   charge[0] = -1;
   charge[1] = -2;
   charge[2] = 2;
   tree->Fill();
   tree->ResetBranchAddresses(); 
   return tree;
}

int runcscan() {
   TTree *p = create(); 
   p->Scan("charge*1.0:1.0*charge","","colsize=13");
   return 0;
}



