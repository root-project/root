#include "TTree.h"



TTree* modulo() {

   TTree *tree = new TTree("T","T");
   UInt_t i = 1<<31;
   Int_t k = 1<<20;
   tree->Branch("unsignedInt",&i,"unsignedInt/i");
   tree->Branch("signedInt",&k,"signedInt/I");
   for(int j=0;j<2;++j) {
      tree->Fill();
      k = -k;
   }
   i = 2763;
   k = 11*i;
   tree->Fill();

   tree->Scan("signedInt:signedInt%235");
   tree->Scan("unsignedInt:unsignedInt%235","","colsize=15");
   tree->Scan("(signedInt/unsignedInt):(signedInt/unsignedInt)%10");
   return tree;
}

int runmodulo() {
   modulo();
   return 0;
}