#include "TTree.h"
#include "TClonesArray.h"
#include "TParticle.h"

void execWriteRead() {

   TTree *tree = new TTree("T","T");
   TParticle named;
   tree->Branch("Particles",&named);
   tree->Fill();
   tree->GetEntry(0);
}
