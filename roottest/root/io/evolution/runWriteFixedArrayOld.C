#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

#include "FixedArrayOld.C"

using namespace std;

void runWriteFixedArrayOld()
{
   TFile* f = new TFile("FixedArrayOld.root", "recreate");
   FixedArrayContainer* fa = new FixedArrayContainer();
   TTree* t1 = new TTree("t1", "Test Tree");
   t1->Branch("br1.", "FixedArrayContainer", &fa);
   t1->Fill();
   t1->Write();
   delete t1;
   t1 = 0;
   delete fa;
   fa = 0;
   f->Close();
   delete f;
   f = 0;
}

