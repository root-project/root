#include "abcsetup.h"

void abcwrite(const char* mode) {
   abcsetup(mode);
   Holder h;

   TFile f(TString::Format("abc_%s.root", mode), "recreate");
   TTree* tree = new TTree("tree", "abc tree");
   tree->Branch("h", &h);
   for (int e = 0; e < 100; ++e) {
      h.Set(e);
//      gDebug = 3;
	tree->Fill();
	gDebug = 0;
   }
   tree->Write();
   delete tree;
}
