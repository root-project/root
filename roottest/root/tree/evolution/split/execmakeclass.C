#include "TTree.h"

TTree *create() {
   TTree *t = new TTree("t","t");
   TNamed data("name","title");
   t->Branch("data.",&data);
   t->Fill();
   t->Fill();
   t->ResetBranchAddresses();
   return t;
}

void execmakeclass() {
   TTree *t = create();
   t->SetMakeClass(1);
   Int_t bits = 0;
   t->Show(0);
   t->SetBranchAddress("data.TObject.fBits",&bits);
   t->Show(0);
}
