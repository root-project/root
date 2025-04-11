#include "TObject.h"
#include "TChain.h"
#include <stdio.h>

struct CallBack : public TObject {
  CallBack(TChain *ch) : fChain(ch) {};
  TChain *fChain; // We do not own this.
  Bool_t Notify() override {
     printf("Notify called for a tree with %lld entries\n",fChain->GetTree()->GetEntries());  
     return kTRUE;
  }
  ClassDefOverride(CallBack,0);
};

int execEmpty() {
  auto mychain = new TChain("ntup");
  CallBack callback(mychain);
  mychain->SetNotify(&callback);
  mychain->AddFile("n_empty.root");
  mychain->AddFile("n_full.root");
  mychain->AddFile("n_empty.root");
  mychain->AddFile("n_full.root");
  mychain->AddFile("n_empty.root");
  for(Long64_t e = 0; ; ++e) {
    if (0 > mychain->LoadTree(e)) break;
  }
  return 0;
}
