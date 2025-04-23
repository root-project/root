#define tree_cxx
#include "tree.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

void tree::Loop()
{
  if (fChain == 0) return;
  
  Int_t nentries = Int_t(fChain->GetEntriesFast());
  
  Int_t nbytes = 0, nb = 0;
  for (Int_t jentry=0; jentry<nentries;jentry++) {
    Int_t ientry = LoadTree(jentry);
    if (ientry < 0) 
      break;
    nbytes = fChain->GetEntry(jentry);   

    for (Int_t i = 0; i < foo_; i++) 
      cout << "Foo: " << foo_fFoo[i] << endl;
  }
}
