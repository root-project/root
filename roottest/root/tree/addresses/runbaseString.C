#include "TFile.h"
#include "TTree.h"

#if !defined(__CINT__) || defined(__MAKECINT__)
#include "runbaseString.h"
#endif

bool baseString() {

  Final* f     = new Final();
  f->i = 99;
  f->SetString("this is a test");
  f->print(cout);

  TFile fo("tree.root","RECREATE");

  TTree* tree = new TTree("tree","Test Tree");

  tree->Branch("Final.","Final",&f, 32000,99);

  tree->Fill();

  fo.Write();

  fo.Close();

  f->i = -99;
  f->SetString("erased value");
  f->print(cout);

  TFile fi("tree.root","READ");
  fi.GetObject("tree",tree);
  tree->SetBranchAddress("Final.",&f);
  tree->GetEntry(0);
  
  f->print(cout);

  return true;
}

bool runbaseString() { 
   return !baseString();
}

