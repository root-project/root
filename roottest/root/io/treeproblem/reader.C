#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"

#include <iostream>

int reader()
{
  gSystem->Load("libTreeProblemFoo");

  Int_t  foo_     = 0;
  Int_t  foo_fFoo[3];
  TFile* file     = TFile::Open("file.root", "READ");
  TTree* tree     = (TTree *) file->Get("tree");
  tree->SetMakeClass(1);
  tree->SetBranchAddress("foo", (void*)&foo_);
  tree->SetBranchAddress("foo.fFoo", &foo_fFoo);

  Int_t  n = (Int_t)tree->GetEntries();

  for (Int_t i = 0; i < n; i++) {
    tree->GetEntry(i);
    for (Int_t j = 0; j < foo_; j++)
      std::cout << "Foo # " << i << "," << j << ": " << foo_fFoo[j] << std::endl;
  }
  file->Close();

  return 0;
}

