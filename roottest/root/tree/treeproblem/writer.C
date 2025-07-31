#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "Foo.h"
#include "TROOT.h"


int writer()
{
  gROOT->LoadClass("Foo", "Foo");

  TClonesArray* array = new TClonesArray("Foo");
  TFile* file         = TFile::Open("file.root", "RECREATE");
  TTree* tree         = new TTree("tree", "tree");
  tree->Branch("foo", &array);

  Int_t  n = 100;
  for (Int_t i = 0; i < n; i++) {
    array->Clear();
    for (Int_t j = 0; j < 3; j++)
      new ((*array)[j]) Foo(i * 10 + j);
    tree->Fill();
  }

  tree->Print();
  file->Write();
  file->Close();

  return 0;
}

