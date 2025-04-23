#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif
#ifndef TREEPROBLEM_Foo
#include "Foo.h"
#endif
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif
#ifndef __IOSTREAM__
#include <iostream>
#endif

int writer()
{
#if defined(__CLING__) && !defined(ClingWorkAroundMissingDynamicScope) 
  gSystem->Load("Foo");
#endif

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

#ifndef __CLING__
int main(int argc, char** argv)
{
  return writer();
}
#endif
