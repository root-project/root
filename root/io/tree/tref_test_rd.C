#ifndef __CINT__
#include "TFile.h"
#include "TApplication.h"
#include "TSystem.h"
#include "TTree.h"
#include "TRef.h"
#include "TH1.h"

void tref_test_rd()
#endif
{
#ifdef __CINT__
  gROOT->Reset();
#endif

  Int_t bufSize = 64000;
  Int_t nEvents = 6;

  TFile hFile("test2.root");

#ifdef ClingReinstateImplicitDynamicCast
  TTree* tree = hFile.Get("tree");
#else
  TTree* tree = (TTree*)hFile.Get("tree");
#endif

  TObject* obj = new TObject();
  TRef*  ref = new TRef();

  tree->SetBranchAddress("Abc",&obj);
  tree->SetBranchAddress("Refs",&ref);

  //------------------------------------------------------

  // Main 'event loop'.
  for (Int_t eventNum = 0; eventNum != tree->GetEntriesFast(); ++eventNum) {

    tree->GetEntry(eventNum);

    cout << " bits: " << Form("%x",obj->TestBits(0xffffffff)) << endl;
  } // for (eventNum)

  //------------------------------------------------------

  // Close the output file.
  hFile.Close();

  // Done.
  return;
}
