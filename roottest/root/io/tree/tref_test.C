#include "TFile.h"
#include "TApplication.h"
#include "TSystem.h"
#include "TTree.h"
#include "TRef.h"
#include "TH1.h"

void tref_test()
{
  Int_t bufSize = 4000;
  Int_t nEvents = 2000;

  // Create a new file for the output.
  TFile hFile("test2.root","RECREATE");

  // Create the ROOT tree.
  TTree* tree = new TTree("tree","TRef tester tree");

  TObject* obj = new TObject();
  TRef*  ref = new TRef();

  // Create the branches in the tree.
  TBranch* b =tree->Branch("Abc","TObject",&obj,bufSize,99);
  // Follow Axel's advice and do not split the TRef's branch.
  TBranch* bb =tree->Branch("Refs","TRef",&ref,bufSize,0);

  tree->SetBranchAddress("Abc",&obj);
  tree->SetBranchAddress("Refs",&ref);

  //------------------------------------------------------

  // Main 'event loop'.
  for (Int_t eventNum = 0; eventNum != nEvents; ++eventNum) {

    if (eventNum % 3 == 0) {
      (*ref) = obj;
    } else {
      (*ref) = 0;
      obj->ResetBit(kIsReferenced);
      obj->SetUniqueID(0);
    }

    tree->Fill();

  } // for (eventNum)

  //------------------------------------------------------

  // Write the tree to file.
  tree->Write();

  // Close the output file.
  hFile.Close();

  // Done.
  return;
}
