//
// Driver code to open file, create tree, and write 3 Simple objects to     //
// tree.
//
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TTUBE.h"
#include "TBRIK.h"
#include "Simple.h"

int Create(bool debug)
{

// Open output file to hold events
  TFile *simplefile = new TFile("simple.root","RECREATE","Simple root file");

// Create a ROOT tree to hold the simple data
  TTree *simpletree = new TTree("SimpleTree","Simple Tree");

  Simple *simple = 0;
  // Split branches to store fShape objects & fID on separate branches
  TTree::SetBranchStyle(1);
  simpletree->Branch("Simple","Simple",&simple,16000,1);

  for (Int_t ient=0; ient < 3; ient++) {
// Create 3 Simple objects containing TTube's and store each in tree
    TTUBE *tube = new TTUBE("TUBE","TUBE","void",150,200,400);
    // tube is adopted & deleted by Simple
    simple = new Simple(ient,tube);
    if (debug) simple -> Print();
    simpletree -> Fill();
    delete simple;
  }
  TBRIK * brik = new TBRIK("BRIK","BRIK","void",10,20,30);
  simple = new Simple(4,brik);
  if (debug) simple -> Print();
  simpletree -> Fill();
  delete simple;

// Write the simple tree to file
  simplefile->Write();
// Print final summary of simple tree contents
  if (debug) simpletree->Print();

  return 0;
}
