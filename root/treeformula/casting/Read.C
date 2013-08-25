//
// Code to open root file containing tree, Get tree, and attempt to read in //
// 3 Simple objects from tree making use of TTreeFormula to apply selection
// cuts.
//
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TTUBE.h"
#include "Simple.h"
#include "TTreeFormula.h"
#include <iostream>
using namespace std;

int Read(bool debug = false) 
{

// Open file to read Simple entries
  TFile *file = new TFile("simple.root","READ");

// Retrieve ROOT tree holding the data
  TTree *tree = dynamic_cast<TTree*>(file -> Get("SimpleTree"));

  // const char* select = "fID > 0"; // this works
  // The following selection cut should select all 3 entries, but doesn't work
  const char* select1 = "((TBRIK*)fShape)->GetVisibility() > 0";
  const char* select2 = "((TTUBE*)fShape)->GetRmin() > 0";

  TTreeFormula* treeformula1 = new TTreeFormula("myselection",select1,tree);
  if (treeformula1->GetNdim()==0) return 0;

  TTreeFormula* treeformula2 = new TTreeFormula("myselection",select2,tree);
  if (treeformula2->GetNdim()==0) return 0;

  Int_t npass1 = 0;
  Int_t npass2 = 0;

  Int_t nent = (Int_t)tree -> GetEntries();

  Simple *simple = new Simple();
  tree -> SetBranchAddress("Simple",&simple);
  for (Int_t ient=0; ient < nent; ient++) {

    // change TTree fReadEntry to entry of interest
    tree -> LoadTree(ient);

    // Ask the TTreeFormula to evaluate if this entry passes selection cut
    if (treeformula1->GetNdata() && treeformula1 -> EvalInstance() > 0) {
      // entry passes the cuts, retrieve the entire record
      tree -> GetEntry(ient);
      npass1++;
      if (debug) simple -> Print();
    }

    // Ask the TTreeFormula to evaluate if this entry passes selection cut
    if (treeformula2->GetNdata() && ( treeformula2 -> EvalInstance()) > 0) {
      // entry passes the cuts, retrieve the entire record
      tree -> GetEntry(ient);
      npass2++;
      if (debug) simple -> Print();
    }

  }
  delete simple;

  // Finished all entries
  if (debug) {
    cout << "Total entries in tree = " << nent << ". " << endl;
    cout << "Total passing cuts " << select1 << " = " << npass1 << endl;
    cout << "Total passing cuts " << select2 << " = " << npass2 << endl;
  }

  bool success = true;
  if (npass1!=1) {
    cerr << "ERROR: " << select1 << " selected " << npass1 << " entries instead of 1\n";
    success = false;
  }
  if (npass2!=3) {
    cerr << "ERROR: " << select2 << " selected " << npass2 << " entries instead of 3\n";
    success = false;
  }

  return success;

}
