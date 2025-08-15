// runvaryingArrayNoClassDefWrite.C

#include "BNoClassDef.h"
#include "ANoClassDef.h"
#include "BNoClassDef.C"
#include "ANoClassDef.C"

#include "Rtypes.h"
#include "TBranch.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>

using namespace std;

const char* testFile = "varyingArrayNoClassDef.root";
const int nTestEntries = 3;


void runvaryingArrayNoClassDefWrite()
{
  //
  // Create and write a test tree.
  //

  A* a = new A();

  TFile* f1 = new TFile(testFile, "recreate");
  TTree* t1 = new TTree("t1", "test tree");
  TBranch* b1 = t1->Branch("A.", &a);
  b1->SetAddress(&a);
  for (int i = 0; i < nTestEntries; ++i) {
    a->Init();
    a->repr();
    t1->Fill();
  }
  t1->Write();
  f1->Close();
  delete f1;
  f1 = 0;
  b1 = 0;
  delete a;
  a = 0;
}
