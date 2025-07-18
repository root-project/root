// runvaryingArrayRead.C

#include "B.h"
#include "A.h"
#include "B.C"
#include "A.C"

#include "Rtypes.h"
#include "TBranch.h"
#include "TFile.h"
#include "TTree.h"

using namespace std;

const char* testFile = "varyingArray.root";
const int nTestEntries = 3;


void runvaryingArrayRead()
{
  //
  // Read and dump a test tree.
  //

  A* a = new A();

  TFile* f1 = new TFile(testFile);
  TTree* t1 = 0;
  f1->GetObject("t1;1", t1);
  TBranch* b1 = t1->GetBranch("A.");
  b1->SetAddress(&a);

  for (int i = 0; i < nTestEntries; ++i) {
    t1->GetEntry(i);
    a->repr();
  }

  f1->Close();
  delete f1;
  f1 = 0;

  delete a;
  a = 0;
}
