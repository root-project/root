// runvaryingArrayNoClassDefRead.C

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


void runvaryingArrayNoClassDefRead()
{
  //
  // Now read and dump test tree.
  //

  A* a = new A();
  TTree* t1 = 0;

  TFile* f1 = new TFile(testFile);
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
  b1 = 0;
  delete a;
  a = 0;
}

#ifdef TEST
int main()
{
  runvaryingArrayNoClassDefRead();
  return 0;
}
#endif // TEST

