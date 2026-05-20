#include "TBranchElement.h"
#include "TFile.h"
#include "TTree.h"

#include "FixedArrayNew.C"

#include <iostream>

using namespace std;

void runReadFixedArrayOldWithNew()
{
   TFile* f = new TFile("FixedArrayOld.root");
   FixedArrayContainer* fa = new FixedArrayContainer();
   TTree* t1 = (TTree*) f->Get("t1");
   TBranchElement* br1 = (TBranchElement*) t1->GetBranch("br1.");
   //br1->GetListOfBranches()->ls();
   br1->SetAddress(&fa);
   TBranchElement* br2 = (TBranchElement*) br1->GetListOfBranches()->At(1);
   //cerr << "-----" << endl;
   //br2->GetListOfBranches()->ls();
   int* offsetTable = br2->GetBranchOffset();
   //cerr << "-----" << endl;
   cerr << "offset: " << offsetTable[3] << endl;
}

