// runIgnoreTObject.C
//
// Make sure that the TObject base branch
// is not created for a class which has had
// TClass::IgnoreTObjectStreamer() called.
//

#include "TTree.h"
#include "TBranchElement.h"

#include "A.C"

#include <iostream>

using namespace std;

void runIgnoreTObject()
{
   A* a = new A();
   TTree* t1 = new TTree("t1", "Test Tree");
   TBranchElement* br1 = reinterpret_cast<TBranchElement*>(t1->Branch("br1.", "A", &a));
   TBranchElement* brx = reinterpret_cast<TBranchElement*>(br1->GetListOfBranches()->At(0));
   cout << "name: " << brx->GetName() << " id: " << brx->GetID() << endl;
   delete t1;
   t1 = 0;
   delete a; 
   a = 0;
}

