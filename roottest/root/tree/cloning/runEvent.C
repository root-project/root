#include "TApplication.h"
#include "TBranch.h"
#include "TChain.h"
#include "TTree.h"

#include <iostream>

void runEvent()
{
   bool result = true;

   TChain* chain = new TChain("T");
   chain->Add("event1.root");
   chain->Add("event2.root");

   // Get a clone of the first tree in the chain
   // filled with all the entries from the chain.
   TTree* clone1 = chain->CopyTree("");

   // Make sure the clone got the branches.
   TBranch* br = clone1->GetBranch("event");

   // Reset the chain to the first tree (it was
   // on the second tree after the copy above).
   chain->GetEntry(0);

   // Create a second clone of the first tree
   // in the chain, also filled with all the
   // entries from the chain.
   TTree* clone2 = chain->CopyTree("");

   // Switch to the first tree in the chain.
   chain->LoadTree(0);

   // Remember the number of entries in the
   // first tree in the chain.
   Long64_t n = chain->GetTree()->GetEntries();

   // Create a clone of the first tree
   // in the chain, with no entries.
   // Note: The chain considers this tree to be
   //       a clone of the chain itself.
   TTree* clone3 = chain->CloneTree(0);

   // Make sure the clone has the same branch
   // addresses as the chain.
   void* chainadd = chain->GetBranch("event")->GetAddress();
   void* clone3add = clone3->GetBranch("event")->GetAddress();
   if (chainadd != clone3add) {
      cerr << "clone3 is not connected to the chain\n";
      result = false;
   }

   double a;
   clone3->Branch("MT",&a,"MT/D");

   // Switch to the second tree in the chain.
   // Note: This should force the branch addresses to
   //       be changed in clone3.
   chain->LoadTree(n+1);

   // Make sure the clone has the same branch
   // addresses as the chain.
   chainadd = chain->GetBranch("event")->GetAddress();
   clone3add = clone3->GetBranch("event")->GetAddress();
   if (chainadd != clone3add) {
      cerr << "clone3 is not well connected to the chain\n";
      result = false;
   }

   if (clone3->GetBranch("MT")->GetAddress() != (char*)&a) {
      cerr << "We have a problem since the address of the branch MT (" << (void*)(clone3->GetBranch("MT")->GetAddress())
           << " is not the address of the variable (" << (void*)&a << ")" << endl;
   }
    
   // We are deleting a clone of a chain, it should not
   // attempt to delete any allocated objects.
   delete clone3;
   clone3 = 0;

   // Create a clone of the first tree in the chain,
   // and fill it with all the entries from the first tree.
   // Note: The chain considers this tree to be
   //       a clone of the chain itself.
   TTree* clone4 = chain->CloneTree();

   if (!result) {
      gApplication->Terminate(1);
   }
}

