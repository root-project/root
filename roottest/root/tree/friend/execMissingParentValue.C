#include "TTree.h"
#include "Riostream.h"

int execMissingParentValue() 
{
   int run,event;
   double val;
   TTree *one = new TTree("one","TTree with run/event");
   one->Branch("run",&run);
   one->Branch("event",&event);
   one->Branch("val",&val);

   TTree *two = new TTree("two","TTree with run2/event2");
   two->Branch("run2",&run);
   two->Branch("event2",&event);
   two->Branch("val2",&val);
   
   for(Int_t i = 0; i < 100; ++i) {
      run = i / 100;
      event = i % 100;
      val = i / 2.0;
      if ( (i%2)==0 ) two->Fill();
      one->Fill();
   }
   one->BuildIndex("run","event");
   two->BuildIndex("run2","event2");

   one->ResetBranchAddresses();
   two->ResetBranchAddresses();
   cout <<  "The run2 branch is not yet found: ";
   cout << (one->GetBranch("run2")!=0) << '\n';

   one->AddFriend(two);
   cout << "The run2 branch is found because they are now friends: ";
   cout << (one->GetBranch("run2")!=0) << '\n';
   one->GetEntry(2);

   cout << "The following should give a negative number or the entry number (2) of the parent tree\n";
   cout << two->GetTreeIndex()->GetEntryNumberFriend(one) << '\n';

   cout << "Without the finding the proper indices, we see the entries uncorrelated\n";
   one->SetScanField(-1);
   one->Scan("val:two.val2","","",100);

   // Rebuild the index to make sure the formula looking at the parent.
   two->BuildIndex("run2","event2");
   one->SetAlias("run2","run");
   one->SetAlias("event2","event");
   one->SetScanField(-1);
   cout << "With setting the aliases, we see the entries properly correlated\n";
   one->Scan("val:two.val2","","",100);
   return 0;
}
