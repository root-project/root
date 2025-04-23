#include "TTree.h"
#include "Riostream.h"

int runcircular() {
   int i = 0;
   int j = 1;
   // int k = 2;;
   TTree *master = new TTree("master","master");
   master->Branch("i",&i);
   TTree *slave = new TTree("slave","slave");
   slave->Branch("j",&j);
   for(int s=0; s<10; ++s) master->Fill();
   for(int s=0; s<15; ++s) slave->Fill();
   cout << "Alone\n";
   cout << master->LoadTree(3) << endl;
   cout << master->LoadTree(12) << endl;
   cout << master->LoadTree(20) << endl;

   master->AddFriend(slave);
   cout << "Friend\n";
   cout << master->LoadTree(3) << endl;
   cout << master->LoadTree(12) << endl;
   cout << master->LoadTree(20) << endl;

   slave->AddFriend(master);
   cout << "Indirect Circular\n";
   cout << master->LoadTree(3) << endl;
   cout << master->LoadTree(12) << endl;
   cout << master->LoadTree(20) << endl;

   master->AddFriend(master);
   cout << "Direct Circular\n";
   cout << master->LoadTree(3) << endl;
   cout << master->LoadTree(12) << endl;
   cout << master->LoadTree(20) << endl;

   if (master->LoadTree(20) != -2) {
     cout << "A circular TTree friendship leads to LoadTree incorrectly returning: " << master->LoadTree(20) << " when it should return -2 (out of bound entry)\n";
     return 1;
   }
   return 0; 
}
