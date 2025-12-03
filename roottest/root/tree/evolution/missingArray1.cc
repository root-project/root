// testclass.cc 
#define REQUIRED_VERSION 1
#include "myclass.cc" 
#include "TFile.h" 
#include "TTree.h" 
int missingArray1() 
{ 
   TFile f("missingArray.root","recreate"); 
   TTree *tree = new TTree("tree", "tree"); 
   myclass *my = new myclass();
   tree->Branch("mybranch", "myclass", &my, 32000, 99); 

   my->SeteSize(100);
   my->SetgSize(100);
   for (Int_t i = 0; i < 100; i++) {
      my->SeteAt(i, i);
      my->SetgAt(i, i);
   }

   Int_t numbers[] = {14, 3, 2, 12, 1, 0, 3, 12, 4}; 
   for (Int_t i = 0; i < 9; i++) {
      my->Setn(numbers[i]);
      tree->Fill(); 
   }

   f.Write();
   return 0;
}
