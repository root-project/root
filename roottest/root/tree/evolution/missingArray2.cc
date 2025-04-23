// testclass.cc
#define REQUIRED_VERSION 2
#include "myclass.cc" 
#include "TFile.h" 
#include "TTree.h" 
int missingArray2() 
{ 
   TFile f("missingArray.root"); 
   TTree *tree = (TTree*)f.FindObjectAny("tree"); 

   myclass *my = new myclass();
   tree->SetBranchAddress("mybranch", &my); 

   Int_t numbers[] = {14, 3, 2, 12, 1, 0, 3, 12, 4}; 

   for (Int_t i = 0; i < 9; i++) {
      tree->GetEntry(i);
      if (my->Getn() == 0) {
         my->Sete(0);
         my->Setf(0);
      }
      my->SeteSize(numbers[i]);
      my->SetfSize(numbers[i]);
      for (Int_t j = 0; j < my->Getn(); j++) {
         my->SeteAt(j, j);
      }
      for (Int_t j = 0; j < my->Getn(); j++) {
         my->SetfAt(j, j);
      }
   }
   return 0;
}
