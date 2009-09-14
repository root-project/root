#include "TTree.h"



TTree* ternary() {

   TTree *tree = new TTree("T","T");
   float cond, first, second;
   
   tree->Branch("cond", &cond);
   tree->Branch("first",&first);
   tree->Branch("second",&second);
   for(int j=0;j<5;++j) {
      cond = j;
      first = 100 + j;
      second = -j;
      tree->Fill();
      
   }
   tree->Scan("cond:first:second:cond<2?first:second","","col=:::20");
   return tree;
}

int runternary() {
   ternary();
   return 0;
}