#include <vector>
#include "TTree.h"


void runsimple(Int_t len = 10, Int_t vlen = 3) {
   TTree *tree = new TTree("tree","simple tree");
   
   std::vector<int> vec;
   Int_t value;

   tree->Branch("vec",&vec);
   tree->Branch("value",&value);

   for(Int_t i = 0; i<len; ++i) {

      vec.clear();

      value = i;
      for(Int_t j = 0; j<vlen; ++j) {
         vec.push_back( i*vlen*10 + j );
      }

      tree->Fill();

   }

   tree->SetScanField(0);
   tree->Scan();

   tree->ResetBranchAddresses();

   tree->Scan();
}
