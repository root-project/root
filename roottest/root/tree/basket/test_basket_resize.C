#include "TTree.h"
#include "TMemFile.h"
#include "TBranch.h"
#include "TBasket.h"
#include <iostream>

void test_basket_resize() {
   TMemFile f("test_resize.root", "RECREATE");
   TTree tree("T", "Tree for testing basket resizing");

   int value;
   // Set small initial basket size to force automatic buffer resizing
   TBranch *branch = tree.Branch("b", &value, "value/I", 64); 
   for (int i = 0; i < 100; ++i) {
      value = i;
      tree.Fill();
   }
   tree.Write();
   TBasket *basket = branch->GetBasket(0);
   if (basket) {
      Int_t newSize = basket->GetBufferSize();
      std::cout << "Basket buffer size: " << newSize << std::endl;
   }
   f.Close();
}
