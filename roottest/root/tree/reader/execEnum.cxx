#include "TTree.h"
#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include <vector>
#include <iostream>
#include <memory>

enum ESomeEnum {
  kOneValue = 1,
  kTwoValue = 2
};

struct Holder {
  ESomeEnum value;
  std::vector<ESomeEnum> vec;
};

#ifdef __ROOTCLING__
#pragma link C++ enum ESomeEnum;
#pragma link C++ class std::vector<ESomeEnum>+;
#pragma link C++ class Holder+;
#endif

TTree *CreateTree()
{
   auto tree = new TTree("t","t");
   Holder h;
   h.value = kTwoValue;
   h.vec.push_back(kTwoValue);
   tree->Branch("obj", &h);
   tree->Fill();
   tree->ResetBranchAddresses();
   return tree;
}

int execEnum() {
   std::unique_ptr<TTree> tree { CreateTree() };

   TTreeReader     fReader(tree.get());

   TTreeReaderValue<ESomeEnum> test_someEnum = {fReader, "value"};
   TTreeReaderArray<ESomeEnum> test_anotherEnum = {fReader, "vec"};

   if (! fReader.Next()) {
     std::cout << "Error could not read/load the first entry\n";
     return 1;
   }

   if (*test_someEnum != 2) {
     std::cout << "Error in reading 'value' expected 2 and got " << *test_someEnum << '\n';
     return 2;
   }
   if (test_anotherEnum.GetSize() != 1) {
     std::cout << "Error in reading 'vec' the size is: " << test_anotherEnum.GetSize() << '\n';
     return 3;
   }
   if (test_anotherEnum[0] != 2) {
     std::cout << "Error in reading 'vec' expected 2 and got " << test_anotherEnum[1] << '\n';
     return 4;
   }
   return 0;
}
