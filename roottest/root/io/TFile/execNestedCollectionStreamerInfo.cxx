#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "TMemFile.h"
#include <iostream>
#include <memory>

struct Inside {
  int fValue = 0;
};

struct Middle {
  std::vector<Inside> fInside;
};

struct Outer {
  std::vector<Middle> fMiddle;
};

bool CheckNestedCollectionStreamerInfo(bool empty = true)
{
   std::unique_ptr<TFile> f{ new TMemFile("forceinfo.root", "RECREATE") };
   TTree *t = new TTree("t", "t");
   Outer object;
   t->Branch("object.", &object);

   if (!empty) {
      Inside in;
      Middle m;
      m.fInside.push_back(in);
      object.fMiddle.push_back(m);
   }

   t->Fill();
   f->Write();
   auto si = f->GetStreamerInfoList()->FindObject("Inside");
   if (!si) {
     cerr << "Could not find the StreamerInfo for the class 'Inside' in the TMemFile file with the collection " 
	  << (empty ? "empty" : "with content") << "\n";
     return false;
   }
   return true;
}

int execNestedCollectionStreamerInfo()
{
   // Test both empty and filled collections as the StreamerInfo handling is different
   // in some cases (split vs non-split, object-wise streaming vs member-wise streaming).
   // In the case of the original issue (https://github.com/root-project/root/issues/11436)
   // having a filled collection did not help.
   bool success = CheckNestedCollectionStreamerInfo(true) && CheckNestedCollectionStreamerInfo(false);
   return success ? 0 : 1;
}

