#include "TClonesArray.h"
#include "TTree.h"
#include "TClass.h"
#include "TLeaf.h"
#include "TBranch.h"
#include <iostream>

class Holder {
public:
   std::vector<TNamed> fVector;
   TClonesArray        fArray{"TNamed"};

   void Init(int size) {
      for (int i = 0; i < size; ++i) {
         auto *n = static_cast<TNamed *>(fArray.ConstructedAt(i));
         n->SetName(TString::Format("named#%d", i));
         fVector.push_back(*n);
      }
   }
};


int execLeafFullName() {
   TClonesArray arr("TNamed", 3);
   for (int i = 0; i < 3; ++i) {
      auto *n = static_cast<TNamed *>(arr.ConstructedAt(i));
      n->SetName(TString::Format("named#%d", i));
   }
   Holder h;
   h.Init(5);

   TTree t("t", "t");
   t.Branch("arr", &arr);
   t.Branch("arrdot.", &arr);
   t.Branch("nested.", &h);
   t.Branch("vec", &h.fVector);
   t.Branch("vecdot.", &h.fVector);
   t.Branch("nested_nodot", &h);
   t.Fill();

   t.GetListOfLeaves()->ls("noaddr");
   TIter next(t.GetListOfLeaves());
   TObject *obj = nullptr;
   while( (obj = next()) ) {
      TLeaf *leaf = (TLeaf*)obj;
      //fprintf(stderr, "The fullname for %s in branch %s\n", leaf->GetName(), leaf->GetBranch()->GetFullName().Data());

      fprintf(stdout, "The fullname for %s is %s in branch %s of type %s\n", leaf->GetName(), leaf->GetFullName().Data(), leaf->GetBranch()->GetFullName().Data(), leaf->GetBranch()->IsA()->GetName() );
   }
   auto l = t.GetLeaf("arr");
   if (l != nullptr) {
      // Something went wrong.
      // In the old implementation this printed "Int_t"
      std::cout << "leaf has type name " << l->GetTypeName() << std::endl;
      l->Print();
      l->ls();
      return 1;
   }

   std::cout << "leaf is null" << std::endl; // this is what happens before this PR
   return 0;
}


