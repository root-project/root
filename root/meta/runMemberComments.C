#include "TError.h"

// from ROOT-5660
class baseCl {
public: 
   virtual int fGetIndex(Int_t aIndex=0) { return 42; } // Title Base
};
class childCl : baseCl {
public:
   virtual int fGetIndex(Int_t aIndex); // Title Derived
};

void runMemberComments() {
   printf("\nTH1F::Class()->GetListOfAllPublicDataMembers():\n");
   TH1F::Class()->GetListOfAllPublicDataMembers()->ls("noaddr");

   printf("\nTArrow::Class()->GetListOfAllPublicMethods():\n");
   TArrow::Class()->GetListOfAllPublicMethods()->ls("noaddr");

   TList menuItems;
   printf("\nTH1::Class()->GetMenuItems():\n");
   TH1::Class()->GetMenuItems(&menuItems);
   menuItems.ls("noaddr");

   printf("\nchildCl::Class()->GetListOfAllPublicMethods():\n");
   const TCollection* pubMeth =  TClass::GetClass("childCl")->GetListOfAllPublicMethods();
   TIter iPubMeth(pubMeth);
   TMethod* meth = 0;
   while ((meth = (TMethod*)iPubMeth())) {
      printf("childCl::%s%s // %s\n", meth->GetName(), meth->GetSignature(),
             meth->GetTitle());
   }
}
