#include "TError.h"

// from ROOT-5660
class baseCl {
public:
   baseCl() {}
   baseCl(const baseCl&) {}
#if __cplusplus >= 201103L
   baseCl &operator=(baseCl&&) = delete;
#endif
public:
   virtual int fGetIndex(Int_t aIndex=0) { return 42; } // Title Base
};

class basePrCl {
public:
   basePrCl() {}
   basePrCl(const basePrCl&) {}
#if __cplusplus >= 201103L
   basePrCl &operator=(basePrCl&&) = delete;
#endif
public:
   virtual int fGetPrIndex(Int_t aIndex=0) { return 52; } // Title Base
};

class childCl : private basePrCl, public baseCl {
public:
   childCl() {}
   childCl(const childCl&) {}
   virtual int fGetIndex(Int_t aIndex); // Title Derived
};

void runMemberComments() {
   printf("\nTH1F::Class()->GetListOfAllPublicDataMembers():\n");
   TH1F::Class()->GetListOfAllPublicDataMembers()->ls("noaddr");

   printf("\nTArrow::Class()->GetListOfAllPublicMethods():\n");
   TList arrowPubMeths;
   for (TObject* pubMeth: *TArrow::Class()->GetListOfAllPublicMethods()) {
      arrowPubMeths.AddLast(pubMeth);
   }
   arrowPubMeths.Sort();
   arrowPubMeths.ls("noaddr");

   TList menuItems;
   printf("\nTH1::Class()->GetMenuItems():\n");
   TH1::Class()->GetMenuItems(&menuItems);
   menuItems.ls("noaddr");

   printf("\nchildCl::Class()->GetListOfAllPublicMethods():\n");
   const TCollection* pubMeth =  TClass::GetClass("childCl")->GetListOfAllPublicMethods();
   TIter iPubMeth(pubMeth);
   TMethod* meth = 0;
   while ((meth = (TMethod*)iPubMeth())) {
      // skip C++11 move c'tor, move op=
      if (strstr(meth->GetSignature(), "&&")) continue;
      printf("%s::%s%s // %s\n", meth->GetClass()->GetName(), meth->GetName(), meth->GetSignature(),
             meth->GetTitle());
   }
}
