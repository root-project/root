#include "TTree.h"

class WithEnum {
public:
   enum MyEnum { kOne, kTwo };
   MyEnum val;
   WithEnum() : val(kTwo) {};
#ifdef ClingWorkAroundCallfuncAndInline
   MyEnum GetVal();   
#else
   MyEnum GetVal() { return val; }
#endif
};

#ifdef ClingWorkAroundCallfuncAndInline
WithEnum::MyEnum WithEnum::GetVal() { return val; }
#endif

void Enum() {
   TTree *t = new TTree;
   WithEnum *e = new WithEnum;
   t->Branch("test",&e);
   t->Fill();
   t->Fill();
#ifdef ClingWorkAroundCallfuncAndInline2
   t->Scan("test.val");
#else
   t->Scan("test.val:test.GetVal()");
#endif
}
