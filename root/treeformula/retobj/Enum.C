#include "TTree.h"

class WithEnum {
public:
  enum MyEnum { kOne, kTwo };
  MyEnum val;
  WithEnum() : val(kTwo) {};
  MyEnum GetVal() { return val; }
};

void Enum() {
  TTree *t = new TTree;
  WithEnum *e = new WithEnum;
  t->Branch("test",&e);
  t->Fill();
  t->Fill();
  t->Scan("test.val:test.GetVal()");
}
