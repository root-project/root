#include "TROOT.h"
#include "TClonesArray.h"

class foo: public TObject {
public:
  foo() { i = 1; f = 2*i; }
  foo(Int_t I) {i = I; f = 2*i; }
  ~foo() {}

  Int_t i;
  Float_t f;
  
  ClassDef(foo,1)
};
