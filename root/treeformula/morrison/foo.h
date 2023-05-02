#include "TROOT.h"
#include "TClonesArray.h"

int run();

class foo: public TObject {
public:
  foo() { i = 1; f = 2*i; }
  foo(Int_t I) {i = I; f = 2*i; }
  ~foo() override {}

  Int_t i;
  Float_t f;
  
  static int run() { return ::run(); }
  ClassDefOverride(foo,1)
};

