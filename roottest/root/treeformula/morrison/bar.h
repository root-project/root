#include "TROOT.h"
#include "TClonesArray.h"

class bar: public TObject {
public:
  bar() {
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
    f = 0;
  }
  ~bar() override {if (f) f->Clear();}

  Int_t v[3];
  TClonesArray *f;

  ClassDefOverride(bar,1)
};
