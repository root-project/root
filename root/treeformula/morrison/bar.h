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
  ~bar() {if (f) f->Clear();}

  Int_t v[3];
  TClonesArray *f;

  ClassDef(bar,1)
};
