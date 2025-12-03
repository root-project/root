#include "tmplt.h"

class Holder {
public:
  Wrapper<Double32_t> fValue;
 
  virtual ~Holder() {}
  ClassDef(Holder,4);
};

#ifdef __ROOTCLING__
#pragma link C++ class Wrapper<Double32_t>+;
#endif

int execTmpltD32()
{
   writeFile<Holder>("tmpltd32.root");
   return 0;
}

