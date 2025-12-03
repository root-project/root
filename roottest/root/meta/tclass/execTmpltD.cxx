#include "tmplt.h"

class Holder {
public:
  Wrapper<double> fValue;
  Wrapper<long long> fIntValue;
  virtual ~Holder() {}
  ClassDef(Holder,3);
};

#ifdef __ROOTCLING__
#pragma link C++ class Wrapper<double>+;
#pragma link C++ class Wrapper<long long>+;
#endif

int execTmpltD()
{
   writeFile<Holder>("tmpltd.root");
   return 0;
}

