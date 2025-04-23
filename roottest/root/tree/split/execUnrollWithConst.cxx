class Content {
   int fOne;
   int fTwo;
};

class Owner {
public:
  Owner() : fValue() {}
  const Content fValue;
};

class Wrapper {
public:
  Wrapper() : fObj() {}
  Owner fObj;
};

#include "TTree.h"

void execUnrollWithConst()
{
  TTree t("T","T");
  Owner o;
  Wrapper w;
  
  t.Branch("object",&o,32000,99);
  t.Branch("wrapper",&w,32000,99);
  t.Print();
}


