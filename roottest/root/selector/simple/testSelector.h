#include "TSelector.h"
#include "TDatime.h"
#include "TH1.h"
#include <cstdio>

class testSelector: public TSelector {
 public:
   testSelector();

   ClassDef(testSelector,1);
};

