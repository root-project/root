#include "testSelector.h"


testSelector::testSelector() {
#ifdef __CLING__
  fprintf(stderr,"Executed testSelector ctor from JITed code\n");
#else
   fprintf(stderr,"Executed testSelector ctor from externally compiled code\n");
#endif
}

