namespace edm {
  template <int I> class Hash {};
  const int typeN =1;
  typedef Hash<typeN + 1> ParentageID;
}

#ifdef __ROOTCLING__
#pragma link C++ class edm::ParentageID+;
#endif

#include "TClass.h"
int execTemplateExpr() {
  TClass *cl = TClass::GetClass("edm::ParentageID");
  if (!cl) {
     printf("Can't find edm::ParentageID's TClass\n");
  }
  printf("Parentage's name is %s\n",cl->GetName());
  return 0;
}
