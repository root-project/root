#include "template.h"


//newClassImpT(MyTemplate)

#define newClassImpT2(name) \
   template <> char * R__tInit<name >::fgImplFileName = __FILE__; \
   template <> int R__tInit<name >::fgImplFileLine = __LINE__;

//newClassImpT2(MyTemplate<const int*>)

static MyTemplate<const int*> dummy;
static R__tInit<MyTemplate<const int*> > d2(0);

int f2() {
  fprintf(stderr,"%s %d \n", d2.GetImplFileName(), d2.GetImplFileLine());
  return 0;
}

//optClassImp(MyTemplate<const int*>)
  //optClassImp(MyTemplate<const double*>)


  //optClassImp(MyTemplate)
