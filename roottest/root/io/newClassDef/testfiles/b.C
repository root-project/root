#include "template.h"


ClassImpTGeneric(MyTemplate)

//newClassImpT(MyTemplate)

static MyTemplate<const int*> dummy;
static R__tInit<MyTemplate<const int* > > d2(0);
static R__tInit<MyTemplate<const double* > > d3(0);

R__Setter1<const int*, MyTemplate<const int* > > a1;
R__Setter1<const int*, MyTemplate<const double* > > a2;

int f1() {
  fprintf(stderr,"d1 %s %d \n", d2.GetImplFileName(), d2.GetImplFileLine());
  fprintf(stderr,"d2 %s %d \n", d3.GetImplFileName(), d3.GetImplFileLine());
  return 0;
}



