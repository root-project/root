#include "./Rtypes.h"
#include "template.h"
#include "TClass.h"

//ClassImpT(MyTemplate,const int)

ClassImp(MyTemplate<const double*> )

ClassImpT(MyTemplate,T)
  //ClassImp2T(MyPairTemplate,T,T2)
ClassImpT(MyPairTemplate,T)

static MyTemplate<const int*> dummy;
static MyPairTemplate<int,int> dummy2(0,0);
static MyPairTemplate<int, double> dummy3(0,0);

static MyTemplate<const int*> *pdummy(0);
static MyPairTemplate<int,int> *pdummy2(0);
static MyPairTemplate<int, double> *pdummy3(0);

void func() {
  fprintf(stderr,"the class name of MyTemplate<const int*> is %s\n",
          MyTemplate<const int*>::Class_Name());
}
void func1() {
  fprintf(stderr,"the class name of MyTemplate<const int*> is %s\n",
          MyTemplate<const int*>::Class()->GetName());
}

void func2() {
  fprintf(stderr,"the class name of MyTemplate<const int*> is %s\n",
          dummy.IsA()->GetName());  
}

void writetest() 
{
  TBuffer b(TBuffer::kWrite);
  b << pdummy;
  b << pdummy2;
  b << pdummy3;
}

void readtest() 
{
  TBuffer b(TBuffer::kRead);
  b >> pdummy;
  b >> pdummy2;
  b >> pdummy3;
}
