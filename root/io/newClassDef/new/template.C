#include "template.h"
#include "TBufferFile.h"

const double dvalue = 33.3;

ClassImp( MyTemplate<const double*> )

typedef  MyPairTemplate<int, double> type001;
ClassImp( type001 )

templateClassImp(MyTemplate)
templateClassImp(MyPairTemplate)

static MyTemplate<int> dummy(1);
static MyPairTemplate<int,int> dummy2(1,2);
static MyPairTemplate<int, double> dummy3(3,4);
static MyTemplate<const double*> dummy4(&dvalue);

static MyTemplate< int> *pdummy = 0;
static MyPairTemplate<int,int> *pdummy2 = 0;
static MyPairTemplate<int, double> *pdummy3 = 0;
static MyTemplate<const double*> *pdummy4 = 0;

#include "TClass.h"

void func() {
  fprintf(stderr,"the class name of MyTemplate<int> is %s\n",
          MyTemplate<int>::Class_Name());
}
void func1() {
  fprintf(stderr,"the class name of MyTemplate<int> is %s\n",
          MyTemplate<int>::Class()->GetName());
}

void func2() {
  fprintf(stderr,"the class name of MyTemplate<int> is %s\n",
          dummy.IsA()->GetName());  
}

TBuffer* t_writetest() 
{
  TBuffer *b = new TBufferFile(TBuffer::kWrite);
  *b << &dummy;
  *b << &dummy2;
  *b << &dummy3;
  *b << &dummy4;
  return b;
}

void t_readtest(TBuffer & b) 
{
  // TBuffer b(TBuffer::kRead);
  b >> pdummy;
  if (pdummy->variable!=dummy.variable) {
     fprintf(stderr,"Error: MyTemplate<int> not read properly!");
     fprintf(stderr,"Expected %d and got %d\n", 
             dummy.variable,
             pdummy->variable);
  }
  b >> pdummy2;
  if (pdummy2->var1!=dummy2.var1) {
     fprintf(stderr,"Error: MyPairTemplate<int,int> not read properly!");
     fprintf(stderr,"Expected %d and got %d\n", 
             dummy2.var1,
             pdummy2->var1);
  }
  if (pdummy2->var2!=dummy2.var2) {
     fprintf(stderr,"Error: MyPairTemplate<int,int> not read properly!");
     fprintf(stderr,"Expected %d and got %d\n", 
             dummy2.var2,
             pdummy2->var2);
  }


  b >> pdummy3;
  if (pdummy3->var1!=dummy3.var1) {
     fprintf(stderr,"Error: MyPairTemplate<int,int> not read properly!");
     fprintf(stderr,"Expected %f and got %f\n", 
             dummy3.var1,
             pdummy3->var1);
  }
  if (pdummy3->var2!=dummy3.var2) {
     fprintf(stderr,"Error: MyPairTemplate<int,int> not read properly!");
     fprintf(stderr,"Expected %f and got %f\n", 
             dummy3.var2,
             pdummy3->var2);
  }

  b >> pdummy4;
  if (pdummy4->variable!=dummy4.variable) {
     fprintf(stderr,"Error: MyTemplate<const double*> not read properly!");
     fprintf(stderr,"Expected %f and got %f\n", 
             dummy4.variable,
             pdummy4->variable);
  }
}

void template_driver() {
  TBuffer *buf = t_writetest();
  buf->SetReadMode();
  buf->Reset();
  t_readtest(*buf);
}
