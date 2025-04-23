#include "template.h"
#include "TBufferFile.h"

//ClassImpT(MyTemplate,const int)

#ifdef _ClassInit_
// Old Style
#define _ClassImpTS_(name,Tmpl) \
   TClass *name<Tmpl>::Class() \
      { if (!fgIsA) name<Tmpl>::Dictionary(); return fgIsA; } \
   const char *name<Tmpl>::ImplFileName() \
      { return __FILE__; } \
   int name<Tmpl>::ImplFileLine() { return __LINE__; } \
   TClass *name<Tmpl>::fgIsA = 0;

#define ClassImpTS(name,Tmpl) \
   void name<Tmpl>::Dictionary() { \
      fgIsA = CreateClass(Class_Name(),   Class_Version(), \
                          DeclFileName(), ImplFileName(), \
                          DeclFileLine(), ImplFileLine()); \
   } \
   _ClassImpTS_(name,Tmpl)
#else
// New style
#define ClassImpTS(name,Tmpl) templateClassImp( name< Tmpl > )
#endif

ClassImpTS(MyTemplate,const double*)

ClassImpT(MyTemplate,T)
ClassImp2T(MyPairTemplate,T,T2)

const double dvalue = 33.3;

static MyTemplate<int> dummy(1);
static MyPairTemplate<int,int> dummy2(1,2);

static MyTemplate<const double*> dummy3(&dvalue);

static MyTemplate<int> *pdummy = 0;
static MyPairTemplate<int,int> *pdummy2 = 0;

static MyTemplate<const double*> *pdummy3 = 0;

TBuffer* t_writetest() 
{
  TBuffer *b = new TBufferFile(TBuffer::kWrite);
  *b << &dummy;
  *b << &dummy2;
  *b << &dummy3;
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

  // Operator>> is screwer for this example! b >> pdummy3;
  pdummy3 = (MyTemplate<const double*> *) b.ReadObject(MyTemplate<const double*>::Class());
  if (pdummy3->variable!=dummy3.variable) {
     fprintf(stderr,"Error: MyTemplate<const double*> not read properly!");
     fprintf(stderr,"Expected %f and got %f\n", 
             dummy3.variable,
             pdummy3->variable);
  }
}

void template_driver() {
  TBuffer *buf = t_writetest();
  buf->SetReadMode();
  buf->Reset();
  t_readtest(*buf);
}
