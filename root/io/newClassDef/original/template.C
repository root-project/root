#include "template.h"

//ClassImpT(MyTemplate,const int)

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


ClassImpTS(MyTemplate,const double*)

ClassImpT(MyTemplate,T)
ClassImp2T(MyPairTemplate,T,T2)

static MyTemplate<const int*> dummy;
static MyPairTemplate<int,int> dummy2(0,0);
static MyPairTemplate<int, double> dummy3(0,0);

static MyTemplate<const int*> *pdummy(0);
static MyPairTemplate<int,int> *pdummy2(0);
static MyPairTemplate<int, double> *pdummy3(0);

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
