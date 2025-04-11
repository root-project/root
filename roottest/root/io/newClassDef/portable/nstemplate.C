#include "nstemplate.h"

//ClassImpT(MyTemplate,const int)

#if 0
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
#endif

ClassImpT(MySpace::MyTemplate,T)

namespace MySpace {

  //ClassImpTS(MyTemplate,const double*)

  MyTemplate<const int*> dummy; // this prevents rootcint from working on vector!?

}

