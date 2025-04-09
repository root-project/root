#include "class1.h"

int execInterpClassNew()
{
   gROOT->ProcessLine(".L classlib.cxx+");
   TClass * c = TClass::GetClass("class1");
   if (!c) {
     fprintf(stderr,"Error: Could retrieve the TClass for class1\n");
     return 0;
   }
   if (c->IsLoaded()) {
     fprintf(stderr,"Error: The TClass for class1 is marker as loaded.n");
     return 0;
   }
   void *p = c->New();
   if (!p) {
     fprintf(stderr,"Error: Could not create an object of type class1.\n");
     return 1;
  }
  return 0;
}
