#include "TInterpreter.h"

bool runtest() {
   ClassInfo_t *cl = gInterpreter->ClassInfo_Factory("testSelector");
   int  valid = gInterpreter->ClassInfo_IsValid(cl);
   long property = gInterpreter->ClassInfo_Property(cl);
   void *p =  gInterpreter->ClassInfo_New(cl);
   fprintf(stderr,"testSelector result is %d %ld %d\n",valid, property&kIsCPPCompiled,p!=0);
   gInterpreter->ClassInfo_Delete(cl);
   return p!=0;
}
