#include "Api.h"

bool runtest() {
   {
      G__ClassInfo cl("testSelector");
      int  valid = cl.IsValid();
      long property = cl.Property();
      void *p = cl.New();
      fprintf(stderr,"testSelector result is %d %d %d\n",valid, property&G__BIT_ISCPPCOMPILED,p!=0);
      return p!=0;
   }
}
