#include <stdio.h>

bool tester(int i, bool &b,int expected) {
   if (b!=expected) {
      const char *got = b ? "true" : "false";
      const char *exp = expected ? "true" : "false";
      printf("Problem at %d, expected %s but got %s\n",i,exp,got);
      return false;
   }
   return true;
}

int runReadbool() {
   bool t = 1;
   bool f = 0;
   
   int i = 0;
   tester(i++,t,1);

   tester(i++,f,0);
   
   t = true;
   tester(i++,t,1);

   f = false;
   tester(i++,f,0);

   f = !t;
   tester(i++,f,0);

   t = !f;
   tester(i++,t,1);

   if (t) t = false;
   tester(i++,t,0);

   if (!f) f = true;
   tester(i++,f,1);

   t = kTRUE;
   tester(i++,t,1);

   f = kFALSE;
   tester(i++,f,0);

#ifdef ClingWorkAroundErracticValuePrinter
   printf("(int)0\n");
#endif
   return 0;
}
