/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
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

int main() {
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

   return 0;
}
