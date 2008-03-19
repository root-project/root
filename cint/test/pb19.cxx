/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

void exa2()
{
  // this should output 3 times the x array!
   short x[] = { 10,20,30,40,50 };
   int d; double y;
   short * p = x;
   int i;
   for (i = 0; i < 5; i++) {
     printf("x[0] %d x[1] %d x[2] %d ",x[0],x[1],x[2]);
     //printf("at %p : ",p);
     d = *p++;
     printf("%d\n",d);

   }
   printf("next\n");
   p = x;
   for (i = 0; i < 5; i++) {
     printf("x[0] %d x[1] %d x[2] %d ",x[0],x[1],x[2]);
     //printf("at %p : ",p);
     d = (int) *p++; 
     printf("%d\n",d);
   }
   printf("next\n");
   p = x;
   for (i = 0; i < 5; i++) {
     printf("x[0] %d x[1] %d x[2] %d ",x[0],x[1],x[2]);
     //printf("at %p : ",p);
     d = *p++;
     printf("%d\n",d);
   }
   // Also this did not work:
   y = -3.5;
   for(i=0;i<3;i++) {
     d = (int)++y;
     printf("should be -2: %d\n",d);
   }
}

int main() {
  exa2();
  return 0;
}
