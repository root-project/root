/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

int tree(int x)
{
   static int y=0;
   static int z=0;
   static int qq = printf("%5s","*");

   x = x>0 ? -9: x;
   z = (z=x+5)>0 ? z:-z;
   /* printf("qq=%d\n",qq); */
   printf(!x&&++y ? "\n":(z?(z>y%3+y/3?" ":(x<-5 ?"/" :"\\")):"|"));
   return (y-9) ? tree(++x) : printf("  _|_|_\n  \\___/\n");
}

int main() {
  tree(0);
  return 0;
}
