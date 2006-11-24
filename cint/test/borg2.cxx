/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
typedef int Int_t ;

void borg2() {
  Int_t flag=1;
  Int_t i=0;
  do  {
    i++;
    if( i%2==flag) {
      printf("%i: first\n",i);
      if( 1 ) continue;
    } else {
      printf("%i: second\n",i);
    }
  } while(i<10);
}

int main() {
  borg2();
  borg2();
  return 0;
}
