/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
typedef int Int_t ;

Int_t flag=0;

void borg1() {
  for (Int_t i=0;i<10;i++) {
    if( i%2==flag) {
      printf("%i: first\n",i);
      if( 1 ) continue;
    } else {
      printf("%i: second\n",i);
    }
  }
}

int main() {
  borg1();
  borg1();
  return 0;
}
