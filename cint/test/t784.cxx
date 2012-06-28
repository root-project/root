/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <stdio.h>

typedef const char TItemName[20];
const int kShapes =7;
TItemName a[kShapes+1] = 
{"undefined","tau_C","tau","B","rho","C","y_kt","y_fJ"};

int main() {
  int i;
  for(i=0;i<kShapes+1;i++) {
    printf("%s\n",a[i]);
  }
  return 0;
}
