/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#include <iostream>
using namespace std;

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1187.h"
#endif

void toto() {

  TMatrix * mat = new TMatrix(30,30);

  for (int y = 1;y<20;y++) {
    printf("Iter %d ", y);
    float number = TMath::Sqrt((*mat)(1,y));
    printf("%g\n",number);
  }
}

int main() {
  toto();
  Double_t a=1.234;
  cout << a << endl;
  return 0;
}
