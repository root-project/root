/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
typedef double Double_t;
typedef int Int_t;

void g() {
  const Int_t SIZE=10;
  static Double_t a[SIZE] = {1,2,3,4,5};
  printf("%d\n",SIZE);
  for(int i=0;i<SIZE;i++) {
    printf("%g ",a[i]);
    a[i] = i ;
  }
  printf("\n");
}

void f() {
  Double_t  a= 12.2,b;
  Int_t c=2;
  b = Double_t(c)/a;
  printf("f() b=%g\n",b);
}

int main() {
  for(int i=0;i<3;i++) {
    g();
    f();
  }
  return(0);
}

