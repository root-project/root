/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
typedef int Int_t;

int x=0;

class TTRAP {
 int a;
public:
  TTRAP() { a=x++; }
  void disp() { printf("%d\n",a); }
};

void f() {
  Int_t i,k;
  printf("f-----------------\n");
  const Int_t nSec=2;
  const Int_t nPads=3;
  TTRAP *** trap;
  trap = new TTRAP** [nSec];
  for (i=0;i<nSec;i++){
    trap[i] = new TTRAP* [nPads];
  }
  
  for (k=0;k<nSec;k++){
    for (i=0;i<nPads;i++){
      trap[k][i] = new TTRAP();
      trap[k][i]->disp();
      trap[k][i][0].disp();
    }
  }
  for (k=0;k<nSec;k++){
    for (i=0;i<nPads;i++){
      delete trap[k][i];
    }
  }
  for (i=0;i<nSec;i++){
    delete[] trap[i];
  }
  delete[] trap;
}

void ff() {
  printf("ff-----------------\n");
  TTRAP y[7],x;
  x.disp();
  TTRAP *** trap;
  trap = new TTRAP** [3];
  trap[0] = new TTRAP* [4];
  TTRAP *p = new TTRAP[2];
  trap[0][0] = p;
  trap[0][0][0].disp();
  trap[0][0][1].disp();
  trap[0][0]->disp();
  p->disp();
  trap[0][0][1] = x;
  trap[0][0][0].disp();
  trap[0][0][1].disp();

  delete[] p;
  delete[] trap[0];
  delete[] trap;
}

int main() {
  ff();
  f();
  return 0;
}
