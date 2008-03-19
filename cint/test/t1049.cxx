/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// t1049.cxx

#include <stdio.h>

class A{
  int x,y;
public:
  A(int a,int b) : x(a), y(b) { }
  void disp() const { printf("x=%d y=%d\n",x,y); }
};

void fit(A* i, int j) {
  int a[1]={1};
  printf("fit(%d) %d ",j,a[0]);
  i->disp();
}

void fit2(A* i, int j) {
  printf("fit2(%d) ",j);
  i->disp();
}

int main(){
  A* h[3][5]; 
  int k,m;
  for(k=0;k<3;k++) for(m=0;m<5;m++) h[k][m] = new A(k,m);
  for(k=0;k<3;k++) for(m=0;m<5;m++) fit(h[k][m],10);
  for(k=0;k<3;k++) for(m=0;m<5;m++) fit2(h[k][m],10);
  for(k=0;k<3;k++) for(m=0;m<5;m++) delete h[k][m];
  return 0;
}

