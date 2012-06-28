/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
typedef int Int_t;

#if 0
template<class T*> void print1(Int_t N, T x, T y)
{
  for(int i=0;i<N;i+=1) {
    printf("%g  %g\n",x[i],y[i]);
  }
}
#else
template<class T> void print1(Int_t N, T* x, T* y)
{
  for(int i=0;i<N;i+=1) {
    printf("%g  %g\n",x[i],y[i]);
  }
}
#endif

template<class T> void print2(Int_t N, T* x, T* y)
{
  for(int i=0;i<N;i+=1) {
    printf("%g  %g\n",x[i],y[i]);
  }
}

int main() {
  double x[10] = { 1,2,3,4,5 };
  double y[10] = { 6,7,8,9,0 };
  //print1(5,x,y);
  print2(5,x,y);
  return 0;
}

