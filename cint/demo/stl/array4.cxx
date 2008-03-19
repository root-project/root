/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream.h>
#include <algo.h>

int main()
{
  const int N=10;
  int vector1[N];
  for(int k=0;k!=N;++k) vector1[k]=rand()%10;
  for(int k=0;k!=N;++k) cout << vector1[k] << " ";
  cout << endl;

  sort(&vector1[0],&vector1[N]);
  
  for(int k=0;k!=N;++k) cout << vector1[k] << " ";
  cout << endl;
}

