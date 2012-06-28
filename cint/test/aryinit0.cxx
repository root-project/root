/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// aryinit0.cxx
#include <iostream>
using namespace std;

int main() 
{
  const int ASIZE=10; 
  int a[ASIZE] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };

  int i;
  for(i=0;i<ASIZE;++i) { 
    cout << a[i] << " ";
  }
  cout << endl;

  return(0);
}

