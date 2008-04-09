/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// aryinit1.cxx
#include <iostream>
using namespace std;

int main() 
{
  int a[] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };
  const int ASIZE = sizeof(a)/sizeof(int); 

  int i;
  cout << "ASIZE = " << ASIZE << endl;
  for(i=0;i<ASIZE;++i) { 
    cout << a[i] << " ";
  }
  cout << endl;

  return(0);
}

