/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#if defined(__CINT__) && !defined(INTERPRET)
#pragma include "test.dll"
#else
#include "t674.h"
#endif

int main() {
  int i;
  string a[3] = { "abc", "def", "hij" };
  string b[] = { "ABC", "DEF", "HIJ" };

  for(i=0;i<3;i++) printf("%d:%s,%s ",i,a[i].c_str(),b[i].c_str());
  printf("\n");

  string x = "XYZ";
  string y = "YZ";
  string z = "Z";
  string c[3] = { x,y,z };
  for(i=0;i<3;i++) printf("%d:%s ",i,c[i].c_str());
  printf("\n");

  return(0);
}
