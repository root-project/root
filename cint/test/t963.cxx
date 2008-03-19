/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 021211rootcint.txt, Maarten 
// 021214rootcint.txt
#include "t963.h"

int main() {
  TPhTopoDataI a;
  TPhTopoDataI b(a);
  TPhTopoDataI::TError e;
  a.disp();
  b.disp();
  e.disp();
  return 0;
}
