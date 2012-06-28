/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

class TA {
public:
  void Bugged(const double x) const; };
void TA::Bugged(const double x) const {
  printf("x:%f\n",x);
  if (x<15           ) return;
  if (x>80           ) return;
  int    i   = 0;        // Removing this line solve the problem
  printf("OK x:%f\n",x); // This line is never reached
}

void bug() {
  TA a;
  int i;
  for (i=0;i<10;i++) a.Bugged(i*10);
}

int main() { 
  bug(); 
  return 0;
}
