/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <stdio.h>

//long long x;

typedef long long Long64_t;
typedef unsigned long long  ULong64_t;


int main() {
   printf("%d\n",(int)sizeof(Long64_t));
   printf("%d\n",(int)sizeof(ULong64_t));
  return 0;
}

