/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file main/G__cppmain.C
 ************************************************************************
 * Description:
 *  C++ version main function
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

extern "C" {
extern void G__setothermain(int othermain);
extern int G__main(int argc,char **argv);
}

int main(int argc,char **argv)
{
  G__setothermain(0);
  // coverity[fun_call_w_exception]: we don't care.
  return(G__main(argc,argv));
}

